#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include <nccl_device/gin/proxy/gin_proxy.h>

#include "common/checks.hpp"
#include "common/mpi.hpp"
#include "common/parse.hpp"
#include "common/stats.hpp"
#include "common/timer.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace nccl_deviceapi_test;

constexpr int kThreads = 32;
constexpr int kGinResourceCount = 16;
constexpr ncclGinSignal_t kPingSignal = 0;
constexpr ncclGinSignal_t kPongSignal = 1;

enum Phase : int {
  kPhaseGpuMetadataLookup = 0,
  kPhaseGpuPrepareEnqueueGfd = 1,
  kPhaseProxyToCreditObserved = 2,
  kPhaseRemoteWaitSignalSatisfied = 3,
  kPhaseCount = 4,
};

constexpr const char* kPhaseNames[kPhaseCount] = {
    "phase1_gpu_metadata_lookup",
    "phase2_gpu_prepare_enqueue_gfd",
    "phase3_to_phase7_proxy_to_cq_credit_observed",
    "phase8_remote_wait_signal_satisfied",
};

struct Options {
  size_t bytes = 64;
  int warmup = 100;
  int iters = 1000;
  int device = -1;
};

struct ProxyPostResult {
  ncclGinProxyGpuCtx_t* proxy_ctx;
  int gin_peer;
  uint32_t next_gfd_idx;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--bytes N] [--warmup N] [--iters N] [--device ID]\n"
               "  Breaks down the observable NCCL GIN proxy put+SignalInc path.\n"
               "  Requires exactly 2 MPI ranks and NCCL_GIN_TYPE=2/proxy.\n",
               argv0);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        usage(argv[0]);
        std::exit(3);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--bytes") == 0) {
      opt.bytes = parse_size(need(argv[i]));
    } else if (std::strcmp(argv[i], "--warmup") == 0 || std::strcmp(argv[i], "--warmup-iters") == 0) {
      opt.warmup = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      opt.iters = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--device") == 0) {
      opt.device = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument(std::string("unknown argument: ") + argv[i]);
    }
  }
  if (opt.bytes == 0 || opt.bytes > nccl::gin::proxy::DataChunkSize || opt.warmup < 0 || opt.iters <= 0) {
    throw std::invalid_argument("invalid benchmark options; bytes must be in (0, 1GB]");
  }
  return opt;
}

__global__ void fill_kernel(char* sendbuf, char* recvbuf, size_t bytes, int rank) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = idx; i < bytes; i += stride) {
    sendbuf[i] = static_cast<char>((i + rank * 17) & 0xff);
    recvbuf[i] = 0;
  }
}

__device__ __forceinline__ ProxyPostResult manual_proxy_put_signal(ncclGin const& gin, ncclTeam team, int peer,
                                                                   ncclWindow_t dstwin, ncclWindow_t srcwin,
                                                                   size_t bytes, ncclGinSignal_t signal,
                                                                   uint64_t* samples, int iters, int iter,
                                                                   bool record) {
  using nccl::gin::internal::teamRankToGinRank;
  using nccl::utility::loadConst;

  uint64_t t0 = global_timer();

  ncclGinCtx ctx = gin._makeCtx();
  int gin_peer = teamRankToGinRank(gin.comm, team, peer);
  auto* proxy_ctx = &reinterpret_cast<ncclGinProxyGpuCtx_t*>(ctx.handle)[ctx.contextId];
  ncclGinWindow_t dst_gin_win = loadConst(&dstwin->ginWins[gin.connectionId]);
  ncclGinWindow_t src_gin_win = loadConst(&srcwin->ginWins[gin.connectionId]);
  size_t dst_off = 4096 * static_cast<size_t>(loadConst(&dstwin->ginOffset4K));
  size_t src_off = 4096 * static_cast<size_t>(loadConst(&srcwin->ginOffset4K));
  bool is_strong_signal = gin.comm.ginStrongLegacySignals;

  uint64_t t1 = global_timer();

  ncclGinProxyGfd_t gfd;
  ncclGinProxyOp_t op;
  nccl::gin::proxy::constructProxyOp(op, /*isGet=*/false, /*isFlush=*/false, /*hasInline=*/false,
                                     NCCL_GIN_SIGNAL_TYPE_INDEXED, ncclGinSignalInc, /*hasCounter=*/false);
  nccl::gin::proxy::buildGfd(&gfd, op, /*srcVal=*/uint64_t{0}, /*hasInline=*/false, src_off, src_gin_win,
                             dst_off, dst_gin_win, bytes, /*counterId=*/0, signal, /*signalVal=*/1,
                             /*signalWindow=*/nullptr, /*signalOff=*/0, is_strong_signal);

  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pi(loadConst(&proxy_ctx->pis)[gin_peer]);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ci(loadConst(&proxy_ctx->cis)[gin_peer]);
  ncclGinProxyGfd_t* queue = &loadConst(&proxy_ctx->queues)[gin_peer * proxy_ctx->queueSize];
  uint32_t queue_size = loadConst(&proxy_ctx->queueSize);
  uint32_t idx = pi.fetch_add(1, cuda::memory_order_relaxed);
  while (queue_size <= idx - ci.load(cuda::memory_order_relaxed)) {
  }
  uint32_t gfd_idx = idx & (queue_size - 1);

  NVCC_PRAGMA_UNROLL_AUTO
  for (uint8_t i = 0; i < sizeof(ncclGinProxyGfd_t) / sizeof(uint4); i++) {
    __stwt((uint4*)&queue[gfd_idx] + i, ((uint4*)&gfd)[i]);
  }

  uint64_t t2 = global_timer();

  if (record && samples != nullptr) {
    samples[static_cast<size_t>(kPhaseGpuMetadataLookup) * iters + iter] = t1 - t0;
    samples[static_cast<size_t>(kPhaseGpuPrepareEnqueueGfd) * iters + iter] = t2 - t1;
  }

  return {proxy_ctx, gin_peer, idx + 1};
}

__device__ __forceinline__ void wait_proxy_credit(ProxyPostResult const& post, uint64_t* samples, int iters, int iter,
                                                  bool record) {
  uint64_t t0 = global_timer();
  nccl::gin::proxy::waitForGfdComplete(post.proxy_ctx, post.gin_peer, post.next_gfd_idx,
                                       cuda::memory_order_acquire, nullptr);
  uint64_t t1 = global_timer();
  if (record && samples != nullptr) {
    samples[static_cast<size_t>(kPhaseProxyToCreditObserved) * iters + iter] = t1 - t0;
  }
}

__global__ void proxy_phase_breakdown_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin, ncclDevComm dev_comm,
                                             size_t bytes, int warmup, int iters, uint64_t* samples) {
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclCoopWarpSpan warp(0, 1, 0);
  int peer = world.rank == 0 ? 1 : 0;

  __shared__ uint64_t signal_bases[2];
  if (threadIdx.x == 0) {
    signal_bases[0] = gin.readSignal(kPingSignal);
    signal_bases[1] = gin.readSignal(kPongSignal);
  }
  __syncthreads();

  auto one_iter = [&](int iter, bool record) {
    uint64_t ping_expected = signal_bases[0] + static_cast<uint64_t>(iter + 1);
    uint64_t pong_expected = signal_bases[1] + static_cast<uint64_t>(iter + 1);

    if (world.rank == 0) {
      if (threadIdx.x == 0) {
        ProxyPostResult post = manual_proxy_put_signal(gin, world, peer, recvwin, sendwin, bytes, kPingSignal,
                                                       samples, iters, iter, record);
        wait_proxy_credit(post, samples, iters, iter, record);
      }
      __syncwarp();
      gin.waitSignal(warp, kPongSignal, pong_expected, 64);
    } else {
      gin.waitSignal(warp, kPingSignal, ping_expected, 64);
      uint64_t t0 = 0;
      if (threadIdx.x == 0 && record) t0 = global_timer();
      gin.waitSignal(warp, kPingSignal, ping_expected, 64);
      if (threadIdx.x == 0 && record) {
        uint64_t t1 = global_timer();
        samples[static_cast<size_t>(kPhaseRemoteWaitSignalSatisfied) * iters + iter] = t1 - t0;
      }
      if (threadIdx.x == 0) {
        ProxyPostResult post = manual_proxy_put_signal(gin, world, peer, recvwin, sendwin, bytes, kPongSignal,
                                                       nullptr, iters, iter, false);
        wait_proxy_credit(post, nullptr, iters, iter, false);
      }
    }
  };

  for (int i = 0; i < warmup; i++) {
    one_iter(i, false);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    signal_bases[0] = gin.readSignal(kPingSignal);
    signal_bases[1] = gin.readSignal(kPongSignal);
  }
  __syncthreads();

  for (int i = 0; i < iters; i++) {
    one_iter(i, true);
  }
}

std::vector<double> convert_phase_ticks_to_us(const std::vector<uint64_t>& ticks,
                                              int phase, int iters, double ticks_per_us) {
  std::vector<double> values;
  values.reserve(iters);
  const size_t base = static_cast<size_t>(phase) * iters;
  for (int i = 0; i < iters; i++) {
    values.push_back(static_cast<double>(ticks[base + i]) / ticks_per_us);
  }
  return values;
}

void print_phase_stats(const char* name, const char* scope, bool measured, const std::vector<double>& values,
                       const char* note) {
  std::printf("%s,%s,%s,", name, scope, measured ? "yes" : "no");
  if (measured) {
    MetricStats s = compute_stats(values);
    std::printf("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n", s.min, s.p50, s.p90, s.p99, s.max, s.avg, note);
  } else {
    std::printf("nan,nan,nan,nan,nan,nan,%s\n", note);
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool mpi_initialized = false;
  try {
    Options opt = parse_args(argc, argv);

    MPI_CHECK(MPI_Init(&argc, &argv));
    mpi_initialized = true;

    int rank = 0;
    int nranks = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    if (nranks != 2) {
      if (rank == 0) {
        std::cerr << "gin_proxy_phase_breakdown requires exactly 2 ranks, got " << nranks << "\n";
      }
      MPI_CHECK(MPI_Finalize());
      return 2;
    }

    int dev = opt.device >= 0 ? opt.device : mpi_local_rank(MPI_COMM_WORLD);
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0 || dev >= dev_count) {
      throw std::runtime_error("selected CUDA device is not visible");
    }
    CUDA_CHECK(cudaSetDevice(dev));

    ncclUniqueId id = mpi_bcast_nccl_unique_id(rank, MPI_COMM_WORLD);
    ncclComm_t comm = nullptr;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    if (!props.deviceApiSupport) {
      throw std::runtime_error("NCCL device API is not supported by this communicator");
    }
    if (props.ginType != NCCL_GIN_TYPE_PROXY) {
      throw std::runtime_error("this benchmark only supports NCCL_GIN_TYPE_PROXY");
    }

    char* sendbuf = nullptr;
    char* recvbuf = nullptr;
    ncclWindow_t sendwin = nullptr;
    ncclWindow_t recvwin = nullptr;
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&sendbuf), opt.bytes));
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&recvbuf), opt.bytes));
    NCCL_CHECK(ncclCommWindowRegister(comm, sendbuf, opt.bytes, &sendwin, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, recvbuf, opt.bytes, &recvwin, NCCL_WIN_COLL_SYMMETRIC));

    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = kGinResourceCount;
    reqs.ginContextCount = 1;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
    reqs.ginForceEnable = true;
#endif
    ncclDevComm dev_comm{};
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    int fill_blocks = static_cast<int>(std::min<size_t>((opt.bytes + 255) / 256, 1024));
    fill_blocks = std::max(fill_blocks, 1);
    fill_kernel<<<fill_blocks, 256, 0, stream>>>(sendbuf, recvbuf, opt.bytes, rank);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    double ticks_per_us = calibrate_ticks_per_us();

    uint64_t* d_samples = nullptr;
    const size_t sample_count = static_cast<size_t>(kPhaseCount) * opt.iters;
    CUDA_CHECK(cudaMalloc(&d_samples, sample_count * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_samples, 0, sample_count * sizeof(uint64_t)));

    proxy_phase_breakdown_kernel<<<1, kThreads, 0, stream>>>(sendwin, recvwin, dev_comm, opt.bytes,
                                                             opt.warmup, opt.iters, d_samples);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<uint64_t> h_samples(sample_count);
    CUDA_CHECK(cudaMemcpy(h_samples.data(), d_samples, sample_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    std::vector<double> rank1_phase8_us(opt.iters);
    if (rank == 1) {
      std::vector<double> local_phase8 = convert_phase_ticks_to_us(h_samples, kPhaseRemoteWaitSignalSatisfied,
                                                                   opt.iters, ticks_per_us);
      MPI_CHECK(MPI_Send(local_phase8.data(), opt.iters * sizeof(double), MPI_BYTE, 0, 1008,
                         MPI_COMM_WORLD));
    } else {
      MPI_CHECK(MPI_Recv(rank1_phase8_us.data(), opt.iters * sizeof(double), MPI_BYTE, 1, 1008,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    if (rank == 0) {
      std::printf("# gin_proxy_phase_breakdown bytes=%zu warmup=%d iters=%d ranks=2 ginType=%d\n",
                  opt.bytes, opt.warmup, opt.iters, props.ginType);
      std::printf("# measured GPU rows are direct globaltimer samples.\n");
      std::printf("# phase3_to_phase7_proxy_to_cq_credit_observed is one combined wait for local GFD credit return;\n");
      std::printf("# splitting CPU proxy poll/copy, WR build, ibv_post_send, network, and CQ requires NCCL host-proxy instrumentation.\n");
      std::printf("phase,scope,measured,min_us,p50_us,p90_us,p99_us,max_us,avg_us,note\n");
      print_phase_stats(kPhaseNames[kPhaseGpuMetadataLookup], "rank0_gpu", true,
                        convert_phase_ticks_to_us(h_samples, kPhaseGpuMetadataLookup, opt.iters, ticks_per_us),
                        "direct_gpu_sample");
      print_phase_stats(kPhaseNames[kPhaseGpuPrepareEnqueueGfd], "rank0_gpu", true,
                        convert_phase_ticks_to_us(h_samples, kPhaseGpuPrepareEnqueueGfd, opt.iters, ticks_per_us),
                        "includes_gfd_build_queue_credit_wait_and_write_through_publish");
      print_phase_stats(kPhaseNames[kPhaseProxyToCreditObserved], "rank0_gpu_observed", true,
                        convert_phase_ticks_to_us(h_samples, kPhaseProxyToCreditObserved, opt.iters, ticks_per_us),
                        "combined_cpu_proxy_wr_post_network_cq_credit");
      print_phase_stats("phase3_cpu_proxy_poll_copy_gfd", "host_proxy", false, {},
                        "requires_timestamps_in_nccl_src_gin_gin_host_proxy_cc");
      print_phase_stats("phase4_cpu_translate_gfd_to_chained_ib_wrs", "host_proxy", false, {},
                        "requires_timestamps_around_rmaBackend_iputSignal");
      print_phase_stats("phase5_ibv_post_send_nic_doorbell", "host_proxy_verbs", false, {},
                        "requires_timestamps_in_nccl_src_transport_net_ib_gin_cc");
      print_phase_stats("phase6_network_transfer", "nic_network", false, {},
                        "not_directly_visible_without_NIC_or_remote_timestamp");
      print_phase_stats("phase7_cq_completion_gfd_credit_update", "host_proxy", false, {},
                        "included_only_in_phase3_to_phase7_combined_observation");
      print_phase_stats(kPhaseNames[kPhaseRemoteWaitSignalSatisfied], "rank1_gpu", true, rank1_phase8_us,
                        "satisfied_waitSignal_poll_cost_after_signal_visible");
    }

    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    NCCL_CHECK(ncclCommWindowDeregister(comm, sendwin));
    NCCL_CHECK(ncclCommWindowDeregister(comm, recvwin));
    NCCL_CHECK(ncclMemFree(sendbuf));
    NCCL_CHECK(ncclMemFree(recvbuf));
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    if (mpi_initialized) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
  }
}

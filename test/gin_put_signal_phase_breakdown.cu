#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include <nccl_device/gin/gdaki/gin_gdaki.h>
#include <transport/net_ib/gdaki/doca-gpunetio/include/device/doca_gpunetio_dev_verbs_onesided.cuh>

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
  kPhaseMetadataLookup = 0,
  kPhaseReserveWqeSlots = 1,
  kPhaseConstructWqes = 2,
  kPhaseMarkWqesReady = 3,
  kPhaseDoorbellSubmit = 4,
  kPhaseRemotePollSignal = 5,
  kPhaseCount = 6,
};

constexpr const char* kPhaseNames[kPhaseCount] = {
    "phase1_metadata_lookup",
    "phase2_reserve_wqe_slots",
    "phase3_construct_data_signal_wqe",
    "phase4_mark_data_signal_wqe_ready",
    "phase5_doorbell_submit",
    "phase7_remote_poll_signal",
};

struct Options {
  size_t bytes = 64;
  int warmup = 100;
  int iters = 1000;
  int device = -1;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--bytes N] [--warmup N] [--iters N] [--device ID]\n"
               "  Breaks down NCCL GIN GDAKI put+SignalInc issue-path phases.\n"
               "  Requires exactly 2 MPI ranks and NCCL_GIN_TYPE=3/GDAKI.\n",
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
  if (opt.bytes == 0 || opt.warmup < 0 || opt.iters <= 0) {
    throw std::invalid_argument("invalid benchmark options");
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

template <enum doca_gpu_dev_verbs_resource_sharing_mode ResourceSharingMode>
__device__ __forceinline__ void manual_gdaki_put_signal_mode(ncclGin const& gin, ncclTeam team, int peer,
                                                             ncclWindow_t dstwin, ncclWindow_t srcwin,
                                                             size_t bytes, ncclGinSignal_t signal,
                                                             uint64_t* samples, int iters, int iter,
                                                             bool record) {
  using nccl::gin::internal::teamRankToGinRank;
  using nccl::utility::loadConst;

  uint64_t t0 = global_timer();

  ncclGinCtx ctx = gin._makeCtx();
  int gin_peer = teamRankToGinRank(gin.comm, team, peer);
  auto* gdaki = &reinterpret_cast<ncclGinGdakiGPUContext*>(ctx.handle)[ctx.contextId];
  doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + gin_peer;

  auto* dst_mh = reinterpret_cast<ncclGinGdakiMemHandle*>(loadConst(&dstwin->ginWins[gin.connectionId]));
  auto* src_mh = reinterpret_cast<ncclGinGdakiMemHandle*>(loadConst(&srcwin->ginWins[gin.connectionId]));

  doca_gpu_dev_verbs_addr raddr{};
  raddr.addr = 4096 * size_t(loadConst(&dstwin->ginOffset4K));
  raddr.key = loadConst(loadConst(&dst_mh->rkeys) + gin_peer);

  doca_gpu_dev_verbs_addr laddr{};
  laddr.addr = 4096 * size_t(loadConst(&srcwin->ginOffset4K));
  laddr.key = loadConst(&src_mh->lkey);

  doca_gpu_dev_verbs_addr sig_raddr{};
  sig_raddr.addr = sizeof(uint64_t) * (signal + loadConst(&gdaki->signals_table.offset));
  sig_raddr.key = loadConst(loadConst(&gdaki->signals_table.rkeys) + gin_peer);

  doca_gpu_dev_verbs_addr sig_laddr{};
  sig_laddr.addr = 0;
  sig_laddr.key = loadConst(&gdaki->sink_buffer_lkey);

  uint32_t code_opt = nccl::gin::gdaki::docaOptFlagsFromGinOptFlags(ncclGinOptFlagsDefault);
  uint64_t t1 = global_timer();

  uint32_t num_chunks = doca_gpu_dev_verbs_div_ceil_aligned_pow2_32bits(
      bytes, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
  num_chunks = num_chunks > 1 ? num_chunks : 1;

  uint64_t base_wqe_idx =
      doca_gpu_dev_verbs_reserve_wq_slots<ResourceSharingMode>(qp, num_chunks + 1, code_opt);
  uint64_t t2 = global_timer();

  struct doca_gpu_dev_verbs_wqe* wqe_ptr = nullptr;
  uint64_t wqe_idx = base_wqe_idx;
  size_t remaining_size = bytes;
  for (uint64_t i = 0; i < num_chunks; i++) {
    wqe_idx = base_wqe_idx + i;
    size_t size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                       ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                       : remaining_size;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, static_cast<uint16_t>(wqe_idx));
    if (size_ > 0) {
      doca_gpu_dev_verbs_wqe_prepare_write(
          qp, wqe_ptr, static_cast<uint16_t>(wqe_idx), DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
          DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
          raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), raddr.key,
          laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), laddr.key, size_);
    } else {
      doca_gpu_dev_verbs_wqe_prepare_nop(qp, wqe_ptr, static_cast<uint16_t>(wqe_idx),
                                         DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
    }
    remaining_size -= size_;
  }

  ++wqe_idx;
  wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, static_cast<uint16_t>(wqe_idx));
  doca_gpu_dev_verbs_wqe_prepare_atomic(
      qp, wqe_ptr, static_cast<uint16_t>(wqe_idx), DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
      DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, sig_raddr.addr, sig_raddr.key, sig_laddr.addr,
      sig_laddr.key, sizeof(uint64_t), 1, 0);
  uint64_t t3 = global_timer();

  doca_gpu_dev_verbs_mark_wqes_ready<ResourceSharingMode>(qp, base_wqe_idx, wqe_idx);
  uint64_t t4 = global_timer();

  constexpr enum doca_gpu_dev_verbs_sync_scope submit_sync_scope =
      (ResourceSharingMode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU)
          ? DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD
          : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
  doca_gpu_dev_verbs_submit<ResourceSharingMode, submit_sync_scope, DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
      qp, wqe_idx + 1, code_opt);
  uint64_t t5 = global_timer();

  if (record && samples != nullptr) {
    samples[static_cast<size_t>(kPhaseMetadataLookup) * iters + iter] = t1 - t0;
    samples[static_cast<size_t>(kPhaseReserveWqeSlots) * iters + iter] = t2 - t1;
    samples[static_cast<size_t>(kPhaseConstructWqes) * iters + iter] = t3 - t2;
    samples[static_cast<size_t>(kPhaseMarkWqesReady) * iters + iter] = t4 - t3;
    samples[static_cast<size_t>(kPhaseDoorbellSubmit) * iters + iter] = t5 - t4;
  }
}

__device__ __forceinline__ void manual_gdaki_put_signal(ncclGin const& gin, ncclTeam team, int peer,
                                                        ncclWindow_t dstwin, ncclWindow_t srcwin, size_t bytes,
                                                        ncclGinSignal_t signal, uint64_t* samples,
                                                        int iters, int iter, bool record) {
  ncclGinCtx ctx = gin._makeCtx();
  switch (static_cast<ncclGinResourceSharingMode>(ctx.resourceSharingMode)) {
    case NCCL_GIN_RESOURCE_SHARING_THREAD:
      manual_gdaki_put_signal_mode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
          gin, team, peer, dstwin, srcwin, bytes, signal, samples, iters, iter, record);
      break;
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      manual_gdaki_put_signal_mode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(
          gin, team, peer, dstwin, srcwin, bytes, signal, samples, iters, iter, record);
      break;
    default:
      manual_gdaki_put_signal_mode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
          gin, team, peer, dstwin, srcwin, bytes, signal, samples, iters, iter, record);
      break;
  }
}

__global__ void phase_breakdown_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin, ncclDevComm dev_comm,
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
        manual_gdaki_put_signal(gin, world, peer, recvwin, sendwin, bytes, kPingSignal,
                                samples, iters, iter, record);
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
        samples[static_cast<size_t>(kPhaseRemotePollSignal) * iters + iter] = t1 - t0;
      }
      if (threadIdx.x == 0) {
        manual_gdaki_put_signal(gin, world, peer, recvwin, sendwin, bytes, kPongSignal,
                                nullptr, iters, iter, false);
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

void print_phase_stats(const char* name, const std::vector<double>& values) {
  MetricStats s = compute_stats(values);
  std::printf("%s,%.6f,%.6f,%.6f,%.6f\n", name, s.min, s.p50, s.max, s.avg);
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
        std::cerr << "gin_put_signal_phase_breakdown requires exactly 2 ranks, got " << nranks << "\n";
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
    if (props.ginType != NCCL_GIN_TYPE_GDAKI) {
      throw std::runtime_error("this benchmark only supports NCCL_GIN_TYPE_GDAKI");
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

    phase_breakdown_kernel<<<1, kThreads, 0, stream>>>(sendwin, recvwin, dev_comm, opt.bytes,
                                                       opt.warmup, opt.iters, d_samples);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<uint64_t> h_samples(sample_count);
    CUDA_CHECK(cudaMemcpy(h_samples.data(), d_samples, sample_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    std::vector<double> rank1_phase7_us(opt.iters);
    if (rank == 1) {
      std::vector<double> local_phase7 = convert_phase_ticks_to_us(h_samples, kPhaseRemotePollSignal,
                                                                   opt.iters, ticks_per_us);
      MPI_CHECK(MPI_Send(local_phase7.data(), opt.iters * sizeof(double), MPI_BYTE, 0, 1007,
                         MPI_COMM_WORLD));
    } else {
      MPI_CHECK(MPI_Recv(rank1_phase7_us.data(), opt.iters * sizeof(double), MPI_BYTE, 1, 1007,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    if (rank == 0) {
      std::printf("# gin_put_signal_phase_breakdown bytes=%zu warmup=%d iters=%d ranks=2 ginType=%d\n",
                  opt.bytes, opt.warmup, opt.iters, props.ginType);
      std::printf("# phase1-5 are rank0 issue-path timings; phase7 is rank1 satisfied waitSignal timing after ping is visible.\n");
      std::printf("phase,min_us,p50_us,max_us,avg_us\n");
      for (int phase = 0; phase <= kPhaseDoorbellSubmit; phase++) {
        print_phase_stats(kPhaseNames[phase],
                          convert_phase_ticks_to_us(h_samples, phase, opt.iters, ticks_per_us));
      }
      print_phase_stats(kPhaseNames[kPhaseRemotePollSignal], rank1_phase7_us);
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

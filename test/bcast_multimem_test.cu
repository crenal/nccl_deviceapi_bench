// Standalone NCCL bcastMultimem latency/bandwidth benchmark.

#include <cuda_runtime.h>
#include <nccl.h>

#include "common/checks.hpp"
#include "common/env.hpp"
#include "common/parse.hpp"
#include "common/socket.hpp"
#include "common/stats.hpp"
#include "common/symk.hpp"
#include "common/sweep.hpp"
#include "common/timer.cuh"
#include "nccl_device.h"
#include "device/symmetric/all_gather.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace nccl_deviceapi_test;

struct Options {
  size_t min_bytes = 64;
  size_t max_bytes = 512ull << 20;
  int factor = 2;
  int warmup = 100;
  int iters = 1000;
  int threads = 512;
  int num_blocks = 1;
  int device = -1;
  int port = 22340;
  double ticks_per_us = 0.0;
  std::string master_addr;
#if defined(BCAST_MM_DEFAULT_COLLECTIVE_STATS)
  bool collective_stats = true;
#else
  bool collective_stats = false;
#endif
};

void usage(const char* argv0) {
  std::fprintf(stderr,
      "Usage: %s [--min-bytes N] [--max-bytes N] [--factor N]\n"
      "          [--warmup N] [--iters N] [--threads N] [--num-blocks N]\n"
      "          [--device ID] [--master ADDR] [--port PORT] [--ticks-per-us X]\n"
      "          [--stats-mode rank|collective]\n",
      argv0);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  opt.master_addr = env_string("BCAST_MM_MASTER_ADDR", "127.0.0.1");
  opt.port = env_int("BCAST_MM_MASTER_PORT", opt.port);

  for (int i = 1; i < argc; i++) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        usage(argv[0]);
        std::exit(3);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--min-bytes") == 0) {
      opt.min_bytes = parse_size(need(argv[i]));
    } else if (std::strcmp(argv[i], "--max-bytes") == 0) {
      opt.max_bytes = parse_size(need(argv[i]));
    } else if (std::strcmp(argv[i], "--factor") == 0) {
      opt.factor = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      opt.warmup = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      opt.iters = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--threads") == 0) {
      opt.threads = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--num-blocks") == 0) {
      opt.num_blocks = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--device") == 0) {
      opt.device = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--master") == 0) {
      opt.master_addr = need(argv[i]);
    } else if (std::strcmp(argv[i], "--port") == 0) {
      opt.port = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--ticks-per-us") == 0) {
      opt.ticks_per_us = std::atof(need(argv[i]));
    } else if (std::strcmp(argv[i], "--stats-mode") == 0) {
      const char* mode = need(argv[i]);
      if (std::strcmp(mode, "rank") == 0) {
        opt.collective_stats = false;
      } else if (std::strcmp(mode, "collective") == 0) {
        opt.collective_stats = true;
      } else {
        std::fprintf(stderr, "Unknown --stats-mode: %s\n", mode);
        usage(argv[0]);
        std::exit(3);
      }
    } else if (std::strcmp(argv[i], "--collective-stats") == 0) {
      opt.collective_stats = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(3);
    }
  }

  if (opt.min_bytes == 0 || opt.max_bytes < opt.min_bytes || opt.factor <= 1 ||
      opt.warmup < 0 || opt.iters <= 0 || opt.threads <= 0 || opt.threads > 1024 ||
      (opt.threads % 32) != 0 || opt.num_blocks <= 0 || opt.num_blocks > ncclSymkMaxBlocks ||
      opt.port <= 0 || opt.ticks_per_us < 0.0) {
    std::fprintf(stderr, "Invalid arguments\n");
    std::exit(3);
  }
  return opt;
}

__device__ __forceinline__ void run_bcast_multimem(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  int const rank = handler.comm.rank;

  handler.forEachWork<char>([&] __device__(int block, int n_blocks, size_t n_elts, size_t n_all_elts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
    int t = flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE,
                      block, n_blocks,
                      threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int tn = n_blocks * blockDim.x;
    bcastMultimem<char, false>(handler, tn, t, input, output + rank * n_all_elts, n_elts);
  });
}

__global__ void bcast_multimem_bench_kernel(ncclSymkDevWorkArgs4K NCCL_GRID_CONSTANT const args4k,
                                            uint64_t* block_samples, int warmup, int iters) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler{&args4k.args};
  ncclLsaBarrierSession<ncclCoopCta> bar(cta, handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true);

  for (int i = 0; i < warmup; i++) {
    bar.sync(cta, cuda::memory_order_release);
    run_bcast_multimem(&args4k.args);
    bar.sync(cta, cuda::memory_order_release);
  }

  for (int i = 0; i < iters; i++) {
    bar.sync(cta, cuda::memory_order_release);
    uint64_t t0 = global_timer();
    run_bcast_multimem(&args4k.args);
    bar.sync(cta, cuda::memory_order_release);
    uint64_t t1 = global_timer();
    if (threadIdx.x == 0) {
      block_samples[static_cast<size_t>(i) * gridDim.x + blockIdx.x] = t1 - t0;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options opt = parse_args(argc, argv);
    std::vector<size_t> sizes = make_sizes(opt.min_bytes, opt.max_bytes, opt.factor);

    int rank = env_int("OMPI_COMM_WORLD_RANK", env_int("PMI_RANK", env_int("RANK", 0)));
    int nranks = env_int("OMPI_COMM_WORLD_SIZE", env_int("PMI_SIZE", env_int("WORLD_SIZE", 1)));
    int local_rank = env_int("OMPI_COMM_WORLD_LOCAL_RANK", env_int("MPI_LOCALRANKID", env_int("LOCAL_RANK", rank)));

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int device = opt.device >= 0 ? opt.device : (local_rank % std::max(1, ndev));
    CUDA_CHECK(cudaSetDevice(device));
    if (opt.ticks_per_us == 0.0) opt.ticks_per_us = calibrate_ticks_per_us();

    ncclUniqueId id = exchange_unique_id_socket(rank, nranks, opt.master_addr, opt.port);

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    if (!props.multimemSupport) {
      std::fprintf(stderr, "Rank %d: NCCL communicator reports multimemSupport=0\n", rank);
      std::exit(5);
    }

    ncclTeam_t lsa = ncclTeamLsa(comm);
    int lsa_ranks = lsa.nRanks;

    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaMultimem = true;
    reqs.lsaBarrierCount = opt.num_blocks;
    reqs.ginContextCount = 0;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_NONE;
    reqs.ginQueueDepth = 0;

    ncclDevComm_t dev_comm;
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    size_t send_bytes = opt.max_bytes;
    size_t recv_bytes = opt.max_bytes * static_cast<size_t>(nranks);
    void* sendbuff = nullptr;
    void* recvbuff = nullptr;
    NCCL_CHECK(ncclMemAlloc(&sendbuff, send_bytes));
    NCCL_CHECK(ncclMemAlloc(&recvbuff, recv_bytes));
    CUDA_CHECK(cudaMemsetAsync(sendbuff, 0x5a, send_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(recvbuff, 0, recv_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ncclWindow_t send_win = nullptr;
    ncclWindow_t recv_win = nullptr;
    NCCL_CHECK(ncclCommWindowRegister(comm, sendbuff, send_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, recvbuff, recv_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));

    uint64_t* d_block_samples = nullptr;
    std::vector<uint64_t> h_block_samples(static_cast<size_t>(opt.iters) * opt.num_blocks);
    CUDA_CHECK(cudaMalloc(&d_block_samples, h_block_samples.size() * sizeof(uint64_t)));

    std::vector<double> local_samples;
    local_samples.reserve(static_cast<size_t>(sizes.size()) * opt.iters);

    for (size_t size_bytes : sizes) {
      ncclSymkDevWorkArgs4K args4k = make_single_work_args(dev_comm, send_win, recv_win, size_bytes, opt.num_blocks);
      CUDA_CHECK(cudaMemsetAsync(d_block_samples, 0, h_block_samples.size() * sizeof(uint64_t), stream));
      bcast_multimem_bench_kernel<<<opt.num_blocks, opt.threads, 0, stream>>>(
          args4k, d_block_samples, opt.warmup, opt.iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(h_block_samples.data(), d_block_samples, h_block_samples.size() * sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      for (int i = 0; i < opt.iters; i++) {
        uint64_t max_cycles = 0;
        for (int b = 0; b < opt.num_blocks; b++) {
          max_cycles = std::max(max_cycles, h_block_samples[static_cast<size_t>(i) * opt.num_blocks + b]);
        }
        local_samples.push_back(static_cast<double>(max_cycles) / opt.ticks_per_us);
      }
    }

    CUDA_CHECK(cudaFree(d_block_samples));
    NCCL_CHECK(ncclCommWindowDeregister(comm, send_win));
    NCCL_CHECK(ncclCommWindowDeregister(comm, recv_win));
    NCCL_CHECK(ncclMemFree(sendbuff));
    NCCL_CHECK(ncclMemFree(recvbuff));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    NCCL_CHECK(ncclCommDestroy(comm));

    std::vector<double> all = gather_samples_socket(rank, nranks, opt.master_addr, opt.port + 1, local_samples);

    if (rank == 0) {
      size_t rank_chunk = static_cast<size_t>(sizes.size()) * opt.iters;
      if (rank_chunk == 0 || all.size() % rank_chunk != 0) {
        std::fprintf(stderr, "Unexpected gathered sample count: %zu rankChunk=%zu\n", all.size(), rank_chunk);
        std::exit(6);
      }
      std::printf("# bcast-multimem-test ranks=%d lsa_ranks=%d threads=%d num_blocks=%d warmup=%d iters=%d "
                  "stats_mode=%s ticks_per_us_rank0=%.3f\n",
                  nranks, lsa_ranks, opt.threads, opt.num_blocks, opt.warmup, opt.iters,
                  opt.collective_stats ? "collective" : "rank", opt.ticks_per_us);
      if (opt.collective_stats) {
        std::printf("size_B,num_blocks,min_us,p50_us,p90_us,p99_us,max_us,"
                    "per_rank_inj_bw_GBps,aggregate_inj_bw_GBps,aggregate_delivered_bw_GBps\n");
      } else {
        std::printf("size_B,num_blocks,min_us,p50_us,p90_us,p99_us,max_us,inj_bw_GBps,delivered_bw_GBps\n");
      }
      for (size_t sidx = 0; sidx < sizes.size(); sidx++) {
        std::vector<double> samples = samples_for_size(all, rank_chunk, sidx, opt.iters, opt.collective_stats);
        MetricStats stats = compute_stats(samples);
        double per_rank_inj_bw = gib_per_second(sizes[sidx], stats.p50);
        if (opt.collective_stats) {
          double aggregate_inj_bw = per_rank_inj_bw * static_cast<double>(lsa_ranks);
          double aggregate_delivered_bw = aggregate_inj_bw * static_cast<double>(lsa_ranks);
          std::printf("%zu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                      sizes[sidx], opt.num_blocks, stats.min, stats.p50, stats.p90, stats.p99, stats.max,
                      per_rank_inj_bw, aggregate_inj_bw, aggregate_delivered_bw);
        } else {
          double delivered_bw = per_rank_inj_bw * static_cast<double>(lsa_ranks);
          std::printf("%zu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                      sizes[sidx], opt.num_blocks, stats.min, stats.p50, stats.p90, stats.p99, stats.max,
                      per_rank_inj_bw, delivered_bw);
        }
      }
    }

    return 0;
  } catch (const std::exception& e) {
    nccl_deviceapi_test::fail_fast(e);
    return 1;
  }
}

// Standalone benchmark for the ported NCCL AllGather_RailRing_LsaSTMC device API kernel.

#include <cuda_runtime.h>
#include <nccl.h>

#include "coll/all_gather_gin.cuh"
#include "coll/all_gather_gin_oneshot_rail.cuh"
#include "common/checks.hpp"
#include "common/env.hpp"
#include "common/parse.hpp"
#include "common/socket.hpp"
#include "common/stats.hpp"
#include "common/symk.hpp"
#include "common/sweep.hpp"
#include "common/timer.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace nccl_deviceapi_test;

enum AllGatherKernelKind {
  kRailRing = 0,
  kOneshotRail = 1,
};

struct Options {
  size_t min_bytes = 32ull << 10;
  size_t max_bytes = 8ull << 20;
  int factor = 2;
  int warmup = 100;
  int iters = 1000;
  int threads = ncclSymkMaxThreads;
  int num_blocks = 1;
  int gin_contexts = 4;
  int device = -1;
  int port = 22540;
  double ticks_per_us = 0.0;
  bool collective_stats = true;
  bool check = false;
  int kernel_kind = kRailRing;
  const char* kernel_name = "railring";
  std::string master_addr;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
      "Usage: %s [--min-bytes N] [--max-bytes N] [--factor N]\n"
      "          [--warmup N] [--iters N] [--threads N] [--num-blocks N]\n"
      "          [--gin-contexts N] [--device ID] [--master ADDR] [--port PORT]\n"
      "          [--ticks-per-us X] [--stats-mode rank|collective]\n"
      "          [--kernel railring|oneshot-rail] [--check]\n",
      argv0);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  opt.master_addr = env_string("AG_GIN_MASTER_ADDR", "127.0.0.1");
  opt.port = env_int("AG_GIN_MASTER_PORT", opt.port);

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
    } else if (std::strcmp(argv[i], "--gin-contexts") == 0) {
      opt.gin_contexts = std::atoi(need(argv[i]));
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
    } else if (std::strcmp(argv[i], "--kernel") == 0) {
      const char* kernel = need(argv[i]);
      if (std::strcmp(kernel, "railring") == 0) {
        opt.kernel_kind = kRailRing;
        opt.kernel_name = "railring";
      } else if (std::strcmp(kernel, "oneshot-rail") == 0) {
        opt.kernel_kind = kOneshotRail;
        opt.kernel_name = "oneshot-rail";
      } else {
        std::fprintf(stderr, "Unknown --kernel: %s\n", kernel);
        usage(argv[0]);
        std::exit(3);
      }
    } else if (std::strcmp(argv[i], "--check") == 0) {
      opt.check = true;
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
      opt.warmup < 0 || opt.iters <= 0 || opt.threads <= WARP_SIZE || opt.threads > 1024 ||
      (opt.threads % WARP_SIZE) != 0 || opt.num_blocks <= 0 || opt.num_blocks > ncclSymkMaxBlocks ||
      opt.gin_contexts <= 0 || opt.port <= 0 || opt.ticks_per_us < 0.0) {
    std::fprintf(stderr, "Invalid arguments\n");
    std::exit(3);
  }
  return opt;
}

ncclSymkDevWorkArgs4K make_args(ncclDevComm_t dev_comm, ncclGinSyncHandle gin_sync,
                                ncclWindow_t input_win, ncclWindow_t output_win,
                                size_t send_bytes, int num_blocks) {
  ncclSymkDevWorkArgs4K args4k = make_single_work_args(dev_comm, input_win, output_win, send_bytes, num_blocks);
  args4k.args.kcomm.ginSyncHandle = gin_sync;
  return args4k;
}

__global__ void allgather_gin_bench_kernel(ncclSymkDevWorkArgs4K NCCL_GRID_CONSTANT const args4k,
                                           uint64_t* body_samples, int sample_idx, int kernel_kind) {
  if (kernel_kind == kOneshotRail) {
    ncclSymkRun_AllGather_OneshotRail_Timed(&args4k.args, body_samples, sample_idx);
  } else {
    ncclSymkRun_AllGather_RailRing_LsaSTMC_Timed(&args4k.args, body_samples, sample_idx);
  }
}

void check_allgather_result(void* recvbuff, cudaStream_t stream, size_t size_bytes, int rank, int nranks) {
  std::vector<uint8_t> h_recv(size_bytes * static_cast<size_t>(nranks));
  CUDA_CHECK(cudaMemcpyAsync(h_recv.data(), recvbuff, h_recv.size(), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int peer = 0; peer < nranks; peer++) {
    uint8_t expected = static_cast<uint8_t>(peer & 0xff);
    size_t base = static_cast<size_t>(peer) * size_bytes;
    for (size_t i = 0; i < size_bytes; i++) {
      uint8_t got = h_recv[base + i];
      if (got != expected) {
        std::fprintf(stderr,
                     "Rank %d check failed: send_B=%zu peer=%d offset=%zu got=%u expected=%u\n",
                     rank, size_bytes, peer, i, static_cast<unsigned>(got), static_cast<unsigned>(expected));
        std::exit(7);
      }
    }
  }
}

std::vector<double> mean_samples_for_size(const std::vector<double>& all_samples, size_t rank_chunk,
                                          size_t size_index, int iters) {
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iters));
  size_t nranks = rank_chunk == 0 ? 0 : all_samples.size() / rank_chunk;
  for (int i = 0; i < iters; i++) {
    double sum_us = 0.0;
    for (size_t base = 0; base < all_samples.size(); base += rank_chunk) {
      size_t off = base + size_index * static_cast<size_t>(iters) + static_cast<size_t>(i);
      sum_us += all_samples[off];
    }
    samples.push_back(nranks == 0 ? 0.0 : sum_us / static_cast<double>(nranks));
  }
  return samples;
}

void print_metric_rows(const char* kind, const std::vector<double>& all_samples, size_t rank_chunk,
                       const std::vector<size_t>& sizes, int iters, int nranks, int num_blocks,
                       bool collective_stats) {
  for (size_t sidx = 0; sidx < sizes.size(); sidx++) {
    std::vector<double> samples = collective_stats
                                      ? mean_samples_for_size(all_samples, rank_chunk, sidx, iters)
                                      : samples_for_size(all_samples, rank_chunk, sidx, iters, false);
    MetricStats stats = compute_stats(samples);
    double send_bw = gib_per_second(sizes[sidx], stats.p50);
    double recv_bw = gib_per_second(sizes[sidx] * static_cast<size_t>(nranks), stats.p50);
    std::printf("%s,%zu,%zu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                kind, sizes[sidx], sizes[sidx] * static_cast<size_t>(nranks), num_blocks,
                stats.min, stats.p50, stats.p90, stats.p99, stats.max, stats.avg, send_bw, recv_bw);
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
    ncclTeam_t rail = ncclTeamRail(comm);
    if (rail.nRanks <= 1 && rank == 0) {
      std::fprintf(stderr, "Warning: rail team has one rank; GIN ring path will not exercise cross-node put.\n");
    }

    ncclGinSyncHandle gin_sync = {};
    ncclDevResourceRequirements_t rail_signal_req = {};
    rail_signal_req.ginSignalCount = rail.nRanks * opt.num_blocks;
    rail_signal_req.outGinSignalStart = &gin_sync.railSignals;

    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.resourceRequirementsList = &rail_signal_req;
    reqs.lsaMultimem = true;
    reqs.barrierCount = opt.num_blocks;
    reqs.lsaBarrierCount = opt.num_blocks;
    reqs.ginContextCount = opt.gin_contexts;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
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
    CUDA_CHECK(cudaMemsetAsync(sendbuff, rank & 0xff, send_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(recvbuff, 0, recv_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ncclWindow_t send_win = nullptr;
    ncclWindow_t recv_win = nullptr;
    NCCL_CHECK(ncclCommWindowRegister(comm, sendbuff, send_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, recvbuff, recv_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));

    uint64_t* d_body_samples = nullptr;
    std::vector<uint64_t> h_body_samples(static_cast<size_t>(opt.iters) * opt.num_blocks);
    CUDA_CHECK(cudaMalloc(&d_body_samples, h_body_samples.size() * sizeof(uint64_t)));

    std::vector<cudaEvent_t> event_start(opt.iters);
    std::vector<cudaEvent_t> event_stop(opt.iters);
    for (int i = 0; i < opt.iters; i++) {
      CUDA_CHECK(cudaEventCreate(&event_start[i]));
      CUDA_CHECK(cudaEventCreate(&event_stop[i]));
    }

    std::vector<double> local_kernel_samples;
    std::vector<double> local_body_samples;
    local_kernel_samples.reserve(static_cast<size_t>(sizes.size()) * opt.iters);
    local_body_samples.reserve(static_cast<size_t>(sizes.size()) * opt.iters);

    for (size_t size_bytes : sizes) {
      ncclSymkDevWorkArgs4K args4k = make_args(dev_comm, gin_sync, send_win, recv_win, size_bytes, opt.num_blocks);
      CUDA_CHECK(cudaMemsetAsync(d_body_samples, 0, h_body_samples.size() * sizeof(uint64_t), stream));
      for (int i = 0; i < opt.warmup; i++) {
        allgather_gin_bench_kernel<<<opt.num_blocks, opt.threads, 0, stream>>>(
            args4k, nullptr, 0, opt.kernel_kind);
        CUDA_CHECK(cudaGetLastError());
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));

      for (int i = 0; i < opt.iters; i++) {
        CUDA_CHECK(cudaEventRecord(event_start[i], stream));
        allgather_gin_bench_kernel<<<opt.num_blocks, opt.threads, 0, stream>>>(
            args4k, d_body_samples, i, opt.kernel_kind);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(event_stop[i], stream));
      }
      CUDA_CHECK(cudaMemcpyAsync(h_body_samples.data(), d_body_samples, h_body_samples.size() * sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (opt.check) {
        check_allgather_result(recvbuff, stream, size_bytes, rank, nranks);
      }

      for (int i = 0; i < opt.iters; i++) {
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, event_start[i], event_stop[i]));
        local_kernel_samples.push_back(static_cast<double>(elapsed_ms) * 1000.0);

        uint64_t max_cycles = 0;
        for (int b = 0; b < opt.num_blocks; b++) {
          max_cycles = std::max(max_cycles, h_body_samples[static_cast<size_t>(i) * opt.num_blocks + b]);
        }
        local_body_samples.push_back(static_cast<double>(max_cycles) / opt.ticks_per_us);
      }
    }

    for (int i = 0; i < opt.iters; i++) {
      CUDA_CHECK(cudaEventDestroy(event_start[i]));
      CUDA_CHECK(cudaEventDestroy(event_stop[i]));
    }
    CUDA_CHECK(cudaFree(d_body_samples));
    NCCL_CHECK(ncclCommWindowDeregister(comm, send_win));
    NCCL_CHECK(ncclCommWindowDeregister(comm, recv_win));
    NCCL_CHECK(ncclMemFree(sendbuff));
    NCCL_CHECK(ncclMemFree(recvbuff));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    NCCL_CHECK(ncclCommDestroy(comm));

    std::vector<double> all_kernel =
        gather_samples_socket(rank, nranks, opt.master_addr, opt.port + 1, local_kernel_samples);
    std::vector<double> all_body =
        gather_samples_socket(rank, nranks, opt.master_addr, opt.port + 2, local_body_samples);

    if (rank == 0) {
      size_t rank_chunk = static_cast<size_t>(sizes.size()) * opt.iters;
      if (rank_chunk == 0 || all_kernel.size() % rank_chunk != 0 || all_body.size() % rank_chunk != 0) {
        std::fprintf(stderr, "Unexpected gathered sample count: kernel=%zu body=%zu rankChunk=%zu\n",
                     all_kernel.size(), all_body.size(), rank_chunk);
        std::exit(6);
      }
      std::printf("# allgather-gin-deviceapi-perf kernel=%s ranks=%d lsa_ranks=%d rail_ranks=%d threads=%d "
                  "num_blocks=%d gin_contexts=%d warmup=%d iters=%d stats_mode=%s check=%s "
                  "ticks_per_us_rank0=%.3f\n",
                  opt.kernel_name, nranks, lsa.nRanks, rail.nRanks, opt.threads, opt.num_blocks, opt.gin_contexts,
                  opt.warmup, opt.iters, opt.collective_stats ? "collective_mean" : "rank",
                  opt.check ? "passed" : "off", opt.ticks_per_us);
      std::printf("time_kind,send_B,recv_B,num_blocks,min_us,p50_us,p90_us,p99_us,max_us,avg_us,"
                  "send_bw_GBps,recv_bw_GBps\n");
      print_metric_rows("kernel", all_kernel, rank_chunk, sizes, opt.iters, nranks, opt.num_blocks,
                        opt.collective_stats);
      print_metric_rows("kernel_body", all_body, rank_chunk, sizes, opt.iters, nranks, opt.num_blocks,
                        opt.collective_stats);
    }

    return 0;
  } catch (const std::exception& e) {
    nccl_deviceapi_test::fail_fast(e);
    return 1;
  }
}

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "coll/all_gather_gin_oneshot_rail.cuh"
#include "common/checks.hpp"
#include "common/mpi.hpp"
#include "common/parse.hpp"
#include "common/stats.hpp"
#include "common/sweep.hpp"
#include "common/symk.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace nccl_deviceapi_test;

constexpr int kGinPingpongThreads = 32;
constexpr int kNvshmemPingpongThreads = 256;
constexpr int kGinResourceCount = 16;
constexpr ncclGinSignal_t kGinPingSignal = 0;
constexpr ncclGinSignal_t kGinPongSignal = 1;

#define NVSHMEM_CHECK(stmt)                                                                         \
  do {                                                                                              \
    int nvshmem_err__ = (stmt);                                                                     \
    if (nvshmem_err__ != 0) {                                                                       \
      throw ::nccl_deviceapi_test::make_error("NVSHMEM", __FILE__, __LINE__,                        \
                                              std::to_string(nvshmem_err__));                       \
    }                                                                                               \
  } while (0)

enum class Backend {
  kGin,
  kNvshmem,
};

enum class Op {
  kPingpong,
  kAllGather,
};

struct SemanticsCase {
  const char* name;
  bool init_bar;
  bool flush;
};

constexpr SemanticsCase kSemanticsCases[] = {
    {"no24", false, false},
    {"initbar", true, false},
    {"flush", false, true},
    {"full", true, true},
};

struct Options {
  Backend backend = Backend::kGin;
  Op op = Op::kPingpong;
  std::string semantics = "all";
  size_t min_bytes = 1ull << 10;
  size_t max_bytes = 1ull << 20;
  int factor = 2;
  int warmup = 100;
  int iters = 1000;
  int threads = ncclSymkMaxThreads;
  int split_blocks = 4;
  int gin_contexts = 4;
  int device = -1;
  bool check = false;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--backend gin|nvshmem] [--op pingpong|ag]\n"
               "          [--semantics all|no24|initbar|flush|full]\n"
               "          [--bytes N] [--min-bytes N] [--max-bytes N] [--factor N]\n"
               "          [--warmup N] [--iters N] [--threads N]\n"
               "          [--split-blocks N] [--gin-contexts N] [--device ID] [--check]\n",
               argv0);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        usage(argv[0]);
        std::exit(3);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--backend") == 0) {
      const char* backend = need(argv[i]);
      if (std::strcmp(backend, "gin") == 0 || std::strcmp(backend, "nccl") == 0 ||
          std::strcmp(backend, "ncclgin") == 0) {
        opt.backend = Backend::kGin;
      } else if (std::strcmp(backend, "nvshmem") == 0) {
        opt.backend = Backend::kNvshmem;
      } else {
        throw std::invalid_argument(std::string("unknown backend: ") + backend);
      }
    } else if (std::strcmp(argv[i], "--op") == 0) {
      const char* op = need(argv[i]);
      if (std::strcmp(op, "pingpong") == 0 || std::strcmp(op, "pp") == 0) {
        opt.op = Op::kPingpong;
      } else if (std::strcmp(op, "ag") == 0 || std::strcmp(op, "allgather") == 0) {
        opt.op = Op::kAllGather;
      } else {
        throw std::invalid_argument(std::string("unknown op: ") + op);
      }
    } else if (std::strcmp(argv[i], "--semantics") == 0) {
      opt.semantics = need(argv[i]);
    } else if (std::strcmp(argv[i], "--bytes") == 0) {
      opt.min_bytes = opt.max_bytes = parse_size(need(argv[i]));
    } else if (std::strcmp(argv[i], "--min-bytes") == 0) {
      opt.min_bytes = parse_size(need(argv[i]));
    } else if (std::strcmp(argv[i], "--max-bytes") == 0) {
      opt.max_bytes = parse_size(need(argv[i]));
    } else if (std::strcmp(argv[i], "--factor") == 0) {
      opt.factor = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      opt.warmup = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      opt.iters = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--threads") == 0) {
      opt.threads = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--split-blocks") == 0) {
      opt.split_blocks = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--gin-contexts") == 0) {
      opt.gin_contexts = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--device") == 0) {
      opt.device = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--check") == 0) {
      opt.check = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument(std::string("unknown argument: ") + argv[i]);
    }
  }

  if (opt.min_bytes == 0 || opt.max_bytes < opt.min_bytes || opt.factor <= 1 || opt.warmup < 0 ||
      opt.iters <= 0 || opt.threads <= 0 || opt.threads > 1024 || (opt.threads % 32) != 0 ||
      opt.split_blocks <= 0 || opt.split_blocks > ncclSymkMaxBlocks || opt.gin_contexts <= 0) {
    throw std::invalid_argument("invalid benchmark options");
  }
  if (opt.backend == Backend::kNvshmem && opt.op == Op::kAllGather && opt.min_bytes < 16) {
    throw std::invalid_argument("NVSHMEM allgather needs at least 16 bytes so data and flag words do not alias");
  }
  return opt;
}

std::vector<SemanticsCase> selected_semantics(const std::string& name) {
  if (name == "all") {
    return std::vector<SemanticsCase>(std::begin(kSemanticsCases), std::end(kSemanticsCases));
  }
  for (const SemanticsCase& s : kSemanticsCases) {
    if (name == s.name) return {s};
  }
  if (name == "init") return {kSemanticsCases[1]};
  if (name == "quiet") return {kSemanticsCases[2]};
  throw std::invalid_argument("unknown semantics: " + name);
}

const char* backend_name(Backend backend) {
  return backend == Backend::kGin ? "gin" : "nvshmem";
}

const char* op_name(Op op) {
  return op == Op::kPingpong ? "pingpong" : "ag";
}

double gib_per_second_bytes(size_t bytes, double us) {
  return us > 0.0 ? static_cast<double>(bytes) / 1024.0 / 1024.0 / 1024.0 / us * 1000000.0 : 0.0;
}

__global__ void fill_u8_kernel(uint8_t* ptr, size_t bytes, uint8_t value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < bytes; i += stride) {
    ptr[i] = value;
  }
}

__global__ void fill_u64_kernel(uint64_t* ptr, size_t words, uint64_t value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < words; i += stride) {
    ptr[i] = value;
  }
}

__global__ void fill_float_kernel(float* ptr, size_t n, float value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < n; i += stride) {
    ptr[i] = value;
  }
}

__global__ void gin_pingpong_loop_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                         ncclDevComm dev_comm, size_t bytes, int iters,
                                         bool init_bar, bool flush_after_put) {
  ncclCoopCta cta;
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclBarrierSession<ncclCoopCta> world_bar{
      cta, ncclTeamTagWorld(), gin, static_cast<uint32_t>(blockIdx.x)};
  ncclCoopWarpSpan warps(0, 1, 0);

  __shared__ uint64_t signal_bases[2];
  if (threadIdx.x == 0) {
    signal_bases[0] = gin.readSignal(kGinPingSignal);
    signal_bases[1] = gin.readSignal(kGinPongSignal);
  }
  __syncthreads();

  const int peer = world.rank == 0 ? 1 : 0;
  for (int i = 1; i <= iters; i++) {
    if (init_bar) {
      world_bar.sync(cta, cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
    }

    uint64_t ping_expected = signal_bases[0] + static_cast<uint64_t>(i);
    uint64_t pong_expected = signal_bases[1] + static_cast<uint64_t>(i);
    if (world.rank == 0) {
      if (threadIdx.x < kGinPingpongThreads) {
        gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_SignalInc{kGinPingSignal},
                ncclGin_None{}, warps);
        if (flush_after_put) {
          gin.flush(warps);
        }
        gin.waitSignal(warps, kGinPongSignal, pong_expected, kGinPingpongThreads);
      }
    } else {
      if (threadIdx.x < kGinPingpongThreads) {
        gin.waitSignal(warps, kGinPingSignal, ping_expected, kGinPingpongThreads);
        gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_SignalInc{kGinPongSignal},
                ncclGin_None{}, warps);
        if (flush_after_put) {
          gin.flush(warps);
        }
      }
    }
    __syncthreads();
  }
}

__global__ void gin_allgather_loop_kernel(ncclSymkDevWorkArgs4K NCCL_GRID_CONSTANT const args4k,
                                          int iters, bool init_bar, bool flush_after_put) {
  for (int i = 0; i < iters; i++) {
    ncclSymkRun_AllGather_OneshotRail_Semantics(&args4k.args, init_bar, flush_after_put);
  }
}

__global__ void nvshmem_pingpong_loop_kernel(float* dst, float* src, size_t elems, int peer,
                                             int iters, uint64_t base, int is_sender, bool init_bar,
                                             bool quiet_after_put) {
  volatile float* flag = dst + (elems - 1);
  for (int i = 1; i <= iters; i++) {
    if (init_bar) {
      nvshmemx_barrier_all_block();
    }
    float expected = static_cast<float>(base + static_cast<uint64_t>(i));
    if (is_sender) {
      if (threadIdx.x == 0) {
        src[elems - 1] = expected;
      }
      __syncthreads();
      nvshmemx_putmem_nbi_block(dst, src, elems * sizeof(float), peer);
      if (quiet_after_put) {
        nvshmem_quiet();
      }
      if (threadIdx.x == 0) {
        while (*flag < expected) {
        }
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0) {
        while (*flag < expected) {
        }
        src[elems - 1] = expected;
      }
      __syncthreads();
      nvshmemx_putmem_nbi_block(dst, src, elems * sizeof(float), peer);
      if (quiet_after_put) {
        nvshmem_quiet();
      }
    }
  }
}

__global__ void nvshmem_allgather_loop_kernel(uint64_t* dst, uint64_t* src, size_t words, int npes,
                                              int mype, int iters, uint64_t base, bool init_bar,
                                              bool quiet_after_put) {
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int nwarps = blockDim.x / 32;
  int nsend_peers = npes - 1;

  for (int i = 1; i <= iters; i++) {
    if (init_bar) {
      nvshmemx_barrier_all_block();
    }

    uint64_t expected = base + static_cast<uint64_t>(i);
    if (threadIdx.x == 0) {
      src[words - 1] = expected;
    }
    __syncthreads();

    for (size_t idx = static_cast<size_t>(threadIdx.x); idx < words; idx += blockDim.x) {
      dst[static_cast<size_t>(mype) * words + idx] = src[idx];
    }
    __syncthreads();

    for (int peer_slot = warp; peer_slot < nsend_peers; peer_slot += nwarps) {
      int peer = (mype + 1 + peer_slot) % npes;
      nvshmemx_putmem_nbi_warp(dst + static_cast<size_t>(mype) * words, src, words * sizeof(uint64_t), peer);
    }
    if (quiet_after_put) {
      __syncthreads();
      nvshmem_quiet();
    }
    for (int peer_slot = warp; peer_slot < nsend_peers; peer_slot += nwarps) {
      int peer = (mype + 1 + peer_slot) % npes;
      volatile uint64_t* flag = dst + static_cast<size_t>(peer) * words + (words - 1);
      if (lane == 0) {
        while (*flag < expected) {
        }
      }
    }
    __syncthreads();
  }
}

bool check_first_byte(uint8_t* ptr, uint8_t expected, cudaStream_t stream) {
  uint8_t value = 0;
  CUDA_CHECK(cudaMemcpyAsync(&value, ptr, sizeof(value), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return value == expected;
}

bool check_first_float(float* ptr, float expected, cudaStream_t stream) {
  float value = 0.0f;
  CUDA_CHECK(cudaMemcpyAsync(&value, ptr, sizeof(value), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return value == expected;
}

bool check_gin_allgather(uint8_t* recvbuf, size_t bytes, int nranks, cudaStream_t stream) {
  std::vector<uint8_t> host(bytes * static_cast<size_t>(nranks));
  CUDA_CHECK(cudaMemcpyAsync(host.data(), recvbuf, host.size(), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int peer = 0; peer < nranks; peer++) {
    if (host[static_cast<size_t>(peer) * bytes] != static_cast<uint8_t>(peer & 0xff)) {
      return false;
    }
  }
  return true;
}

bool check_nvshmem_allgather(uint64_t* recvbuf, size_t words, int npes, cudaStream_t stream) {
  std::vector<uint64_t> host(words * static_cast<size_t>(npes));
  CUDA_CHECK(cudaMemcpyAsync(host.data(), recvbuf, host.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int peer = 0; peer < npes; peer++) {
    uint64_t got = host[static_cast<size_t>(peer) * words];
    uint64_t expected = 0x100000000ull + static_cast<uint64_t>(peer);
    if (got != expected) {
      return false;
    }
  }
  return true;
}

struct TimingResult {
  double local_us = 0.0;
  bool check_ok = true;
};

TimingResult run_gin_pingpong(const Options& opt, const SemanticsCase& sem, size_t bytes, int rank,
                              int nranks, cudaStream_t stream) {
  if (nranks != 2) {
    throw std::runtime_error("GIN pingpong requires exactly 2 ranks");
  }

  ncclUniqueId id = mpi_bcast_nccl_unique_id(rank, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(comm, &props));
  if (!props.deviceApiSupport || props.ginType == NCCL_GIN_TYPE_NONE) {
    throw std::runtime_error("NCCL communicator does not support GIN device API");
  }

  uint8_t* sendbuf = nullptr;
  uint8_t* recvbuf = nullptr;
  ncclWindow_t sendwin = nullptr;
  ncclWindow_t recvwin = nullptr;
  NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&sendbuf), bytes));
  NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&recvbuf), bytes));
  NCCL_CHECK(ncclCommWindowRegister(comm, sendbuf, bytes, &sendwin, NCCL_WIN_COLL_SYMMETRIC));
  NCCL_CHECK(ncclCommWindowRegister(comm, recvbuf, bytes, &recvwin, NCCL_WIN_COLL_SYMMETRIC));

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = kGinResourceCount;
  reqs.barrierCount = 1;
  reqs.lsaBarrierCount = kGinResourceCount;
  reqs.railGinBarrierCount = kGinResourceCount;
  reqs.ginContextCount = 1;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
  reqs.ginForceEnable = true;
#endif
  ncclDevComm dev_comm{};
  NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

  int fill_blocks = std::max(1, std::min<int>(1024, static_cast<int>((bytes + 255) / 256)));
  fill_u8_kernel<<<fill_blocks, 256, 0, stream>>>(sendbuf, bytes, static_cast<uint8_t>(rank & 0xff));
  fill_u8_kernel<<<fill_blocks, 256, 0, stream>>>(recvbuf, bytes, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (opt.warmup > 0) {
    gin_pingpong_loop_kernel<<<1, kGinPingpongThreads, 0, stream>>>(sendwin, recvwin, dev_comm, bytes, opt.warmup,
                                                                    sem.init_bar, sem.flush);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  gin_pingpong_loop_kernel<<<1, kGinPingpongThreads, 0, stream>>>(sendwin, recvwin, dev_comm, bytes, opt.iters,
                                                                 sem.init_bar, sem.flush);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  TimingResult result;
  result.local_us = (static_cast<double>(elapsed_ms) * 1000.0) / static_cast<double>(opt.iters) / 2.0;
  if (opt.check) {
    result.check_ok = check_first_byte(recvbuf, static_cast<uint8_t>((rank == 0 ? 1 : 0) & 0xff), stream);
  }

  NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
  NCCL_CHECK(ncclCommWindowDeregister(comm, sendwin));
  NCCL_CHECK(ncclCommWindowDeregister(comm, recvwin));
  NCCL_CHECK(ncclMemFree(sendbuf));
  NCCL_CHECK(ncclMemFree(recvbuf));
  NCCL_CHECK(ncclCommDestroy(comm));
  return result;
}

TimingResult run_gin_allgather(const Options& opt, const SemanticsCase& sem, size_t bytes, int rank,
                               int nranks, cudaStream_t stream) {
  ncclUniqueId id = mpi_bcast_nccl_unique_id(rank, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(comm, &props));
  if (!props.deviceApiSupport || props.ginType == NCCL_GIN_TYPE_NONE || !props.multimemSupport) {
    throw std::runtime_error("NCCL communicator does not support GIN symmetric allgather");
  }

  ncclTeam_t rail = ncclTeamRail(comm);
  if (rail.nRanks != 4) {
    throw std::runtime_error("GIN allgather oneshot-rail benchmark currently expects rail.nRanks == 4");
  }

  void* sendbuf = nullptr;
  void* recvbuf = nullptr;
  ncclWindow_t sendwin = nullptr;
  ncclWindow_t recvwin = nullptr;
  size_t recv_bytes = bytes * static_cast<size_t>(nranks);
  NCCL_CHECK(ncclMemAlloc(&sendbuf, bytes));
  NCCL_CHECK(ncclMemAlloc(&recvbuf, recv_bytes));
  NCCL_CHECK(ncclCommWindowRegister(comm, sendbuf, bytes, &sendwin, NCCL_WIN_COLL_SYMMETRIC));
  NCCL_CHECK(ncclCommWindowRegister(comm, recvbuf, recv_bytes, &recvwin, NCCL_WIN_COLL_SYMMETRIC));

  int fill_blocks = std::max(1, std::min<int>(1024, static_cast<int>((recv_bytes + 255) / 256)));
  fill_u8_kernel<<<std::max(1, std::min<int>(1024, static_cast<int>((bytes + 255) / 256))), 256, 0, stream>>>(
      static_cast<uint8_t*>(sendbuf), bytes, static_cast<uint8_t>(rank & 0xff));
  fill_u8_kernel<<<fill_blocks, 256, 0, stream>>>(static_cast<uint8_t*>(recvbuf), recv_bytes, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  ncclGinSyncHandle gin_sync{};
  ncclDevResourceRequirements_t rail_signal_req = {};
  rail_signal_req.ginSignalCount = rail.nRanks * opt.split_blocks;
  rail_signal_req.outGinSignalStart = &gin_sync.railSignals;

  ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.resourceRequirementsList = &rail_signal_req;
  reqs.lsaMultimem = true;
  reqs.barrierCount = opt.split_blocks;
  reqs.lsaBarrierCount = opt.split_blocks;
  reqs.ginContextCount = opt.gin_contexts;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  reqs.ginQueueDepth = 0;

  ncclDevComm_t dev_comm{};
  NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));
  ncclSymkDevWorkArgs4K args4k = make_single_work_args(dev_comm, sendwin, recvwin, bytes, opt.split_blocks);
  args4k.args.kcomm.ginSyncHandle = gin_sync;

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (opt.warmup > 0) {
    gin_allgather_loop_kernel<<<opt.split_blocks, opt.threads, 0, stream>>>(args4k, opt.warmup, sem.init_bar,
                                                                            sem.flush);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  gin_allgather_loop_kernel<<<opt.split_blocks, opt.threads, 0, stream>>>(args4k, opt.iters, sem.init_bar, sem.flush);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  TimingResult result;
  result.local_us = (static_cast<double>(elapsed_ms) * 1000.0) / static_cast<double>(opt.iters);
  if (opt.check) {
    result.check_ok = check_gin_allgather(static_cast<uint8_t*>(recvbuf), bytes, nranks, stream);
  }

  NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
  NCCL_CHECK(ncclCommWindowDeregister(comm, sendwin));
  NCCL_CHECK(ncclCommWindowDeregister(comm, recvwin));
  NCCL_CHECK(ncclMemFree(sendbuf));
  NCCL_CHECK(ncclMemFree(recvbuf));
  NCCL_CHECK(ncclCommDestroy(comm));
  return result;
}

TimingResult run_nvshmem_pingpong(const Options& opt, const SemanticsCase& sem, size_t bytes, int mype,
                                  int npes, cudaStream_t stream) {
  if (npes != 2) {
    throw std::runtime_error("NVSHMEM pingpong requires exactly 2 PEs");
  }

  size_t nelems = (bytes + sizeof(float) - 1) / sizeof(float);
  size_t put_bytes = nelems * sizeof(float);
  float* sendbuf = static_cast<float*>(nvshmem_malloc(put_bytes));
  float* recvbuf = static_cast<float*>(nvshmem_malloc(put_bytes));
  if (sendbuf == nullptr || recvbuf == nullptr) {
    throw std::runtime_error("nvshmem_malloc failed");
  }

  int fill_blocks = std::max(1, std::min<int>(1024, static_cast<int>((nelems + 255) / 256)));
  fill_float_kernel<<<fill_blocks, 256, 0, stream>>>(sendbuf, nelems, 42.0f + static_cast<float>(mype));
  fill_float_kernel<<<fill_blocks, 256, 0, stream>>>(recvbuf, nelems, 0.0f);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int peer = mype == 0 ? 1 : 0;
  int is_sender = mype == 0 ? 1 : 0;
  if (opt.warmup > 0) {
    nvshmem_pingpong_loop_kernel<<<1, kNvshmemPingpongThreads, 0, stream>>>(
        recvbuf, sendbuf, nelems, peer, opt.warmup, 0, is_sender, sem.init_bar, sem.flush);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  nvshmem_pingpong_loop_kernel<<<1, kNvshmemPingpongThreads, 0, stream>>>(
      recvbuf, sendbuf, nelems, peer, opt.iters, static_cast<uint64_t>(opt.warmup), is_sender, sem.init_bar,
      sem.flush);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  TimingResult result;
  result.local_us = (static_cast<double>(elapsed_ms) * 1000.0) / static_cast<double>(opt.iters) / 2.0;
  if (opt.check) {
    result.check_ok = check_first_float(recvbuf, 42.0f + static_cast<float>(peer), stream);
  }

  nvshmem_free(sendbuf);
  nvshmem_free(recvbuf);
  return result;
}

TimingResult run_nvshmem_allgather(const Options& opt, const SemanticsCase& sem, size_t bytes, int mype,
                                   int npes, cudaStream_t stream) {
  size_t words = (bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  uint64_t* sendbuf = static_cast<uint64_t*>(nvshmem_malloc(words * sizeof(uint64_t)));
  uint64_t* recvbuf = static_cast<uint64_t*>(nvshmem_malloc(words * static_cast<size_t>(npes) * sizeof(uint64_t)));
  if (sendbuf == nullptr || recvbuf == nullptr) {
    throw std::runtime_error("nvshmem_malloc failed");
  }

  int send_blocks = std::max(1, std::min<int>(1024, static_cast<int>((words + 255) / 256)));
  int recv_blocks = std::max(1, std::min<int>(1024, static_cast<int>((words * static_cast<size_t>(npes) + 255) / 256)));
  fill_u64_kernel<<<send_blocks, 256, 0, stream>>>(sendbuf, words, 0x100000000ull + static_cast<uint64_t>(mype));
  fill_u64_kernel<<<recv_blocks, 256, 0, stream>>>(recvbuf, words * static_cast<size_t>(npes), 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int ag_threads = std::max(32, std::min(opt.threads, 256));

  if (opt.warmup > 0) {
    nvshmem_allgather_loop_kernel<<<1, ag_threads, 0, stream>>>(recvbuf, sendbuf, words, npes, mype, opt.warmup, 0,
                                                                sem.init_bar, sem.flush);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  nvshmem_allgather_loop_kernel<<<1, ag_threads, 0, stream>>>(recvbuf, sendbuf, words, npes, mype, opt.iters,
                                                              static_cast<uint64_t>(opt.warmup), sem.init_bar,
                                                              sem.flush);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  TimingResult result;
  result.local_us = (static_cast<double>(elapsed_ms) * 1000.0) / static_cast<double>(opt.iters);
  if (opt.check) {
    result.check_ok = check_nvshmem_allgather(recvbuf, words, npes, stream);
  }

  nvshmem_free(sendbuf);
  nvshmem_free(recvbuf);
  return result;
}

void print_row(const Options& opt, const SemanticsCase& sem, size_t bytes, int nranks, int num_blocks,
               double local_us, bool local_ok, int rank) {
  std::vector<double> gathered;
  if (rank == 0) gathered.resize(static_cast<size_t>(nranks));
  MPI_CHECK(MPI_Gather(&local_us, 1, MPI_DOUBLE, gathered.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));

  int ok_int = local_ok ? 1 : 0;
  int all_ok = 0;
  MPI_CHECK(MPI_Allreduce(&ok_int, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

  if (rank == 0) {
    MetricStats stats = compute_stats(gathered);
    size_t recv_bytes = opt.op == Op::kAllGather ? bytes * static_cast<size_t>(nranks) : bytes;
    size_t bw_bytes = opt.op == Op::kAllGather ? bytes * static_cast<size_t>(nranks - 1) : bytes;
    std::printf("%s,%s,%s,%zu,%zu,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n",
                backend_name(opt.backend), op_name(opt.op), sem.name, bytes, recv_bytes, num_blocks,
                sem.init_bar ? 1 : 0, sem.flush ? 1 : 0, stats.avg, stats.min, stats.p50, stats.p90,
                stats.max, gib_per_second_bytes(bw_bytes, stats.avg), all_ok ? "passed" : "failed");
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool mpi_initialized = false;
  bool nvshmem_initialized = false;
  try {
    Options opt = parse_args(argc, argv);
    std::vector<size_t> sizes = make_sizes(opt.min_bytes, opt.max_bytes, opt.factor);
    std::vector<SemanticsCase> semantics = selected_semantics(opt.semantics);

    MPI_CHECK(MPI_Init(&argc, &argv));
    mpi_initialized = true;

    int rank = 0;
    int nranks = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    int local_rank = mpi_local_rank(MPI_COMM_WORLD);

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int device = opt.device >= 0 ? opt.device : local_rank % std::max(1, ndev);
    CUDA_CHECK(cudaSetDevice(device));

    if (opt.op == Op::kPingpong && nranks != 2) {
      throw std::runtime_error("pingpong benchmark requires exactly 2 MPI ranks");
    }

    if (opt.backend == Backend::kNvshmem) {
      nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
      MPI_Comm nvshmem_comm = MPI_COMM_WORLD;
      NVSHMEM_CHECK(nvshmemx_set_attr_mpi_comm_args(&nvshmem_comm, &attr));
      NVSHMEM_CHECK(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
      nvshmem_initialized = true;
      if (nvshmem_my_pe() != rank || nvshmem_n_pes() != nranks) {
        throw std::runtime_error("NVSHMEM PE layout does not match MPI_COMM_WORLD");
      }
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (rank == 0) {
      std::printf("# symm-semantics-perf backend=%s op=%s ranks=%d warmup=%d iters=%d "
                  "timing=single_kernel_loop rank_result=avg\n",
                  backend_name(opt.backend), op_name(opt.op), nranks, opt.warmup, opt.iters);
      std::printf("backend,op,semantics,send_B,recv_B,num_blocks,init_bar,flush_or_quiet,"
                  "avg_us,min_us,p50_us,p90_us,max_us,bw_GBps,check\n");
    }

    for (const SemanticsCase& sem : semantics) {
      for (size_t bytes : sizes) {
        TimingResult result;
        int num_blocks = 1;
        if (opt.backend == Backend::kGin && opt.op == Op::kPingpong) {
          result = run_gin_pingpong(opt, sem, bytes, rank, nranks, stream);
        } else if (opt.backend == Backend::kGin && opt.op == Op::kAllGather) {
          num_blocks = opt.split_blocks;
          result = run_gin_allgather(opt, sem, bytes, rank, nranks, stream);
        } else if (opt.backend == Backend::kNvshmem && opt.op == Op::kPingpong) {
          result = run_nvshmem_pingpong(opt, sem, bytes, rank, nranks, stream);
        } else {
          result = run_nvshmem_allgather(opt, sem, bytes, rank, nranks, stream);
        }
        print_row(opt, sem, bytes, nranks, num_blocks, result.local_us, result.check_ok, rank);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    if (nvshmem_initialized) {
      nvshmem_finalize();
      nvshmem_initialized = false;
    }
    MPI_CHECK(MPI_Finalize());
    mpi_initialized = false;
    return 0;
  } catch (const std::exception& e) {
    fail_fast(e);
    if (nvshmem_initialized) {
      nvshmem_global_exit(1);
    }
    if (mpi_initialized) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
  }
}

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include "common/checks.hpp"
#include "common/csv.hpp"
#include "common/mpi.hpp"
#include "common/parse.hpp"
#include "common/timer.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
#error "NCCL GIN pingpong requires NCCL 2.29.0 or newer"
#endif

namespace {

using namespace nccl_deviceapi_test;

constexpr int kKernelThreads = 32;
constexpr int kSignalBits = 32;
constexpr int kGinResourceCount = 16;
constexpr ncclGinSignal_t kPingSignal = 0;
constexpr ncclGinSignal_t kPongSignal = 1;
constexpr uint64_t kPollBase = 0x100000000ULL;

enum PingpongMode {
  kSignalMode = 0,
  kPollMode = 1,
};

enum ThreadGroup {
  kThreadGroupThread = 0,
  kThreadGroupWarp = 1,
};

struct Options {
  size_t bytes = 1024;
  int warmup_iters = 100;
  int iters = 1000;
  int device = -1;
  double ticks_per_us = 0.0;
  std::string csv = "nccl_gin_pingpong.csv";
  bool check = false;
  bool profile_kernel = false;
  int mode = kSignalMode;
  int thread_group = kThreadGroupThread;
};

__global__ void init_buffers_kernel(char *sendbuf, char *recvbuf, size_t bytes, int rank) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < bytes; i += stride) {
    sendbuf[i] = static_cast<char>((i + rank * 17) & 0xff);
    recvbuf[i] = 0;
  }
}

__global__ void read_signal_bases_kernel(ncclDevComm dev_comm, uint64_t *signal_bases) {
  ncclGin gin{dev_comm, 0};
  if (threadIdx.x == 0) {
    signal_bases[0] = gin.readSignal(kPingSignal);
    signal_bases[1] = gin.readSignal(kPongSignal);
  }
}

__global__ void nccl_gin_pingpong_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                         ncclDevComm dev_comm, const uint64_t *signal_bases,
                                         size_t bytes, int loop_iters,
                                         uint64_t *roundtrip_ticks) {
  if (threadIdx.x != 0) {
    return;
  }

  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclCoopThread thread;

  const int peer = world.rank == 0 ? 1 : 0;
  const uint64_t ping_base = signal_bases[0];
  const uint64_t pong_base = signal_bases[1];

  for (int iter = 0; iter < loop_iters; ++iter) {
    const uint64_t ping_expected = ping_base + static_cast<uint64_t>(iter) + 1;
    const uint64_t pong_expected = pong_base + static_cast<uint64_t>(iter) + 1;
    if (world.rank == 0) {
      const uint64_t t0 = roundtrip_ticks != nullptr ? global_timer() : 0;
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes,
              ncclGin_SignalInc{kPingSignal}, ncclGin_None{}, thread);
      gin.waitSignal(thread, kPongSignal, pong_expected, kSignalBits);
      if (roundtrip_ticks != nullptr) {
        roundtrip_ticks[iter] = global_timer() - t0;
      }
    } else {
      gin.waitSignal(thread, kPingSignal, ping_expected, kSignalBits);
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes,
              ncclGin_SignalInc{kPongSignal}, ncclGin_None{}, thread);
    }
  }
}

__global__ void nccl_gin_pingpong_warp_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                              ncclDevComm dev_comm,
                                              const uint64_t *signal_bases, size_t bytes,
                                              int loop_iters, uint64_t *roundtrip_ticks) {
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclCoopWarpSpan warps(0, 1, 0);

  if (threadIdx.x >= kKernelThreads) {
    return;
  }

  const int peer = world.rank == 0 ? 1 : 0;
  const uint64_t ping_base = signal_bases[0];
  const uint64_t pong_base = signal_bases[1];

  for (int iter = 0; iter < loop_iters; ++iter) {
    const uint64_t ping_expected = ping_base + static_cast<uint64_t>(iter) + 1;
    const uint64_t pong_expected = pong_base + static_cast<uint64_t>(iter) + 1;
    if (world.rank == 0) {
      const bool lane0 = warps.thread_rank() == 0;
      const uint64_t t0 = roundtrip_ticks != nullptr && lane0 ? global_timer() : 0;
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes,
              ncclGin_SignalInc{kPingSignal}, ncclGin_None{}, warps);
      gin.waitSignal(warps, kPongSignal, pong_expected, kSignalBits);
      if (roundtrip_ticks != nullptr && lane0) {
        roundtrip_ticks[iter] = global_timer() - t0;
      }
    } else {
      gin.waitSignal(warps, kPingSignal, ping_expected, kSignalBits);
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes,
              ncclGin_SignalInc{kPongSignal}, ncclGin_None{}, warps);
    }
  }
}

__global__ void nccl_gin_pingpong_poll_kernel(char *sendbuf, char *recvbuf,
                                              ncclWindow_t sendwin, ncclWindow_t recvwin,
                                              ncclDevComm dev_comm, size_t bytes,
                                              uint64_t marker_base, int loop_iters,
                                              uint64_t *roundtrip_ticks) {
  if (threadIdx.x != 0) {
    return;
  }

  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclCoopThread thread;

  const int peer = world.rank == 0 ? 1 : 0;
  volatile uint64_t *recv_flag = reinterpret_cast<volatile uint64_t *>(recvbuf);
  uint64_t *send_flag = reinterpret_cast<uint64_t *>(sendbuf);

  for (int iter = 0; iter < loop_iters; ++iter) {
    const uint64_t marker = marker_base + static_cast<uint64_t>(iter) + 1;
    if (world.rank == 0) {
      const uint64_t t0 = roundtrip_ticks != nullptr ? global_timer() : 0;
      *send_flag = marker;
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_None{}, ncclGin_None{},
              thread);
      while (*recv_flag != marker) {
      }
      if (roundtrip_ticks != nullptr) {
        roundtrip_ticks[iter] = global_timer() - t0;
      }
    } else {
      while (*recv_flag != marker) {
      }
      *send_flag = marker;
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_None{}, ncclGin_None{},
              thread);
    }
  }
}

__global__ void nccl_gin_pingpong_poll_warp_kernel(char *sendbuf, char *recvbuf,
                                                   ncclWindow_t sendwin, ncclWindow_t recvwin,
                                                   ncclDevComm dev_comm, size_t bytes,
                                                   uint64_t marker_base, int loop_iters,
                                                   uint64_t *roundtrip_ticks) {
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclCoopWarpSpan warps(0, 1, 0);

  const int peer = world.rank == 0 ? 1 : 0;
  const bool active = threadIdx.x < kKernelThreads;
  const bool lane0 = active && warps.thread_rank() == 0;
  volatile uint64_t *recv_flag = reinterpret_cast<volatile uint64_t *>(recvbuf);
  uint64_t *send_flag = reinterpret_cast<uint64_t *>(sendbuf);

  if (!active) {
    return;
  }

  for (int iter = 0; iter < loop_iters; ++iter) {
    const uint64_t marker = marker_base + static_cast<uint64_t>(iter) + 1;
    if (world.rank == 0) {
      const uint64_t t0 = roundtrip_ticks != nullptr && lane0 ? global_timer() : 0;
      if (lane0) {
        *send_flag = marker;
      }
      __syncwarp();
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_None{}, ncclGin_None{},
              warps);
      if (lane0) {
        while (*recv_flag != marker) {
        }
        if (roundtrip_ticks != nullptr) {
          roundtrip_ticks[iter] = global_timer() - t0;
        }
      }
      __syncwarp();
    } else {
      if (lane0) {
        while (*recv_flag != marker) {
        }
        *send_flag = marker;
      }
      __syncwarp();
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_None{}, ncclGin_None{},
              warps);
    }
  }
}

static const char *thread_group_name(int thread_group) {
  return thread_group == kThreadGroupWarp ? "warp" : "thread";
}

static Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--bytes") {
      opt.bytes = parse_size(need_value("--bytes"));
    } else if (arg == "--warmup-iters" || arg == "--warmup") {
      opt.warmup_iters = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--iters") {
      opt.iters = parse_int(need_value("--iters"), "--iters");
    } else if (arg == "--device") {
      opt.device = parse_int(need_value("--device"), "--device");
    } else if (arg == "--ticks-per-us") {
      opt.ticks_per_us = std::stod(need_value("--ticks-per-us"));
    } else if (arg == "--thread-group") {
      std::string group = need_value("--thread-group");
      if (group == "thread") {
        opt.thread_group = kThreadGroupThread;
      } else if (group == "warp") {
        opt.thread_group = kThreadGroupWarp;
      } else {
        throw std::invalid_argument("unknown --thread-group; expected thread or warp");
      }
    } else if (arg == "--mode") {
      std::string mode = need_value("--mode");
      if (mode == "signal") {
        opt.mode = kSignalMode;
      } else if (mode == "poll" || mode == "put-without-signal" ||
                 mode == "put_without_signal") {
        opt.mode = kPollMode;
      } else {
        throw std::invalid_argument("unknown --mode; expected signal or poll");
      }
    } else if (arg == "--put-without-signal" || arg == "--put_without_signal") {
      opt.mode = kPollMode;
    } else if (arg == "--csv") {
      opt.csv = need_value("--csv");
    } else if (arg == "--check") {
      opt.check = true;
    } else if (arg == "--profile-kernel" || arg == "--profile_kernel") {
      opt.profile_kernel = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: nccl_gin_pingpong_perf [options]\n"
          << "  --bytes <N|64B|1KB|1MB>       payload bytes per one-way put\n"
          << "  --warmup-iters <N>             warmup iterations\n"
          << "  --iters <N>                    measured iterations\n"
          << "  --device <N>                   CUDA device index; defaults to MPI local rank\n"
          << "  --ticks-per-us <F>             override GPU globaltimer ticks/us for profiling\n"
          << "  --thread-group thread|warp     GIN cooperative group; default thread\n"
          << "  --mode signal|poll             poll uses put without signal/counter\n"
          << "  --put-without-signal           alias for --mode poll\n"
          << "  --csv <path>                   output CSV path\n"
          << "  --check                        verify final returned payload on rank 0\n"
          << "  --profile-kernel               record per-iteration rank0 RTT ticks\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (opt.bytes == 0 || opt.iters <= 0 || opt.warmup_iters < 0 || opt.device < -1) {
    throw std::invalid_argument("bytes/iters must be positive and warmup non-negative");
  }
  if (opt.ticks_per_us < 0.0) {
    throw std::invalid_argument("--ticks-per-us must be positive");
  }
  if (opt.mode == kPollMode && opt.bytes < sizeof(uint64_t)) {
    throw std::invalid_argument("--mode poll requires --bytes >= 8");
  }
  return opt;
}

static void write_csv(const std::string &path, int rank, const Options &opt, double elapsed_us,
                      bool ok) {
  CsvFile csv(path, 6);
  std::ofstream &out = csv.stream();

  out << std::fixed << std::setprecision(6);
  const double pingpong_us = elapsed_us / static_cast<double>(opt.iters);
  const double one_way_us = pingpong_us / 2.0;
  const double bw_gib_s = one_way_us > 0.0
                              ? static_cast<double>(opt.bytes) / 1024.0 / 1024.0 / 1024.0 /
                                    one_way_us * 1000000.0
                              : 0.0;
  const char *mode_name = opt.mode == kPollMode ? "poll" : "signal";
  write_csv_row(out, "rank", "bytes", "mode", "thread_group", "warmup_iters", "iters",
                "elapsed_us", "pingpong_us", "one_way_us", "bw_gib_s", "check");
  write_csv_row(out, rank, opt.bytes, mode_name, thread_group_name(opt.thread_group),
                opt.warmup_iters, opt.iters, elapsed_us, pingpong_us, one_way_us, bw_gib_s,
                ok ? "ok" : "failed");
}

static void print_summary(int rank, const Options &opt, double elapsed_us, bool ok) {
  if (rank != 0) {
    return;
  }
  const double pingpong_us = elapsed_us / static_cast<double>(opt.iters);
  const double one_way_us = pingpong_us / 2.0;
  const double bw_gib_s = one_way_us > 0.0
                              ? static_cast<double>(opt.bytes) / 1024.0 / 1024.0 / 1024.0 /
                                    one_way_us * 1000000.0
                              : 0.0;
  const char *mode_name = opt.mode == kPollMode ? "poll" : "signal";
  std::cout << std::fixed << std::setprecision(3)
            << "nccl_gin_pingpong_perf result bytes=" << opt.bytes << " mode=" << mode_name
            << " thread_group=" << thread_group_name(opt.thread_group)
            << " warmup_iters=" << opt.warmup_iters << " iters=" << opt.iters
            << " elapsed_us=" << elapsed_us << " pingpong_us=" << pingpong_us
            << " one_way_us=" << one_way_us << " one_way_bw_gib_s=" << bw_gib_s
            << " check=" << (ok ? "ok" : "failed") << "\n";
}

static double percentile_sorted(const std::vector<double> &values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  const double rank = percentile / 100.0 * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(rank);
  const size_t hi = std::min(lo + 1, values.size() - 1);
  const double frac = rank - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

static void print_kernel_profile(int rank, const Options &opt,
                                 const std::vector<uint64_t> &roundtrip_ticks,
                                 double ticks_per_us) {
  if (rank != 0 || !opt.profile_kernel) {
    return;
  }
  std::vector<double> one_way_us;
  one_way_us.reserve(roundtrip_ticks.size());
  double sum_one_way_us = 0.0;
  for (uint64_t ticks : roundtrip_ticks) {
    const double value = static_cast<double>(ticks) / ticks_per_us / 2.0;
    one_way_us.push_back(value);
    sum_one_way_us += value;
  }
  std::sort(one_way_us.begin(), one_way_us.end());
  const double mean_one_way_us = sum_one_way_us / static_cast<double>(one_way_us.size());
  const char *mode_name = opt.mode == kPollMode ? "poll" : "signal";
  std::cout << std::fixed << std::setprecision(3)
            << "nccl_gin_pingpong_kernel_profile bytes=" << opt.bytes
            << " mode=" << mode_name
            << " thread_group=" << thread_group_name(opt.thread_group)
            << " iters=" << opt.iters
            << " ticks_per_us=" << ticks_per_us
            << " min_one_way_us=" << one_way_us.front()
            << " mean_one_way_us=" << mean_one_way_us
            << " p50_one_way_us=" << percentile_sorted(one_way_us, 50.0)
            << " p99_one_way_us=" << percentile_sorted(one_way_us, 99.0)
            << " max_one_way_us=" << one_way_us.back() << "\n";
}

static bool verify_rank0(char *recvbuf, size_t bytes) {
  std::vector<char> host(bytes);
  CUDA_CHECK(cudaMemcpy(host.data(), recvbuf, bytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < bytes; ++i) {
    char expected = static_cast<char>((i + 17) & 0xff);
    if (host[i] != expected) {
      std::cerr << "check failed at byte " << i << ": got "
                << static_cast<int>(static_cast<unsigned char>(host[i])) << ", expected "
                << static_cast<int>(static_cast<unsigned char>(expected)) << "\n";
      return false;
    }
  }
  return true;
}

static bool verify_rank0_poll(char *recvbuf, int warmup_iters, int iters) {
  uint64_t host = 0;
  CUDA_CHECK(cudaMemcpy(&host, recvbuf, sizeof(host), cudaMemcpyDeviceToHost));
  const uint64_t expected =
      kPollBase + static_cast<uint64_t>(warmup_iters) + static_cast<uint64_t>(iters);
  if (host != expected) {
    std::cerr << "poll check failed: got " << host << ", expected " << expected << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
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
        std::cerr << "nccl_gin_pingpong_perf requires exactly 2 ranks, got " << nranks << "\n";
      }
      MPI_CHECK(MPI_Finalize());
      return 2;
    }

    int dev = opt.device >= 0 ? opt.device : mpi_local_rank(MPI_COMM_WORLD);
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0 || dev >= dev_count) {
      throw std::runtime_error("local rank exceeds visible CUDA device count");
    }
    CUDA_CHECK(cudaSetDevice(dev));
    double timer_ticks_per_us = 0.0;
    if (opt.profile_kernel) {
      timer_ticks_per_us =
          opt.ticks_per_us > 0.0 ? opt.ticks_per_us : calibrate_ticks_per_us(100000000ULL);
    }

    ncclUniqueId id = mpi_bcast_nccl_unique_id(rank, MPI_COMM_WORLD);

    ncclComm_t comm = nullptr;
    ncclDevComm dev_comm{};
    char *sendbuf = nullptr;
    char *recvbuf = nullptr;
    ncclWindow_t sendwin = nullptr;
    ncclWindow_t recvwin = nullptr;
    uint64_t *signal_bases_d = nullptr;
    uint64_t *roundtrip_ticks_d = nullptr;
    cudaStream_t stream{};
    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};

    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    if (!props.deviceApiSupport) {
      throw std::runtime_error("NCCL device API is not supported by this communicator");
    }
    if (props.ginType == NCCL_GIN_TYPE_NONE) {
      throw std::runtime_error("NCCL GIN is not enabled for this communicator");
    }

    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void **>(&sendbuf), opt.bytes));
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void **>(&recvbuf), opt.bytes));
    NCCL_CHECK(ncclCommWindowRegister(comm, sendbuf, opt.bytes, &sendwin,
                                      NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, recvbuf, opt.bytes, &recvwin,
                                      NCCL_WIN_COLL_SYMMETRIC));

    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = kGinResourceCount;
    reqs.lsaBarrierCount = kGinResourceCount;
    reqs.railGinBarrierCount = kGinResourceCount;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
    reqs.ginForceEnable = true;
#endif
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&signal_bases_d), sizeof(uint64_t) * 2));
    if (opt.profile_kernel && rank == 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&roundtrip_ticks_d),
                            sizeof(uint64_t) * static_cast<size_t>(opt.iters)));
    }
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    int init_blocks = static_cast<int>(std::min<size_t>((opt.bytes + 255) / 256, 1024));
    init_blocks = std::max(init_blocks, 1);
    init_buffers_kernel<<<init_blocks, 256, 0, stream>>>(sendbuf, recvbuf, opt.bytes, rank);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (opt.warmup_iters > 0) {
      if (opt.mode == kPollMode) {
        if (opt.thread_group == kThreadGroupWarp) {
          nccl_gin_pingpong_poll_warp_kernel<<<1, kKernelThreads, 0, stream>>>(
              sendbuf, recvbuf, sendwin, recvwin, dev_comm, opt.bytes, kPollBase,
              opt.warmup_iters, nullptr);
        } else {
          nccl_gin_pingpong_poll_kernel<<<1, kKernelThreads, 0, stream>>>(
              sendbuf, recvbuf, sendwin, recvwin, dev_comm, opt.bytes, kPollBase,
              opt.warmup_iters, nullptr);
        }
      } else {
        read_signal_bases_kernel<<<1, kKernelThreads, 0, stream>>>(dev_comm, signal_bases_d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (opt.thread_group == kThreadGroupWarp) {
          nccl_gin_pingpong_warp_kernel<<<1, kKernelThreads, 0, stream>>>(
              sendwin, recvwin, dev_comm, signal_bases_d, opt.bytes, opt.warmup_iters,
              nullptr);
        } else {
          nccl_gin_pingpong_kernel<<<1, kKernelThreads, 0, stream>>>(
              sendwin, recvwin, dev_comm, signal_bases_d, opt.bytes, opt.warmup_iters,
              nullptr);
        }
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (opt.mode == kPollMode) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      CUDA_CHECK(cudaEventRecord(start_event, stream));
      if (opt.thread_group == kThreadGroupWarp) {
        nccl_gin_pingpong_poll_warp_kernel<<<1, kKernelThreads, 0, stream>>>(
            sendbuf, recvbuf, sendwin, recvwin, dev_comm, opt.bytes,
            kPollBase + static_cast<uint64_t>(opt.warmup_iters), opt.iters,
            roundtrip_ticks_d);
      } else {
        nccl_gin_pingpong_poll_kernel<<<1, kKernelThreads, 0, stream>>>(
            sendbuf, recvbuf, sendwin, recvwin, dev_comm, opt.bytes,
            kPollBase + static_cast<uint64_t>(opt.warmup_iters), opt.iters,
            roundtrip_ticks_d);
      }
    } else {
      read_signal_bases_kernel<<<1, kKernelThreads, 0, stream>>>(dev_comm, signal_bases_d);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      CUDA_CHECK(cudaEventRecord(start_event, stream));
      if (opt.thread_group == kThreadGroupWarp) {
        nccl_gin_pingpong_warp_kernel<<<1, kKernelThreads, 0, stream>>>(
            sendwin, recvwin, dev_comm, signal_bases_d, opt.bytes, opt.iters,
            roundtrip_ticks_d);
      } else {
        nccl_gin_pingpong_kernel<<<1, kKernelThreads, 0, stream>>>(
            sendwin, recvwin, dev_comm, signal_bases_d, opt.bytes, opt.iters,
            roundtrip_ticks_d);
      }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;

    bool ok = true;
    if (opt.check && rank == 0) {
      ok = opt.mode == kPollMode ? verify_rank0_poll(recvbuf, opt.warmup_iters, opt.iters)
                                 : verify_rank0(recvbuf, opt.bytes);
    }

    write_csv(rank_csv_path(opt.csv, rank, nranks), rank, opt, elapsed_us, ok);
    print_summary(rank, opt, elapsed_us, ok);
    if (opt.profile_kernel && rank == 0) {
      std::vector<uint64_t> roundtrip_ticks(static_cast<size_t>(opt.iters));
      CUDA_CHECK(cudaMemcpy(roundtrip_ticks.data(), roundtrip_ticks_d,
                            sizeof(uint64_t) * roundtrip_ticks.size(),
                            cudaMemcpyDeviceToHost));
      print_kernel_profile(rank, opt, roundtrip_ticks, timer_ticks_per_us);
    }

    if (rank == 0) {
      std::cout << "nccl_gin_pingpong_perf complete: bytes=" << opt.bytes
                << " warmup_iters=" << opt.warmup_iters << " iters=" << opt.iters
                << " mode=" << (opt.mode == kPollMode ? "poll" : "signal")
                << " thread_group=" << thread_group_name(opt.thread_group)
                << " csv_prefix=" << opt.csv
                << " check=" << (ok ? "ok" : "failed") << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (roundtrip_ticks_d != nullptr) {
      CUDA_CHECK(cudaFree(roundtrip_ticks_d));
    }
    CUDA_CHECK(cudaFree(signal_bases_d));
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    NCCL_CHECK(ncclCommWindowDeregister(comm, sendwin));
    NCCL_CHECK(ncclCommWindowDeregister(comm, recvwin));
    NCCL_CHECK(ncclMemFree(sendbuf));
    NCCL_CHECK(ncclMemFree(recvbuf));
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
    return ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    if (mpi_initialized) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
  }
}

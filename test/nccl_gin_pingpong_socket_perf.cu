#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "common/checks.hpp"
#include "common/env.hpp"
#include "common/parse.hpp"
#include "common/socket.hpp"
#include "common/timer.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
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
  kNegZeroMode = 2,
};

enum ThreadGroup {
  kThreadGroupThread = 0,
  kThreadGroupWarp = 1,
};

struct Options {
  size_t bytes = 8;
  int warmup_iters = 1000;
  int iters = 100000;
  int device = -1;
  int src_dev = -1;
  int dst_dev = -1;
  int poll_blocks = 0;
  int port = 22690;
  double ticks_per_us = 0.0;
  bool check = false;
  bool profile_kernel = false;
  int mode = kSignalMode;
  int thread_group = kThreadGroupThread;
  std::string master_addr;
};

__device__ __forceinline__ uint32_t negative_zero_word() {
  return 0x80000000u;
}

__global__ void init_buffers_kernel(char *sendbuf, char *recvbuf, size_t bytes, int rank) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < bytes; i += stride) {
    sendbuf[i] = static_cast<char>((i + rank * 17) & 0xff);
    recvbuf[i] = 0;
  }
}

__global__ void init_negzero_buffers_kernel(char *sendbuf, char *recvbuf,
                                            size_t elems_per_slot, int slots) {
  const size_t total_elems = elems_per_slot * static_cast<size_t>(slots);
  uint32_t *send = reinterpret_cast<uint32_t *>(sendbuf);
  uint32_t *recv = reinterpret_cast<uint32_t *>(recvbuf);
  const uint32_t negzero = negative_zero_word();

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elems;
       idx += blockDim.x * gridDim.x) {
    send[idx] = 0u;
    recv[idx] = negzero;
  }
}

__global__ void read_signal_bases_kernel(ncclDevComm dev_comm, uint64_t *signal_bases) {
  ncclGin gin{dev_comm, 0};
  if (threadIdx.x == 0) {
    signal_bases[0] = gin.readSignal(kPingSignal);
    signal_bases[1] = gin.readSignal(kPongSignal);
  }
}

__global__ void nccl_gin_pingpong_signal_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                                ncclDevComm dev_comm,
                                                const uint64_t *signal_bases, size_t bytes,
                                                int loop_iters,
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
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_SignalInc{kPingSignal},
              ncclGin_None{}, thread);
      gin.waitSignal(thread, kPongSignal, pong_expected, kSignalBits);
      if (roundtrip_ticks != nullptr) {
        roundtrip_ticks[iter] = global_timer() - t0;
      }
    } else {
      gin.waitSignal(thread, kPingSignal, ping_expected, kSignalBits);
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_SignalInc{kPongSignal},
              ncclGin_None{}, thread);
    }
  }
}

__global__ void nccl_gin_pingpong_signal_warp_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                                     ncclDevComm dev_comm,
                                                     const uint64_t *signal_bases, size_t bytes,
                                                     int loop_iters,
                                                     uint64_t *roundtrip_ticks) {
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclCoopWarpSpan warps(0, 1, 0);

  const int peer = world.rank == 0 ? 1 : 0;
  const uint64_t ping_base = signal_bases[0];
  const uint64_t pong_base = signal_bases[1];

  if (threadIdx.x >= kKernelThreads) {
    return;
  }

  for (int iter = 0; iter < loop_iters; ++iter) {
    const uint64_t ping_expected = ping_base + static_cast<uint64_t>(iter) + 1;
    const uint64_t pong_expected = pong_base + static_cast<uint64_t>(iter) + 1;
    if (world.rank == 0) {
      const bool lane0 = warps.thread_rank() == 0;
      const uint64_t t0 = roundtrip_ticks != nullptr && lane0 ? global_timer() : 0;
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_SignalInc{kPingSignal},
              ncclGin_None{}, warps);
      gin.waitSignal(warps, kPongSignal, pong_expected, kSignalBits);
      if (roundtrip_ticks != nullptr && lane0) {
        roundtrip_ticks[iter] = global_timer() - t0;
      }
    } else {
      gin.waitSignal(warps, kPingSignal, ping_expected, kSignalBits);
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_SignalInc{kPongSignal},
              ncclGin_None{}, warps);
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

  if (active) {
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
}

__device__ __forceinline__ void wait_blocks_done(volatile unsigned int *counter,
                                                 unsigned int expected) {
  while (*counter < expected) {
  }
}

__device__ void poll_slot_not_negzero(char *recv_slot, size_t elems_per_slot,
                                      unsigned int *done_counts, int slot) {
  volatile uint32_t *recv = reinterpret_cast<volatile uint32_t *>(recv_slot);
  const uint32_t negzero = negative_zero_word();

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elems_per_slot;
       idx += blockDim.x * gridDim.x) {
    while (recv[idx] == negzero) {
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(done_counts + slot, 1u);
  }
}

__global__ void nccl_gin_pingpong_negzero_kernel(char *sendbuf, char *recvbuf,
                                                 ncclWindow_t sendwin,
                                                 ncclWindow_t recvwin,
                                                 ncclDevComm dev_comm,
                                                 size_t bytes_per_slot,
                                                 size_t elems_per_slot, int start_slot,
                                                 int loop_iters,
                                                 unsigned int *done_counts,
                                                 uint64_t *roundtrip_ticks) {
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclCoopThread thread;

  const int peer = world.rank == 0 ? 1 : 0;

  for (int iter = 0; iter < loop_iters; ++iter) {
    const int slot = start_slot + iter;
    const size_t byte_offset = static_cast<size_t>(slot) * bytes_per_slot;
    uint64_t t0 = 0;

    if (world.rank == 0 && blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        t0 = roundtrip_ticks != nullptr ? global_timer() : 0;
        gin.put(world, peer, recvwin, byte_offset, sendwin, byte_offset, bytes_per_slot,
                ncclGin_None{}, ncclGin_None{}, thread);
      }
      __syncthreads();
    }

    poll_slot_not_negzero(recvbuf + byte_offset, elems_per_slot, done_counts, slot);

    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        wait_blocks_done(reinterpret_cast<volatile unsigned int *>(done_counts + slot),
                         static_cast<unsigned int>(gridDim.x));
        if (world.rank == 1) {
          gin.put(world, peer, recvwin, byte_offset, sendwin, byte_offset, bytes_per_slot,
                  ncclGin_None{}, ncclGin_None{}, thread);
        } else if (roundtrip_ticks != nullptr) {
          roundtrip_ticks[iter] = global_timer() - t0;
        }
      }
      __syncthreads();
    }
  }
}

static const char *thread_group_name(int thread_group) {
  return thread_group == kThreadGroupWarp ? "warp" : "thread";
}

static const char *mode_name(int mode) {
  if (mode == kSignalMode) {
    return "signal";
  }
  if (mode == kPollMode) {
    return "poll";
  }
  return "negzero";
}

static Options parse_args(int argc, char **argv) {
  Options opt;
  opt.master_addr = env_string("NCCL_GIN_MASTER_ADDR", "127.0.0.1");
  opt.port = env_int("NCCL_GIN_MASTER_PORT", opt.port);

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
    } else if (arg == "--src-dev" || arg == "--src_dev") {
      opt.src_dev = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--dst-dev" || arg == "--dst_dev") {
      opt.dst_dev = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--poll-blocks") {
      opt.poll_blocks = parse_int(need_value("--poll-blocks"), "--poll-blocks");
    } else if (arg == "--master") {
      opt.master_addr = need_value("--master");
    } else if (arg == "--port") {
      opt.port = parse_int(need_value("--port"), "--port");
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
      } else if (mode == "negzero" || mode == "negative-zero" ||
                 mode == "negative_zero") {
        opt.mode = kNegZeroMode;
      } else {
        throw std::invalid_argument("unknown --mode; expected signal, poll, or negzero");
      }
    } else if (arg == "--negzero-poll" || arg == "--negzero_poll") {
      opt.mode = kNegZeroMode;
    } else if (arg == "--put-without-signal" || arg == "--put_without_signal") {
      opt.mode = kPollMode;
    } else if (arg == "--check") {
      opt.check = true;
    } else if (arg == "--profile-kernel" || arg == "--profile_kernel") {
      opt.profile_kernel = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: nccl_gin_pingpong_socket_perf [options]\n"
          << "  --bytes <N|64B|1KB|1MB>       payload bytes per one-way put\n"
          << "  --warmup-iters <N>             warmup iterations\n"
          << "  --iters <N>                    measured iterations\n"
          << "  --device <N>                   CUDA device index fallback\n"
          << "  --src-dev <N>                  CUDA device for rank 0\n"
          << "  --dst-dev <N>                  CUDA device for rank 1\n"
          << "  --poll-blocks <N>              negzero poll blocks; default SM count\n"
          << "  --master <IPv4|IPv6>           rank 0 address for socket bootstrap\n"
          << "  --port <N>                     base TCP port for socket bootstrap\n"
          << "  --ticks-per-us <F>             override GPU globaltimer ticks/us for profiling\n"
          << "  --thread-group thread|warp     GIN cooperative group; default thread\n"
          << "  --mode signal|poll|negzero     negzero polls all float elements for != -0\n"
          << "  --put-without-signal           alias for --mode poll\n"
          << "  --negzero-poll                 alias for --mode negzero\n"
          << "  --check                        verify final returned payload on rank 0\n"
          << "  --profile-kernel               record per-iteration rank0 RTT ticks\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (opt.bytes == 0 || opt.iters <= 0 || opt.warmup_iters < 0 || opt.device < -1 ||
      opt.src_dev < -1 || opt.dst_dev < -1 || opt.poll_blocks < 0 || opt.port <= 0) {
    throw std::invalid_argument("bytes/iters/port must be positive and warmup non-negative");
  }
  if (opt.ticks_per_us < 0.0) {
    throw std::invalid_argument("--ticks-per-us must be positive");
  }
  if (opt.mode == kPollMode && opt.bytes < sizeof(uint64_t)) {
    throw std::invalid_argument("--mode poll requires --bytes >= 8");
  }
  if (opt.mode == kNegZeroMode && opt.bytes % sizeof(uint32_t) != 0) {
    throw std::invalid_argument("--mode negzero requires --bytes to be a multiple of 4");
  }
  return opt;
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

static int get_rank() {
  return env_int("RANK", env_int("OMPI_COMM_WORLD_RANK", env_int("PMI_RANK", 0)));
}

static int get_nranks() {
  return env_int("WORLD_SIZE", env_int("OMPI_COMM_WORLD_SIZE", env_int("PMI_SIZE", 1)));
}

static int get_local_rank(int rank) {
  return env_int("LOCAL_RANK", env_int("OMPI_COMM_WORLD_LOCAL_RANK", rank));
}

static void socket_barrier(int rank, const std::string &master_addr, int port) {
  char token = 1;
  if (rank == 0) {
    int listen_fd = listen_socket(port);
    int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0) {
      throw std::runtime_error("accept barrier failed");
    }
    if (!recv_all(fd, &token, sizeof(token)) || !send_all(fd, &token, sizeof(token))) {
      throw std::runtime_error("socket barrier transfer failed");
    }
    ::close(fd);
    ::close(listen_fd);
  } else {
    int fd = connect_socket(master_addr, port);
    if (!send_all(fd, &token, sizeof(token)) || !recv_all(fd, &token, sizeof(token))) {
      throw std::runtime_error("socket barrier transfer failed");
    }
    ::close(fd);
  }
}

static int socket_allreduce_min(int rank, const std::string &master_addr, int port, int value) {
  if (rank == 0) {
    int listen_fd = listen_socket(port);
    int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0) {
      throw std::runtime_error("accept allreduce failed");
    }
    int peer_value = 0;
    if (!recv_all(fd, &peer_value, sizeof(peer_value))) {
      throw std::runtime_error("recv allreduce value failed");
    }
    int result = std::min(value, peer_value);
    if (!send_all(fd, &result, sizeof(result))) {
      throw std::runtime_error("send allreduce result failed");
    }
    ::close(fd);
    ::close(listen_fd);
    return result;
  }

  int fd = connect_socket(master_addr, port);
  if (!send_all(fd, &value, sizeof(value))) {
    throw std::runtime_error("send allreduce value failed");
  }
  int result = 0;
  if (!recv_all(fd, &result, sizeof(result))) {
    throw std::runtime_error("recv allreduce result failed");
  }
  ::close(fd);
  return result;
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

static bool verify_rank0_negzero(char *recvbuf, size_t bytes_per_slot, int slot) {
  const size_t elems_per_slot = bytes_per_slot / sizeof(uint32_t);
  std::vector<uint32_t> host(elems_per_slot);
  CUDA_CHECK(cudaMemcpy(host.data(), recvbuf + static_cast<size_t>(slot) * bytes_per_slot,
                        bytes_per_slot, cudaMemcpyDeviceToHost));
  const uint32_t negzero = 0x80000000u;
  for (size_t i = 0; i < host.size(); ++i) {
    if (host[i] == negzero) {
      std::cerr << "negzero check failed at slot " << slot << " element " << i << "\n";
      return false;
    }
  }
  return true;
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
  std::cout << std::fixed << std::setprecision(3)
            << "nccl_gin_pingpong_socket_kernel_profile bytes=" << opt.bytes
            << " mode=" << mode_name(opt.mode)
            << " thread_group=" << thread_group_name(opt.thread_group)
            << " iters=" << opt.iters
            << " ticks_per_us=" << ticks_per_us
            << " min_one_way_us=" << one_way_us.front()
            << " mean_one_way_us=" << mean_one_way_us
            << " p50_one_way_us=" << percentile_sorted(one_way_us, 50.0)
            << " p99_one_way_us=" << percentile_sorted(one_way_us, 99.0)
            << " max_one_way_us=" << one_way_us.back() << "\n";
}

static void print_summary(int rank, const Options &opt, double elapsed_us, bool ok) {
  if (rank != 0) {
    return;
  }
  const double pingpong_us = elapsed_us / static_cast<double>(opt.iters);
  const double one_way_us = pingpong_us / 2.0;
  const double one_way_bw_gib_s = one_way_us > 0.0
                                      ? static_cast<double>(opt.bytes) / 1024.0 / 1024.0 /
                                            1024.0 / one_way_us * 1000000.0
                                      : 0.0;
  std::cout << std::fixed << std::setprecision(3)
            << "nccl_gin_pingpong_socket_perf result bytes=" << opt.bytes
            << " mode=" << mode_name(opt.mode)
            << " thread_group=" << thread_group_name(opt.thread_group)
            << " warmup_iters=" << opt.warmup_iters << " iters=" << opt.iters
            << " elapsed_us=" << elapsed_us << " pingpong_us=" << pingpong_us
            << " one_way_us=" << one_way_us << " one_way_bw_gib_s=" << one_way_bw_gib_s
            << " check=" << (ok ? "ok" : "failed") << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Options opt = parse_args(argc, argv);

    int rank = get_rank();
    int nranks = get_nranks();
    if (nranks != 2 || rank < 0 || rank >= nranks) {
      std::cerr << "nccl_gin_pingpong_socket_perf requires exactly 2 ranks, got rank="
                << rank << " nranks=" << nranks << "\n";
      return 2;
    }

    int dev = -1;
    if (rank == 0 && opt.src_dev >= 0) {
      dev = opt.src_dev;
    } else if (rank == 1 && opt.dst_dev >= 0) {
      dev = opt.dst_dev;
    } else {
      dev = opt.device >= 0 ? opt.device : get_local_rank(rank);
    }
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0 || dev < 0 || dev >= dev_count) {
      throw std::runtime_error("selected CUDA device is outside visible device count");
    }
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp device_prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev));
    const int poll_blocks =
        opt.poll_blocks > 0 ? opt.poll_blocks : std::max(device_prop.multiProcessorCount, 1);
    double timer_ticks_per_us = 0.0;
    if (opt.profile_kernel) {
      timer_ticks_per_us =
          opt.ticks_per_us > 0.0 ? opt.ticks_per_us : calibrate_ticks_per_us(100000000ULL);
    }

    ncclUniqueId id = exchange_unique_id_socket(rank, nranks, opt.master_addr, opt.port);

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
    unsigned int *done_counts_d = nullptr;

    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    if (!props.deviceApiSupport) {
      throw std::runtime_error("NCCL device API is not supported by this communicator");
    }
    if (props.ginType == NCCL_GIN_TYPE_NONE) {
      throw std::runtime_error("NCCL GIN is not enabled for this communicator");
    }

    const int total_slots = opt.warmup_iters + opt.iters;
    const size_t bytes_per_window =
        opt.mode == kNegZeroMode ? opt.bytes * static_cast<size_t>(total_slots) : opt.bytes;
    const size_t elems_per_slot = opt.bytes / sizeof(uint32_t);

    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void **>(&sendbuf), bytes_per_window));
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void **>(&recvbuf), bytes_per_window));
    NCCL_CHECK(ncclCommWindowRegister(comm, sendbuf, bytes_per_window, &sendwin,
                                      NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, recvbuf, bytes_per_window, &recvwin,
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
    if (opt.mode == kNegZeroMode) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&done_counts_d),
                            sizeof(unsigned int) * static_cast<size_t>(total_slots)));
    }
    if (opt.profile_kernel && rank == 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&roundtrip_ticks_d),
                            sizeof(uint64_t) * static_cast<size_t>(opt.iters)));
    }
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    if (opt.mode == kNegZeroMode) {
      CUDA_CHECK(cudaMemsetAsync(done_counts_d, 0,
                                 sizeof(unsigned int) * static_cast<size_t>(total_slots),
                                 stream));
      const size_t total_elems = elems_per_slot * static_cast<size_t>(total_slots);
      int init_blocks = static_cast<int>(std::min<size_t>((total_elems + 255) / 256, 1024));
      init_blocks = std::max(init_blocks, 1);
      init_negzero_buffers_kernel<<<init_blocks, 256, 0, stream>>>(
          sendbuf, recvbuf, elems_per_slot, total_slots);
    } else {
      int init_blocks = static_cast<int>(std::min<size_t>((opt.bytes + 255) / 256, 1024));
      init_blocks = std::max(init_blocks, 1);
      init_buffers_kernel<<<init_blocks, 256, 0, stream>>>(sendbuf, recvbuf, opt.bytes, rank);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    socket_barrier(rank, opt.master_addr, opt.port + 1);

    if (opt.warmup_iters > 0) {
      if (opt.mode == kNegZeroMode) {
        nccl_gin_pingpong_negzero_kernel<<<poll_blocks, 256, 0, stream>>>(
            sendbuf, recvbuf, sendwin, recvwin, dev_comm, opt.bytes, elems_per_slot, 0,
            opt.warmup_iters, done_counts_d, nullptr);
      } else if (opt.mode == kPollMode) {
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
        socket_barrier(rank, opt.master_addr, opt.port + 2);
        if (opt.thread_group == kThreadGroupWarp) {
          nccl_gin_pingpong_signal_warp_kernel<<<1, kKernelThreads, 0, stream>>>(
              sendwin, recvwin, dev_comm, signal_bases_d, opt.bytes, opt.warmup_iters,
              nullptr);
        } else {
          nccl_gin_pingpong_signal_kernel<<<1, kKernelThreads, 0, stream>>>(
              sendwin, recvwin, dev_comm, signal_bases_d, opt.bytes, opt.warmup_iters,
              nullptr);
        }
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (opt.mode == kNegZeroMode) {
      socket_barrier(rank, opt.master_addr, opt.port + 3);
      CUDA_CHECK(cudaEventRecord(start_event, stream));
      nccl_gin_pingpong_negzero_kernel<<<poll_blocks, 256, 0, stream>>>(
          sendbuf, recvbuf, sendwin, recvwin, dev_comm, opt.bytes, elems_per_slot,
          opt.warmup_iters, opt.iters, done_counts_d, roundtrip_ticks_d);
    } else if (opt.mode == kPollMode) {
      socket_barrier(rank, opt.master_addr, opt.port + 3);
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
      socket_barrier(rank, opt.master_addr, opt.port + 3);
      CUDA_CHECK(cudaEventRecord(start_event, stream));
      if (opt.thread_group == kThreadGroupWarp) {
        nccl_gin_pingpong_signal_warp_kernel<<<1, kKernelThreads, 0, stream>>>(
            sendwin, recvwin, dev_comm, signal_bases_d, opt.bytes, opt.iters,
            roundtrip_ticks_d);
      } else {
        nccl_gin_pingpong_signal_kernel<<<1, kKernelThreads, 0, stream>>>(
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

    bool ok_local = true;
    if (opt.check && rank == 0) {
      if (opt.mode == kNegZeroMode) {
        ok_local = verify_rank0_negzero(recvbuf, opt.bytes, total_slots - 1);
      } else {
        ok_local = opt.mode == kPollMode
                       ? verify_rank0_poll(recvbuf, opt.warmup_iters, opt.iters)
                       : verify_rank0(recvbuf, opt.bytes);
      }
    }
    int ok_int = ok_local ? 1 : 0;
    int ok_all = socket_allreduce_min(rank, opt.master_addr, opt.port + 4, ok_int);
    bool ok = ok_all == 1;

    print_summary(rank, opt, elapsed_us, ok);
    if (opt.profile_kernel && rank == 0) {
      std::vector<uint64_t> roundtrip_ticks(static_cast<size_t>(opt.iters));
      CUDA_CHECK(cudaMemcpy(roundtrip_ticks.data(), roundtrip_ticks_d,
                            sizeof(uint64_t) * roundtrip_ticks.size(),
                            cudaMemcpyDeviceToHost));
      print_kernel_profile(rank, opt, roundtrip_ticks, timer_ticks_per_us);
    }

    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (roundtrip_ticks_d != nullptr) {
      CUDA_CHECK(cudaFree(roundtrip_ticks_d));
    }
    if (done_counts_d != nullptr) {
      CUDA_CHECK(cudaFree(done_counts_d));
    }
    CUDA_CHECK(cudaFree(signal_bases_d));
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    NCCL_CHECK(ncclCommWindowDeregister(comm, sendwin));
    NCCL_CHECK(ncclCommWindowDeregister(comm, recvwin));
    NCCL_CHECK(ncclMemFree(sendbuf));
    NCCL_CHECK(ncclMemFree(recvbuf));
    NCCL_CHECK(ncclCommDestroy(comm));
    return ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

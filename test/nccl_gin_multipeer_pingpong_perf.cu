#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "common/checks.hpp"
#include "common/csv.hpp"
#include "common/env.hpp"
#include "common/parse.hpp"
#include "common/socket.hpp"
#include "common/timer.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
#error "NCCL GIN multipeer pingpong requires NCCL 2.29.0 or newer"
#endif

namespace {

using namespace nccl_deviceapi_test;

constexpr int kSignalBits = 32;
constexpr int kGinResourceCount = 16;
constexpr int kMaxPeers = 8;
constexpr int kWarpSize = 32;
constexpr int kKernelThreads = kWarpSize;
constexpr ncclGinSignal_t kPingSignal = 0;
constexpr ncclGinSignal_t kPongSignal = 1;

enum Direction {
  kFusedRoundtrip = 0,
  kOneToMany = 1,
  kManyToOne = 2,
};

struct Options {
  size_t bytes = 1024;
  size_t min_bytes = 0;
  size_t max_bytes = 0;
  int warmup_iters = 100;
  int iters = 1000;
  int device = -1;
  int port = 43021;
  double ticks_per_us = 0.0;
  std::string csv = "nccl_gin_multipeer_pingpong.csv";
  std::string master_addr;
  bool check = false;
  bool profile_kernel = false;
  bool adaptive_iters = false;
  int direction = kFusedRoundtrip;
};

__global__ void init_buffers_kernel(char *sendbuf, char *recvbuf, size_t total_bytes,
                                    int rank) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < total_bytes; i += stride) {
    sendbuf[i] = static_cast<char>((i + static_cast<size_t>(rank) * 17) & 0xff);
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

__global__ void nccl_gin_multipeer_pingpong_kernel(
    ncclWindow_t sendwin, ncclWindow_t recvwin, ncclDevComm dev_comm,
    const uint64_t *signal_bases, size_t bytes, int peer_count, int direction,
    int loop_iters, uint64_t *roundtrip_ticks) {
  if (threadIdx.x != 0) {
    return;
  }

  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclCoopThread thread;

  const uint64_t ping_base = signal_bases[0];
  const uint64_t pong_base = signal_bases[1];

  for (int iter = 0; iter < loop_iters; ++iter) {
    if (direction == kOneToMany) {
      if (world.rank == 0) {
        const uint64_t t0 = roundtrip_ticks != nullptr ? global_timer() : 0;
        for (int peer = 1; peer <= peer_count; ++peer) {
          const size_t offset = static_cast<size_t>(peer - 1) * bytes;
          gin.put(world, peer, recvwin, offset, sendwin, offset, bytes,
                  ncclGin_SignalInc{kPingSignal}, ncclGin_None{}, thread);
        }
        const uint64_t expected =
            pong_base + static_cast<uint64_t>(iter + 1) * peer_count;
        gin.waitSignal(thread, kPongSignal, expected, kSignalBits);
        if (roundtrip_ticks != nullptr) {
          roundtrip_ticks[iter] = global_timer() - t0;
        }
      } else {
        const size_t offset = static_cast<size_t>(world.rank - 1) * bytes;
        const uint64_t expected = ping_base + static_cast<uint64_t>(iter) + 1;
        gin.waitSignal(thread, kPingSignal, expected, kSignalBits);
        gin.put(world, 0, recvwin, offset, sendwin, offset, bytes,
                ncclGin_SignalInc{kPongSignal}, ncclGin_None{}, thread);
      }
    } else {
      if (world.rank == 0) {
        const uint64_t expected =
            ping_base + static_cast<uint64_t>(iter + 1) * peer_count;
        gin.waitSignal(thread, kPingSignal, expected, kSignalBits);
        for (int peer = 1; peer <= peer_count; ++peer) {
          const size_t offset = static_cast<size_t>(peer - 1) * bytes;
          gin.put(world, peer, recvwin, offset, sendwin, offset, bytes,
                  ncclGin_SignalInc{kPongSignal}, ncclGin_None{}, thread);
        }
      } else {
        const size_t offset = static_cast<size_t>(world.rank - 1) * bytes;
        const uint64_t t0 = roundtrip_ticks != nullptr ? global_timer() : 0;
        gin.put(world, 0, recvwin, offset, sendwin, offset, bytes,
                ncclGin_SignalInc{kPingSignal}, ncclGin_None{}, thread);
        const uint64_t expected = pong_base + static_cast<uint64_t>(iter) + 1;
        gin.waitSignal(thread, kPongSignal, expected, kSignalBits);
        if (roundtrip_ticks != nullptr) {
          roundtrip_ticks[iter] = global_timer() - t0;
        }
      }
    }
  }
}

// One fused iteration is hub -> all peers -> hub.  Rank 0 assigns one warp to
// each peer so the independent GIN puts are issued concurrently.  Peers return
// their payload as soon as their own forward transfer completes; this is the
// intentional half-roundtrip model used by this benchmark.
__global__ void nccl_gin_multipeer_fused_warp_kernel(
    ncclWindow_t sendwin, ncclWindow_t recvwin, ncclDevComm dev_comm,
    const uint64_t *signal_bases, size_t bytes, int peer_count, int loop_iters,
    uint64_t *roundtrip_ticks) {
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const uint64_t ping_base = signal_bases[0];
  const uint64_t pong_base = signal_bases[1];

  for (int iter = 0; iter < loop_iters; ++iter) {
    if (world.rank == 0) {
      uint64_t t0 = 0;
      if (warp_id < peer_count) {
        ncclCoopWarpSpan warp(warp_id, 1, warp_id);
        const int peer = warp_id + 1;
        const size_t offset = static_cast<size_t>(warp_id) * bytes;
        if (warp_id == 0 && warp.thread_rank() == 0 &&
            roundtrip_ticks != nullptr) {
          t0 = global_timer();
        }
        gin.put(world, peer, recvwin, offset, sendwin, offset, bytes,
                ncclGin_SignalInc{kPingSignal}, ncclGin_None{}, warp);
      }

      // Do not start the next iteration until every peer put has been issued.
      __syncthreads();
      if (warp_id == 0) {
        ncclCoopWarpSpan warp(0, 1, 0);
        const uint64_t expected =
            pong_base + static_cast<uint64_t>(iter + 1) * peer_count;
        gin.waitSignal(warp, kPongSignal, expected, kSignalBits);
        if (warp.thread_rank() == 0 && roundtrip_ticks != nullptr) {
          roundtrip_ticks[iter] = global_timer() - t0;
        }
      }
      __syncthreads();
    } else {
      if (warp_id == 0) {
        ncclCoopWarpSpan warp(0, 1, 0);
        const size_t offset = static_cast<size_t>(world.rank - 1) * bytes;
        const uint64_t expected = ping_base + static_cast<uint64_t>(iter) + 1;
        gin.waitSignal(warp, kPingSignal, expected, kSignalBits);
        gin.put(world, 0, recvwin, offset, sendwin, offset, bytes,
                ncclGin_SignalInc{kPongSignal}, ncclGin_None{}, warp);
      }
      __syncthreads();
    }
  }
}

static const char *direction_name(int direction) {
  if (direction == kManyToOne) {
    return "many-to-one";
  }
  if (direction == kOneToMany) {
    return "one-to-many";
  }
  return "one-to-many-to-one";
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
    } else if (arg == "--min-bytes") {
      opt.min_bytes = parse_size(need_value("--min-bytes"));
    } else if (arg == "--max-bytes") {
      opt.max_bytes = parse_size(need_value("--max-bytes"));
    } else if (arg == "--warmup-iters" || arg == "--warmup") {
      opt.warmup_iters = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--iters") {
      opt.iters = parse_int(need_value("--iters"), "--iters");
    } else if (arg == "--device") {
      opt.device = parse_int(need_value("--device"), "--device");
    } else if (arg == "--master") {
      opt.master_addr = need_value("--master");
    } else if (arg == "--port") {
      opt.port = parse_int(need_value("--port"), "--port");
    } else if (arg == "--ticks-per-us") {
      opt.ticks_per_us = std::stod(need_value("--ticks-per-us"));
    } else if (arg == "--direction") {
      std::string direction = need_value("--direction");
      if (direction == "roundtrip" || direction == "fused" ||
          direction == "one-to-many-to-one") {
        opt.direction = kFusedRoundtrip;
      } else if (direction == "one-to-many" || direction == "1ton") {
        opt.direction = kOneToMany;
      } else if (direction == "many-to-one" || direction == "nto1") {
        opt.direction = kManyToOne;
      } else {
        throw std::invalid_argument(
            "unknown --direction; expected roundtrip, one-to-many, or many-to-one");
      }
    } else if (arg == "--csv") {
      opt.csv = need_value("--csv");
    } else if (arg == "--check") {
      opt.check = true;
    } else if (arg == "--profile-kernel" || arg == "--profile_kernel") {
      opt.profile_kernel = true;
    } else if (arg == "--adaptive-iters" || arg == "--adaptive_iters") {
      opt.adaptive_iters = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: nccl_gin_multipeer_pingpong_perf [options]\n"
          << "  Ranks: rank 0 is the hub; ranks 1..N are peers (1 <= N <= 8)\n"
          << "  --bytes <N|64B|1KB|1MB>       payload bytes per peer per direction\n"
          << "  --min-bytes/--max-bytes <N>    factor-2 size sweep (inclusive)\n"
          << "  --warmup-iters <N>             warmup iterations\n"
          << "  --iters <N>                    measured iterations\n"
          << "  --device <N>                   CUDA device; defaults to LOCAL_RANK\n"
          << "  --master <IPv4|IPv6>           rank-0 address for socket bootstrap\n"
          << "  --port <N>                     base TCP port; benchmark uses port..port+7\n"
          << "  --direction roundtrip|one-to-many|many-to-one\n"
          << "                                  default roundtrip: hub fans out, peers return\n"
          << "  --ticks-per-us <F>             override GPU globaltimer ticks/us\n"
          << "  --csv <path>                   rank-0 output CSV path\n"
          << "  --check                        verify returned payloads\n"
          << "  --profile-kernel               report per-iteration batch percentiles\n"
          << "  --adaptive-iters               reduce iterations for large sweep sizes\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (opt.bytes == 0 || opt.iters <= 0 || opt.warmup_iters < 0 || opt.device < -1 ||
      opt.port <= 0 || opt.port > 65528) {
    throw std::invalid_argument("bytes/iters must be positive and warmup non-negative");
  }
  if (opt.ticks_per_us < 0.0) {
    throw std::invalid_argument("--ticks-per-us must be positive");
  }
  if ((opt.min_bytes == 0) != (opt.max_bytes == 0) ||
      (opt.min_bytes != 0 && opt.min_bytes > opt.max_bytes)) {
    throw std::invalid_argument(
        "--min-bytes and --max-bytes must be specified together with min <= max");
  }
  return opt;
}

static std::vector<size_t> build_sizes(const Options &opt) {
  if (opt.min_bytes == 0) {
    return {opt.bytes};
  }
  std::vector<size_t> sizes;
  for (size_t value = opt.min_bytes;;) {
    sizes.push_back(value);
    if (value >= opt.max_bytes) {
      break;
    }
    if (value > std::numeric_limits<size_t>::max() / 2) {
      throw std::overflow_error("size sweep overflow");
    }
    value *= 2;
    if (value > opt.max_bytes) {
      value = opt.max_bytes;
    }
  }
  return sizes;
}

static int adaptive_iters(const Options &opt, size_t bytes) {
  if (!opt.adaptive_iters) {
    return opt.iters;
  }
  if (bytes <= 1024ULL * 1024ULL) {
    return opt.iters;
  }
  if (bytes <= 16ULL * 1024ULL * 1024ULL) {
    return std::min(opt.iters, 100);
  }
  if (bytes <= 64ULL * 1024ULL * 1024ULL) {
    return std::min(opt.iters, 30);
  }
  return std::min(opt.iters, 10);
}

static int adaptive_warmup(const Options &opt, int iters) {
  return opt.adaptive_iters ? std::min(opt.warmup_iters, std::max(2, iters / 10))
                            : opt.warmup_iters;
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
  return env_int("WORLD_SIZE",
                 env_int("OMPI_COMM_WORLD_SIZE", env_int("PMI_SIZE", 1)));
}

static int get_local_rank(int rank) {
  return env_int("LOCAL_RANK", env_int("OMPI_COMM_WORLD_LOCAL_RANK", rank));
}

static void socket_barrier(int rank, int nranks, const std::string &master_addr,
                           int port) {
  char token = 1;
  if (rank == 0) {
    int listen_fd = listen_socket(port);
    std::vector<int> peer_fds;
    peer_fds.reserve(static_cast<size_t>(nranks - 1));
    for (int i = 1; i < nranks; ++i) {
      int fd = ::accept(listen_fd, nullptr, nullptr);
      if (fd < 0) {
        throw std::runtime_error("accept barrier failed");
      }
      int peer_rank = -1;
      if (!recv_all(fd, &peer_rank, sizeof(peer_rank)) || peer_rank <= 0 ||
          peer_rank >= nranks) {
        throw std::runtime_error("recv barrier rank failed");
      }
      peer_fds.push_back(fd);
    }
    for (int fd : peer_fds) {
      if (!send_all(fd, &token, sizeof(token))) {
        throw std::runtime_error("send barrier token failed");
      }
      ::close(fd);
    }
    ::close(listen_fd);
    return;
  }

  int fd = connect_socket(master_addr, port);
  if (!send_all(fd, &rank, sizeof(rank)) ||
      !recv_all(fd, &token, sizeof(token))) {
    throw std::runtime_error("peer barrier transfer failed");
  }
  ::close(fd);
}

static double socket_allreduce_max(int rank, int nranks,
                                   const std::string &master_addr, int port,
                                   double value) {
  if (rank == 0) {
    double result = value;
    int listen_fd = listen_socket(port);
    std::vector<int> peer_fds;
    peer_fds.reserve(static_cast<size_t>(nranks - 1));
    for (int i = 1; i < nranks; ++i) {
      int fd = ::accept(listen_fd, nullptr, nullptr);
      int peer_rank = -1;
      double peer_value = 0.0;
      if (fd < 0 || !recv_all(fd, &peer_rank, sizeof(peer_rank)) ||
          !recv_all(fd, &peer_value, sizeof(peer_value))) {
        throw std::runtime_error("recv max-reduction value failed");
      }
      result = std::max(result, peer_value);
      peer_fds.push_back(fd);
    }
    for (int fd : peer_fds) {
      if (!send_all(fd, &result, sizeof(result))) {
        throw std::runtime_error("send max-reduction result failed");
      }
      ::close(fd);
    }
    ::close(listen_fd);
    return result;
  }

  int fd = connect_socket(master_addr, port);
  if (!send_all(fd, &rank, sizeof(rank)) ||
      !send_all(fd, &value, sizeof(value))) {
    throw std::runtime_error("send max-reduction value failed");
  }
  double result = 0.0;
  if (!recv_all(fd, &result, sizeof(result))) {
    throw std::runtime_error("recv max-reduction result failed");
  }
  ::close(fd);
  return result;
}

static int socket_allreduce_min(int rank, int nranks,
                                const std::string &master_addr, int port,
                                int value) {
  if (rank == 0) {
    int result = value;
    int listen_fd = listen_socket(port);
    std::vector<int> peer_fds;
    peer_fds.reserve(static_cast<size_t>(nranks - 1));
    for (int i = 1; i < nranks; ++i) {
      int fd = ::accept(listen_fd, nullptr, nullptr);
      int peer_rank = -1;
      int peer_value = 0;
      if (fd < 0 || !recv_all(fd, &peer_rank, sizeof(peer_rank)) ||
          !recv_all(fd, &peer_value, sizeof(peer_value))) {
        throw std::runtime_error("recv min-reduction value failed");
      }
      result = std::min(result, peer_value);
      peer_fds.push_back(fd);
    }
    for (int fd : peer_fds) {
      if (!send_all(fd, &result, sizeof(result))) {
        throw std::runtime_error("send min-reduction result failed");
      }
      ::close(fd);
    }
    ::close(listen_fd);
    return result;
  }

  int fd = connect_socket(master_addr, port);
  if (!send_all(fd, &rank, sizeof(rank)) ||
      !send_all(fd, &value, sizeof(value))) {
    throw std::runtime_error("send min-reduction value failed");
  }
  int result = 0;
  if (!recv_all(fd, &result, sizeof(result))) {
    throw std::runtime_error("recv min-reduction result failed");
  }
  ::close(fd);
  return result;
}

static std::vector<double> socket_reduce_max_vector(
    int rank, int nranks, const std::string &master_addr, int port,
    const std::vector<double> &local) {
  char ack = 1;
  if (rank == 0) {
    std::vector<double> result = local;
    int listen_fd = listen_socket(port);
    std::vector<int> peer_fds;
    peer_fds.reserve(static_cast<size_t>(nranks - 1));
    for (int i = 1; i < nranks; ++i) {
      int fd = ::accept(listen_fd, nullptr, nullptr);
      int peer_rank = -1;
      uint64_t count = 0;
      if (fd < 0 || !recv_all(fd, &peer_rank, sizeof(peer_rank)) ||
          !recv_all(fd, &count, sizeof(count)) || count != result.size()) {
        throw std::runtime_error("recv vector-reduction header failed");
      }
      std::vector<double> peer(static_cast<size_t>(count));
      if (count != 0 &&
          !recv_all(fd, peer.data(), static_cast<size_t>(count) * sizeof(double))) {
        throw std::runtime_error("recv vector-reduction payload failed");
      }
      for (size_t j = 0; j < result.size(); ++j) {
        result[j] = std::max(result[j], peer[j]);
      }
      peer_fds.push_back(fd);
    }
    for (int fd : peer_fds) {
      if (!send_all(fd, &ack, sizeof(ack))) {
        throw std::runtime_error("send vector-reduction ack failed");
      }
      ::close(fd);
    }
    ::close(listen_fd);
    return result;
  }

  int fd = connect_socket(master_addr, port);
  const uint64_t count = local.size();
  if (!send_all(fd, &rank, sizeof(rank)) ||
      !send_all(fd, &count, sizeof(count)) ||
      (count != 0 &&
       !send_all(fd, local.data(), static_cast<size_t>(count) * sizeof(double))) ||
      !recv_all(fd, &ack, sizeof(ack))) {
    throw std::runtime_error("peer vector-reduction transfer failed");
  }
  ::close(fd);
  return {};
}

static bool verify_slot(char *recvbuf, size_t bytes, int slot, int expected_rank) {
  std::vector<char> host(bytes);
  const size_t offset = static_cast<size_t>(slot) * bytes;
  CUDA_CHECK(cudaMemcpy(host.data(), recvbuf + offset, bytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < bytes; ++i) {
    const char expected = static_cast<char>(
        (offset + i + static_cast<size_t>(expected_rank) * 17) & 0xff);
    if (host[i] != expected) {
      std::cerr << "check failed slot=" << slot << " byte=" << i << ": got "
                << static_cast<int>(static_cast<unsigned char>(host[i]))
                << ", expected "
                << static_cast<int>(static_cast<unsigned char>(expected)) << "\n";
      return false;
    }
  }
  return true;
}

static void write_csv_header(std::ostream &out) {
  write_csv_row(out, "direction", "peers", "bytes", "warmup_iters", "iters",
                "elapsed_us", "pingpong_us", "half_roundtrip_us",
                "p50_pingpong_us", "p50_half_roundtrip_us", "aggregate_gib_s",
                "check");
}

static void write_csv_result(std::ostream &out, const Options &opt, int peer_count,
                             double elapsed_us, double p50_pingpong_us, bool ok) {
  const double pingpong_us = elapsed_us / static_cast<double>(opt.iters);
  const double half_roundtrip_us = pingpong_us / 2.0;
  const double aggregate_gib_s =
      half_roundtrip_us > 0.0
          ? static_cast<double>(opt.bytes) * peer_count / 1024.0 / 1024.0 / 1024.0 /
                half_roundtrip_us * 1000000.0
          : 0.0;
  out << std::fixed << std::setprecision(6);
  write_csv_row(out, direction_name(opt.direction), peer_count, opt.bytes,
                opt.warmup_iters, opt.iters, elapsed_us, pingpong_us,
                half_roundtrip_us, p50_pingpong_us, p50_pingpong_us / 2.0,
                aggregate_gib_s, ok ? "ok" : "failed");
}

static void print_summary(const Options &opt, int peer_count, double elapsed_us,
                          double p50_pingpong_us, bool ok) {
  const double pingpong_us = elapsed_us / static_cast<double>(opt.iters);
  const double half_roundtrip_us = pingpong_us / 2.0;
  const double aggregate_gib_s =
      half_roundtrip_us > 0.0
          ? static_cast<double>(opt.bytes) * peer_count / 1024.0 / 1024.0 / 1024.0 /
                half_roundtrip_us * 1000000.0
          : 0.0;
  std::cout << std::fixed << std::setprecision(3)
            << "nccl_gin_multipeer_pingpong_perf result direction="
            << direction_name(opt.direction) << " peers=" << peer_count
            << " bytes=" << opt.bytes << " warmup_iters=" << opt.warmup_iters
            << " iters=" << opt.iters << " elapsed_us=" << elapsed_us
            << " pingpong_us=" << pingpong_us
            << " half_roundtrip_us=" << half_roundtrip_us
            << " p50_pingpong_us=" << p50_pingpong_us
            << " p50_half_roundtrip_us=" << p50_pingpong_us / 2.0
            << " aggregate_gib_s=" << aggregate_gib_s
            << " check=" << (ok ? "ok" : "failed") << "\n";
}

static void print_kernel_profile(const Options &opt, int peer_count,
                                 std::vector<double> batch_us) {
  if (!opt.profile_kernel || batch_us.empty()) {
    return;
  }
  double sum = 0.0;
  for (double value : batch_us) {
    sum += value;
  }
  std::sort(batch_us.begin(), batch_us.end());
  std::cout << std::fixed << std::setprecision(3)
            << "nccl_gin_multipeer_kernel_profile direction="
            << direction_name(opt.direction) << " peers=" << peer_count
            << " bytes=" << opt.bytes << " iters=" << opt.iters
            << " min_batch_us=" << batch_us.front()
            << " mean_batch_us=" << sum / static_cast<double>(batch_us.size())
            << " p50_batch_us=" << percentile_sorted(batch_us, 50.0)
            << " p99_batch_us=" << percentile_sorted(batch_us, 99.0)
            << " max_batch_us=" << batch_us.back() << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Options opt = parse_args(argc, argv);

    const int rank = get_rank();
    const int nranks = get_nranks();
    const int peer_count = nranks - 1;
    if (rank < 0 || rank >= nranks || peer_count < 1 || peer_count > kMaxPeers) {
      if (rank == 0) {
        std::cerr << "nccl_gin_multipeer_pingpong_perf requires 2.."
                  << (kMaxPeers + 1) << " ranks, got " << nranks << "\n";
      }
      return 2;
    }
    const std::vector<size_t> sizes = build_sizes(opt);
    const size_t max_bytes = sizes.back();
    if (max_bytes > std::numeric_limits<size_t>::max() /
                        static_cast<size_t>(peer_count)) {
      throw std::overflow_error("bytes * peer_count overflows size_t");
    }
    const size_t total_bytes = max_bytes * static_cast<size_t>(peer_count);

    const int dev = opt.device >= 0 ? opt.device : get_local_rank(rank);
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0 || dev < 0 || dev >= dev_count) {
      throw std::runtime_error("local rank exceeds visible CUDA device count");
    }
    CUDA_CHECK(cudaSetDevice(dev));

    double timer_ticks_per_us = 0.0;
    if (opt.profile_kernel) {
      timer_ticks_per_us =
          opt.ticks_per_us > 0.0 ? opt.ticks_per_us : calibrate_ticks_per_us(100000000ULL);
    }

    ncclUniqueId id =
        exchange_unique_id_socket(rank, nranks, opt.master_addr, opt.port);
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

    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void **>(&sendbuf), total_bytes));
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void **>(&recvbuf), total_bytes));
    NCCL_CHECK(ncclCommWindowRegister(comm, sendbuf, total_bytes, &sendwin,
                                      NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, recvbuf, total_bytes, &recvwin,
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

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&signal_bases_d),
                          sizeof(uint64_t) * 2));
    if (opt.profile_kernel) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&roundtrip_ticks_d),
                            sizeof(uint64_t) * static_cast<size_t>(opt.iters)));
      CUDA_CHECK(cudaMemset(roundtrip_ticks_d, 0,
                            sizeof(uint64_t) * static_cast<size_t>(opt.iters)));
    }
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    int init_blocks =
        static_cast<int>(std::min<size_t>((total_bytes + 255) / 256, 1024));
    init_blocks = std::max(init_blocks, 1);
    init_buffers_kernel<<<init_blocks, 256, 0, stream>>>(sendbuf, recvbuf, total_bytes,
                                                         rank);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    socket_barrier(rank, nranks, opt.master_addr, opt.port + 1);

    auto profile_ptr_for_rank = [&]() -> uint64_t * {
      if (!opt.profile_kernel) {
        return nullptr;
      }
      const bool records = opt.direction == kManyToOne ? rank != 0 : rank == 0;
      return records ? roundtrip_ticks_d : nullptr;
    };

    auto launch_pingpong = [&](const Options &run_opt, int loop_iters,
                               uint64_t *ticks) {
      if (run_opt.direction == kFusedRoundtrip) {
        const int threads = peer_count * kWarpSize;
        nccl_gin_multipeer_fused_warp_kernel<<<1, threads, 0, stream>>>(
            sendwin, recvwin, dev_comm, signal_bases_d, run_opt.bytes, peer_count,
            loop_iters, ticks);
      } else {
        nccl_gin_multipeer_pingpong_kernel<<<1, kKernelThreads, 0, stream>>>(
            sendwin, recvwin, dev_comm, signal_bases_d, run_opt.bytes, peer_count,
            run_opt.direction, loop_iters, ticks);
      }
    };

    std::ofstream csv_out;
    if (rank == 0) {
      csv_out.open(opt.csv);
      if (!csv_out) {
        throw std::runtime_error("failed to open csv output: " + opt.csv);
      }
      write_csv_header(csv_out);
    }

    bool overall_ok = true;
    for (size_t bytes : sizes) {
      Options run_opt = opt;
      run_opt.bytes = bytes;
      run_opt.iters = adaptive_iters(opt, bytes);
      run_opt.warmup_iters = adaptive_warmup(opt, run_opt.iters);

      if (run_opt.warmup_iters > 0) {
        read_signal_bases_kernel<<<1, kKernelThreads, 0, stream>>>(dev_comm,
                                                                   signal_bases_d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        socket_barrier(rank, nranks, opt.master_addr, opt.port + 2);
        launch_pingpong(run_opt, run_opt.warmup_iters, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        socket_barrier(rank, nranks, opt.master_addr, opt.port + 3);
      }

      read_signal_bases_kernel<<<1, kKernelThreads, 0, stream>>>(dev_comm,
                                                                 signal_bases_d);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (opt.profile_kernel) {
        CUDA_CHECK(cudaMemset(roundtrip_ticks_d, 0,
                              sizeof(uint64_t) * static_cast<size_t>(run_opt.iters)));
      }
      socket_barrier(rank, nranks, opt.master_addr, opt.port + 4);

      CUDA_CHECK(cudaEventRecord(start_event, stream));
      launch_pingpong(run_opt, run_opt.iters, profile_ptr_for_rank());
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(stop_event, stream));
      CUDA_CHECK(cudaEventSynchronize(stop_event));

      float local_elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&local_elapsed_ms, start_event, stop_event));
      const double local_elapsed_us =
          static_cast<double>(local_elapsed_ms) * 1000.0;
      const double elapsed_us = socket_allreduce_max(
          rank, nranks, opt.master_addr, opt.port + 5, local_elapsed_us);

      bool local_ok = true;
      if (run_opt.check) {
        if ((run_opt.direction == kFusedRoundtrip ||
             run_opt.direction == kOneToMany) &&
            rank == 0) {
          for (int peer = 1; peer <= peer_count && local_ok; ++peer) {
            local_ok = verify_slot(recvbuf, run_opt.bytes, peer - 1, peer);
          }
        } else if (run_opt.direction == kManyToOne && rank != 0) {
          local_ok = verify_slot(recvbuf, run_opt.bytes, rank - 1, 0);
        }
      }
      const int local_ok_int = local_ok ? 1 : 0;
      const int all_ok_int = socket_allreduce_min(
          rank, nranks, opt.master_addr, opt.port + 6, local_ok_int);
      const bool ok = all_ok_int != 0;
      overall_ok = overall_ok && ok;

      std::vector<double> max_batch_us;
      if (run_opt.profile_kernel) {
        std::vector<uint64_t> local_ticks(static_cast<size_t>(run_opt.iters), 0);
        if (profile_ptr_for_rank() != nullptr) {
          CUDA_CHECK(cudaMemcpy(local_ticks.data(), roundtrip_ticks_d,
                                sizeof(uint64_t) * local_ticks.size(),
                                cudaMemcpyDeviceToHost));
        }
        std::vector<double> local_batch_us(static_cast<size_t>(run_opt.iters), 0.0);
        for (int i = 0; i < run_opt.iters; ++i) {
          local_batch_us[static_cast<size_t>(i)] =
              static_cast<double>(local_ticks[static_cast<size_t>(i)]) /
              timer_ticks_per_us;
        }
        max_batch_us = socket_reduce_max_vector(
            rank, nranks, opt.master_addr, opt.port + 7, local_batch_us);
      }

      if (rank == 0) {
        double p50_pingpong_us = std::numeric_limits<double>::quiet_NaN();
        if (!max_batch_us.empty()) {
          std::vector<double> sorted_batch_us = max_batch_us;
          std::sort(sorted_batch_us.begin(), sorted_batch_us.end());
          p50_pingpong_us = percentile_sorted(sorted_batch_us, 50.0);
        }
        write_csv_result(csv_out, run_opt, peer_count, elapsed_us,
                         p50_pingpong_us, ok);
        csv_out.flush();
        print_summary(run_opt, peer_count, elapsed_us, p50_pingpong_us, ok);
        print_kernel_profile(run_opt, peer_count, std::move(max_batch_us));
      }
    }

    if (rank == 0) {
      std::cout << "nccl_gin_multipeer_pingpong_perf complete csv=" << opt.csv
                << "\n";
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
    return overall_ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

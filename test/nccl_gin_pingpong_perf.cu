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
constexpr int kGinResourceCount = 16;
constexpr ncclGinSignal_t kPingSignal = 0;
constexpr ncclGinSignal_t kPongSignal = 1;

struct Options {
  size_t bytes = 1024;
  int warmup_iters = 100;
  int iters = 1000;
  double ticks_per_us = 0.0;
  std::string csv = "nccl_gin_pingpong.csv";
  bool check = false;
};

struct Sample {
  uint64_t t0;
  uint64_t t_after_put;
  uint64_t t_after_quiet;
  uint64_t t_after_signal;
  uint64_t t_wait_done;
};

__device__ __forceinline__ uint64_t globaltimer() {
  return global_timer();
}

__global__ void init_buffers_kernel(char *sendbuf, char *recvbuf, size_t bytes, int rank) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < bytes; i += stride) {
    sendbuf[i] = static_cast<char>((i + rank * 17) & 0xff);
    recvbuf[i] = 0;
  }
}

__global__ void nccl_gin_pingpong_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                         ncclDevComm dev_comm, Sample *samples, size_t bytes,
                                         int sample_index, int record_sample) {
  ncclCoopCta cta;
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclBarrierSession<ncclCoopCta> world_bar{
      cta, ncclTeamTagWorld(), gin, static_cast<uint32_t>(blockIdx.x)};
  ncclCoopWarpSpan warps(0, 1, 0);

  __shared__ uint64_t signal_bases[2];
  if (threadIdx.x == 0) {
    signal_bases[0] = gin.readSignal(kPingSignal);
    signal_bases[1] = gin.readSignal(kPongSignal);
  }
  __syncthreads();

  const uint64_t ping_expected = signal_bases[0] + 1;
  const uint64_t pong_expected = signal_bases[1] + 1;

  world_bar.sync(cta, cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  Sample s{};
  if (threadIdx.x < kKernelThreads) {
    const int peer = world.rank == 0 ? 1 : 0;
    const bool lane0 = warps.thread_rank() == 0;

    if (world.rank == 0) {
      if (lane0) {
        s.t0 = globaltimer();
      }

      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes,
              ncclGin_SignalInc{kPingSignal}, ncclGin_None{}, warps);

      if (lane0) {
        s.t_after_put = globaltimer();
        s.t_after_quiet = s.t_after_put;
        s.t_after_signal = s.t_after_put;
      }

      gin.waitSignal(warps, kPongSignal, pong_expected, kKernelThreads);

      if (lane0) {
        s.t_wait_done = globaltimer();
      }
    } else {
      if (lane0) {
        s.t0 = globaltimer();
      }

      gin.waitSignal(warps, kPingSignal, ping_expected, kKernelThreads);

      if (lane0) {
        s.t_wait_done = globaltimer();
      }

      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes,
              ncclGin_SignalInc{kPongSignal}, ncclGin_None{}, warps);

      if (lane0) {
        s.t_after_put = globaltimer();
        s.t_after_quiet = s.t_after_put;
        s.t_after_signal = s.t_after_put;
      }
    }

    if (record_sample && lane0) {
      samples[sample_index] = s;
    }
  }

  world_bar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
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
    } else if (arg == "--ticks-per-us") {
      opt.ticks_per_us = std::stod(need_value("--ticks-per-us"));
    } else if (arg == "--thread-group") {
      std::string group = need_value("--thread-group");
      if (group != "warp") {
        throw std::invalid_argument("NCCL GIN pingpong only supports --thread-group warp");
      }
    } else if (arg == "--csv") {
      opt.csv = need_value("--csv");
    } else if (arg == "--check") {
      opt.check = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: nccl_gin_pingpong_perf [options]\n"
          << "  --bytes <N|64B|1KB|1MB>       payload bytes per one-way put\n"
          << "  --warmup-iters <N>             warmup iterations\n"
          << "  --iters <N>                    measured iterations\n"
          << "  --ticks-per-us <F>             override GPU globaltimer ticks/us\n"
          << "  --thread-group warp            accepted for CSV compatibility\n"
          << "  --csv <path>                   output CSV path\n"
          << "  --check                        verify final returned payload on rank 0\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (opt.bytes == 0 || opt.iters <= 0 || opt.warmup_iters < 0) {
    throw std::invalid_argument("bytes/iters must be positive and warmup non-negative");
  }
  if (opt.ticks_per_us < 0.0) {
    throw std::invalid_argument("--ticks-per-us must be positive");
  }
  return opt;
}

static void write_csv(const std::string &path, int rank, const Options &opt,
                      const std::vector<Sample> &samples, double timer_ticks_per_us) {
  CsvFile csv(path, 6);
  std::ofstream &out = csv.stream();

  out << std::fixed << std::setprecision(6);
  write_csv_row(out, "iter", "rank", "bytes", "thread_group", "timer_ticks_per_us",
                "t0", "t_after_put", "t_after_quiet", "t_after_signal", "t_wait_done",
                "put_issue_us", "quiet_us", "signal_us", "wait_us", "roundtrip_us",
                "one_way_us", "bw_p50_formula_gib_s");
  for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
    const Sample &s = samples[i];
    auto ticks_to_us = [&](uint64_t ticks) {
      return static_cast<double>(ticks) / timer_ticks_per_us;
    };

    double put_issue_us = rank == 0 ? ticks_to_us(s.t_after_put - s.t0)
                                    : ticks_to_us(s.t_after_put - s.t_wait_done);
    double quiet_us = ticks_to_us(s.t_after_quiet - s.t_after_put);
    double signal_us = ticks_to_us(s.t_after_signal - s.t_after_quiet);
    double wait_us = 0.0;
    double roundtrip_us = 0.0;
    double one_way_us = 0.0;

    if (rank == 0) {
      wait_us = ticks_to_us(s.t_wait_done - s.t_after_signal);
      roundtrip_us = ticks_to_us(s.t_wait_done - s.t0);
      one_way_us = roundtrip_us / 2.0;
    } else {
      wait_us = ticks_to_us(s.t_wait_done - s.t0);
    }

    double bw = one_way_us > 0.0
                    ? static_cast<double>(opt.bytes) / 1024.0 / 1024.0 / 1024.0 / one_way_us *
                          1000000.0
                    : 0.0;

    write_csv_row(out, i, rank, opt.bytes, "warp", timer_ticks_per_us, s.t0, s.t_after_put,
                  s.t_after_quiet, s.t_after_signal, s.t_wait_done, put_issue_us, quiet_us,
                  signal_us, wait_us, roundtrip_us, one_way_us, bw);
  }
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

    int dev = mpi_local_rank(MPI_COMM_WORLD);
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0 || dev >= dev_count) {
      throw std::runtime_error("local rank exceeds visible CUDA device count");
    }
    CUDA_CHECK(cudaSetDevice(dev));

    double timer_ticks_per_us =
        opt.ticks_per_us > 0.0 ? opt.ticks_per_us : calibrate_ticks_per_us(100000000ULL);

    ncclUniqueId id = mpi_bcast_nccl_unique_id(rank, MPI_COMM_WORLD);

    ncclComm_t comm = nullptr;
    ncclDevComm dev_comm{};
    char *sendbuf = nullptr;
    char *recvbuf = nullptr;
    ncclWindow_t sendwin = nullptr;
    ncclWindow_t recvwin = nullptr;
    Sample *samples_d = nullptr;
    cudaStream_t stream{};

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

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&samples_d), sizeof(Sample) * opt.iters));
    CUDA_CHECK(cudaStreamCreate(&stream));

    int init_blocks = static_cast<int>(std::min<size_t>((opt.bytes + 255) / 256, 1024));
    init_blocks = std::max(init_blocks, 1);
    init_buffers_kernel<<<init_blocks, 256, 0, stream>>>(sendbuf, recvbuf, opt.bytes, rank);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int total_iters = opt.warmup_iters + opt.iters;
    for (int iter = 0; iter < total_iters; ++iter) {
      int sample_index = iter - opt.warmup_iters;
      int record_sample = iter >= opt.warmup_iters ? 1 : 0;
      nccl_gin_pingpong_kernel<<<1, kKernelThreads, 0, stream>>>(
          sendwin, recvwin, dev_comm, samples_d, opt.bytes, sample_index, record_sample);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<Sample> samples(opt.iters);
    CUDA_CHECK(cudaMemcpy(samples.data(), samples_d, sizeof(Sample) * opt.iters,
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    if (opt.check && rank == 0) {
      ok = verify_rank0(recvbuf, opt.bytes);
    }

    write_csv(rank_csv_path(opt.csv, rank, nranks), rank, opt, samples, timer_ticks_per_us);

    if (rank == 0) {
      std::cout << "nccl_gin_pingpong_perf complete: bytes=" << opt.bytes
                << " warmup_iters=" << opt.warmup_iters << " iters=" << opt.iters
                << " thread_group=warp csv_prefix=" << opt.csv
                << " check=" << (ok ? "ok" : "failed") << "\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(samples_d));
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

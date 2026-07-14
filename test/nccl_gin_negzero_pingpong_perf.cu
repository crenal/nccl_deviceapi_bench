#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include "common/checks.hpp"
#include "common/mpi.hpp"
#include "common/parse.hpp"
#include "common/timer.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
#error "NCCL GIN negzero pingpong requires NCCL 2.29.0 or newer"
#endif

namespace {

using namespace nccl_deviceapi_test;

constexpr int kGinResourceCount = 16;

enum class DataType {
  kFloat,
  kBFloat16,
};

struct Options {
  size_t bytes = 1024;
  int warmup_iters = 100;
  int iters = 1000;
  int device = -1;
  int poll_blocks = 0;
  int threads = 256;
  double ticks_per_us = 0.0;
  bool check = false;
  DataType dtype = DataType::kFloat;
};

template <typename Word>
__device__ __forceinline__ Word negative_zero_word();

template <>
__device__ __forceinline__ uint32_t negative_zero_word<uint32_t>() {
  return 0x80000000u;
}

template <>
__device__ __forceinline__ uint16_t negative_zero_word<uint16_t>() {
  return 0x8000u;
}

template <typename Word>
__global__ void init_negzero_buffers_kernel(char *sendbuf, char *recvbuf,
                                            size_t elems_per_slot, int slots) {
  const size_t total_elems = elems_per_slot * static_cast<size_t>(slots);
  Word *send = reinterpret_cast<Word *>(sendbuf);
  Word *recv = reinterpret_cast<Word *>(recvbuf);
  const Word negzero = negative_zero_word<Word>();

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elems;
       idx += blockDim.x * gridDim.x) {
    // Positive zero is the Lamport-style escaped payload for negative zero.
    send[idx] = static_cast<Word>(0);
    recv[idx] = negzero;
  }
}

__device__ __forceinline__ void wait_blocks_done(volatile unsigned int *counter,
                                                 unsigned int expected) {
  while (*counter < expected) {
  }
}

template <typename Word>
__device__ void poll_slot_not_negzero(char *recv_slot, size_t elems_per_slot,
                                      unsigned int *done_counts, int slot) {
  volatile Word *recv = reinterpret_cast<volatile Word *>(recv_slot);
  const Word negzero = negative_zero_word<Word>();

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

template <typename Word>
__global__ void negzero_pingpong_kernel(char *sendbuf, char *recvbuf, ncclWindow_t sendwin,
                                        ncclWindow_t recvwin, ncclDevComm dev_comm,
                                        size_t bytes_per_slot, size_t elems_per_slot,
                                        int start_slot, int loop_iters,
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

    poll_slot_not_negzero<Word>(recvbuf + byte_offset, elems_per_slot, done_counts, slot);

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

static const char *dtype_name(DataType dtype) {
  return dtype == DataType::kFloat ? "float" : "bf16";
}

static size_t dtype_size(DataType dtype) {
  return dtype == DataType::kFloat ? sizeof(uint32_t) : sizeof(uint16_t);
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
    } else if (arg == "--poll-blocks") {
      opt.poll_blocks = parse_int(need_value("--poll-blocks"), "--poll-blocks");
    } else if (arg == "--threads") {
      opt.threads = parse_int(need_value("--threads"), "--threads");
    } else if (arg == "--ticks-per-us") {
      opt.ticks_per_us = std::stod(need_value("--ticks-per-us"));
    } else if (arg == "--dtype") {
      std::string dtype = need_value("--dtype");
      if (dtype == "float" || dtype == "fp32") {
        opt.dtype = DataType::kFloat;
      } else if (dtype == "bf16" || dtype == "bfloat16") {
        opt.dtype = DataType::kBFloat16;
      } else {
        throw std::invalid_argument("unknown --dtype; expected float or bf16");
      }
    } else if (arg == "--check") {
      opt.check = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: nccl_gin_negzero_pingpong_perf [options]\n"
          << "  --bytes <N|1KB|1MB>      payload bytes per one-way put\n"
          << "  --dtype float|bf16       element type used for -0 polling\n"
          << "  --warmup-iters <N>       warmup pingpong iterations\n"
          << "  --iters <N>              measured pingpong iterations\n"
          << "  --device <N>             CUDA device index; defaults to MPI local rank\n"
          << "  --poll-blocks <N>        poll blocks; default is device SM count\n"
          << "  --threads <N>            threads per poll block; default 256\n"
          << "  --ticks-per-us <F>       override GPU globaltimer ticks/us\n"
          << "  --check                  verify final rank0 slot is no longer -0\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (opt.bytes == 0 || opt.iters <= 0 || opt.warmup_iters < 0 || opt.device < -1) {
    throw std::invalid_argument("bytes/iters must be positive and warmup non-negative");
  }
  if (opt.poll_blocks < 0 || opt.threads <= 0 || opt.threads > 1024) {
    throw std::invalid_argument("poll-blocks must be non-negative and threads must be 1..1024");
  }
  if (opt.ticks_per_us < 0.0) {
    throw std::invalid_argument("--ticks-per-us must be positive");
  }
  const size_t elem_size = dtype_size(opt.dtype);
  if (opt.bytes % elem_size != 0) {
    throw std::invalid_argument("--bytes must be a multiple of the selected dtype size");
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

static void print_result(const Options &opt, int poll_blocks, double elapsed_us,
                         double ticks_per_us, const std::vector<uint64_t> &roundtrip_ticks) {
  std::vector<double> one_way_us;
  one_way_us.reserve(roundtrip_ticks.size());
  for (uint64_t ticks : roundtrip_ticks) {
    one_way_us.push_back(static_cast<double>(ticks) / ticks_per_us / 2.0);
  }
  std::sort(one_way_us.begin(), one_way_us.end());
  const double mean_one_way_us =
      std::accumulate(one_way_us.begin(), one_way_us.end(), 0.0) /
      static_cast<double>(one_way_us.size());
  const double event_one_way_us =
      elapsed_us / static_cast<double>(opt.iters) / 2.0;

  std::cout << std::fixed << std::setprecision(3)
            << "nccl_gin_negzero_pingpong_perf dtype=" << dtype_name(opt.dtype)
            << " bytes=" << opt.bytes << " warmup_iters=" << opt.warmup_iters
            << " iters=" << opt.iters << " poll_blocks=" << poll_blocks
            << " threads=" << opt.threads << " elapsed_us=" << elapsed_us
            << " event_one_way_us=" << event_one_way_us
            << " min_one_way_us=" << one_way_us.front()
            << " mean_one_way_us=" << mean_one_way_us
            << " p50_one_way_us=" << percentile_sorted(one_way_us, 50.0)
            << " p99_one_way_us=" << percentile_sorted(one_way_us, 99.0)
            << " max_one_way_us=" << one_way_us.back() << "\n";
}

template <typename Word>
static bool verify_no_negzero(char *recvbuf, size_t bytes_per_slot, int slot) {
  const size_t elems_per_slot = bytes_per_slot / sizeof(Word);
  std::vector<Word> host(elems_per_slot);
  CUDA_CHECK(cudaMemcpy(host.data(), recvbuf + static_cast<size_t>(slot) * bytes_per_slot,
                        bytes_per_slot, cudaMemcpyDeviceToHost));
  Word negzero = 0;
  if constexpr (sizeof(Word) == sizeof(uint32_t)) {
    negzero = static_cast<Word>(0x80000000u);
  } else {
    negzero = static_cast<Word>(0x8000u);
  }
  for (size_t i = 0; i < host.size(); ++i) {
    if (host[i] == negzero) {
      std::cerr << "check failed: slot " << slot << " element " << i
                << " is still negative zero\n";
      return false;
    }
  }
  return true;
}

template <typename Word>
static void launch_init(char *sendbuf, char *recvbuf, size_t elems_per_slot, int slots,
                        cudaStream_t stream) {
  const size_t total_elems = elems_per_slot * static_cast<size_t>(slots);
  int blocks = static_cast<int>(std::min<size_t>((total_elems + 255) / 256, 1024));
  blocks = std::max(blocks, 1);
  init_negzero_buffers_kernel<Word><<<blocks, 256, 0, stream>>>(sendbuf, recvbuf,
                                                                elems_per_slot, slots);
}

template <typename Word>
static void launch_pingpong(char *sendbuf, char *recvbuf, ncclWindow_t sendwin,
                            ncclWindow_t recvwin, ncclDevComm dev_comm,
                            size_t bytes_per_slot, size_t elems_per_slot, int start_slot,
                            int loop_iters, unsigned int *done_counts,
                            uint64_t *roundtrip_ticks, int poll_blocks, int threads,
                            cudaStream_t stream) {
  negzero_pingpong_kernel<Word><<<poll_blocks, threads, 0, stream>>>(
      sendbuf, recvbuf, sendwin, recvwin, dev_comm, bytes_per_slot, elems_per_slot,
      start_slot, loop_iters, done_counts, roundtrip_ticks);
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
        std::cerr << "nccl_gin_negzero_pingpong_perf requires exactly 2 ranks, got "
                  << nranks << "\n";
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

    cudaDeviceProp device_prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev));
    const int poll_blocks =
        opt.poll_blocks > 0 ? opt.poll_blocks : std::max(device_prop.multiProcessorCount, 1);
    const double ticks_per_us =
        opt.ticks_per_us > 0.0 ? opt.ticks_per_us : calibrate_ticks_per_us(100000000ULL);

    ncclUniqueId id = mpi_bcast_nccl_unique_id(rank, MPI_COMM_WORLD);

    ncclComm_t comm = nullptr;
    ncclDevComm dev_comm{};
    char *sendbuf = nullptr;
    char *recvbuf = nullptr;
    ncclWindow_t sendwin = nullptr;
    ncclWindow_t recvwin = nullptr;
    unsigned int *done_counts = nullptr;
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

    const size_t elem_size = dtype_size(opt.dtype);
    const size_t elems_per_slot = opt.bytes / elem_size;
    const int total_slots = opt.warmup_iters + opt.iters;
    const size_t window_bytes = opt.bytes * static_cast<size_t>(total_slots);

    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void **>(&sendbuf), window_bytes));
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void **>(&recvbuf), window_bytes));
    NCCL_CHECK(ncclCommWindowRegister(comm, sendbuf, window_bytes, &sendwin,
                                      NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, recvbuf, window_bytes, &recvwin,
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

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&done_counts),
                          sizeof(unsigned int) * static_cast<size_t>(total_slots)));
    if (rank == 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&roundtrip_ticks_d),
                            sizeof(uint64_t) * static_cast<size_t>(opt.iters)));
    }
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaMemsetAsync(done_counts, 0,
                               sizeof(unsigned int) * static_cast<size_t>(total_slots),
                               stream));
    if (opt.dtype == DataType::kFloat) {
      launch_init<uint32_t>(sendbuf, recvbuf, elems_per_slot, total_slots, stream);
    } else {
      launch_init<uint16_t>(sendbuf, recvbuf, elems_per_slot, total_slots, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (opt.warmup_iters > 0) {
      if (opt.dtype == DataType::kFloat) {
        launch_pingpong<uint32_t>(sendbuf, recvbuf, sendwin, recvwin, dev_comm, opt.bytes,
                                  elems_per_slot, 0, opt.warmup_iters, done_counts, nullptr,
                                  poll_blocks, opt.threads, stream);
      } else {
        launch_pingpong<uint16_t>(sendbuf, recvbuf, sendwin, recvwin, dev_comm, opt.bytes,
                                  elems_per_slot, 0, opt.warmup_iters, done_counts, nullptr,
                                  poll_blocks, opt.threads, stream);
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    CUDA_CHECK(cudaEventRecord(start_event, stream));
    if (opt.dtype == DataType::kFloat) {
      launch_pingpong<uint32_t>(sendbuf, recvbuf, sendwin, recvwin, dev_comm, opt.bytes,
                                elems_per_slot, opt.warmup_iters, opt.iters, done_counts,
                                roundtrip_ticks_d, poll_blocks, opt.threads, stream);
    } else {
      launch_pingpong<uint16_t>(sendbuf, recvbuf, sendwin, recvwin, dev_comm, opt.bytes,
                                elems_per_slot, opt.warmup_iters, opt.iters, done_counts,
                                roundtrip_ticks_d, poll_blocks, opt.threads, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;

    bool ok = true;
    if (opt.check && rank == 0) {
      const int last_slot = total_slots - 1;
      ok = opt.dtype == DataType::kFloat
               ? verify_no_negzero<uint32_t>(recvbuf, opt.bytes, last_slot)
               : verify_no_negzero<uint16_t>(recvbuf, opt.bytes, last_slot);
    }

    if (rank == 0) {
      std::vector<uint64_t> roundtrip_ticks(static_cast<size_t>(opt.iters));
      CUDA_CHECK(cudaMemcpy(roundtrip_ticks.data(), roundtrip_ticks_d,
                            sizeof(uint64_t) * roundtrip_ticks.size(),
                            cudaMemcpyDeviceToHost));
      print_result(opt, poll_blocks, elapsed_us, ticks_per_us, roundtrip_ticks);
      std::cout << "nccl_gin_negzero_pingpong_perf complete check="
                << (ok ? "ok" : "failed") << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (roundtrip_ticks_d != nullptr) {
      CUDA_CHECK(cudaFree(roundtrip_ticks_d));
    }
    CUDA_CHECK(cudaFree(done_counts));
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

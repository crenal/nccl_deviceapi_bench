#include <cuda/atomic>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
#error "waitsignal_perf requires NCCL 2.29.0 or newer"
#endif

#define CUDA_CHECK(stmt)                                                     \
  do {                                                                       \
    cudaError_t _err = (stmt);                                               \
    if (_err != cudaSuccess) {                                                \
      std::ostringstream _oss;                                                \
      _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
           << cudaGetErrorString(_err);                                      \
      throw std::runtime_error(_oss.str());                                  \
    }                                                                        \
  } while (0)

#define NCCL_CHECK(stmt)                                                     \
  do {                                                                       \
    ncclResult_t _err = (stmt);                                              \
    if (_err != ncclSuccess) {                                                \
      std::ostringstream _oss;                                                \
      _oss << "NCCL error at " << __FILE__ << ":" << __LINE__ << ": "       \
           << ncclGetErrorString(_err);                                      \
      throw std::runtime_error(_oss.str());                                  \
    }                                                                        \
  } while (0)

#define MPI_CHECK(stmt)                                                      \
  do {                                                                       \
    int _err = (stmt);                                                       \
    if (_err != MPI_SUCCESS) {                                               \
      char _msg[MPI_MAX_ERROR_STRING];                                       \
      int _len = 0;                                                          \
      MPI_Error_string(_err, _msg, &_len);                                   \
      std::ostringstream _oss;                                                \
      _oss << "MPI error at " << __FILE__ << ":" << __LINE__ << ": "        \
           << std::string(_msg, _len);                                       \
      throw std::runtime_error(_oss.str());                                  \
    }                                                                        \
  } while (0)

namespace {

constexpr int kTraceSlots = 4;
constexpr int kGinSignalCount = 8;

enum TraceSlot {
  kWaiterReady = 0,
  kSignalBefore = 1,
  kSignalAfter = 2,
  kWaitDone = 3,
};

struct Options {
  int warmup_iters = 100;
  int iters = 1000;
  double ticks_per_us = 0.0;
  std::string csv_path = "waitsignal_rank0_trace.csv";
};

struct Sample {
  uint64_t timestamps[kTraceSlots] = {};
  uint64_t local_atomic_add_ticks = 0;
  uint64_t signal_to_wait_done_ticks = 0;
  uint64_t signal_after_to_wait_done_ticks = 0;
};

struct Stats {
  double min_us = 0.0;
  double max_us = 0.0;
  double p50_us = 0.0;
  double p99_us = 0.0;
};

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void calibrate_timer_kernel(uint64_t *ticks, uint64_t spin_ticks) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  uint64_t t0 = globaltimer();
  uint64_t t1 = t0;
  while (t1 - t0 < spin_ticks) {
    t1 = globaltimer();
  }
  ticks[0] = t0;
  ticks[1] = t1;
}

__global__ __launch_bounds__(32, 2) void waitsignal_trace_kernel(
    ncclDevComm dev_comm, uint64_t *trace_ticks) {
  constexpr ncclGinSignal_t signal_index = 2;
  ncclGin gin{dev_comm, 0};
  volatile uint64_t *volatile_trace = reinterpret_cast<volatile uint64_t *>(trace_ticks);

  if (threadIdx.x != 0 || trace_ticks == nullptr) {
    return;
  }

  if (blockIdx.x == 0) {
    const uint64_t signal_base = gin.readSignal(signal_index);
    trace_ticks[kWaiterReady] = globaltimer();
    __threadfence_system();

    gin.waitSignal(ncclCoopThread(), signal_index, signal_base + 1);
    trace_ticks[kWaitDone] = globaltimer();
  } else if (blockIdx.x == 1) {
    while (volatile_trace[kWaiterReady] == 0) {
    }

    auto signal_ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(gin._makeCtx(), signal_index);

    trace_ticks[kSignalBefore] = globaltimer();
    cuda::atomic_ref<uint64_t>{*signal_ptr.ptr}.fetch_add(1, cuda::memory_order_release);
    trace_ticks[kSignalAfter] = globaltimer();
  }
}

static int parse_int(const std::string &text, const char *flag) {
  char *end = nullptr;
  long v = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0' || v < std::numeric_limits<int>::min() ||
      v > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid integer for ") + flag + ": " + text);
  }
  return static_cast<int>(v);
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

    if (arg == "--warmup-iters" || arg == "--warmup") {
      opt.warmup_iters = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--iters") {
      opt.iters = parse_int(need_value("--iters"), "--iters");
    } else if (arg == "--ticks-per-us") {
      opt.ticks_per_us = std::stod(need_value("--ticks-per-us"));
    } else if (arg == "--csv") {
      opt.csv_path = need_value("--csv");
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: waitsignal_perf [options]\n"
          << "  --warmup-iters <N>   warmup samples\n"
          << "  --iters <N>          measured samples\n"
          << "  --ticks-per-us <F>   override GPU globaltimer ticks/us\n"
          << "  --csv <path>         rank0 CSV output\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (opt.warmup_iters < 0 || opt.iters <= 0 || opt.ticks_per_us < 0.0 ||
      opt.csv_path.empty()) {
    throw std::invalid_argument("invalid waitsignal options");
  }
  return opt;
}

static int local_rank(MPI_Comm world) {
  MPI_Comm local = MPI_COMM_NULL;
  MPI_CHECK(MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local));
  int rank = 0;
  MPI_CHECK(MPI_Comm_rank(local, &rank));
  MPI_CHECK(MPI_Comm_free(&local));
  return rank;
}

static double calibrate_timer_ticks_per_us() {
  constexpr uint64_t kSpinTicks = 100000000ULL;
  uint64_t *ticks_d = nullptr;
  uint64_t ticks_h[2] = {};
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ticks_d), sizeof(ticks_h)));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  calibrate_timer_kernel<<<1, 1>>>(ticks_d, kSpinTicks);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(ticks_h, ticks_d, sizeof(ticks_h), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(ticks_d));

  uint64_t elapsed_ticks = ticks_h[1] - ticks_h[0];
  if (elapsed_ticks == 0 || elapsed_ms <= 0.0f) {
    throw std::runtime_error("failed to calibrate globaltimer");
  }
  return static_cast<double>(elapsed_ticks) / (static_cast<double>(elapsed_ms) * 1000.0);
}

static double ticks_to_us(uint64_t ticks, double ticks_per_us) {
  return static_cast<double>(ticks) / ticks_per_us;
}

static double percentile_nearest(const std::vector<double> &sorted_values, double percentile) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  const double rank = std::ceil((percentile / 100.0) * sorted_values.size());
  const size_t index = static_cast<size_t>(std::max(1.0, rank)) - 1;
  return sorted_values[std::min(index, sorted_values.size() - 1)];
}

static Stats compute_stats(const std::vector<uint64_t> &ticks, double ticks_per_us) {
  std::vector<double> values;
  values.reserve(ticks.size());
  for (uint64_t tick : ticks) {
    values.push_back(ticks_to_us(tick, ticks_per_us));
  }
  std::sort(values.begin(), values.end());

  Stats stats;
  stats.min_us = values.front();
  stats.max_us = values.back();
  stats.p50_us = percentile_nearest(values, 50.0);
  stats.p99_us = percentile_nearest(values, 99.0);
  return stats;
}

static void write_raw_row(std::ofstream &csv, size_t iter, const char *metric,
                          uint64_t ticks, double ticks_per_us, const Sample &sample) {
  csv << "raw," << iter << ',' << metric << ',' << ticks << ','
      << ticks_to_us(ticks, ticks_per_us) << ",,,,,"
      << sample.timestamps[kWaiterReady] << ','
      << sample.timestamps[kSignalBefore] << ','
      << sample.timestamps[kSignalAfter] << ','
      << sample.timestamps[kWaitDone] << '\n';
}

static void write_csv(const std::string &path, const std::vector<Sample> &samples,
                      double ticks_per_us) {
  std::ofstream csv(path);
  if (!csv) {
    throw std::runtime_error("failed to open csv output: " + path);
  }

  csv << "row_type,iteration,metric,ticks,us,min_us,max_us,p50_us,p99_us,"
      << "t_waiter_ready,t_signal_before,t_signal_after,t_wait_done\n";
  csv << std::setprecision(10);

  std::vector<uint64_t> local_atomic_add;
  std::vector<uint64_t> signal_to_wait_done;
  std::vector<uint64_t> signal_after_to_wait_done;
  local_atomic_add.reserve(samples.size());
  signal_to_wait_done.reserve(samples.size());
  signal_after_to_wait_done.reserve(samples.size());

  for (size_t i = 0; i < samples.size(); ++i) {
    const Sample &sample = samples[i];
    local_atomic_add.push_back(sample.local_atomic_add_ticks);
    signal_to_wait_done.push_back(sample.signal_to_wait_done_ticks);
    signal_after_to_wait_done.push_back(sample.signal_after_to_wait_done_ticks);

    write_raw_row(csv, i, "local_atomic_add", sample.local_atomic_add_ticks, ticks_per_us, sample);
    write_raw_row(csv, i, "signal_to_wait_done", sample.signal_to_wait_done_ticks, ticks_per_us, sample);
    write_raw_row(csv, i, "signal_after_to_wait_done",
                  sample.signal_after_to_wait_done_ticks, ticks_per_us, sample);
  }

  auto write_summary = [&](const char *metric, const Stats &stats) {
    csv << "summary,," << metric << ",,," << stats.min_us << ','
        << stats.max_us << ',' << stats.p50_us << ',' << stats.p99_us << ",,,,\n";
  };
  write_summary("local_atomic_add", compute_stats(local_atomic_add, ticks_per_us));
  write_summary("signal_to_wait_done", compute_stats(signal_to_wait_done, ticks_per_us));
  write_summary("signal_after_to_wait_done", compute_stats(signal_after_to_wait_done, ticks_per_us));
}

static void print_summary(const std::vector<Sample> &samples, int warmup_iters,
                          double ticks_per_us, const std::string &csv_path) {
  std::vector<uint64_t> local_atomic_add;
  std::vector<uint64_t> signal_to_wait_done;
  std::vector<uint64_t> signal_after_to_wait_done;
  local_atomic_add.reserve(samples.size());
  signal_to_wait_done.reserve(samples.size());
  signal_after_to_wait_done.reserve(samples.size());
  for (const Sample &sample : samples) {
    local_atomic_add.push_back(sample.local_atomic_add_ticks);
    signal_to_wait_done.push_back(sample.signal_to_wait_done_ticks);
    signal_after_to_wait_done.push_back(sample.signal_after_to_wait_done_ticks);
  }

  auto print_metric = [&](const char *name, const Stats &stats) {
    std::cout << std::left << std::setw(28) << name << std::setw(16)
              << stats.min_us << std::setw(16) << stats.max_us
              << std::setw(16) << stats.p50_us << std::setw(16)
              << stats.p99_us << '\n';
  };

  std::cout << "# GIN waitSignal same-GPU local-atomic trace\n";
  std::cout << "# warmup_samples " << warmup_iters << " samples " << samples.size()
            << " ticks_per_us " << ticks_per_us << " csv " << csv_path << '\n';
  std::cout << std::left << std::setw(28) << "metric" << std::setw(16) << "min_us"
            << std::setw(16) << "max_us" << std::setw(16) << "p50_us"
            << std::setw(16) << "p99_us" << '\n';
  std::cout << std::setprecision(4) << std::fixed;
  print_metric("local_atomic_add", compute_stats(local_atomic_add, ticks_per_us));
  print_metric("signal_to_wait_done", compute_stats(signal_to_wait_done, ticks_per_us));
  print_metric("signal_after_to_wait_done", compute_stats(signal_after_to_wait_done, ticks_per_us));
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

    int dev = local_rank(MPI_COMM_WORLD);
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0 || dev >= dev_count) {
      throw std::runtime_error("local rank exceeds visible CUDA device count");
    }
    CUDA_CHECK(cudaSetDevice(dev));

    double timer_ticks_per_us =
        opt.ticks_per_us > 0.0 ? opt.ticks_per_us : calibrate_timer_ticks_per_us();

    ncclUniqueId id{};
    if (rank == 0) {
      NCCL_CHECK(ncclGetUniqueId(&id));
    }
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm = nullptr;
    ncclDevComm dev_comm{};
    uint64_t *trace_d = nullptr;
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

    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = kGinSignalCount;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&trace_d), kTraceSlots * sizeof(uint64_t)));
    CUDA_CHECK(cudaStreamCreate(&stream));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    for (int iter = 0; iter < opt.warmup_iters; ++iter) {
      CUDA_CHECK(cudaMemsetAsync(trace_d, 0, kTraceSlots * sizeof(uint64_t), stream));
      waitsignal_trace_kernel<<<2, 32, 0, stream>>>(dev_comm, trace_d);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<Sample> rank0_samples;
    if (rank == 0) {
      rank0_samples.reserve(static_cast<size_t>(opt.iters));
    }

    for (int iter = 0; iter < opt.iters; ++iter) {
      CUDA_CHECK(cudaMemsetAsync(trace_d, 0, kTraceSlots * sizeof(uint64_t), stream));
      waitsignal_trace_kernel<<<2, 32, 0, stream>>>(dev_comm, trace_d);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));

      if (rank == 0) {
        Sample sample;
        CUDA_CHECK(cudaMemcpyAsync(sample.timestamps, trace_d, sizeof(sample.timestamps),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        sample.local_atomic_add_ticks = sample.timestamps[kSignalAfter] - sample.timestamps[kSignalBefore];
        sample.signal_to_wait_done_ticks = sample.timestamps[kWaitDone] - sample.timestamps[kSignalBefore];
        sample.signal_after_to_wait_done_ticks = sample.timestamps[kWaitDone] - sample.timestamps[kSignalAfter];
        rank0_samples.push_back(sample);
      }
    }

    if (rank == 0) {
      write_csv(opt.csv_path, rank0_samples, timer_ticks_per_us);
      print_summary(rank0_samples, opt.warmup_iters, timer_ticks_per_us, opt.csv_path);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(trace_d));
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    if (mpi_initialized) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
  }
}

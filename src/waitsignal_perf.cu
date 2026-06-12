#include <cuda/atomic>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include "common/checks.hpp"
#include "common/mpi.hpp"
#include "common/parse.hpp"
#include "common/stats.hpp"
#include "common/timer.cuh"

#include <algorithm>
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

namespace {

using namespace nccl_deviceapi_test;

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

__device__ __forceinline__ uint64_t globaltimer() {
  return global_timer();
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

static double ticks_to_us(uint64_t ticks, double ticks_per_us) {
  return static_cast<double>(ticks) / ticks_per_us;
}

static MetricStats compute_tick_stats(const std::vector<uint64_t> &ticks, double ticks_per_us) {
  std::vector<double> values = nccl_deviceapi_test::ticks_to_us(ticks, ticks_per_us);
  std::sort(values.begin(), values.end());
  MetricStats stats;
  stats.min = values.front();
  stats.max = values.back();
  stats.p50 = percentile_nearest_percent(values, 50.0);
  stats.p99 = percentile_nearest_percent(values, 99.0);
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

  auto write_summary = [&](const char *metric, const MetricStats &stats) {
    csv << "summary,," << metric << ",,," << stats.min << ','
        << stats.max << ',' << stats.p50 << ',' << stats.p99 << ",,,,\n";
  };
  write_summary("local_atomic_add", compute_tick_stats(local_atomic_add, ticks_per_us));
  write_summary("signal_to_wait_done", compute_tick_stats(signal_to_wait_done, ticks_per_us));
  write_summary("signal_after_to_wait_done", compute_tick_stats(signal_after_to_wait_done, ticks_per_us));
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

  auto print_metric = [&](const char *name, const MetricStats &stats) {
    std::cout << std::left << std::setw(28) << name << std::setw(16)
              << stats.min << std::setw(16) << stats.max
              << std::setw(16) << stats.p50 << std::setw(16)
              << stats.p99 << '\n';
  };

  std::cout << "# GIN waitSignal same-GPU local-atomic trace\n";
  std::cout << "# warmup_samples " << warmup_iters << " samples " << samples.size()
            << " ticks_per_us " << ticks_per_us << " csv " << csv_path << '\n';
  std::cout << std::left << std::setw(28) << "metric" << std::setw(16) << "min_us"
            << std::setw(16) << "max_us" << std::setw(16) << "p50_us"
            << std::setw(16) << "p99_us" << '\n';
  std::cout << std::setprecision(4) << std::fixed;
  print_metric("local_atomic_add", compute_tick_stats(local_atomic_add, ticks_per_us));
  print_metric("signal_to_wait_done", compute_tick_stats(signal_to_wait_done, ticks_per_us));
  print_metric("signal_after_to_wait_done", compute_tick_stats(signal_after_to_wait_done, ticks_per_us));
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

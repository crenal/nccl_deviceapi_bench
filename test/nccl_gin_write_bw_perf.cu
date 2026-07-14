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
#error "NCCL GIN write bandwidth requires NCCL 2.29.0 or newer"
#endif

namespace {

using namespace nccl_deviceapi_test;

constexpr int kKernelThreads = 32;
constexpr int kGinResourceCount = 16;

struct Options {
  size_t bytes = 256ULL * 1024ULL * 1024ULL;
  int warmup_iters = 5;
  int iters = 20;
  int tx_depth = 128;
  int post_list = 1;
  int slots = 1;
  int qps = 1;
  int device = -1;
  int src_dev = -1;
  int dst_dev = -1;
  double ticks_per_us = 0.0;
  std::string csv = "nccl_gin_write_bw.csv";
  bool check = false;
};

struct Sample {
  uint64_t t_start;
  uint64_t t_stop;
};

__device__ __forceinline__ uint64_t globaltimer() {
  return global_timer();
}

__global__ void init_buffers_kernel(char *sendbuf, char *recvbuf, size_t payload_bytes,
                                    size_t window_bytes, int rank) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < window_bytes; i += stride) {
    const size_t byte_in_msg = i % payload_bytes;
    sendbuf[i] = static_cast<char>((byte_in_msg + rank * 17) & 0xff);
    recvbuf[i] = 0;
  }
}

__global__ void read_gin_bases_kernel(ncclDevComm dev_comm, uint64_t *signal_bases,
                                      uint64_t *counter_bases) {
  const int qp = static_cast<int>(blockIdx.x);
  ncclGin gin{dev_comm, qp};
  const ncclGinSignal_t done_signal = static_cast<ncclGinSignal_t>(qp);
  const ncclGinCounter_t put_counter = static_cast<ncclGinCounter_t>(qp);
  if (threadIdx.x == 0) {
    signal_bases[qp] = gin.readSignal(done_signal);
    counter_bases[qp] = gin.readCounter(put_counter);
  }
}

__global__ void nccl_gin_write_bw_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                         ncclDevComm dev_comm, Sample *sample,
                                         const uint64_t *signal_bases,
                                         const uint64_t *counter_bases,
                                         size_t bytes, int iters, int tx_depth, int post_list,
                                         int slots) {
  const int qp = static_cast<int>(blockIdx.x);
  ncclGin gin{dev_comm, qp};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclCoopWarpSpan warps(0, 1, 0);
  const ncclGinSignal_t done_signal = static_cast<ncclGinSignal_t>(qp);
  const ncclGinCounter_t put_counter = static_cast<ncclGinCounter_t>(qp);

  const uint64_t signal_base = signal_bases[qp];
  const uint64_t counter_base = counter_bases[qp];
  const uint64_t done_expected = signal_base + 1;

  if (threadIdx.x < kKernelThreads) {
    const int peer = world.rank == 0 ? 1 : 0;
    const bool lane0 = warps.thread_rank() == 0;

    if (world.rank == 0) {
      if (lane0 && qp == 0) {
        sample->t_start = globaltimer();
      }

      int posted = 0;
      while (posted < iters) {
        const int remaining = iters - posted;
        const int batch = post_list < remaining ? post_list : remaining;
        const int completion_floor = posted + batch - tx_depth;
        if (completion_floor > 0) {
          gin.waitCounter(warps, put_counter, counter_base + completion_floor, 56);
        }

        for (int i = 0; i < batch; ++i) {
          const int put_index = posted + i;
          const size_t slot = static_cast<size_t>(put_index % slots);
          const size_t offset = (static_cast<size_t>(qp) * static_cast<size_t>(slots) + slot) * bytes;
          const bool signal_final = (put_index + 1 == iters);
          if (signal_final) {
            gin.put(world, peer, recvwin, offset, sendwin, offset, bytes,
                    ncclGin_StrongSignalInc{done_signal}, ncclGin_WeakCounterInc{put_counter},
                    warps);
          } else {
            gin.put(world, peer, recvwin, offset, sendwin, offset, bytes, ncclGin_None{},
                    ncclGin_WeakCounterInc{put_counter}, warps);
          }
        }

        posted += batch;
      }

      gin.waitCounter(warps, put_counter, counter_base + static_cast<uint64_t>(iters), 56);

      if (lane0) {
        if (qp == 0) {
          sample->t_stop = globaltimer();
        }
      }
    } else {
      gin.waitSignal(warps, done_signal, done_expected, kKernelThreads);
    }
  }

}

static size_t checked_window_bytes(size_t bytes, int slots, int qps) {
  if (slots <= 0 || qps <= 0) {
    throw std::invalid_argument("--slots and --qps must be positive");
  }
  if (bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(slots)) {
    throw std::invalid_argument("bytes * slots overflows size_t");
  }
  size_t per_qp = bytes * static_cast<size_t>(slots);
  if (per_qp > std::numeric_limits<size_t>::max() / static_cast<size_t>(qps)) {
    throw std::invalid_argument("bytes * slots * qps overflows size_t");
  }
  return per_qp * static_cast<size_t>(qps);
}

static int touched_slots(const Options &opt) {
  return std::min(opt.slots, std::max(opt.warmup_iters, opt.iters));
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

    if (arg == "--bytes" || arg == "-s") {
      opt.bytes = parse_size(need_value(arg.c_str()));
    } else if (arg == "--warmup-iters" || arg == "--warmup") {
      opt.warmup_iters = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--iters" || arg == "-n") {
      opt.iters = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--tx-depth" || arg == "--tx_depth") {
      opt.tx_depth = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--post-list" || arg == "--post_list") {
      opt.post_list = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--slots") {
      opt.slots = parse_int(need_value("--slots"), "--slots");
    } else if (arg == "--qps" || arg == "--contexts") {
      opt.qps = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--device") {
      opt.device = parse_int(need_value("--device"), "--device");
    } else if (arg == "--src-dev" || arg == "--src_dev") {
      opt.src_dev = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--dst-dev" || arg == "--dst_dev") {
      opt.dst_dev = parse_int(need_value(arg.c_str()), arg.c_str());
    } else if (arg == "--ticks-per-us") {
      opt.ticks_per_us = std::stod(need_value("--ticks-per-us"));
    } else if (arg == "--csv") {
      opt.csv = need_value("--csv");
    } else if (arg == "--check") {
      opt.check = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: nccl_gin_write_bw_perf [options]\n"
          << "  --bytes, -s <N|256MB>          payload bytes per one-way put\n"
          << "  --warmup-iters <N>             warmup write iterations\n"
          << "  --iters, -n <N>                measured write iterations\n"
          << "  --tx-depth <N>                 max locally incomplete puts\n"
          << "  --post-list <N>                puts per posting batch / completion check\n"
          << "  --slots <N>                    payload slots cycled in the symmetric window\n"
          << "  --qps <N>                      GIN contexts/QPs used concurrently\n"
          << "  --device <N>                   CUDA device index; defaults to MPI local rank\n"
          << "  --src-dev <N>                  CUDA device for rank 0\n"
          << "  --dst-dev <N>                  CUDA device for rank 1\n"
          << "  --ticks-per-us <F>             override GPU globaltimer ticks/us\n"
          << "  --csv <path>                   output CSV path\n"
          << "  --check                        verify rank 1 received rank 0 pattern\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (opt.bytes == 0 || opt.iters <= 0 || opt.warmup_iters < 0 || opt.tx_depth <= 0 ||
      opt.post_list <= 0 || opt.slots <= 0 || opt.qps <= 0 || opt.device < -1 ||
      opt.src_dev < -1 || opt.dst_dev < -1) {
    throw std::invalid_argument(
        "bytes/iters/tx-depth/post-list/slots/qps must be positive and warmup non-negative");
  }
  if (opt.post_list > opt.tx_depth) {
    throw std::invalid_argument("--post-list must be <= --tx-depth");
  }
  if (opt.qps > kGinResourceCount) {
    throw std::invalid_argument("--qps must be <= 16 for this microbench");
  }
  if (opt.ticks_per_us < 0.0) {
    throw std::invalid_argument("--ticks-per-us must be positive");
  }
  checked_window_bytes(opt.bytes, opt.slots, opt.qps);
  return opt;
}

static bool verify_rank1(char *recvbuf, size_t bytes, int slots, int qps, int slots_to_check) {
  const size_t verify_bytes = checked_window_bytes(bytes, slots, qps);
  std::vector<char> host(verify_bytes);
  CUDA_CHECK(cudaMemcpy(host.data(), recvbuf, verify_bytes, cudaMemcpyDeviceToHost));
  for (int qp = 0; qp < qps; ++qp) {
    for (int slot = 0; slot < slots_to_check; ++slot) {
      const size_t base = (static_cast<size_t>(qp) * static_cast<size_t>(slots) + slot) * bytes;
      for (size_t i = 0; i < bytes; ++i) {
        char expected = static_cast<char>(i & 0xff);
        if (host[base + i] != expected) {
          std::cerr << "check failed at qp " << qp << " slot " << slot << " byte " << i
                    << ": got " << static_cast<int>(static_cast<unsigned char>(host[base + i]))
                    << ", expected " << static_cast<int>(static_cast<unsigned char>(expected))
                    << "\n";
          return false;
        }
      }
    }
  }
  return true;
}

static void write_csv(const std::string &path, int rank, const Options &opt, const Sample &sample,
                      double timer_ticks_per_us) {
  CsvFile csv(path, 6);
  std::ofstream &out = csv.stream();
  out << std::fixed << std::setprecision(6);
  write_csv_row(out, "rank", "bytes", "iters", "warmup_iters", "tx_depth", "post_list", "qps",
                "remote_signals", "slots", "window_bytes", "timer_ticks_per_us", "elapsed_us",
                "bandwidth_GBps", "bandwidth_GiBps", "t_start", "t_stop");

  double elapsed_us = 0.0;
  double bw_gbps = 0.0;
  double bw_gibps = 0.0;
  if (rank == 0) {
    elapsed_us = static_cast<double>(sample.t_stop - sample.t_start) / timer_ticks_per_us;
    double total_bytes =
        static_cast<double>(opt.bytes) * static_cast<double>(opt.iters) * static_cast<double>(opt.qps);
    bw_gbps = total_bytes / elapsed_us / 1000.0;
    bw_gibps = total_bytes / elapsed_us * 1000000.0 / 1024.0 / 1024.0 / 1024.0;
  }

  write_csv_row(out, rank, opt.bytes, opt.iters, opt.warmup_iters, opt.tx_depth, opt.post_list,
                opt.qps, opt.qps, opt.slots, checked_window_bytes(opt.bytes, opt.slots, opt.qps),
                timer_ticks_per_us, elapsed_us, bw_gbps, bw_gibps, sample.t_start, sample.t_stop);
}

static void print_summary(int rank, const Options &opt, const Sample &sample,
                          double timer_ticks_per_us, bool ok) {
  if (rank != 0) {
    return;
  }
  const double elapsed_us =
      static_cast<double>(sample.t_stop - sample.t_start) / timer_ticks_per_us;
  const double total_bytes =
      static_cast<double>(opt.bytes) * static_cast<double>(opt.iters) * static_cast<double>(opt.qps);
  const double bw_gbps = total_bytes / elapsed_us / 1000.0;
  const double bw_gibps = total_bytes / elapsed_us * 1000000.0 / 1024.0 / 1024.0 / 1024.0;
  std::cout << std::fixed << std::setprecision(3)
            << "nccl_gin_write_bw_perf result bytes=" << opt.bytes << " iters=" << opt.iters
            << " tx_depth=" << opt.tx_depth << " post_list=" << opt.post_list
            << " slots=" << opt.slots << " qps=" << opt.qps << " elapsed_us=" << elapsed_us
            << " bandwidth_GBps=" << bw_gbps << " bandwidth_GiBps=" << bw_gibps
            << " check=" << (ok ? "ok" : "failed") << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  bool mpi_initialized = false;
  try {
    Options opt = parse_args(argc, argv);
    const size_t window_bytes = checked_window_bytes(opt.bytes, opt.slots, opt.qps);

    MPI_CHECK(MPI_Init(&argc, &argv));
    mpi_initialized = true;

    int rank = 0;
    int nranks = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    if (nranks != 2) {
      if (rank == 0) {
        std::cerr << "nccl_gin_write_bw_perf requires exactly 2 ranks, got " << nranks
                  << "\n";
      }
      MPI_CHECK(MPI_Finalize());
      return 2;
    }

    int dev = -1;
    if (rank == 0 && opt.src_dev >= 0) {
      dev = opt.src_dev;
    } else if (rank == 1 && opt.dst_dev >= 0) {
      dev = opt.dst_dev;
    } else {
      dev = opt.device >= 0 ? opt.device : mpi_local_rank(MPI_COMM_WORLD);
    }
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
    Sample *sample_d = nullptr;
    uint64_t *signal_bases_d = nullptr;
    uint64_t *counter_bases_d = nullptr;
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

    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void **>(&sendbuf), window_bytes));
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void **>(&recvbuf), window_bytes));
    NCCL_CHECK(ncclCommWindowRegister(comm, sendbuf, window_bytes, &sendwin,
                                      NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, recvbuf, window_bytes, &recvwin,
                                      NCCL_WIN_COLL_SYMMETRIC));

    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = std::max(kGinResourceCount, opt.qps);
    reqs.ginCounterCount = opt.qps;
    reqs.ginContextCount = std::max(4, opt.qps);
    reqs.lsaBarrierCount = kGinResourceCount;
    reqs.railGinBarrierCount = kGinResourceCount;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
    reqs.ginForceEnable = true;
#endif
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sample_d), sizeof(Sample)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&signal_bases_d), sizeof(uint64_t) * opt.qps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&counter_bases_d), sizeof(uint64_t) * opt.qps));
    CUDA_CHECK(cudaMemset(sample_d, 0, sizeof(Sample)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    int init_blocks = static_cast<int>(std::min<size_t>((window_bytes + 255) / 256, 1024));
    init_blocks = std::max(init_blocks, 1);
    init_buffers_kernel<<<init_blocks, 256, 0, stream>>>(sendbuf, recvbuf, opt.bytes,
                                                         window_bytes, rank);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (opt.warmup_iters > 0) {
      read_gin_bases_kernel<<<opt.qps, kKernelThreads, 0, stream>>>(dev_comm, signal_bases_d,
                                                                    counter_bases_d);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      nccl_gin_write_bw_kernel<<<opt.qps, kKernelThreads, 0, stream>>>(
          sendwin, recvwin, dev_comm, sample_d, signal_bases_d, counter_bases_d, opt.bytes,
          opt.warmup_iters, opt.tx_depth, opt.post_list, opt.slots);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaMemset(sample_d, 0, sizeof(Sample)));
    read_gin_bases_kernel<<<opt.qps, kKernelThreads, 0, stream>>>(dev_comm, signal_bases_d,
                                                                  counter_bases_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CHECK(cudaEventRecord(start_event, stream));
    nccl_gin_write_bw_kernel<<<opt.qps, kKernelThreads, 0, stream>>>(
        sendwin, recvwin, dev_comm, sample_d, signal_bases_d, counter_bases_d, opt.bytes,
        opt.iters, opt.tx_depth, opt.post_list, opt.slots);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    Sample sample{};
    CUDA_CHECK(cudaMemcpy(&sample, sample_d, sizeof(Sample), cudaMemcpyDeviceToHost));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    sample.t_start = 0;
    sample.t_stop = static_cast<uint64_t>(static_cast<double>(elapsed_ms) * 1000.0 * timer_ticks_per_us);

    bool ok_local = true;
    if (opt.check && rank == 1) {
      ok_local = verify_rank1(recvbuf, opt.bytes, opt.slots, opt.qps, touched_slots(opt));
    }
    int ok_int = ok_local ? 1 : 0;
    int ok_all = 0;
    MPI_CHECK(MPI_Allreduce(&ok_int, &ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    bool ok = ok_all == 1;

    write_csv(rank_csv_path(opt.csv, rank, nranks), rank, opt, sample, timer_ticks_per_us);
    print_summary(rank, opt, sample, timer_ticks_per_us, ok);

    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(counter_bases_d));
    CUDA_CHECK(cudaFree(signal_bases_d));
    CUDA_CHECK(cudaFree(sample_d));
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

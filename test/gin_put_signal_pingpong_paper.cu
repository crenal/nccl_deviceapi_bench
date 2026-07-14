#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include "common/checks.hpp"
#include "common/mpi.hpp"
#include "common/parse.hpp"
#include "common/sweep.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace nccl_deviceapi_test;

constexpr int kThreads = 32;
constexpr int kGinResourceCount = 16;
constexpr ncclGinSignal_t kPingSignal = 0;
constexpr ncclGinSignal_t kPongSignal = 1;

struct Options {
  size_t min_bytes = 4;
  size_t max_bytes = 4ull << 20;
  int factor = 2;
  int warmup = 1000;
  int iters = 10000;
  int device = -1;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--min-bytes N] [--max-bytes N] [--factor N]\n"
               "          [--warmup N] [--iters N] [--device ID]\n"
               "  Paper-style NCCL GIN put-with-signal pingpong latency benchmark.\n"
               "  Timing uses CUDA event elapsed time divided by --iters; no kernel-side timer instrumentation.\n"
               "  Requires exactly 2 MPI ranks.\n",
               argv0);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        usage(argv[0]);
        std::exit(3);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--min-bytes") == 0) {
      opt.min_bytes = parse_size(need(argv[i]));
    } else if (std::strcmp(argv[i], "--max-bytes") == 0) {
      opt.max_bytes = parse_size(need(argv[i]));
    } else if (std::strcmp(argv[i], "--bytes") == 0) {
      opt.min_bytes = opt.max_bytes = parse_size(need(argv[i]));
    } else if (std::strcmp(argv[i], "--factor") == 0) {
      opt.factor = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 || std::strcmp(argv[i], "--warmup-iters") == 0) {
      opt.warmup = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      opt.iters = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--device") == 0) {
      opt.device = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument(std::string("unknown argument: ") + argv[i]);
    }
  }
  if (opt.min_bytes == 0 || opt.max_bytes < opt.min_bytes || opt.factor <= 1 || opt.warmup < 0 ||
      opt.iters <= 0) {
    throw std::invalid_argument("invalid benchmark options");
  }
  return opt;
}

__global__ void fill_kernel(char* sendbuf, char* recvbuf, size_t bytes, int rank) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = idx; i < bytes; i += stride) {
    sendbuf[i] = static_cast<char>((i + rank * 17) & 0xff);
    recvbuf[i] = 0;
  }
}

__global__ void gin_put_signal_pingpong_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                               ncclDevComm dev_comm, size_t bytes, int iters) {
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclCoopWarpSpan warp(0, 1, 0);
  const int peer = world.rank == 0 ? 1 : 0;

  __shared__ uint64_t signal_bases[2];
  if (threadIdx.x == 0) {
    signal_bases[0] = gin.readSignal(kPingSignal);
    signal_bases[1] = gin.readSignal(kPongSignal);
  }
  __syncthreads();

  for (int i = 0; i < iters; i++) {
    uint64_t ping_expected = signal_bases[0] + static_cast<uint64_t>(i + 1);
    uint64_t pong_expected = signal_bases[1] + static_cast<uint64_t>(i + 1);

    if (world.rank == 0) {
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_SignalInc{kPingSignal},
              ncclGin_None{}, warp);
      gin.waitSignal(warp, kPongSignal, pong_expected, 64);
    } else {
      gin.waitSignal(warp, kPingSignal, ping_expected, 64);
      gin.put(world, peer, recvwin, 0, sendwin, 0, bytes, ncclGin_SignalInc{kPongSignal},
              ncclGin_None{}, warp);
    }
  }
}

bool verify_rank0(char* recvbuf, size_t bytes) {
  std::vector<char> host(bytes);
  CUDA_CHECK(cudaMemcpy(host.data(), recvbuf, bytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < bytes; i++) {
    char expected = static_cast<char>((i + 17) & 0xff);
    if (host[i] != expected) {
      std::cerr << "verify failed at byte " << i << ": got "
                << static_cast<int>(static_cast<unsigned char>(host[i])) << " expected "
                << static_cast<int>(static_cast<unsigned char>(expected)) << "\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
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
        std::cerr << "gin_put_signal_pingpong_paper requires exactly 2 ranks, got " << nranks << "\n";
      }
      MPI_CHECK(MPI_Finalize());
      return 2;
    }

    int dev = opt.device >= 0 ? opt.device : mpi_local_rank(MPI_COMM_WORLD);
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0 || dev >= dev_count) {
      throw std::runtime_error("selected CUDA device is not visible");
    }
    CUDA_CHECK(cudaSetDevice(dev));

    std::vector<size_t> sizes = make_sizes(opt.min_bytes, opt.max_bytes, opt.factor);

    ncclUniqueId id = mpi_bcast_nccl_unique_id(rank, MPI_COMM_WORLD);
    ncclComm_t comm = nullptr;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    if (!props.deviceApiSupport) {
      throw std::runtime_error("NCCL device API is not supported by this communicator");
    }
    if (props.ginType == NCCL_GIN_TYPE_NONE) {
      throw std::runtime_error("NCCL GIN is not enabled for this communicator");
    }

    char* sendbuf = nullptr;
    char* recvbuf = nullptr;
    ncclWindow_t sendwin = nullptr;
    ncclWindow_t recvwin = nullptr;
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&sendbuf), opt.max_bytes));
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&recvbuf), opt.max_bytes));
    NCCL_CHECK(ncclCommWindowRegister(comm, sendbuf, opt.max_bytes, &sendwin, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, recvbuf, opt.max_bytes, &recvwin, NCCL_WIN_COLL_SYMMETRIC));

    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = kGinResourceCount;
    reqs.ginContextCount = 1;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
    reqs.ginForceEnable = true;
#endif
    ncclDevComm dev_comm{};
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    if (rank == 0) {
      std::printf("# gin_put_signal_pingpong_paper ranks=2 warmup=%d iters=%d timing=cuda_event_elapsed_div_iters\n",
                  opt.warmup, opt.iters);
      std::printf("size_B,event_avg_roundtrip_us,oneway_event_avg_us,bw_event_GBps\n");
    }

    bool ok = true;
    for (size_t bytes : sizes) {
      int fill_blocks = static_cast<int>(std::min<size_t>((bytes + 255) / 256, 1024));
      fill_blocks = std::max(fill_blocks, 1);
      fill_kernel<<<fill_blocks, 256, 0, stream>>>(sendbuf, recvbuf, bytes, rank);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      if (opt.warmup > 0) {
        gin_put_signal_pingpong_kernel<<<1, kThreads, 0, stream>>>(sendwin, recvwin, dev_comm, bytes,
                                                                   opt.warmup);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      }

      CUDA_CHECK(cudaEventRecord(start, stream));
      gin_put_signal_pingpong_kernel<<<1, kThreads, 0, stream>>>(sendwin, recvwin, dev_comm, bytes,
                                                                 opt.iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(stop, stream));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
      double event_avg_us = static_cast<double>(elapsed_ms) * 1000.0 / static_cast<double>(opt.iters);

      if (rank == 0) {
        double oneway_event_avg = event_avg_us / 2.0;
        double bw = gib_per_second(bytes, oneway_event_avg);
        std::printf("%zu,%.3f,%.3f,%.3f\n", bytes, event_avg_us, oneway_event_avg, bw);
        ok = ok && verify_rank0(recvbuf, bytes);
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    NCCL_CHECK(ncclCommWindowDeregister(comm, sendwin));
    NCCL_CHECK(ncclCommWindowDeregister(comm, recvwin));
    NCCL_CHECK(ncclMemFree(sendbuf));
    NCCL_CHECK(ncclMemFree(recvbuf));
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    if (mpi_initialized) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
  }
}

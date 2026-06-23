#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include "common/checks.hpp"
#include "common/mpi.hpp"
#include "common/parse.hpp"

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

constexpr int kBlockThreads = 256;
constexpr int kGinWarpThreads = 32;
constexpr int kGinResourceCount = 16;
constexpr ncclGinSignal_t kDataReadySignal = 0;

struct Options {
  size_t bytes = 128ull << 10;
  int iters = 10000;
  int device = -1;
};

__host__ __device__ __forceinline__ float expected_value(uint64_t round, size_t index) {
  return static_cast<float>((round & 0xffffull) * 1024ull + (index & 0x3ffull));
}

__global__ void init_round_kernel(float* src, float* dst, size_t nelems, uint64_t round) {
  for (size_t i = threadIdx.x; i < nelems; i += blockDim.x) {
    src[i] = expected_value(round, i);
    dst[i] = -1.0f;
  }
}

__global__ void read_signal_base_kernel(ncclDevComm dev_comm, uint64_t* signal_base) {
  if (threadIdx.x == 0) {
    ncclGin gin{dev_comm, 0};
    *signal_base = gin.readSignal(kDataReadySignal);
  }
}

__global__ void send_kernel(ncclWindow_t srcwin, ncclWindow_t dstwin, ncclDevComm dev_comm,
                            size_t bytes) {
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  if (world.rank != 0) return;

  ncclCoopWarpSpan warps(0, 1, 0);
  if (threadIdx.x < kGinWarpThreads) {
    gin.put(world, 1, dstwin, 0, srcwin, 0, bytes, ncclGin_SignalInc{kDataReadySignal},
            ncclGin_None{}, warps);
  }
}

__global__ void recv_check_kernel(float* dst, ncclDevComm dev_comm, int* bad,
                                  int* first_idx, float* first_got, float* first_want,
                                  size_t nelems, uint64_t round, uint64_t expected_signal) {
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  if (world.rank != 1) return;

  ncclCoopWarpSpan warps(0, 1, 0);
  if (threadIdx.x < kGinWarpThreads) {
    gin.waitSignal(warps, kDataReadySignal, expected_signal, 64);
  }
  __syncthreads();

  __shared__ int block_bad;
  if (threadIdx.x == 0) block_bad = 0;
  __syncthreads();

  for (size_t i = threadIdx.x; i < nelems; i += blockDim.x) {
    float want = expected_value(round, i);
    float got = dst[i];
    if (got != want) {
      if (atomicCAS(first_idx, -1, static_cast<int>(i)) == -1) {
        *first_got = got;
        *first_want = want;
      }
      atomicExch(&block_bad, 1);
      break;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && block_bad) {
    atomicAdd(bad, 1);
  }
}

__global__ void sender_flush_kernel(ncclDevComm dev_comm) {
  ncclGin gin{dev_comm, 0};
  ncclTeam world = ncclTeamWorld(dev_comm);
  if (world.rank != 0) return;

  ncclCoopWarpSpan warps(0, 1, 0);
  if (threadIdx.x < kGinWarpThreads) {
    gin.flush(warps);
  }
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--bytes N] [--iters N] [--device ID]\n"
               "  Tests whether NCCL GIN put-with-signal data is readable as soon as waitSignal returns.\n"
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
    if (std::strcmp(argv[i], "--bytes") == 0) {
      opt.bytes = parse_size(need(argv[i]));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      opt.iters = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--device") == 0) {
      opt.device = parse_int(need(argv[i]), argv[i]);
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      usage(argv[0]);
      throw std::invalid_argument(std::string("unknown argument: ") + argv[i]);
    }
  }
  if (opt.bytes == 0 || opt.iters <= 0) {
    throw std::invalid_argument("--bytes and --iters must be positive");
  }
  return opt;
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
        std::cerr << "gin_put_signal_visibility requires exactly 2 ranks, got " << nranks << "\n";
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

    size_t nelems = (opt.bytes + sizeof(float) - 1) / sizeof(float);
    opt.bytes = nelems * sizeof(float);

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

    float* src = nullptr;
    float* dst = nullptr;
    ncclWindow_t srcwin = nullptr;
    ncclWindow_t dstwin = nullptr;
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&src), opt.bytes));
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&dst), opt.bytes));
    NCCL_CHECK(ncclCommWindowRegister(comm, src, opt.bytes, &srcwin, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, dst, opt.bytes, &dstwin, NCCL_WIN_COLL_SYMMETRIC));

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

    int* bad = nullptr;
    int* first_idx = nullptr;
    float* first_got = nullptr;
    float* first_want = nullptr;
    uint64_t* signal_base_d = nullptr;
    CUDA_CHECK(cudaMalloc(&bad, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&first_idx, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&first_got, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&first_want, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&signal_base_d, sizeof(uint64_t)));

    read_signal_base_kernel<<<1, 1>>>(dev_comm, signal_base_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint64_t signal_base = 0;
    CUDA_CHECK(cudaMemcpy(&signal_base, signal_base_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank == 0) {
      std::printf("gin_put_signal_visibility: bytes=%zu iters=%d mode=gin.put+SignalInc waitSignal\n",
                  opt.bytes, opt.iters);
    }

    unsigned long long local_bad_rounds = 0;
    int first_bad_round = -1;
    int first_bad_idx = -1;
    float first_bad_got = 0.0f;
    float first_bad_want = 0.0f;

    for (int r = 1; r <= opt.iters; r++) {
      int zero = 0;
      int neg = -1;
      CUDA_CHECK(cudaMemcpy(bad, &zero, sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(first_idx, &neg, sizeof(int), cudaMemcpyHostToDevice));

      init_round_kernel<<<1, kBlockThreads>>>(src, dst, nelems, static_cast<uint64_t>(r));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      uint64_t expected_signal = signal_base + static_cast<uint64_t>(r);
      send_kernel<<<1, kBlockThreads>>>(srcwin, dstwin, dev_comm, opt.bytes);
      recv_check_kernel<<<1, kBlockThreads>>>(dst, dev_comm, bad, first_idx, first_got,
                                              first_want, nelems, static_cast<uint64_t>(r),
                                              expected_signal);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      sender_flush_kernel<<<1, kGinWarpThreads>>>(dev_comm);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      int h_bad = 0;
      CUDA_CHECK(cudaMemcpy(&h_bad, bad, sizeof(int), cudaMemcpyDeviceToHost));
      if (h_bad) {
        local_bad_rounds++;
        if (first_bad_round < 0) {
          first_bad_round = r;
          CUDA_CHECK(cudaMemcpy(&first_bad_idx, first_idx, sizeof(int), cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(&first_bad_got, first_got, sizeof(float), cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(&first_bad_want, first_want, sizeof(float), cudaMemcpyDeviceToHost));
        }
      }
    }

    unsigned long long total_bad_rounds = 0;
    MPI_CHECK(MPI_Reduce(&local_bad_rounds, &total_bad_rounds, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    if (rank == 1 && first_bad_round >= 0) {
      std::printf("receiver first_bad: round=%d idx=%d got=%f want=%f\n",
                  first_bad_round, first_bad_idx, first_bad_got, first_bad_want);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) {
      std::printf("result: %s (%llu bad rounds / %d)\n",
                  total_bad_rounds == 0 ? "ALL OK" : "FAILED", total_bad_rounds, opt.iters);
    }

    CUDA_CHECK(cudaFree(bad));
    CUDA_CHECK(cudaFree(first_idx));
    CUDA_CHECK(cudaFree(first_got));
    CUDA_CHECK(cudaFree(first_want));
    CUDA_CHECK(cudaFree(signal_base_d));
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    NCCL_CHECK(ncclCommWindowDeregister(comm, srcwin));
    NCCL_CHECK(ncclCommWindowDeregister(comm, dstwin));
    NCCL_CHECK(ncclMemFree(src));
    NCCL_CHECK(ncclMemFree(dst));
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
    return total_bad_rounds == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    if (mpi_initialized) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
  }
}

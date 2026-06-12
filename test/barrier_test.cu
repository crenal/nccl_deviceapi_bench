// Standalone NCCL GIN/world barrier latency benchmark.

#include <cuda_runtime.h>
#include <nccl.h>

#include "common/checks.hpp"
#include "common/env.hpp"
#include "common/socket.hpp"
#include "common/stats.hpp"
#include "common/timer.cuh"
#include "nccl_device/core.h"
#include "nccl_device/barrier.h"
#include "nccl_device/impl/core__funcs.h"
#include "nccl_device/impl/gin__funcs.h"
#include "nccl_device/impl/gin_barrier__funcs.h"
#include "nccl_device/impl/barrier__funcs.h"
#include "nccl_device/impl/ptr__funcs.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace nccl_deviceapi_test;

struct Options {
  int warmup = 100;
  int iters = 1000;
  int threads = 512;
  int device = -1;
  int gin_context = 0;
  int port = 48123;
  bool multimem = true;
  bool full_connection = false;
  double ticks_per_us = 0.0;
  std::string master_addr;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
      "Usage: %s [--warmup N] [--iters N] [--threads N] [--device ID]\n"
      "          [--master ADDR] [--port PORT] [--ticks-per-us X]\n"
      "          [--gin-context ID] [--multimem 0|1] [--connection rail|full]\n",
      argv0);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  opt.master_addr = env_string("BARRIER_TEST_MASTER_ADDR", "127.0.0.1");
  opt.port = env_int("BARRIER_TEST_MASTER_PORT", opt.port);

  for (int i = 1; i < argc; i++) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        usage(argv[0]);
        std::exit(3);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--warmup") == 0) {
      opt.warmup = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      opt.iters = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--threads") == 0) {
      opt.threads = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--device") == 0) {
      opt.device = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--master") == 0) {
      opt.master_addr = need(argv[i]);
    } else if (std::strcmp(argv[i], "--port") == 0) {
      opt.port = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--ticks-per-us") == 0) {
      opt.ticks_per_us = std::atof(need(argv[i]));
    } else if (std::strcmp(argv[i], "--gin-context") == 0) {
      opt.gin_context = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--multimem") == 0) {
      opt.multimem = std::atoi(need(argv[i])) != 0;
    } else if (std::strcmp(argv[i], "--connection") == 0) {
      const char* c = need(argv[i]);
      if (std::strcmp(c, "rail") == 0) {
        opt.full_connection = false;
      } else if (std::strcmp(c, "full") == 0) {
        opt.full_connection = true;
      } else {
        std::fprintf(stderr, "Unknown connection type: %s\n", c);
        std::exit(3);
      }
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(3);
    }
  }

  if (opt.warmup < 0 || opt.iters <= 0 || opt.threads <= 0 || opt.threads > 1024 ||
      (opt.threads % 32) != 0 || opt.port <= 0 || opt.ticks_per_us < 0.0 || opt.gin_context < 0) {
    std::fprintf(stderr, "Invalid arguments\n");
    std::exit(3);
  }
  return opt;
}

__global__ void barrier_bench_kernel(ncclDevComm dev_comm, uint64_t* samples, int warmup, int iters,
                                     int gin_context, int use_multimem) {
  ncclCoopCta cta;
  ncclGin gin(dev_comm, gin_context);
  ncclBarrierSession<ncclCoopCta> bar(cta, ncclTeamTagWorld(), gin, 0, use_multimem != 0);

  for (int i = 0; i < warmup; i++) {
    bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
  }

  for (int i = 0; i < iters; i++) {
    uint64_t t0 = global_timer();
    bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
    uint64_t t1 = global_timer();
    bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
    uint64_t t2 = global_timer();
    bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
    if (threadIdx.x == 0) {
      (void)t0;
      samples[i] = t2 - t1;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options opt = parse_args(argc, argv);

    int rank = env_int("OMPI_COMM_WORLD_RANK", env_int("PMI_RANK", env_int("RANK", 0)));
    int nranks = env_int("OMPI_COMM_WORLD_SIZE", env_int("PMI_SIZE", env_int("WORLD_SIZE", 1)));
    int local_rank = env_int("OMPI_COMM_WORLD_LOCAL_RANK", env_int("MPI_LOCALRANKID", env_int("LOCAL_RANK", rank)));

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int device = opt.device >= 0 ? opt.device : (local_rank % std::max(1, ndev));
    CUDA_CHECK(cudaSetDevice(device));
    if (opt.ticks_per_us == 0.0) opt.ticks_per_us = calibrate_ticks_per_us();

    ncclUniqueId id = exchange_unique_id_socket(rank, nranks, opt.master_addr, opt.port);

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));

    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaMultimem = opt.multimem && props.multimemSupport;
    reqs.barrierCount = 1;
    reqs.ginContextCount = std::max(1, opt.gin_context + 1);
    reqs.ginConnectionType = opt.full_connection ? NCCL_GIN_CONNECTION_FULL : NCCL_GIN_CONNECTION_RAIL;
    reqs.ginQueueDepth = 0;

    ncclDevComm_t dev_comm;
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    uint64_t* d_samples = nullptr;
    std::vector<uint64_t> h_samples(opt.iters);
    CUDA_CHECK(cudaMalloc(&d_samples, opt.iters * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemsetAsync(d_samples, 0, opt.iters * sizeof(uint64_t), stream));

    barrier_bench_kernel<<<1, opt.threads, 0, stream>>>(dev_comm, d_samples, opt.warmup, opt.iters,
                                                        opt.gin_context, reqs.lsaMultimem ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_samples.data(), d_samples, opt.iters * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    NCCL_CHECK(ncclCommDestroy(comm));

    std::vector<double> local = ticks_to_us(h_samples, opt.ticks_per_us);
    std::vector<double> all = gather_samples_socket(rank, nranks, opt.master_addr, opt.port + 1, local);

    if (rank == 0) {
      MetricStats stats = compute_stats(all);
      std::printf("# barrier-test ranks=%d threads=%d warmup=%d iters=%d samples=%zu ticks_per_us=%.3f "
                  "connection=%s multimem=%d gin_context=%d\n",
                  nranks, opt.threads, opt.warmup, opt.iters, all.size(), opt.ticks_per_us,
                  opt.full_connection ? "full" : "rail", reqs.lsaMultimem ? 1 : 0, opt.gin_context);
      std::printf("metric,us\n");
      std::printf("avg,%.3f\n", stats.avg);
      std::printf("min,%.3f\n", stats.min);
      std::printf("p50,%.3f\n", stats.p50);
      std::printf("p90,%.3f\n", stats.p90);
      std::printf("p95,%.3f\n", stats.p95);
      std::printf("p99,%.3f\n", stats.p99);
      std::printf("max,%.3f\n", stats.max);
    }
    return 0;
  } catch (const std::exception& e) {
    nccl_deviceapi_test::fail_fast(e);
    return 1;
  }
}

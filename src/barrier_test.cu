// Standalone NCCL GIN barrier latency benchmark.
//
// Build example:
//   nvcc -O2 -std=c++17 -I${NCCL_SRC}/src/include -I${NCCL_HOME}/include \
//        barrier-test.cu -L${NCCL_HOME}/lib -lnccl -lcudart -o barrier-test
//
// Run example with mpirun only as a process launcher:
//   BARRIER_TEST_MASTER_ADDR=node0 BARRIER_TEST_MASTER_PORT=48123 \
//   mpirun --allow-run-as-root -np 32 --hostfile hostfile --map-by ppr:8:node \
//     -x LD_LIBRARY_PATH -x NCCL_IB_HCA -x NCCL_GIN_TYPE \
//     -x BARRIER_TEST_MASTER_ADDR -x BARRIER_TEST_MASTER_PORT \
//     ./barrier-test --warmup 100 --iters 1000

#include <cuda_runtime.h>
#include <nccl.h>

#include "nccl_device/core.h"
#include "nccl_device/barrier.h"
#include "nccl_device/impl/core__funcs.h"
#include "nccl_device/impl/ptr__funcs.h"
#include "nccl_device/impl/gin__funcs.h"
#include "nccl_device/impl/gin_barrier__funcs.h"
#include "nccl_device/impl/barrier__funcs.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define CUDACHECK(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA failure %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

#define NCCLCHECK(cmd) do { \
  ncclResult_t r = (cmd); \
  if (r != ncclSuccess) { \
    std::fprintf(stderr, "NCCL failure %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    std::exit(2); \
  } \
} while (0)

struct Options {
  int warmup = 100;
  int iters = 1000;
  int threads = 512;
  int device = -1;
  int ginContext = 0;
  int port = 48123;
  bool multimem = true;
  bool fullConnection = false;
  double ticksPerUs = 0.0;
  std::string masterAddr;
};

static int envInt(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v == nullptr ? fallback : std::atoi(v);
}

static std::string envString(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return v == nullptr ? std::string(fallback) : std::string(v);
}

static void usage(const char* argv0) {
  std::fprintf(stderr,
      "Usage: %s [--warmup N] [--iters N] [--threads N] [--device ID]\n"
      "          [--master ADDR] [--port PORT] [--ticks-per-us X]\n"
      "          [--gin-context ID] [--multimem 0|1] [--connection rail|full]\n",
      argv0);
}

static Options parseArgs(int argc, char** argv) {
  Options opt;
  opt.masterAddr = envString("BARRIER_TEST_MASTER_ADDR", "127.0.0.1");
  opt.port = envInt("BARRIER_TEST_MASTER_PORT", opt.port);

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
      opt.masterAddr = need(argv[i]);
    } else if (std::strcmp(argv[i], "--port") == 0) {
      opt.port = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--ticks-per-us") == 0) {
      opt.ticksPerUs = std::atof(need(argv[i]));
    } else if (std::strcmp(argv[i], "--gin-context") == 0) {
      opt.ginContext = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--multimem") == 0) {
      opt.multimem = std::atoi(need(argv[i])) != 0;
    } else if (std::strcmp(argv[i], "--connection") == 0) {
      const char* c = need(argv[i]);
      if (std::strcmp(c, "rail") == 0) {
        opt.fullConnection = false;
      } else if (std::strcmp(c, "full") == 0) {
        opt.fullConnection = true;
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
      (opt.threads % 32) != 0 || opt.port <= 0 || opt.ticksPerUs < 0.0 || opt.ginContext < 0) {
    std::fprintf(stderr, "Invalid arguments\n");
    std::exit(3);
  }
  return opt;
}

static bool sendAll(int fd, const void* data, size_t bytes) {
  const char* p = static_cast<const char*>(data);
  while (bytes > 0) {
    ssize_t n = ::send(fd, p, bytes, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
}

static bool recvAll(int fd, void* data, size_t bytes) {
  char* p = static_cast<char*>(data);
  while (bytes > 0) {
    ssize_t n = ::recv(fd, p, bytes, MSG_WAITALL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
}

static int listenSocket(int port) {
  addrinfo hints = {};
  hints.ai_family = AF_INET6;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  addrinfo* res = nullptr;
  std::string portStr = std::to_string(port);
  int rc = ::getaddrinfo(nullptr, portStr.c_str(), &hints, &res);
  if (rc != 0) {
    std::fprintf(stderr, "getaddrinfo listen failed: %s\n", gai_strerror(rc));
    std::exit(4);
  }

  int fd = -1;
  for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
    fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) continue;
    int one = 1;
    int zero = 0;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    ::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof(zero));
    if (::bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && ::listen(fd, 256) == 0) break;
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(res);

  if (fd < 0) {
    std::fprintf(stderr, "Could not listen on port %d: %s\n", port, std::strerror(errno));
    std::exit(4);
  }
  return fd;
}

static int connectSocket(const std::string& host, int port, int retries = 500) {
  addrinfo hints = {};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  std::string portStr = std::to_string(port);

  for (int attempt = 0; attempt < retries; attempt++) {
    addrinfo* res = nullptr;
    int rc = ::getaddrinfo(host.c_str(), portStr.c_str(), &hints, &res);
    if (rc == 0) {
      for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
        int fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
          ::freeaddrinfo(res);
          return fd;
        }
        ::close(fd);
      }
    }
    if (res != nullptr) ::freeaddrinfo(res);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  std::fprintf(stderr, "Could not connect to %s:%d\n", host.c_str(), port);
  std::exit(4);
}

static ncclUniqueId exchangeUniqueId(int rank, int nranks, const Options& opt) {
  ncclUniqueId id;
  if (rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&id));
    int listenFd = listenSocket(opt.port);
    for (int i = 1; i < nranks; i++) {
      int fd = ::accept(listenFd, nullptr, nullptr);
      if (fd < 0) {
        std::fprintf(stderr, "accept unique id failed: %s\n", std::strerror(errno));
        std::exit(4);
      }
      int peerRank = -1;
      recvAll(fd, &peerRank, sizeof(peerRank));
      if (!sendAll(fd, &id, sizeof(id))) {
        std::fprintf(stderr, "send unique id failed\n");
        std::exit(4);
      }
      ::close(fd);
    }
    ::close(listenFd);
  } else {
    int fd = connectSocket(opt.masterAddr, opt.port);
    sendAll(fd, &rank, sizeof(rank));
    if (!recvAll(fd, &id, sizeof(id))) {
      std::fprintf(stderr, "recv unique id failed\n");
      std::exit(4);
    }
    ::close(fd);
  }
  return id;
}

static std::vector<double> gatherSamples(int rank, int nranks, const Options& opt, const std::vector<double>& local) {
  if (rank == 0) {
    std::vector<double> all = local;
    int listenFd = listenSocket(opt.port + 1);
    for (int i = 1; i < nranks; i++) {
      int fd = ::accept(listenFd, nullptr, nullptr);
      if (fd < 0) {
        std::fprintf(stderr, "accept samples failed: %s\n", std::strerror(errno));
        std::exit(4);
      }
      int peerRank = -1;
      uint64_t count = 0;
      recvAll(fd, &peerRank, sizeof(peerRank));
      recvAll(fd, &count, sizeof(count));
      std::vector<double> peer(count);
      if (count != 0 && !recvAll(fd, peer.data(), count * sizeof(double))) {
        std::fprintf(stderr, "recv samples failed from rank %d\n", peerRank);
        std::exit(4);
      }
      all.insert(all.end(), peer.begin(), peer.end());
      ::close(fd);
    }
    ::close(listenFd);
    return all;
  }

  int fd = connectSocket(opt.masterAddr, opt.port + 1);
  uint64_t count = local.size();
  sendAll(fd, &rank, sizeof(rank));
  sendAll(fd, &count, sizeof(count));
  if (count != 0) sendAll(fd, local.data(), count * sizeof(double));
  ::close(fd);
  return {};
}

static double percentile(const std::vector<double>& v, double q) {
  if (v.empty()) return 0.0;
  size_t idx = static_cast<size_t>((v.size() - 1) * q + 0.5);
  return v[std::min(idx, v.size() - 1)];
}

__device__ __forceinline__ uint64_t readGlobalTimer() {
  uint64_t timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}

__global__ void calibrateGlobalTimerKernel(uint64_t targetTicks, uint64_t* deltaOut) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint64_t t0 = readGlobalTimer();
    uint64_t now = t0;
    while (now - t0 < targetTicks) {
      now = readGlobalTimer();
    }
    *deltaOut = now - t0;
  }
}

static double calibrateTicksPerUs() {
  constexpr uint64_t targetTicks = 50000000ULL;
  uint64_t* dDelta = nullptr;
  uint64_t hDelta = 0;
  cudaEvent_t start, stop;
  CUDACHECK(cudaMalloc(&dDelta, sizeof(uint64_t)));
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  CUDACHECK(cudaEventRecord(start));
  calibrateGlobalTimerKernel<<<1, 1>>>(targetTicks, dDelta);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaEventRecord(stop));
  CUDACHECK(cudaEventSynchronize(stop));
  float elapsedMs = 0.0f;
  CUDACHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
  CUDACHECK(cudaMemcpy(&hDelta, dDelta, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));
  CUDACHECK(cudaFree(dDelta));
  if (elapsedMs <= 0.0f || hDelta == 0) return 1000.0;
  return static_cast<double>(hDelta) / (static_cast<double>(elapsedMs) * 1000.0);
}

__global__ void barrierBenchKernel(ncclDevComm devComm, uint64_t* samples, int warmup, int iters,
                                   int ginContext, int useMultimem) {
  ncclCoopCta cta;
  ncclGin gin(devComm, ginContext);
  ncclBarrierSession<ncclCoopCta> bar(cta, ncclTeamTagWorld(), gin, 0, useMultimem != 0);

  for (int i = 0; i < warmup; i++) {
    bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
  }

  for (int i = 0; i < iters; i++) {
    uint64_t t0 = readGlobalTimer();
    bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
    uint64_t t1 = readGlobalTimer();
    bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
    uint64_t t2 = readGlobalTimer();
    bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
    if (threadIdx.x == 0) {
      (void)t0;
      samples[i] = t2 - t1;
    }
  }
}

int main(int argc, char** argv) {
  Options opt = parseArgs(argc, argv);

  int rank = envInt("OMPI_COMM_WORLD_RANK", envInt("PMI_RANK", envInt("RANK", 0)));
  int nranks = envInt("OMPI_COMM_WORLD_SIZE", envInt("PMI_SIZE", envInt("WORLD_SIZE", 1)));
  int localRank = envInt("OMPI_COMM_WORLD_LOCAL_RANK", envInt("MPI_LOCALRANKID", envInt("LOCAL_RANK", rank)));

  int ndev = 0;
  CUDACHECK(cudaGetDeviceCount(&ndev));
  int device = opt.device >= 0 ? opt.device : (localRank % std::max(1, ndev));
  CUDACHECK(cudaSetDevice(device));
  if (opt.ticksPerUs == 0.0) opt.ticksPerUs = calibrateTicksPerUs();

  ncclUniqueId id = exchangeUniqueId(rank, nranks, opt);

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));

  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(comm, &props));

  ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaMultimem = opt.multimem && props.multimemSupport;
  reqs.barrierCount = 1;
  reqs.ginContextCount = std::max(1, opt.ginContext + 1);
  reqs.ginConnectionType = opt.fullConnection ? NCCL_GIN_CONNECTION_FULL : NCCL_GIN_CONNECTION_RAIL;
  reqs.ginQueueDepth = 0;

  ncclDevComm_t devComm;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  uint64_t* dSamples = nullptr;
  std::vector<uint64_t> hSamples(opt.iters);
  CUDACHECK(cudaMalloc(&dSamples, opt.iters * sizeof(uint64_t)));
  CUDACHECK(cudaMemsetAsync(dSamples, 0, opt.iters * sizeof(uint64_t), stream));

  barrierBenchKernel<<<1, opt.threads, 0, stream>>>(devComm, dSamples, opt.warmup, opt.iters,
                                                    opt.ginContext, reqs.lsaMultimem ? 1 : 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaMemcpyAsync(hSamples.data(), dSamples, opt.iters * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  CUDACHECK(cudaFree(dSamples));
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
  NCCLCHECK(ncclCommDestroy(comm));

  std::vector<double> local;
  local.reserve(hSamples.size());
  for (uint64_t cycles : hSamples) {
    local.push_back(static_cast<double>(cycles) / opt.ticksPerUs);
  }

  std::vector<double> all = gatherSamples(rank, nranks, opt, local);

  if (rank == 0) {
    std::sort(all.begin(), all.end());
    double avg = all.empty() ? 0.0 : std::accumulate(all.begin(), all.end(), 0.0) / all.size();
    std::printf("# barrier-test ranks=%d threads=%d warmup=%d iters=%d samples=%zu ticks_per_us=%.3f "
                "connection=%s multimem=%d gin_context=%d\n",
                nranks, opt.threads, opt.warmup, opt.iters, all.size(), opt.ticksPerUs,
                opt.fullConnection ? "full" : "rail", reqs.lsaMultimem ? 1 : 0, opt.ginContext);
    std::printf("metric,us\n");
    std::printf("avg,%.3f\n", avg);
    std::printf("min,%.3f\n", all.empty() ? 0.0 : all.front());
    std::printf("p50,%.3f\n", percentile(all, 0.50));
    std::printf("p90,%.3f\n", percentile(all, 0.90));
    std::printf("p95,%.3f\n", percentile(all, 0.95));
    std::printf("p99,%.3f\n", percentile(all, 0.99));
    std::printf("max,%.3f\n", all.empty() ? 0.0 : all.back());
  }

  return 0;
}

// Standalone NCCL bcastMultimem latency/bandwidth benchmark.
//
// This benchmark does not modify NCCL. It builds against NCCL's device API
// headers and directly calls bcastMultimem<char, false>() from the symmetric
// AllGather implementation.

#include <cuda_runtime.h>
#include <nccl.h>

#include "nccl_device.h"
#include "device/symmetric/all_gather.cuh"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
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
  size_t minBytes = 64;
  size_t maxBytes = 512ull << 20;
  int factor = 2;
  int warmup = 100;
  int iters = 1000;
  int threads = 512;
  int numBlocks = 1;
  int device = -1;
  int port = 22340;
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

static size_t parseSize(const char* text) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 0);
  if (end == text) {
    std::fprintf(stderr, "Invalid size: %s\n", text);
    std::exit(3);
  }
  while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) end++;
  if (*end == '\0') return static_cast<size_t>(value);

  char suffix = static_cast<char>(std::tolower(static_cast<unsigned char>(*end)));
  end++;
  if (*end == 'i' || *end == 'I') end++;
  if (*end == 'b' || *end == 'B') end++;
  if (*end != '\0') {
    std::fprintf(stderr, "Invalid size suffix: %s\n", text);
    std::exit(3);
  }
  switch (suffix) {
  case 'k': return static_cast<size_t>(value) << 10;
  case 'm': return static_cast<size_t>(value) << 20;
  case 'g': return static_cast<size_t>(value) << 30;
  default:
    std::fprintf(stderr, "Invalid size suffix: %s\n", text);
    std::exit(3);
  }
}

static void usage(const char* argv0) {
  std::fprintf(stderr,
      "Usage: %s [--min-bytes N] [--max-bytes N] [--factor N]\n"
      "          [--warmup N] [--iters N] [--threads N] [--num-blocks N]\n"
      "          [--device ID] [--master ADDR] [--port PORT] [--ticks-per-us X]\n",
      argv0);
}

static Options parseArgs(int argc, char** argv) {
  Options opt;
  opt.masterAddr = envString("BCAST_MM_MASTER_ADDR", "127.0.0.1");
  opt.port = envInt("BCAST_MM_MASTER_PORT", opt.port);

  for (int i = 1; i < argc; i++) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        usage(argv[0]);
        std::exit(3);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--min-bytes") == 0) {
      opt.minBytes = parseSize(need(argv[i]));
    } else if (std::strcmp(argv[i], "--max-bytes") == 0) {
      opt.maxBytes = parseSize(need(argv[i]));
    } else if (std::strcmp(argv[i], "--factor") == 0) {
      opt.factor = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      opt.warmup = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      opt.iters = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--threads") == 0) {
      opt.threads = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--num-blocks") == 0) {
      opt.numBlocks = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--device") == 0) {
      opt.device = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--master") == 0) {
      opt.masterAddr = need(argv[i]);
    } else if (std::strcmp(argv[i], "--port") == 0) {
      opt.port = std::atoi(need(argv[i]));
    } else if (std::strcmp(argv[i], "--ticks-per-us") == 0) {
      opt.ticksPerUs = std::atof(need(argv[i]));
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(3);
    }
  }

  if (opt.minBytes == 0 || opt.maxBytes < opt.minBytes || opt.factor <= 1 ||
      opt.warmup < 0 || opt.iters <= 0 || opt.threads <= 0 || opt.threads > 1024 ||
      (opt.threads % 32) != 0 || opt.numBlocks <= 0 || opt.numBlocks > ncclSymkMaxBlocks ||
      opt.port <= 0 || opt.ticksPerUs < 0.0) {
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

static std::vector<size_t> makeSizes(size_t minBytes, size_t maxBytes, int factor) {
  std::vector<size_t> sizes;
  for (size_t s = minBytes; s <= maxBytes; ) {
    sizes.push_back(s);
    if (s > maxBytes / static_cast<size_t>(factor)) break;
    s *= static_cast<size_t>(factor);
  }
  return sizes;
}

static ncclSymkDevWorkArgs4K makeArgs(ncclDevComm_t devComm, ncclWindow_t inputWin, ncclWindow_t outputWin,
                                      size_t nBytes, int numBlocks) {
  ncclSymkDevWorkArgs4K args4K;
  std::memset(&args4K, 0, sizeof(args4K));
  args4K.args.kcomm.devComm = devComm;
  args4K.args.nMaxChannels = numBlocks;
  args4K.args.maxDynamicSmem = 0;

  ncclSymkChannelWorkRange* ranges = args4K.args.getWorkRange();
  for (int b = 0; b < numBlocks; b++) {
    uint32_t end = static_cast<uint32_t>((static_cast<uint64_t>(b + 1) * 0x10000ull) /
                                         static_cast<uint64_t>(numBlocks));
    ranges[b].workHi = 0;
    ranges[b].fracHi = static_cast<uint16_t>(end - 1);
  }

  ncclSymkDevWork* works = args4K.args.getWorks(numBlocks);
  works[0].redOpArg = 0;
  works[0].nElts = nBytes;
  works[0].inputWin = inputWin;
  works[0].outputWin = outputWin;
  works[0].inputOff = 0;
  works[0].outputOff = 0;
  works[0].rootRank = 0;
  works[0].sChannelId = 0;
  works[0].nChannels = numBlocks;
  return args4K;
}

__device__ __forceinline__ void runBcastMultimem(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  int const rank = handler.comm.rank;

  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
    int t = flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE,
                      block, nBlocks,
                      threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int tn = nBlocks * blockDim.x;
    bcastMultimem<char, false>(handler, tn, t, input, output + rank * nAllElts, nElts);
  });
}

__global__ void bcastMultimemBenchKernel(ncclSymkDevWorkArgs4K NCCL_GRID_CONSTANT const args4K,
                                         uint64_t* blockSamples, int warmup, int iters) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler{&args4K.args};
  ncclLsaBarrierSession<ncclCoopCta> bar(cta, handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true);

  for (int i = 0; i < warmup; i++) {
    bar.sync(cta, cuda::memory_order_release);
    runBcastMultimem(&args4K.args);
    bar.sync(cta, cuda::memory_order_release);
  }

  for (int i = 0; i < iters; i++) {
    bar.sync(cta, cuda::memory_order_release);
    uint64_t t0 = readGlobalTimer();
    runBcastMultimem(&args4K.args);
    bar.sync(cta, cuda::memory_order_release);
    uint64_t t1 = readGlobalTimer();
    if (threadIdx.x == 0) {
      blockSamples[static_cast<size_t>(i) * gridDim.x + blockIdx.x] = t1 - t0;
    }
  }
}

int main(int argc, char** argv) {
  Options opt = parseArgs(argc, argv);
  std::vector<size_t> sizes = makeSizes(opt.minBytes, opt.maxBytes, opt.factor);

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
  if (!props.multimemSupport) {
    std::fprintf(stderr, "Rank %d: NCCL communicator reports multimemSupport=0\n", rank);
    std::exit(5);
  }

  ncclTeam_t lsa = ncclTeamLsa(comm);
  int lsaRanks = lsa.nRanks;

  ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaMultimem = true;
  reqs.lsaBarrierCount = opt.numBlocks;
  reqs.ginContextCount = 0;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_NONE;
  reqs.ginQueueDepth = 0;

  ncclDevComm_t devComm;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  size_t sendBytes = opt.maxBytes;
  size_t recvBytes = opt.maxBytes * static_cast<size_t>(nranks);
  void* sendbuff = nullptr;
  void* recvbuff = nullptr;
  NCCLCHECK(ncclMemAlloc(&sendbuff, sendBytes));
  NCCLCHECK(ncclMemAlloc(&recvbuff, recvBytes));
  CUDACHECK(cudaMemsetAsync(sendbuff, 0x5a, sendBytes, stream));
  CUDACHECK(cudaMemsetAsync(recvbuff, 0, recvBytes, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  ncclWindow_t sendWin = nullptr;
  ncclWindow_t recvWin = nullptr;
  NCCLCHECK(ncclCommWindowRegister(comm, sendbuff, sendBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(comm, recvbuff, recvBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));

  uint64_t* dBlockSamples = nullptr;
  std::vector<uint64_t> hBlockSamples(static_cast<size_t>(opt.iters) * opt.numBlocks);
  CUDACHECK(cudaMalloc(&dBlockSamples, hBlockSamples.size() * sizeof(uint64_t)));

  std::vector<double> localSamples;
  localSamples.reserve(static_cast<size_t>(sizes.size()) * opt.iters);

  for (size_t sizeBytes : sizes) {
    ncclSymkDevWorkArgs4K args4K = makeArgs(devComm, sendWin, recvWin, sizeBytes, opt.numBlocks);
    CUDACHECK(cudaMemsetAsync(dBlockSamples, 0, hBlockSamples.size() * sizeof(uint64_t), stream));
    bcastMultimemBenchKernel<<<opt.numBlocks, opt.threads, 0, stream>>>(args4K, dBlockSamples, opt.warmup, opt.iters);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaMemcpyAsync(hBlockSamples.data(), dBlockSamples, hBlockSamples.size() * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < opt.iters; i++) {
      uint64_t maxCycles = 0;
      for (int b = 0; b < opt.numBlocks; b++) {
        maxCycles = std::max(maxCycles, hBlockSamples[static_cast<size_t>(i) * opt.numBlocks + b]);
      }
      localSamples.push_back(static_cast<double>(maxCycles) / opt.ticksPerUs);
    }
  }

  CUDACHECK(cudaFree(dBlockSamples));
  NCCLCHECK(ncclCommWindowDeregister(comm, sendWin));
  NCCLCHECK(ncclCommWindowDeregister(comm, recvWin));
  NCCLCHECK(ncclMemFree(sendbuff));
  NCCLCHECK(ncclMemFree(recvbuff));
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
  NCCLCHECK(ncclCommDestroy(comm));

  std::vector<double> all = gatherSamples(rank, nranks, opt, localSamples);

  if (rank == 0) {
    size_t rankChunk = static_cast<size_t>(sizes.size()) * opt.iters;
    if (rankChunk == 0 || all.size() % rankChunk != 0) {
      std::fprintf(stderr, "Unexpected gathered sample count: %zu rankChunk=%zu\n", all.size(), rankChunk);
      std::exit(6);
    }
    std::printf("# bcast-multimem-test ranks=%d lsa_ranks=%d threads=%d num_blocks=%d warmup=%d iters=%d "
                "ticks_per_us_rank0=%.3f\n",
                nranks, lsaRanks, opt.threads, opt.numBlocks, opt.warmup, opt.iters, opt.ticksPerUs);
    std::printf("size_B,num_blocks,min_us,p50_us,p90_us,p99_us,max_us,inj_bw_GBps,delivered_bw_GBps\n");
    for (size_t sidx = 0; sidx < sizes.size(); sidx++) {
      std::vector<double> samples;
      samples.reserve(static_cast<size_t>(nranks) * opt.iters);
      for (size_t base = 0; base < all.size(); base += rankChunk) {
        size_t off = base + sidx * static_cast<size_t>(opt.iters);
        samples.insert(samples.end(), all.begin() + off, all.begin() + off + opt.iters);
      }
      std::sort(samples.begin(), samples.end());
      double p50 = percentile(samples, 0.50);
      double injBw = p50 > 0.0 ? (static_cast<double>(sizes[sidx]) / (1024.0 * 1024.0 * 1024.0)) / (p50 * 1.0e-6) : 0.0;
      double deliveredBw = injBw * static_cast<double>(lsaRanks);
      std::printf("%zu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                  sizes[sidx], opt.numBlocks,
                  samples.empty() ? 0.0 : samples.front(),
                  p50,
                  percentile(samples, 0.90),
                  percentile(samples, 0.99),
                  samples.empty() ? 0.0 : samples.back(),
                  injBw, deliveredBw);
    }
  }

  return 0;
}

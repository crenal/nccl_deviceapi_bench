#pragma once

#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "gin_scratch__types.h"

#ifndef NCCL_DEVICEAPI_TEST_SYMK_GLOBAL_TIMER_DEFINED
#define NCCL_DEVICEAPI_TEST_SYMK_GLOBAL_TIMER_DEFINED
__device__ __forceinline__ uint64_t ncclSymkReadGlobalTimer() {
  uint64_t timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}
#endif

__device__ __forceinline__ void ncclSymkRun_AllGather_OneshotRail_Timed(
    struct ncclSymkDevWorkArgs const* args, uint64_t* bodySamples, int sampleIdx) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler(args);
  ncclTeam rail = ncclTeamRail(handler.comm);
  ncclGin gin(handler.comm, (int)(blockIdx.x % handler.comm.ginContextCount));
  constexpr int chunkSize = ncclSymkAllGather_RailRing_ChunkSize;
  ncclGinSignal_t railSignals = handler.ginSyncHandle.railSignals + blockIdx.x * rail.nRanks;
  ncclBarrierSession<ncclCoopCta> bar(cta, ncclTeamTagWorld(), gin, blockIdx.x, /*multimem=*/true);
  int warpId = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  bar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::None);
  uint64_t bodyStart = 0;
  if (threadIdx.x == 0 && bodySamples != nullptr) bodyStart = ncclSymkReadGlobalTimer();

  handler.template forEachWorkNoFusion<uint8_t>([&] __device__(size_t nElts, size_t nAllElts, ncclSymPtr<uint8_t> input,
                                                               ncclSymPtr<uint8_t> output) {
    if (warpId == 0) {
      ncclCoopWarpSpan warps(0, 1, 0);
      int dgrank = ncclTeamRankToWorld(handler.comm, rail, rail.rank);
      size_t remainingElts = nElts;
      size_t offset = 0;
      while (remainingElts) {
        size_t chunkElts = min(remainingElts, size_t(chunkSize));
        for (int peer = 0; peer < rail.nRanks; peer++) {
          if (peer == rail.rank) continue;
          gin.put(rail, peer, output + dgrank * nAllElts + offset, input + offset, chunkElts,
                  ncclGin_SignalInc{railSignals + rail.rank}, ncclGin_None{}, warps);
        }
        offset += chunkElts;
        remainingElts -= chunkElts;
      }
      gin.flush(warps);
    } else {
      int dataPeer = warpId - 1;
      if (dataPeer < rail.nRanks) {
        ncclCoopWarpSpan warps(warpId, 1, warpId);
        int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
        size_t remainingElts = nElts;
        size_t offset = 0;
        if (dataPeer == rail.rank) {
          while (remainingElts) {
            size_t chunkElts = min(remainingElts, size_t(chunkSize));
            bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), input + offset,
                          output + dgrank * nAllElts + offset, chunkElts);
            offset += chunkElts;
            remainingElts -= chunkElts;
          }
        } else {
          uint64_t* localSignalPtr = gin.getSignalShadowPtr(railSignals + dataPeer);
          uint64_t localSignalValue = *localSignalPtr;
          while (remainingElts) {
            size_t chunkElts = min(remainingElts, size_t(chunkSize));
            gin.waitSignal(warps, railSignals + dataPeer, localSignalValue + 1, 32);
            bcastMultimem(handler, warps.num_threads(), warps.thread_rank(),
                          output + dgrank * nAllElts + offset, output + dgrank * nAllElts + offset, chunkElts);
            offset += chunkElts;
            remainingElts -= chunkElts;
            localSignalValue++;
          }
          if (lane == 0) {
            *localSignalPtr = localSignalValue;
          }
        }
      }
    }
  });

  bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::None);
  if (threadIdx.x == 0 && bodySamples != nullptr) {
    bodySamples[static_cast<size_t>(sampleIdx) * gridDim.x + blockIdx.x] = ncclSymkReadGlobalTimer() - bodyStart;
  }
}

__device__ __forceinline__ void ncclSymkRun_AllGather_OneshotRail(struct ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_OneshotRail_Timed(args, nullptr, 0);
}

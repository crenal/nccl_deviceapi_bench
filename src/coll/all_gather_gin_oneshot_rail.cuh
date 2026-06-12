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
  int nWarps = blockDim.x / WARP_SIZE;
  int sendWarpCount = rail.nRanks > 1 ? rail.nRanks - 1 : 0;
  int bcastWarp0 = sendWarpCount;
  int bcastWarpCount = nWarps - bcastWarp0;

  bar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::None);
  uint64_t bodyStart = 0;
  if (threadIdx.x == 0 && bodySamples != nullptr) bodyStart = ncclSymkReadGlobalTimer();

  handler.template forEachWorkNoFusion<uint8_t>([&] __device__(size_t nElts, size_t nAllElts, ncclSymPtr<uint8_t> input,
                                                               ncclSymPtr<uint8_t> output) {
    if (warpId < sendWarpCount) {
      int remoteIdx = warpId;
      int peer = remoteIdx >= rail.rank ? remoteIdx + 1 : remoteIdx;
      ncclCoopWarpSpan warps(warpId, 1, warpId);
      int dgrank = ncclTeamRankToWorld(handler.comm, rail, rail.rank);
      size_t remainingElts = nElts;
      size_t offset = 0;
      while (remainingElts) {
        size_t chunkElts = min(remainingElts, size_t(chunkSize));
        gin.put(rail, peer, output + dgrank * nAllElts + offset, input + offset, chunkElts,
                ncclGin_SignalInc{railSignals + rail.rank}, ncclGin_None{}, warps);
        offset += chunkElts;
        remainingElts -= chunkElts;
      }
      gin.flush(warps);
    } else if (bcastWarpCount > 0) {
      for (int dataPeer = 0; dataPeer < rail.nRanks; dataPeer++) {
        int groupWarp0 = bcastWarp0;
        int groupWarps = 0;
        if (rail.nRanks > 1 && bcastWarpCount >= rail.nRanks) {
          if (dataPeer == rail.rank) {
            groupWarp0 = bcastWarp0;
            groupWarps = 1;
          } else {
            int remoteOrdinal = dataPeer < rail.rank ? dataPeer : dataPeer - 1;
            int remoteWarp0 = bcastWarp0 + 1;
            int remoteWarpCount = bcastWarpCount - 1;
            int relStart = (remoteOrdinal * remoteWarpCount) / (rail.nRanks - 1);
            int relEnd = ((remoteOrdinal + 1) * remoteWarpCount) / (rail.nRanks - 1);
            groupWarp0 = remoteWarp0 + relStart;
            groupWarps = relEnd - relStart;
          }
        } else {
          int relStart = (dataPeer * bcastWarpCount) / rail.nRanks;
          int relEnd = ((dataPeer + 1) * bcastWarpCount) / rail.nRanks;
          groupWarp0 = bcastWarp0 + relStart;
          groupWarps = relEnd - relStart;
        }
        if (groupWarps == 0 || warpId < groupWarp0 || warpId >= groupWarp0 + groupWarps) continue;

        ncclCoopWarpSpan warps(groupWarp0, groupWarps, sendWarpCount + dataPeer);
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
        break;
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

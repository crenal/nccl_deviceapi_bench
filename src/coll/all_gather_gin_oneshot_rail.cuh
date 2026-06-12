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

enum NcclSymkAgOneshotBreakdownMetric {
  ncclSymkAgOneshotBreakdownInitBar = 0,
  ncclSymkAgOneshotBreakdownWorkTotal = 1,
  ncclSymkAgOneshotBreakdownSendPut = 2,
  ncclSymkAgOneshotBreakdownSendFlush = 3,
  ncclSymkAgOneshotBreakdownSelfBcast = 4,
  ncclSymkAgOneshotBreakdownRemoteWait = 5,
  ncclSymkAgOneshotBreakdownRemoteBcast = 6,
  ncclSymkAgOneshotBreakdownShadowUpdate = 7,
  ncclSymkAgOneshotBreakdownFinalBar = 8,
  ncclSymkAgOneshotBreakdownBodyTotal = 9,
  ncclSymkAgOneshotBreakdownMetricCount = 10,
};

__device__ __forceinline__ void ncclSymkRecordAgOneshotBreakdown(
    uint64_t* samples, int sampleIdx, int warpId, int metric, uint64_t value) {
  if (samples == nullptr || (threadIdx.x % WARP_SIZE) != 0) return;
  int nWarps = blockDim.x / WARP_SIZE;
  size_t sampleBase = static_cast<size_t>(sampleIdx) * gridDim.x + blockIdx.x;
  size_t warpBase = sampleBase * nWarps + warpId;
  samples[warpBase * ncclSymkAgOneshotBreakdownMetricCount + metric] = value;
}

__device__ __forceinline__ void ncclSymkRun_AllGather_OneshotRail_Timed(
    struct ncclSymkDevWorkArgs const* args, uint64_t* bodySamples, uint64_t* breakdownSamples, int sampleIdx) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler(args);
  ncclTeam rail = ncclTeamRail(handler.comm);
  ncclGin gin(handler.comm, (int)(blockIdx.x % handler.comm.ginContextCount));
  constexpr int chunkSize = ncclSymkAllGather_RailRing_ChunkSize;
  ncclGinSignal_t railSignals = handler.ginSyncHandle.railSignals + blockIdx.x * rail.nRanks;
  ncclBarrierSession<ncclCoopCta> worldBar(cta, ncclTeamTagWorld(), gin, blockIdx.x, /*multimem=*/true);
  ncclLsaBarrierSession<ncclCoopCta> finalLsaBar(cta, handler.comm, ncclTeamTagLsa(), blockIdx.x,
                                                 /*multimem=*/true);
  int warpId = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int nWarps = blockDim.x / WARP_SIZE;
  int sendWarpCount = rail.nRanks > 1 ? rail.nRanks - 1 : 0;
  int bcastWarp0 = sendWarpCount;
  int bcastWarpCount = nWarps - bcastWarp0;

  uint64_t initBarStart = 0;
  if (breakdownSamples != nullptr && lane == 0) initBarStart = ncclSymkReadGlobalTimer();
  worldBar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::None);
  if (breakdownSamples != nullptr && lane == 0) {
    ncclSymkRecordAgOneshotBreakdown(breakdownSamples, sampleIdx, warpId, ncclSymkAgOneshotBreakdownInitBar,
                                     ncclSymkReadGlobalTimer() - initBarStart);
  }

  uint64_t bodyStart = 0;
  if (threadIdx.x == 0 && bodySamples != nullptr) bodyStart = ncclSymkReadGlobalTimer();
  uint64_t bodyStartWarp = 0;
  if (breakdownSamples != nullptr && lane == 0) bodyStartWarp = ncclSymkReadGlobalTimer();

  uint64_t workStart = 0;
  uint64_t sendPutTicks = 0;
  uint64_t sendFlushTicks = 0;
  uint64_t selfBcastTicks = 0;
  uint64_t remoteWaitTicks = 0;
  uint64_t remoteBcastTicks = 0;
  uint64_t shadowUpdateTicks = 0;
  if (breakdownSamples != nullptr && lane == 0) workStart = ncclSymkReadGlobalTimer();

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
        uint64_t t0 = 0;
        if (breakdownSamples != nullptr && lane == 0) t0 = ncclSymkReadGlobalTimer();
        gin.put(rail, peer, output + dgrank * nAllElts + offset, input + offset, chunkElts,
                ncclGin_SignalInc{railSignals + rail.rank}, ncclGin_None{}, warps);
        if (breakdownSamples != nullptr && lane == 0) sendPutTicks += ncclSymkReadGlobalTimer() - t0;
        offset += chunkElts;
        remainingElts -= chunkElts;
      }
      uint64_t flushStart = 0;
      if (breakdownSamples != nullptr && lane == 0) flushStart = ncclSymkReadGlobalTimer();
      gin.flush(warps);
      if (breakdownSamples != nullptr && lane == 0) sendFlushTicks += ncclSymkReadGlobalTimer() - flushStart;
    } else if (bcastWarpCount > 0) {
      for (int dataPeer = 0; dataPeer < rail.nRanks; dataPeer++) {
        int relStart = (dataPeer * bcastWarpCount) / rail.nRanks;
        int relEnd = ((dataPeer + 1) * bcastWarpCount) / rail.nRanks;
        int groupWarp0 = bcastWarp0 + relStart;
        int groupWarps = relEnd - relStart;
        if (groupWarps == 0 || warpId < groupWarp0 || warpId >= groupWarp0 + groupWarps) continue;

        ncclCoopWarpSpan warps(groupWarp0, groupWarps, sendWarpCount + dataPeer);
        int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
        size_t remainingElts = nElts;
        size_t offset = 0;
        if (dataPeer == rail.rank) {
          while (remainingElts) {
            size_t chunkElts = min(remainingElts, size_t(chunkSize));
            uint64_t t0 = 0;
            if (breakdownSamples != nullptr && lane == 0) t0 = ncclSymkReadGlobalTimer();
            bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), input + offset,
                          output + dgrank * nAllElts + offset, chunkElts);
            if (breakdownSamples != nullptr && lane == 0) selfBcastTicks += ncclSymkReadGlobalTimer() - t0;
            offset += chunkElts;
            remainingElts -= chunkElts;
          }
        } else {
          uint64_t* localSignalPtr = gin.getSignalShadowPtr(railSignals + dataPeer);
          uint64_t localSignalValue = *localSignalPtr;
          while (remainingElts) {
            size_t chunkElts = min(remainingElts, size_t(chunkSize));
            uint64_t waitStart = 0;
            if (breakdownSamples != nullptr && lane == 0) waitStart = ncclSymkReadGlobalTimer();
            gin.waitSignal(warps, railSignals + dataPeer, localSignalValue + 1, 32);
            if (breakdownSamples != nullptr && lane == 0) remoteWaitTicks += ncclSymkReadGlobalTimer() - waitStart;
            uint64_t bcastStart = 0;
            if (breakdownSamples != nullptr && lane == 0) bcastStart = ncclSymkReadGlobalTimer();
            bcastMultimem(handler, warps.num_threads(), warps.thread_rank(),
                          output + dgrank * nAllElts + offset, output + dgrank * nAllElts + offset, chunkElts);
            if (breakdownSamples != nullptr && lane == 0) remoteBcastTicks += ncclSymkReadGlobalTimer() - bcastStart;
            offset += chunkElts;
            remainingElts -= chunkElts;
            localSignalValue++;
          }
          if (lane == 0) {
            uint64_t shadowStart = 0;
            if (breakdownSamples != nullptr) shadowStart = ncclSymkReadGlobalTimer();
            *localSignalPtr = localSignalValue;
            if (breakdownSamples != nullptr) shadowUpdateTicks += ncclSymkReadGlobalTimer() - shadowStart;
          }
        }
        break;
      }
    }
  });

  if (breakdownSamples != nullptr && lane == 0) {
    ncclSymkRecordAgOneshotBreakdown(breakdownSamples, sampleIdx, warpId, ncclSymkAgOneshotBreakdownWorkTotal,
                                     ncclSymkReadGlobalTimer() - workStart);
    ncclSymkRecordAgOneshotBreakdown(breakdownSamples, sampleIdx, warpId, ncclSymkAgOneshotBreakdownSendPut,
                                     sendPutTicks);
    ncclSymkRecordAgOneshotBreakdown(breakdownSamples, sampleIdx, warpId, ncclSymkAgOneshotBreakdownSendFlush,
                                     sendFlushTicks);
    ncclSymkRecordAgOneshotBreakdown(breakdownSamples, sampleIdx, warpId, ncclSymkAgOneshotBreakdownSelfBcast,
                                     selfBcastTicks);
    ncclSymkRecordAgOneshotBreakdown(breakdownSamples, sampleIdx, warpId, ncclSymkAgOneshotBreakdownRemoteWait,
                                     remoteWaitTicks);
    ncclSymkRecordAgOneshotBreakdown(breakdownSamples, sampleIdx, warpId, ncclSymkAgOneshotBreakdownRemoteBcast,
                                     remoteBcastTicks);
    ncclSymkRecordAgOneshotBreakdown(breakdownSamples, sampleIdx, warpId, ncclSymkAgOneshotBreakdownShadowUpdate,
                                     shadowUpdateTicks);
  }

  uint64_t finalBarStart = 0;
  if (breakdownSamples != nullptr && lane == 0) finalBarStart = ncclSymkReadGlobalTimer();
  finalLsaBar.sync(cta, cuda::memory_order_release);
  if (breakdownSamples != nullptr && lane == 0) {
    ncclSymkRecordAgOneshotBreakdown(breakdownSamples, sampleIdx, warpId, ncclSymkAgOneshotBreakdownFinalBar,
                                     ncclSymkReadGlobalTimer() - finalBarStart);
    ncclSymkRecordAgOneshotBreakdown(breakdownSamples, sampleIdx, warpId, ncclSymkAgOneshotBreakdownBodyTotal,
                                     ncclSymkReadGlobalTimer() - bodyStartWarp);
  }
  if (threadIdx.x == 0 && bodySamples != nullptr) {
    bodySamples[static_cast<size_t>(sampleIdx) * gridDim.x + blockIdx.x] = ncclSymkReadGlobalTimer() - bodyStart;
  }
}

__device__ __forceinline__ void ncclSymkRun_AllGather_OneshotRail(struct ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_OneshotRail_Timed(args, nullptr, nullptr, 0);
}

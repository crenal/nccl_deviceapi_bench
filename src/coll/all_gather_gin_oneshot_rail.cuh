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
  int warpId = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int nWarps = blockDim.x / WARP_SIZE;
  int sendWarpCount = rail.nRanks > 1 ? rail.nRanks - 1 : 0;
  bool sharedGinLsaSplit = rail.nRanks == 4 && (gridDim.x == 2 || gridDim.x == 4) && nWarps > sendWarpCount;
  ncclGin gin(handler.comm, sharedGinLsaSplit ? 0 : (int)(blockIdx.x % handler.comm.ginContextCount));
  constexpr int chunkSize = ncclSymkAllGather_RailRing_ChunkSize;
  ncclGinSignal_t railSignals =
      handler.ginSyncHandle.railSignals + (sharedGinLsaSplit ? 0 : blockIdx.x * rail.nRanks);
  ncclBarrierSession<ncclCoopCta> worldBar(cta, ncclTeamTagWorld(), gin, blockIdx.x, /*multimem=*/true);
  ncclLsaBarrierSession<ncclCoopCta> finalLsaBar(cta, handler.comm, ncclTeamTagLsa(), blockIdx.x,
                                                 /*multimem=*/true);
  int bcastWarp0 = sendWarpCount;
  int bcastWarpCount = nWarps - bcastWarp0;

  worldBar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::None);
  uint64_t bodyStart = 0;
  if (threadIdx.x == 0 && bodySamples != nullptr) bodyStart = ncclSymkReadGlobalTimer();

  handler.template forEachWorkNoFusion<uint8_t>([&] __device__(size_t nElts, size_t nAllElts, ncclSymPtr<uint8_t> input,
                                                               ncclSymPtr<uint8_t> output) {
    if (sharedGinLsaSplit) {
      struct ncclSymkDevWork const& dw = handler.devWork[0];
      nElts = dw.nElts;
      nAllElts = dw.nElts;
      input = ncclSymPtr<uint8_t>(dw.inputWin, dw.inputOff);
      output = ncclSymPtr<uint8_t>(dw.outputWin, dw.outputOff);

      if (blockIdx.x == 0 && warpId < sendWarpCount) {
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
      }

      int lsaWarp0 = blockIdx.x == 0 ? sendWarpCount : 0;
      int lsaWarpCount = nWarps - lsaWarp0;
      int dataPeerBegin = gridDim.x == 2 ? (int)blockIdx.x * 2 : (int)blockIdx.x;
      int dataPeerCount = gridDim.x == 2 ? 2 : 1;
      for (int localPeer = 0; localPeer < dataPeerCount; localPeer++) {
        int dataPeer = dataPeerBegin + localPeer;
        int relStart = (localPeer * lsaWarpCount) / dataPeerCount;
        int relEnd = ((localPeer + 1) * lsaWarpCount) / dataPeerCount;
        int groupWarp0 = lsaWarp0 + relStart;
        int groupWarps = relEnd - relStart;
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
      return;
    }

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

  finalLsaBar.sync(cta, cuda::memory_order_release);
  if (threadIdx.x == 0 && bodySamples != nullptr) {
    bodySamples[static_cast<size_t>(sampleIdx) * gridDim.x + blockIdx.x] = ncclSymkReadGlobalTimer() - bodyStart;
  }
}

__device__ __forceinline__ void ncclSymkRun_AllGather_OneshotRail(struct ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_OneshotRail_Timed(args, nullptr, 0);
}

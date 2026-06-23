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

__device__ __forceinline__ void ncclSymkAgOneshotRailPutSelfToPeer(ncclSymkArgsHandler& handler, ncclTeam rail,
                                                                    ncclGin& gin, ncclGinSignal_t railSignals,
                                                                    int peer, int spanTag, size_t nElts,
                                                                    size_t nAllElts, ncclSymPtr<uint8_t> input,
                                                                    ncclSymPtr<uint8_t> output,
                                                                    bool flushAfterPut) {
  constexpr int chunkSize = ncclSymkAllGather_RailRing_ChunkSize;
  ncclCoopWarpSpan warps(threadIdx.x / WARP_SIZE, 1, spanTag);
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
  if (flushAfterPut) {
    gin.flush(warps);
  }
}

__device__ __forceinline__ void ncclSymkAgOneshotRailPutSelfToPeerSplitSignals(
    ncclSymkArgsHandler& handler, ncclTeam rail, ncclGin& gin, ncclGinSignal_t railSignals, int peer,
    int blocksPerPeer, int spanTag, size_t nElts, size_t nAllElts, ncclSymPtr<uint8_t> input,
    ncclSymPtr<uint8_t> output, bool flushAfterPut) {
  constexpr int chunkSize = ncclSymkAllGather_RailRing_ChunkSize;
  ncclCoopWarpSpan warps(threadIdx.x / WARP_SIZE, 1, spanTag);
  int dgrank = ncclTeamRankToWorld(handler.comm, rail, rail.rank);
  size_t totalChunks = (nElts + size_t(chunkSize) - 1) / size_t(chunkSize);
  for (int dstBlockInPeer = 0; dstBlockInPeer < blocksPerPeer; dstBlockInPeer++) {
    size_t chunkBegin = (totalChunks * static_cast<size_t>(dstBlockInPeer)) / static_cast<size_t>(blocksPerPeer);
    size_t chunkEnd = (totalChunks * static_cast<size_t>(dstBlockInPeer + 1)) / static_cast<size_t>(blocksPerPeer);
    ncclGinSignal_t blockRailSignals = railSignals + dstBlockInPeer * rail.nRanks;
    for (size_t chunk = chunkBegin; chunk < chunkEnd; chunk++) {
      size_t offset = chunk * size_t(chunkSize);
      size_t chunkElts = min(nElts - offset, size_t(chunkSize));
      gin.put(rail, peer, output + dgrank * nAllElts + offset, input + offset, chunkElts,
              ncclGin_SignalInc{blockRailSignals + rail.rank}, ncclGin_None{}, warps);
    }
  }
  if (flushAfterPut) {
    gin.flush(warps);
  }
}

__device__ __forceinline__ void ncclSymkAgOneshotRailBcastDataPeer(ncclSymkArgsHandler& handler, ncclTeam rail,
                                                                    ncclGin& gin, ncclGinSignal_t railSignals,
                                                                    int dataPeer, int groupWarp0, int groupWarps,
                                                                    int spanTag, size_t nElts, size_t nAllElts,
                                                                    ncclSymPtr<uint8_t> input,
                                                                    ncclSymPtr<uint8_t> output) {
  constexpr int chunkSize = ncclSymkAllGather_RailRing_ChunkSize;
  int lane = threadIdx.x % WARP_SIZE;
  ncclCoopWarpSpan warps(groupWarp0, groupWarps, spanTag);
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
      bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), output + dgrank * nAllElts + offset,
                    output + dgrank * nAllElts + offset, chunkElts);
      offset += chunkElts;
      remainingElts -= chunkElts;
      localSignalValue++;
    }
    if (lane == 0) {
      *localSignalPtr = localSignalValue;
    }
  }
}

__device__ __forceinline__ void ncclSymkAgOneshotRailBcastDataPeerChunkRange(
    ncclSymkArgsHandler& handler, ncclTeam rail, ncclGin& gin, ncclGinSignal_t railSignals, int dataPeer,
    int blockInPeer, int blocksPerPeer, int groupWarp0, int groupWarps, int spanTag, size_t nElts, size_t nAllElts,
    ncclSymPtr<uint8_t> input, ncclSymPtr<uint8_t> output) {
  if (groupWarps <= 0) return;
  constexpr int chunkSize = ncclSymkAllGather_RailRing_ChunkSize;
  ncclCoopWarpSpan warps(groupWarp0, groupWarps, spanTag);
  int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
  size_t totalChunks = (nElts + size_t(chunkSize) - 1) / size_t(chunkSize);
  size_t chunkBegin = (totalChunks * static_cast<size_t>(blockInPeer)) / static_cast<size_t>(blocksPerPeer);
  size_t chunkEnd = (totalChunks * static_cast<size_t>(blockInPeer + 1)) / static_cast<size_t>(blocksPerPeer);

  if (dataPeer == rail.rank) {
    for (size_t chunk = chunkBegin; chunk < chunkEnd; chunk++) {
      size_t offset = chunk * size_t(chunkSize);
      size_t chunkElts = min(nElts - offset, size_t(chunkSize));
      bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), input + offset,
                    output + dgrank * nAllElts + offset, chunkElts);
    }
  } else {
    ncclGinSignal_t blockRailSignals = railSignals + blockInPeer * rail.nRanks;
    uint64_t* localSignalPtr = gin.getSignalShadowPtr(blockRailSignals + dataPeer);
    uint64_t localSignalValue = *localSignalPtr;
    for (size_t chunk = chunkBegin; chunk < chunkEnd; chunk++) {
      size_t offset = chunk * size_t(chunkSize);
      size_t chunkElts = min(nElts - offset, size_t(chunkSize));
      size_t localChunk = chunk - chunkBegin;
      gin.waitSignal(warps, blockRailSignals + dataPeer, localSignalValue + localChunk + 1, 32);
      bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), output + dgrank * nAllElts + offset,
                    output + dgrank * nAllElts + offset, chunkElts);
    }
    if (threadIdx.x == groupWarp0 * WARP_SIZE) {
      *localSignalPtr = localSignalValue + chunkEnd - chunkBegin;
    }
  }
}

__device__ __forceinline__ void ncclSymkRun_AllGather_OneshotRail_SingleBlock(ncclSymkArgsHandler& handler,
                                                                              ncclTeam rail, ncclGin& gin,
                                                                              ncclGinSignal_t railSignals,
                                                                              bool flushAfterPut) {
  int warpId = threadIdx.x / WARP_SIZE;
  int nWarps = blockDim.x / WARP_SIZE;
  int sendWarpCount = rail.nRanks > 1 ? rail.nRanks - 1 : 0;
  int bcastWarp0 = 0;
  int bcastWarpCount = nWarps;

  handler.template forEachWorkNoFusion<uint8_t>([&] __device__(size_t nElts, size_t nAllElts, ncclSymPtr<uint8_t> input,
                                                               ncclSymPtr<uint8_t> output) {
    if (warpId < sendWarpCount) {
      int remoteIdx = warpId;
      int peer = remoteIdx >= rail.rank ? remoteIdx + 1 : remoteIdx;
      ncclSymkAgOneshotRailPutSelfToPeer(handler, rail, gin, railSignals, peer, warpId, nElts, nAllElts, input,
                                         output, flushAfterPut);
    }
    if (bcastWarpCount > 0) {
      for (int dataPeer = 0; dataPeer < rail.nRanks; dataPeer++) {
        int relStart = (dataPeer * bcastWarpCount) / rail.nRanks;
        int relEnd = ((dataPeer + 1) * bcastWarpCount) / rail.nRanks;
        int groupWarp0 = bcastWarp0 + relStart;
        int groupWarps = relEnd - relStart;
        if (groupWarps == 0 || warpId < groupWarp0 || warpId >= groupWarp0 + groupWarps) continue;

        ncclSymkAgOneshotRailBcastDataPeer(handler, rail, gin, railSignals, dataPeer, groupWarp0, groupWarps,
                                           sendWarpCount + dataPeer, nElts, nAllElts, input, output);
        break;
      }
    }
  });
}

__device__ __forceinline__ void ncclSymkRun_AllGather_OneshotRail_SharedSplit(ncclSymkArgsHandler& handler,
                                                                              ncclTeam rail, ncclGin& gin,
                                                                              ncclGinSignal_t railSignals,
                                                                              bool flushAfterPut) {
  int warpId = threadIdx.x / WARP_SIZE;
  int nWarps = blockDim.x / WARP_SIZE;
  int blocksPerPeer = gridDim.x / rail.nRanks;
  int dataPeer = static_cast<int>(blockIdx.x) / blocksPerPeer;
  int blockInPeer = static_cast<int>(blockIdx.x) - dataPeer * blocksPerPeer;
  bool sendBlock = dataPeer != rail.rank && blockInPeer == 0;

  handler.template forEachWorkNoFusion<uint8_t>(
      [&] __device__(size_t, size_t, ncclSymPtr<uint8_t>, ncclSymPtr<uint8_t>) {
        struct ncclSymkDevWork const& dw = handler.devWork[0];
        size_t nElts = dw.nElts;
        size_t nAllElts = dw.nElts;
        ncclSymPtr<uint8_t> input(dw.inputWin, dw.inputOff);
        ncclSymPtr<uint8_t> output(dw.outputWin, dw.outputOff);

        if (sendBlock && warpId == 0) {
          ncclSymkAgOneshotRailPutSelfToPeerSplitSignals(handler, rail, gin, railSignals, dataPeer, blocksPerPeer,
                                                         /*spanTag=*/0, nElts, nAllElts, input, output,
                                                         flushAfterPut);
        }

        int groupWarp0 = 0;
        int groupWarps = nWarps - groupWarp0;
        if (warpId < groupWarp0 || warpId >= groupWarp0 + groupWarps) return;
        ncclSymkAgOneshotRailBcastDataPeerChunkRange(handler, rail, gin, railSignals, dataPeer, blockInPeer,
                                                     blocksPerPeer, groupWarp0, groupWarps, /*spanTag=*/1,
                                                     nElts, nAllElts, input, output);
      });
}

__device__ __forceinline__ bool ncclSymkUseAllGatherOneshotRailSharedSplit(ncclTeam rail) {
  int nWarps = blockDim.x / WARP_SIZE;
  return rail.nRanks == 4 && gridDim.x >= rail.nRanks && (gridDim.x % rail.nRanks) == 0 && nWarps > 1;
}

__device__ __forceinline__ void ncclSymkRun_AllGather_OneshotRail_Timed(
    struct ncclSymkDevWorkArgs const* args, uint64_t* bodySamples, int sampleIdx, bool flushAfterPut) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler(args);
  ncclTeam rail = ncclTeamRail(handler.comm);
  bool sharedSplit = ncclSymkUseAllGatherOneshotRailSharedSplit(rail);
  int ginContext = sharedSplit ? 0 : (int)(blockIdx.x % handler.comm.ginContextCount);
  ncclGin gin(handler.comm, ginContext);
  ncclGinSignal_t railSignals = handler.ginSyncHandle.railSignals + (sharedSplit ? 0 : blockIdx.x * rail.nRanks);
  ncclBarrierSession<ncclCoopCta> worldBar(cta, ncclTeamTagWorld(), gin, blockIdx.x, /*multimem=*/true);
  ncclLsaBarrierSession<ncclCoopCta> finalLsaBar(cta, handler.comm, ncclTeamTagLsa(), blockIdx.x,
                                                 /*multimem=*/true);

  worldBar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::None);
  uint64_t bodyStart = 0;
  if (threadIdx.x == 0 && bodySamples != nullptr) bodyStart = ncclSymkReadGlobalTimer();

  if (sharedSplit) {
    ncclSymkRun_AllGather_OneshotRail_SharedSplit(handler, rail, gin, railSignals, flushAfterPut);
  } else {
    ncclSymkRun_AllGather_OneshotRail_SingleBlock(handler, rail, gin, railSignals, flushAfterPut);
  }

  finalLsaBar.sync(cta, cuda::memory_order_release);
  if (threadIdx.x == 0 && bodySamples != nullptr) {
    bodySamples[static_cast<size_t>(sampleIdx) * gridDim.x + blockIdx.x] = ncclSymkReadGlobalTimer() - bodyStart;
  }
}

__device__ __forceinline__ void ncclSymkRun_AllGather_OneshotRail(struct ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_OneshotRail_Timed(args, nullptr, 0, /*flushAfterPut=*/false);
}

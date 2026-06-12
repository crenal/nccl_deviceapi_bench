#pragma once

#include "common/checks.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace nccl_deviceapi_test {

__device__ __forceinline__ uint64_t global_timer() {
  uint64_t timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}

static __global__ void calibrate_timer_kernel(uint64_t spin_ticks, uint64_t* delta_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint64_t t0 = global_timer();
    uint64_t now = t0;
    while (now - t0 < spin_ticks) {
      now = global_timer();
    }
    *delta_out = now - t0;
  }
}

inline double calibrate_ticks_per_us(uint64_t spin_ticks = 50000000ULL) {
  uint64_t* d_delta = nullptr;
  uint64_t h_delta = 0;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaMalloc(&d_delta, sizeof(uint64_t)));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  calibrate_timer_kernel<<<1, 1>>>(spin_ticks, d_delta);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(&h_delta, d_delta, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_delta));
  if (elapsed_ms <= 0.0f || h_delta == 0) {
    throw std::runtime_error("failed to calibrate GPU globaltimer");
  }
  return static_cast<double>(h_delta) / (static_cast<double>(elapsed_ms) * 1000.0);
}

}  // namespace nccl_deviceapi_test

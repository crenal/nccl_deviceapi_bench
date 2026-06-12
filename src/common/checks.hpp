#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nccl_deviceapi_test {

inline std::runtime_error make_error(const char* kind, const char* file, int line, const std::string& msg) {
  std::ostringstream oss;
  oss << kind << " error at " << file << ":" << line << ": " << msg;
  return std::runtime_error(oss.str());
}

inline void check_cuda(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw make_error("CUDA", file, line, cudaGetErrorString(err));
  }
}

inline void check_nccl(ncclResult_t err, const char* file, int line) {
  if (err != ncclSuccess) {
    throw make_error("NCCL", file, line, ncclGetErrorString(err));
  }
}

inline void fail_fast(const std::exception& e) {
  std::cerr << "error: " << e.what() << "\n";
}

}  // namespace nccl_deviceapi_test

#define CUDA_CHECK(stmt) ::nccl_deviceapi_test::check_cuda((stmt), __FILE__, __LINE__)
#define NCCL_CHECK(stmt) ::nccl_deviceapi_test::check_nccl((stmt), __FILE__, __LINE__)


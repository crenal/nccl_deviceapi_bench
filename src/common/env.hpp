#pragma once

#include <cstdlib>
#include <string>

namespace nccl_deviceapi_test {

inline int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  return value == nullptr ? fallback : std::atoi(value);
}

inline std::string env_string(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string(fallback) : std::string(value);
}

}  // namespace nccl_deviceapi_test


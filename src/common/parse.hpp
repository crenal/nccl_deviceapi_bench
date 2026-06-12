#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace nccl_deviceapi_test {

inline size_t parse_size(const std::string& text) {
  if (text.empty()) {
    throw std::invalid_argument("empty size");
  }
  size_t pos = 0;
  double value = std::stod(text, &pos);
  std::string suffix = text.substr(pos);
  std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  double scale = 1.0;
  if (suffix.empty() || suffix == "b") {
    scale = 1.0;
  } else if (suffix == "k" || suffix == "kb" || suffix == "kib") {
    scale = 1024.0;
  } else if (suffix == "m" || suffix == "mb" || suffix == "mib") {
    scale = 1024.0 * 1024.0;
  } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
    scale = 1024.0 * 1024.0 * 1024.0;
  } else {
    throw std::invalid_argument("unknown size suffix: " + suffix);
  }
  return static_cast<size_t>(value * scale);
}

inline size_t parse_size(const char* text) {
  return parse_size(std::string(text));
}

inline int parse_int(const std::string& text, const char* flag) {
  char* end = nullptr;
  long value = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0' || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid integer for ") + flag + ": " + text);
  }
  return static_cast<int>(value);
}

inline int parse_int(const char* text, const char* flag) {
  return parse_int(std::string(text), flag);
}

}  // namespace nccl_deviceapi_test


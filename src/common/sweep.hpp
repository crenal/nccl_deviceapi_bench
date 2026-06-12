#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace nccl_deviceapi_test {

inline std::vector<size_t> make_sizes(size_t min_bytes, size_t max_bytes, int factor) {
  std::vector<size_t> sizes;
  for (size_t s = min_bytes; s <= max_bytes;) {
    sizes.push_back(s);
    if (s > max_bytes / static_cast<size_t>(factor)) break;
    s *= static_cast<size_t>(factor);
  }
  return sizes;
}

inline std::vector<double> samples_for_size(const std::vector<double>& all_samples, size_t rank_chunk,
                                            size_t size_index, int iters, bool collective_stats) {
  std::vector<double> samples;
  if (collective_stats) {
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; i++) {
      double max_us = 0.0;
      for (size_t base = 0; base < all_samples.size(); base += rank_chunk) {
        size_t off = base + size_index * static_cast<size_t>(iters) + static_cast<size_t>(i);
        max_us = std::max(max_us, all_samples[off]);
      }
      samples.push_back(max_us);
    }
  } else {
    size_t nranks = rank_chunk == 0 ? 0 : all_samples.size() / rank_chunk;
    samples.reserve(nranks * static_cast<size_t>(iters));
    for (size_t base = 0; base < all_samples.size(); base += rank_chunk) {
      size_t off = base + size_index * static_cast<size_t>(iters);
      samples.insert(samples.end(), all_samples.begin() + off, all_samples.begin() + off + iters);
    }
  }
  return samples;
}

inline double gib_per_second(size_t bytes, double us) {
  if (us <= 0.0) return 0.0;
  return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / (us * 1.0e-6);
}

}  // namespace nccl_deviceapi_test

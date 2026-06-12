#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace nccl_deviceapi_test {

struct MetricStats {
  double min = 0.0;
  double max = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double avg = 0.0;
};

inline double percentile_linear(const std::vector<double>& sorted_values, double q) {
  if (sorted_values.empty()) return 0.0;
  size_t idx = static_cast<size_t>((sorted_values.size() - 1) * q + 0.5);
  return sorted_values[std::min(idx, sorted_values.size() - 1)];
}

inline double percentile_nearest_percent(const std::vector<double>& sorted_values, double percentile) {
  if (sorted_values.empty()) return 0.0;
  double rank = std::ceil((percentile / 100.0) * sorted_values.size());
  size_t idx = static_cast<size_t>(std::max(1.0, rank)) - 1;
  return sorted_values[std::min(idx, sorted_values.size() - 1)];
}

inline MetricStats compute_stats(std::vector<double> values) {
  MetricStats stats;
  if (values.empty()) return stats;
  std::sort(values.begin(), values.end());
  stats.min = values.front();
  stats.max = values.back();
  stats.avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  stats.p50 = percentile_linear(values, 0.50);
  stats.p90 = percentile_linear(values, 0.90);
  stats.p95 = percentile_linear(values, 0.95);
  stats.p99 = percentile_linear(values, 0.99);
  return stats;
}

inline std::vector<double> ticks_to_us(const std::vector<uint64_t>& ticks, double ticks_per_us) {
  std::vector<double> values;
  values.reserve(ticks.size());
  for (uint64_t tick : ticks) {
    values.push_back(static_cast<double>(tick) / ticks_per_us);
  }
  return values;
}

}  // namespace nccl_deviceapi_test


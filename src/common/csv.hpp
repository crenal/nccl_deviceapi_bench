#pragma once

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>

namespace nccl_deviceapi_test {

class CsvFile {
 public:
  explicit CsvFile(const std::string& path, int precision = 10) : out_(path) {
    if (!out_) {
      throw std::runtime_error("failed to open csv output: " + path);
    }
    out_ << std::setprecision(precision);
  }

  std::ofstream& stream() { return out_; }

 private:
  std::ofstream out_;
};

template <typename T>
inline void write_csv_value(std::ostream& out, const T& value) {
  out << value;
}

template <typename... Values>
inline void write_csv_row(std::ostream& out, Values&&... values) {
  bool first = true;
  ((out << (std::exchange(first, false) ? "" : ","), write_csv_value(out, values)), ...);
  out << '\n';
}

inline std::string rank_csv_path(const std::string& path, int rank, int nranks) {
  if (nranks <= 1) {
    return path;
  }
  std::string suffix = ".rank" + std::to_string(rank) + ".csv";
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".csv") {
    return path.substr(0, path.size() - 4) + suffix;
  }
  return path + suffix;
}

}  // namespace nccl_deviceapi_test

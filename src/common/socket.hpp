#pragma once

#include "common/checks.hpp"

#include <nccl.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace nccl_deviceapi_test {

inline bool send_all(int fd, const void* data, size_t bytes) {
  const char* p = static_cast<const char*>(data);
  while (bytes > 0) {
    ssize_t n = ::send(fd, p, bytes, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
}

inline bool recv_all(int fd, void* data, size_t bytes) {
  char* p = static_cast<char*>(data);
  while (bytes > 0) {
    ssize_t n = ::recv(fd, p, bytes, MSG_WAITALL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
}

inline int listen_socket(int port) {
  addrinfo hints = {};
  hints.ai_family = AF_INET6;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  addrinfo* res = nullptr;
  std::string port_str = std::to_string(port);
  int rc = ::getaddrinfo(nullptr, port_str.c_str(), &hints, &res);
  if (rc != 0) {
    std::fprintf(stderr, "getaddrinfo listen failed: %s\n", gai_strerror(rc));
    std::exit(4);
  }

  int fd = -1;
  for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
    fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) continue;
    int one = 1;
    int zero = 0;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    ::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof(zero));
    if (::bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && ::listen(fd, 256) == 0) break;
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(res);

  if (fd < 0) {
    std::fprintf(stderr, "Could not listen on port %d: %s\n", port, std::strerror(errno));
    std::exit(4);
  }
  return fd;
}

inline int connect_socket(const std::string& host, int port, int retries = 500) {
  addrinfo hints = {};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  std::string port_str = std::to_string(port);

  for (int attempt = 0; attempt < retries; attempt++) {
    addrinfo* res = nullptr;
    int rc = ::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
    if (rc == 0) {
      for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
        int fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
          ::freeaddrinfo(res);
          return fd;
        }
        ::close(fd);
      }
    }
    if (res != nullptr) ::freeaddrinfo(res);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  std::fprintf(stderr, "Could not connect to %s:%d\n", host.c_str(), port);
  std::exit(4);
}

inline ncclUniqueId exchange_unique_id_socket(int rank, int nranks, const std::string& master_addr, int port) {
  ncclUniqueId id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
    int listen_fd = listen_socket(port);
    for (int i = 1; i < nranks; i++) {
      int fd = ::accept(listen_fd, nullptr, nullptr);
      if (fd < 0) {
        std::fprintf(stderr, "accept unique id failed: %s\n", std::strerror(errno));
        std::exit(4);
      }
      int peer_rank = -1;
      recv_all(fd, &peer_rank, sizeof(peer_rank));
      if (!send_all(fd, &id, sizeof(id))) {
        std::fprintf(stderr, "send unique id failed\n");
        std::exit(4);
      }
      ::close(fd);
    }
    ::close(listen_fd);
  } else {
    int fd = connect_socket(master_addr, port);
    send_all(fd, &rank, sizeof(rank));
    if (!recv_all(fd, &id, sizeof(id))) {
      std::fprintf(stderr, "recv unique id failed\n");
      std::exit(4);
    }
    ::close(fd);
  }
  return id;
}

inline std::vector<double> gather_samples_socket(int rank, int nranks, const std::string& master_addr, int port,
                                                 const std::vector<double>& local) {
  if (rank == 0) {
    std::vector<double> all = local;
    int listen_fd = listen_socket(port);
    for (int i = 1; i < nranks; i++) {
      int fd = ::accept(listen_fd, nullptr, nullptr);
      if (fd < 0) {
        std::fprintf(stderr, "accept samples failed: %s\n", std::strerror(errno));
        std::exit(4);
      }
      int peer_rank = -1;
      uint64_t count = 0;
      recv_all(fd, &peer_rank, sizeof(peer_rank));
      recv_all(fd, &count, sizeof(count));
      std::vector<double> peer(count);
      if (count != 0 && !recv_all(fd, peer.data(), count * sizeof(double))) {
        std::fprintf(stderr, "recv samples failed from rank %d\n", peer_rank);
        std::exit(4);
      }
      all.insert(all.end(), peer.begin(), peer.end());
      ::close(fd);
    }
    ::close(listen_fd);
    return all;
  }

  int fd = connect_socket(master_addr, port);
  uint64_t count = local.size();
  send_all(fd, &rank, sizeof(rank));
  send_all(fd, &count, sizeof(count));
  if (count != 0) send_all(fd, local.data(), count * sizeof(double));
  ::close(fd);
  return {};
}

}  // namespace nccl_deviceapi_test

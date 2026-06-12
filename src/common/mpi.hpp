#pragma once

#include "common/checks.hpp"

#include <mpi.h>
#include <nccl.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace nccl_deviceapi_test {

inline void check_mpi(int err, const char* file, int line) {
  if (err != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(err, msg, &len);
    throw make_error("MPI", file, line, std::string(msg, len));
  }
}

inline int mpi_local_rank(MPI_Comm world) {
  MPI_Comm local = MPI_COMM_NULL;
  check_mpi(MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local), __FILE__, __LINE__);
  int rank = 0;
  check_mpi(MPI_Comm_rank(local, &rank), __FILE__, __LINE__);
  check_mpi(MPI_Comm_free(&local), __FILE__, __LINE__);
  return rank;
}

inline ncclUniqueId mpi_bcast_nccl_unique_id(int world_rank, MPI_Comm world) {
  ncclUniqueId id{};
  if (world_rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }
  check_mpi(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, world), __FILE__, __LINE__);
  return id;
}

}  // namespace nccl_deviceapi_test

#define MPI_CHECK(stmt) ::nccl_deviceapi_test::check_mpi((stmt), __FILE__, __LINE__)


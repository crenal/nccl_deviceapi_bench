cmake_minimum_required(VERSION 3.24)

project(nccl_deviceapi_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)

set(NCCL_HOME "" CACHE PATH "NCCL install/build root containing include/ and lib/")
set(NCCL_SOURCE_DIR "" CACHE PATH "NCCL source root, required for symmetric private headers")

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h nccl_device.h
  HINTS ${NCCL_HOME}
  PATH_SUFFIXES include build/include)

find_library(NCCL_LIBRARY
  NAMES nccl
  HINTS ${NCCL_HOME}
  PATH_SUFFIXES lib lib64 build/lib)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "Could not find NCCL. Set -DNCCL_HOME=/path/to/nccl/install-or-build")
endif()

if(NOT NCCL_SOURCE_DIR)
  if(EXISTS "${NCCL_HOME}/src/include/sym_kernels.h")
    set(NCCL_SOURCE_DIR "${NCCL_HOME}" CACHE PATH "NCCL source root" FORCE)
  elseif(EXISTS "${NCCL_HOME}/../src/include/sym_kernels.h")
    get_filename_component(NCCL_SOURCE_DIR "${NCCL_HOME}/.." ABSOLUTE)
    set(NCCL_SOURCE_DIR "${NCCL_SOURCE_DIR}" CACHE PATH "NCCL source root" FORCE)
  endif()
endif()

if(NOT EXISTS "${NCCL_SOURCE_DIR}/src/include/sym_kernels.h")
  message(FATAL_ERROR
    "NCCL_SOURCE_DIR must point to an NCCL source tree containing src/include/sym_kernels.h. "
    "This is needed by barrier-test and bcast-multimem-test.")
endif()

set(NCCL_PRIVATE_INCLUDE_DIRS
  "${NCCL_SOURCE_DIR}/src/include"
  "${NCCL_SOURCE_DIR}/src/include/plugin"
  "${NCCL_SOURCE_DIR}/src/device/symmetric"
  "${NCCL_SOURCE_DIR}/src")

set(COMMON_CUDA_OPTIONS
  --extended-lambda
  --expt-relaxed-constexpr
  -Xptxas=-maxrregcount=96)

function(configure_cuda_target target)
  target_include_directories(${target} PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${NCCL_INCLUDE_DIR}"
    ${NCCL_PRIVATE_INCLUDE_DIRS})
  target_compile_definitions(${target} PRIVATE NCCL_OS_LINUX=1)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_OPTIONS}>)
  target_link_libraries(${target} PRIVATE CUDA::cudart "${NCCL_LIBRARY}")
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
endfunction()

add_executable(nccl_gin_pingpong_perf
  src/nccl_gin_pingpong_perf.cu)
configure_cuda_target(nccl_gin_pingpong_perf)
target_link_libraries(nccl_gin_pingpong_perf PRIVATE MPI::MPI_CXX)

add_executable(waitsignal_perf
  src/waitsignal_perf.cu)
configure_cuda_target(waitsignal_perf)
target_link_libraries(waitsignal_perf PRIVATE MPI::MPI_CXX)

add_executable(barrier-test
  src/barrier_test.cu)
configure_cuda_target(barrier-test)

add_executable(bcast-multimem-test
  src/bcast_multimem_test.cu)
configure_cuda_target(bcast-multimem-test)

add_executable(bcast-multimem-8way-test
  src/bcast_multimem_test.cu)
configure_cuda_target(bcast-multimem-8way-test)
target_compile_definitions(bcast-multimem-8way-test PRIVATE BCAST_MM_DEFAULT_COLLECTIVE_STATS=1)

add_executable(allgather-gin-deviceapi-perf
  src/allgather_gin_deviceapi_perf.cu)
configure_cuda_target(allgather-gin-deviceapi-perf)

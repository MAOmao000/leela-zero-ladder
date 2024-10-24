# This file is part of Leela Zero.
# Copyright (C) 2017 Marco Calignano
# Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
# Leela Zero is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# Leela Zero is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.5)
cmake_policy(VERSION 3.30)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(leelaz)

# Required Packages
set(Boost_MIN_VERSION "1.58.0")
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.58.0 REQUIRED program_options filesystem)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
if(USE_BLAS OR USE_BACKEND STREQUAL "BLAS")
  # We need OpenBLAS for now, because we make some specific calls.
  # Ideally we'd use OpenBLAS is possible and fall back to
  # not doing those calls if it's not present.
  if(NOT APPLE)
    set(BLA_VENDOR OpenBLAS)
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# See if we can set optimization flags as expected.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(GccSpecificFlags 1)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set(GccSpecificFlags 1)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(GccSpecificFlags 1)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(GccSpecificFlags 0)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(GccSpecificFlags 0)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

if(GccSpecificFlags)
  set(GCC_COMPILE_FLAGS "-Wall -Wextra -ffast-math -flto -march=native")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(GCC_DISABLED_WARNING_COMPILE_FLAGS "-Wno-ignored-attributes -Wno-uninitialized \
        -Wno-mismatched-tags -Wno-deprecated-copy-with-user-provided-copy -Wno-inconsistent-missing-override")
  else()
    set(GCC_DISABLED_WARNING_COMPILE_FLAGS "-Wno-ignored-attributes -Wno-maybe-uninitialized \
        -Wno-mismatched-tags -Wno-deprecated-copy")
  endif()
  set(GCC_FLAGS "${GCC_COMPILE_FLAGS} ${GCC_DISABLED_WARNING_COMPILE_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${GCC_FLAGS} -g -Og")
  set(CMAKE_CXX_FLAGS_RELEASE "${GCC_FLAGS} -g -O3 -DNDEBUG")
  set(CMAKE_EXE_LINKER_FLAGS "-flto -g")
endif(GccSpecificFlags)

if(USE_HALF)
  add_definitions(-DUSE_HALF)
endif()

set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/src/Eigen")
set(SrcPath "${CMAKE_CURRENT_SOURCE_DIR}/src")

include_directories(${IncludePath})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${ZLIB_INCLUDE_DIRS})

set(USE_BACKEND CACHE STRING "EIGEN")
if(USE_BACKEND STREQUAL "TENSORRT" OR USE_BACKEND STREQUAL "CUDNN")
  if(USE_BACKEND STREQUAL "TENSORRT")
    message(STATUS "-DUSE_BACKEND=TENSORRT, using TENSORRT backend.")
    add_definitions(-DUSE_TENSOR_RT)
  else()
    message(STATUS "-DUSE_BACKEND=CUDNN, using CUDNN backend.")
  add_definitions(-DUSE_CUDNN)
  endif()
  enable_language(CUDA)
  set(CUDA_STANDARD 17)
  set(NEURALNET_BACKEND_SOURCES
    ${SrcPath}/SEnet.cu
  )
  enable_language(CUDA)
  set(CUDA_STANDARD 17)
  # https://en.wikipedia.org/wiki/CUDA#GPUs_supported
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 89 90)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86 87 89 90)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.5)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86 87)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2)
    set(CMAKE_CUDA_ARCHITECTURES 30 35 37 50 52 53 60 61 62 70 72 75)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  else()
    message(WARNING "CUDA 10.2 or greater is recommended, but attempting to build anyways")
    set(CMAKE_CUDA_ARCHITECTURES 30 37 53 70)
  endif()

elseif(USE_BACKEND STREQUAL "OPENCL")
  message(STATUS "-DUSE_BACKEND=OPENCL, using OPENCL backend.")

elseif(USE_BACKEND STREQUAL "BLAS")
  message(STATUS "-DUSE_BACKEND=BLAS, using BLAS backend.")
  add_definitions(-DUSE_BLAS)

elseif(USE_BACKEND STREQUAL "EIGEN")
  message(STATUS "-DUSE_BACKEND=EIGEN, using EIGEN backend.")

else()
  message(FATAL_ERROR "Unrecognized backend: " ${USE_BACKEND})
endif()

#--------------------------- LEELA ZERO COMPILING AND LINKING --------------------------------------------------------------

set(leelaz_MAIN "${SrcPath}/Leela.cpp")
file(GLOB leelaz_SRC "${SrcPath}/*.cpp")
list(REMOVE_ITEM leelaz_SRC ${leelaz_MAIN})

# Reuse for leelaz and gtest
add_library(objs OBJECT ${leelaz_SRC})

add_executable(leelaz $<TARGET_OBJECTS:objs> ${NEURALNET_BACKEND_SOURCES} ${leelaz_MAIN})

target_compile_features(leelaz PUBLIC cxx_std_17 c_std_17)

if(USE_BACKEND STREQUAL "TENSORRT")
  message(STATUS "Looking for system TensorRT and CuDNN library.")
  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT_DIR} ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include)
  if((NOT TENSORRT_INCLUDE_DIR))
    message(FATAL_ERROR "NvInfer.h was NOT found, specify TENSORRT_INCLUDE_DIR to indicate where it is.")
  endif()
  find_library(TENSORRT_LIBRARY nvinfer HINTS ${TENSORRT_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib64)
  file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" tensorrt_version_header)
  string(REGEX MATCH "#define NV_TENSORRT_MAJOR ([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  set(TENSORRT_VERSION_MAJOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "#define NV_TENSORRT_MINOR ([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  set(TENSORRT_VERSION_MINOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "#define NV_TENSORRT_PATCH ([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  set(TENSORRT_VERSION_PATCH ${CMAKE_MATCH_1})
  set(TENSORRT_VERSION "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}.${TENSORRT_VERSION_PATCH}")
  if(TENSORRT_VERSION VERSION_LESS 8.6)
    message(FATAL_ERROR "TensorRT 8.6 or greater is required but ${TENSORRT_VERSION} was found.")
  endif()
  find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib64)
  include_directories(${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cudnn-frontend/include)
  target_link_libraries(leelaz CUDA::cudart_static ${TENSORRT_LIBRARY} ${CUDNN_LIBRARY})
  target_link_libraries(leelaz CUDA::cublas ${CUDNN_LIBRARY})
elseif(USE_BACKEND STREQUAL "CUDNN")
  message(STATUS "Looking for system CuDNN library.")
  find_package(CUDAToolkit REQUIRED)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include)
  if((NOT CUDNN_INCLUDE_DIR))
    message(FATAL_ERROR "cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is.")
  endif()
  find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib64)
  include_directories(${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cudnn-frontend/include)
  target_link_libraries(leelaz CUDA::cudart_static ${CUDNN_LIBRARY})
  target_link_libraries(leelaz CUDA::cublas ${CUDNN_LIBRARY})
endif()

if(USE_BACKEND STREQUAL "TENSORRT" OR USE_BACKEND STREQUAL "CUDNN" OR USE_BACKEND STREQUAL "OPENCL")
  find_package(OpenCL REQUIRED)
  include_directories(${OpenCL_INCLUDE_DIRS})
  target_link_libraries(leelaz ${OpenCL_LIBRARIES})
  if(USE_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
      find_path(BLAS_INCLUDE_DIRS openblas_config.h
        /usr/include
        /usr/local/include
        /usr/include/openblas
        /opt/OpenBLAS/include
        /usr/include/x86_64-linux-gnu
        $ENV{BLAS_HOME}/include)
      add_definitions(-DUSE_BLAS)
      if((UNIX AND NOT APPLE) OR WIN32)
        include_directories(${BLAS_INCLUDE_DIRS})
      endif()
      target_link_libraries(leelaz ${BLAS_LIBRARIES})
      if(DNNL_ROOT_DIR)
        add_definitions(-DUSE_DNNL)
        include_directories(${DNNL_ROOT_DIR}/include ${DNNL_ROOT_DIR}/build/include)
        target_link_libraries(leelaz ${DNNL_ROOT_DIR}/build/src/libdnnl.a)
      endif()
    endif()
  endif()
elseif(USE_BACKEND STREQUAL "BLAS")
  message(STATUS "Looking for system BLAS/OpenBLAS library.")
  find_package(BLAS REQUIRED)
  find_path(BLAS_INCLUDE_DIRS openblas_config.h
    /usr/include
    /usr/local/include
    /usr/include/openblas
    /opt/OpenBLAS/include
    /usr/include/x86_64-linux-gnu
    $ENV{BLAS_HOME}/include)
  add_definitions(-DUSE_CPU_ONLY)
  if((UNIX AND NOT APPLE) OR WIN32)
    include_directories(${BLAS_INCLUDE_DIRS})
  endif()
  target_link_libraries(leelaz ${BLAS_LIBRARIES})
  if(DNNL_ROOT_DIR)
    add_definitions(-DUSE_DNNL)
    include_directories(${DNNL_ROOT_DIR}/include ${DNNL_ROOT_DIR}/build/include)
    target_link_libraries(leelaz ${DNNL_ROOT_DIR}/build/src/libdnnl.a)
  endif()
else()
  message(STATUS "Using built-in matrix library.")
  add_definitions(-DUSE_CPU_ONLY)
endif()

if(APPLE)
  include_directories("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers")
endif()

target_link_libraries(leelaz ${Boost_LIBRARIES})
target_link_libraries(leelaz ${ZLIB_LIBRARIES})
target_link_libraries(leelaz ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(leelaz
    ${Boost_STACKTRACE_BASIC_LIBRARY}
    ${Boost_STACKTRACE_BACKTRACE_LIBRARY}
    ${Boost_STACKTRACE_ADDR2LINE_LIBRARY}
    ${Boost_STACKTRACE_NOOP_LIBRARY}
    pthread dl backtrace
)
install(TARGETS leelaz DESTINATION ${CMAKE_INSTALL_BINDIR})

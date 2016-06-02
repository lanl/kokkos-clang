project(IDEAS_CUDA)

cmake_minimum_required(VERSION 2.8.11)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(/usr/local/cuda/include)

cuda_add_library (ideas_runtime_cuda reduce.cu)

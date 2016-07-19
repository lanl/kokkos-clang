cmake_minimum_required(VERSION 3.0)

project(ideas_runtime)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -isystem /projects/opt/centos7/gcc/5.2.0/include/c++/5.2.0 -I/projects/opt/centos7/gcc/5.2.0/include/c++/5.2.0/x86_64-unknown-linux-gnu")

include_directories(include ${CUDA_INCLUDE_DIRS})

add_library (ideas_runtime runtime.cpp)

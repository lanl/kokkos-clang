project(IDEAS)

cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE DEBUG ON CACHE BOOL "Debug mode")

set(LLVM_TARGETS_TO_BUILD "X86;NVPTX" CACHE STRING "LLVM targets")
set(LLVM_ENABLE_CXX1Y ON CACHE BOOL "Enable C++ 14")
set(LLVM_ENABLE_EH ON CACHE BOOL "Enable exception handling")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "Enable RTTI")

set(CMAKE_CXX_FLAGS "-Wl,-fuse-ld=gold")

add_subdirectory(llvm)
add_subdirectory(runtime)

cmake_minimum_required(VERSION 3.19)

if(WIN32)
else()
    set(CMAKE_CXX_COMPILER "g++")
    set(CMAKE_C_COMPILER "gcc")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# cuda
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "72;75")
endif()

# add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# add cuda file to project executables from include/rwkv/cuda/rwkv.cu
project(rwkv_cuda CXX CUDA)

find_package(CUDAToolkit)

add_library(
    rwkv_cuda
    include/rwkv/cuda/rwkv.cu
)

# link cuda lcudart_static
target_link_libraries(rwkv_cuda PRIVATE CUDA::cudart_static)

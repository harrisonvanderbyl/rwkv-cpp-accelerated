cmake_minimum_required(VERSION 3.19)

# ensure CXX at least 11.3.0

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

# set variable for include directory of ../../include
set(RWKV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include CACHE PATH "Path to rwkv include directory")

# add parent directory + ../include to include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${RWKV_INCLUDE_DIR})

# add cuda file to project executables from include/rwkv/cuda/rwkv.cu
project(storygen CXX CUDA)

find_package(CUDAToolkit)

add_executable(storygen
    storygen.cpp
)

# link cuda lcudart_static
target_link_libraries(storygen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../build/librwkv_cuda.a CUDA::cudart_static)

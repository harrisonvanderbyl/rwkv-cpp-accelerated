cmake_minimum_required(VERSION 3.17)

# ensure CXX at least 11.3.0

if(WIN32)
else()
    set(CMAKE_CXX_COMPILER "g++")
    set(CMAKE_C_COMPILER "gcc")
endif()

set(CMAKE_CXX_STANDARD 17)

# set variable for include directory of ../../include
set(RWKV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include CACHE PATH "Path to rwkv include directory")


# add parent directory + ../include to include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${RWKV_INCLUDE_DIR})

# add cuda file to project executables from include/rwkv/cuda/rwkv.cu

add_executable(rwkv
    ../../include/rwkv/tokenizer/tokenizer.cpp
    chat.cpp
    ../../include/rwkv/cuda/rwkv.cu
)

# add ./samplers to include path
target_include_directories(rwkv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samplers)

# link cuda lcudart_static
target_link_libraries(rwkv PRIVATE CUDA::cudart_static)

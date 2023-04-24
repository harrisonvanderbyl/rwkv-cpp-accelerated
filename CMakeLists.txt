cmake_minimum_required(VERSION 3.17)

# ensure CXX at least 11.3.0

if(WIN32)
else()
    set(CMAKE_CXX_COMPILER "g++")
    set(CMAKE_C_COMPILER "gcc")
endif()

set(CMAKE_CXX_STANDARD 17)

project("rwkv.cuda" CXX CUDA)


find_package(CUDAToolkit)

add_executable(rwkv
    chat.cpp
    rwkv.cu

    rwkv.h
    util.h
)

# add ./samplers to include path
target_include_directories(rwkv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samplers)

# link cuda lcudart_static
target_link_libraries(rwkv PRIVATE CUDA::cudart_static)



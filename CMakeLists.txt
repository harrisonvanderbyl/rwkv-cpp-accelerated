cmake_minimum_required(VERSION 3.17)

# ensure CXX at least 11.3.0

project("rwkv.cuda" CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDAToolkit)

add_executable(rwkv
    chat.cpp
    rwkv.cu

    rwkv.h
    util.h
)

# link cuda lcudart_static
target_link_libraries(rwkv PRIVATE CUDA::cudart_static)



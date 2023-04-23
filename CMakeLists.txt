cmake_minimum_required(VERSION 3.17)

project("rwkv.cuda" CXX CUDA)

find_package(CUDAToolkit)

add_executable(rwkv
    chat.cpp
    rwkv.cu

    rwkv.h
    util.h
)


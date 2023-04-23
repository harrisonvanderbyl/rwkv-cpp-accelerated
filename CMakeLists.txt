cmake_minimum_required(VERSION 3.17)

find_package(CUDAToolkit)

project("rwkv.cuda" CXX CUDA)

add_subdirectory(examples/terminalchat)
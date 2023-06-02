#pragma once
#include <iostream>
#include <tuple>
#include "vuda_runtime.hpp"

void setx(
    float* a,
    double b,
    int emb,
    int embsplit,
    int embblock
){
    vuda::dim3 blocks = vuda::dim3(emb, 1, 1);
    const int threads = embsplit;
    const int stream_id = 0;
    vuda::launchKernel("setmemfloat.spv", "main", stream_id, blocks, threads, a, emb, b);
}

void setx(
    double* a,
    double b,
    int emb,
    int embsplit,
    int embblock
){
    vuda::dim3 blocks = vuda::dim3(emb, 1, 1);
    const int threads = embsplit ;
    const int stream_id = 0;
    vuda::launchKernel("setmemdouble.spv", "main", stream_id, blocks,threads, a, emb, b);
}

void setx(
    double* a,
    float* b,
    int emb,
    int embsplit,
    int embblock
){
    vuda::dim3 blocks = vuda::dim3(emb, 1, 1);
    const int threads = embsplit ;
    const int stream_id = 0;
    vuda::launchKernel("movememf2d.spv", "main", stream_id, blocks,threads, a, emb, b);
}

void setx(
    float* a,
    double* b,
    int emb,
    int embsplit,
    int embblock
){
    vuda::dim3 blocks = vuda::dim3(emb, 1, 1);
    const int threads = embsplit ;
    const int stream_id = 0;
    vuda::launchKernel("movememd2f.spv", "main", stream_id, blocks,threads, a, emb, b);
}
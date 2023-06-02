#pragma once
#include <iostream>
#include <tuple>
#include "vuda_runtime.hpp"

void sigmoidrelusquared(
    float* a,
    float* b,
    unsigned long long  emb,
    int embsplit,
    int embblock
){
    vuda::dim3 blocks = vuda::dim3(emb*4, 1, 1);
    const int stream_id = 0;
    vuda::launchKernel("sigmoid.spv", "main", stream_id, blocks, a, emb, b);
}

void blockout(
    double* a,
    float* b,
    float* c,
    unsigned long long emb,
    int embsplit,
    int embblock
){
    vuda::dim3 blocks = vuda::dim3(emb, 1, 1);
    const int stream_id = 0;
    vuda::launchKernel("blockout.spv", "main", stream_id, blocks, emb, a, b, c);
}
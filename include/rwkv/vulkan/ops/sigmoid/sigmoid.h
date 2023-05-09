#pragma once
#include <iostream>
#include <tuple>
#include "vuda_runtime.hpp"

void sigmoidrelusquared(
    float* a,
    float* b,
    int emb,
    int embsplit,
    int embblock
){
    const int blocks = (emb*4 + embsplit - 1);
    const int threads = embsplit / embblock;
    const int stream_id = 0;
    vuda::launchKernel("sigmoid.spv", "main", stream_id, blocks, threads, a, emb, b);
}

void blockout(
    double* a,
    float* b,
    float* c,
    int emb,
    int embsplit,
    int embblock
){
    const int blocks = (emb + embsplit - 1);
    const int threads = embsplit / embblock;
    const int stream_id = 0;
    vuda::launchKernel("blockout.spv", "main", stream_id, blocks, threads, emb, a, b, c);
}
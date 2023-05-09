
#include <iostream>
#include <tuple>
#include "vuda_runtime.hpp"
#include "rwkv/vulkan/ops/set/setx.h"

std::tuple<float *, float *> meanvar(int embsplit, int embblock, unsigned long long emb, double *a, float *buffer, unsigned long long tokenlength)
{
    float *acc = &buffer[0];
    setx(acc, 0.0, tokenlength*2, embsplit, embblock);
    const int blocks = (emb + embsplit - 1);
    const int threads = embsplit;

    const int stream_id = 0;
    const int embint = emb;
    vuda::launchKernel("sum.spv", "main", stream_id, blocks, threads, acc, a, embint, tokenlength);

    float *mean = &buffer[0];
    // cudaMemcpy(mean, acc, sizeof(float)*tokenlength, cudaMemcpyDeviceToDevice);
    // setx(acc, 0, tokenlength, embsplit, embblock);
    vuda::launchKernel("var.spv", "main", stream_id, blocks, threads, acc, a, mean, emb);
    float *var = &buffer[tokenlength];
    // cudaMemcpy(var, acc, sizeof(float)*tokenlength, cudaMemcpyDeviceToDevice);

    return std::make_tuple(mean, var);
}

void layernorm(int embsplit, int embblock, int n_emb, double *x, double *layernorms, int n_layers, int offset, int tokenlength, double *buffer1, float *ffnrbuffer){

    float* mean;
    float* variance;
    std::tie(mean, variance) = meanvar(embsplit, embblock, n_emb, x, ffnrbuffer, tokenlength);

    const int blocks = (n_emb + embsplit - 1);
    const int threads = embsplit;

    const int stream_id = 0;
    vuda::launchKernel("layernorm.spv", "main", stream_id, blocks, threads, n_emb, offset, tokenlength, x, layernorms, mean, buffer1);
    
    }
//     float *mean;
//     float *variance;
//     std::tie(mean, variance) = meanvar(n_emb, x, ffnrbuffer, tokenlength);
//     cuda_layernorm<<<(n_emb + embsplit - 1) / embsplit, embsplit / embblock>>>(n_emb, x, layernorms, 4 * (n_layers) + 2, mean, variance, buffer1, tokenlength);
    
#include <iostream>
#include <tuple>
#include "vuda_runtime.hpp"

void mixatt(int embsplit, int emb, double* buffer1, double* statexy, double* mixk, double* mixv, double* mixr, float* ffnrbuffer, int i, float* buffer2, float* buffer3, float* buffer4, int n_layers, int tokenlength, int mode){
    vuda::dim3 blocks = vuda::dim3(emb, 1, 1);
    const int threads = embsplit;

    const int stream_id = 0;
    const int embint = emb;
    vuda::launchKernel("mixatt.spv", "main", stream_id, blocks, threads, embint, tokenlength, mode, n_layers, i, buffer1, statexy, mixk, mixv, mixr, ffnrbuffer, buffer2, buffer3, buffer4);

}

void mixffn(int embsplit, int emb, double* buffer1, double* statexy, double* mixk, double* mixr, double* kbuffer, double* rbuffer, int i, int n_layers, int tokenlength, int mode){
    vuda::dim3 blocks = vuda::dim3(emb, 1, 1);
    const int threads = embsplit;

    const int stream_id = 0;
    const int embint = emb;
    vuda::launchKernel("mixffn.spv", "main", stream_id, blocks, threads, embint, tokenlength, mode, n_layers, i, buffer1, statexy, mixk, mixr, kbuffer, rbuffer);

}
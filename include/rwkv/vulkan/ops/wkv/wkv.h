#include <iostream>
#include <tuple>
#include "vuda_runtime.hpp"

void kernel_wkvc_forward(int embsplit,  unsigned long long emb,
                                     double *  w,  double *  u,  float *  k,  float *  v,
                                     float *  r, double *  y, double *  _aa, double *  _bb, double *  _pp,
                                    unsigned long long offset, unsigned long long layers, unsigned long long tokenlength, int mode = 1){
    const int blocks = (emb + embsplit - 1);
    const int threads = embsplit;

    const int stream_id = 0;
    const int embint = emb;
    vuda::launchKernel("wkv.spv", "main", stream_id, blocks, threads, embint, tokenlength, mode, layers, offset, w, u, k, v, r, y, _aa, _bb, _pp);
}

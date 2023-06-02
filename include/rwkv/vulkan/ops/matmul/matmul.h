

#include <iostream>
#include <tuple>
#include "vuda_runtime.hpp"
void cuda_mm8_threec(unsigned long long N,
    float *xy,
    uint8_t *w,
    uint8_t *w1,
    uint8_t *w2,
    float *r,
    float *r1,
    float *r2,
    float *o1,
    float *o2,
    float *o3,
    float *y,
    float *y1,
    float *y2,
    unsigned long long offset,
       unsigned long long tokenlength,
     unsigned long long  jsplit,
     unsigned long long jtile

){
const int stream_id = 0;
const int emb = N;
vuda::dim3 kernalparams = vuda::dim3(N/jtile, N/jsplit, jtile);
vuda::launchKernel("kernel_mm8_threec.spv", "main", stream_id, kernalparams, jsplit, jtile, emb, tokenlength, offset, xy, w, w1, w2, r, r1, r2, o1, o2, o3, y, y1, y2);
}

void cuda_mm8_one(unsigned long long N,
    unsigned long long M,
    float *xy,
    uint8_t *w,
    float *r,
    float *o1,
    float *y,
    unsigned long long offset,
       unsigned long long tokenlength
,
     unsigned long long  jsplit,
     unsigned long long jtile
){
const int stream_id = 0;
const int emb = N;
const int oemb = M;
vuda::dim3 kernalparams = vuda::dim3(M/jtile, N/jsplit, jtile);
vuda::launchKernel("kernel_mm8_one.spv", "main", stream_id, kernalparams, jsplit,jtile, emb, oemb, tokenlength, offset, xy, w,  r, o1, y);
}

void cuda_mm8_one(unsigned long long N,
    unsigned long long M,
    double *xy,
    uint8_t *w,
    float *r,
    float *o1,
    float *y,
    unsigned long long offset,
       unsigned long long tokenlength
,
    unsigned long long jsplit,
     unsigned long long jtile
){
const int stream_id = 0;
const int emb = N;
const int oemb = M;
const int jint = 1;
vuda::dim3 kernalparams = vuda::dim3(M/jtile, N/jsplit, jtile);
vuda::launchKernel("kernel_mm8_oned.spv", "main", stream_id, kernalparams, jsplit,jtile, emb, oemb, tokenlength, offset, xy, w,  r, o1, y);
}
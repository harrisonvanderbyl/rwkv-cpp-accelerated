#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <fstream>

#include "rwkv/rwkv/rwkv.h"

#define MM8_ONE_JSPLIT 16
#define MM8_ONE_TILE 1024
#define EMBSPLIT 512
#define EMBBLOCK 16

const char* const names[46] = {
    "xbuf",
    "embed",
    "layernorms",
    "state_xy",
    "state_aa",
    "state_bb",
    "state_pp",
    "state_dd",
    "buffer1",
    "buffer2",
    "buffer3",
    "buffer4",
    "mix_k",
    "mix_v",
    "mix_r",
    "km",
    "vm",
    "rm",
    "kr",
    "vr",
    "rr",
    "o1",
    "o2",
    "o3",
    "att_out",
    "att_out_r",
    "att_out_o",
    "ffn_mix_k",
    "ffn_mix_v",
    "ffn_k",
    "ffn_v",
    "ffn_r",
    "ffn_kr",
    "ffn_vr",
    "ffn_rr",
    "ffn_ko",
    "ffn_vo",
    "ffn_ro",
    "ffn_k_buffer",
    "ffn_v_buffer",
    "ffn_r_buffer",
    "decay",
    "bonus",
    "head",
    "head_r",
    "head_o"
};

#if _MSC_VER >= 1910
namespace std{
template <class Arg, class Result>
struct unary_function
{
    typedef Arg argument_type;
    typedef Result result_type;
};

template <class Arg1, class Arg2, class Result>
struct binary_function
{
    typedef Arg1 first_argument_type;
    typedef Arg2 second_argument_type;
    typedef Result result_type;
};
};
#endif

__global__ void cuda_layernorm(int64_t n_emb, const double *__restrict__ const x, const double *__restrict__ const weight, int64_t offset, float const inmean, float const instd, double *__restrict__ const out)
    {
    double xmean = inmean;
    double x2 = sqrt(instd);

    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
        if(i < n_emb){
            out[i] = weight[n_emb * offset + i] * ((x[i] - xmean) / x2) + weight[n_emb * (offset + 1) + i];
    }}
}
__global__ void kernel_mm8_threec(
    const int64_t N,
    const float *__restrict__ const xy,
    
    const uint8_t *__restrict__ const w,
    const uint8_t *__restrict__ const w1,
    const uint8_t *__restrict__ const w2,
    const float *__restrict__ const r,
    const float *__restrict__ const r1,
    const float *__restrict__ const r2,
    const float *__restrict__ const o1,
    const float *__restrict__ const o2,
    const float *__restrict__ const o3,
    float *__restrict__ const y,
    float *__restrict__ const y1,
    float *__restrict__ const y2,
    int64_t offset)
{

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < N)
    {
        float y_local = 0;
        float y1_local = 0;
        float y2_local = 0;
        for (int j = j0; j < j1; ++j)
        {
            y_local += xy[j] * ((w[j * N + k + offset * N * N] * r[j + offset * N]) + o1[j + offset * N]);
            y1_local += xy[j+N] * ((w1[j * N + k + offset * N * N] * r1[j + offset * N]) + o2[j + offset * N]);
            y2_local += xy[j+N*2] * ((w2[j * N + k + offset * N * N] * r2[j + offset * N]) + o3[j + offset * N]);
        }
        atomicAdd(reinterpret_cast<float *>(&y[k]), *reinterpret_cast<float *>(&y_local));
        atomicAdd(reinterpret_cast<float *>(&y1[k]), *reinterpret_cast<float *>(&y1_local));
        atomicAdd(reinterpret_cast<float *>(&y2[k]), *reinterpret_cast<float *>(&y2_local));
    }
}

// generic T either float or fp16 or fp64

void cuda_mm8_threec(int64_t N,
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
                     int64_t offset = 0

)
{
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (N + blockSize.y - 1) / blockSize.y);
    kernel_mm8_threec<<<gridSize, blockSize>>>(
        N,
        xy,
        w,
        w1,
        w2,
        r,
        r1,
        r2,
        o1,
        o2,
        o3,
        y,
        y1,
        y2,
        offset);
}

__global__ void setx(
    const int emb,
    const float *__restrict__ const a,
    double *__restrict__ const b,
    int64_t offset = 0)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = double(a[i]);
    }}
}
__global__ void setx(
    const int emb,
    double *__restrict__ const a,
    double *__restrict__ const b,
    int64_t offset = 0)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = double(a[i]);
    }}
}
__global__ void setx(
    const int emb,
    double *__restrict__ const a,
    float *__restrict__ const b,
    int64_t offset = 0)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = float(a[i]);
    }}
}
__global__ void setx(
    const int emb,
    float *__restrict__ const a,
    float *__restrict__ const b,
    int64_t offset = 0)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = float(a[i]);
    }}
}
__global__ void setxf(
    const int emb,
    float *__restrict__ const a,
    double *__restrict__ const b,
    int64_t offset = 0)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i] = double(a[i+ offset * emb]);
    }}
}
__global__ void cuda_memset(
    const int emb,
    double *__restrict__ const a,
    double b)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        a[i] = b;
    }}
}

__global__ void cuda_memset(
    const int emb,
    float *__restrict__ const a,
    float b)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        a[i] = b;
    }}
}
__global__ void cuda_relusquared(
    const int emb,
    float *__restrict__ const a,
    float *__restrict__ const b
)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        a[i] = a[i] * (a[i] > 0);
        a[i] = a[i] * a[i];
        if(i%4 == 0){
            b[i/4] = 0.0f;
        }
    }}
}

__global__ void sigmoid(
    const int emb,
    float *__restrict__ const a,
    float *__restrict__ const out,
    float *__restrict__ const d
)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        out[i] = float(1.0 / (1.0 + exp(-double(a[i]))));
        d[i*4] = 0.0;
        d[i*4+1] = 0.0;
        d[i*4+2] = 0.0;
        d[i*4+3] = 0.0;
    }}
}
__global__ void kernel_wkvc_forward(const int C,
                                    const double *__restrict__ const w, const double *__restrict__ const u, const float *__restrict__ const k, const float *__restrict__ const v,
                                    const float *__restrict__ const r, double *__restrict__ const y, double *__restrict__ const _aa, double *__restrict__ const _bb, double *__restrict__ const _pp,
                                    int64_t offset)
{

    int i = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int ii = i + threadIdx.x*EMBBLOCK + c;
    
    if(ii < C){
        double aa = _aa[ii + C * offset];
        double bb = _bb[ii + C * offset];
        double pp = _pp[ii + C * offset];

        const double vv = v[ii];
        const double wr1 = aa + exp(u[ii + C * offset] + w[ii + C * offset] + k[ii]) * vv;
        const double wr2 = bb + exp(u[ii + C * offset] + w[ii + C * offset] + k[ii]);
        y[ii] = (wr1) / wr2;
        y[ii] = (1.0 / (1.0 + exp(-r[ii]))) * y[ii];
        aa = (aa + exp(k[ii]) * vv) * exp(w[ii + C * offset]);
        bb = (bb + exp(k[ii])) * exp(w[ii + C * offset]);
        _aa[ii + C * offset] = aa;
        _bb[ii + C * offset] = bb;
        _pp[ii + C * offset] = pp;
    }}
}

void cuda_wkvc_forward(int B, double *w, double *u, float *k, float *v, float *r, double *y, double *aa, double *bb, double *pp, int64_t offset)
{

    kernel_wkvc_forward<<<(B+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(B, w, u, k, v, r, y, aa, bb, pp, offset);
}
__global__ void kernelc_mm8_one(
    const int N, const int M,
    const double *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    float *__restrict__ const y,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    const int64_t offset)
{

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M)
    {
        float y_local = 0;
        for (int j = j0; j < j1; ++j)
        {
            y_local += float(x[j]) * ((w[j * w_stride + k + offset * N * M] * r[j + offset * N] + o[j + offset * N]));
        }
        atomicAdd(reinterpret_cast<float *>(&y[k]), *reinterpret_cast<float *>(&y_local));
    }
}

void cudac_mm8_one(int N, int M,
                   double *x,
                   uint8_t *w, int w_stride,
                   float *y,
                   float *r,
                   float *o,
                   uint64_t offset)
{
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride, y, r, o, offset);
}

__global__ void kernelc_mm8_one(
    const int N, const int M,
    const float *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    float *__restrict__ const y,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    const int64_t offset)
{

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M)
    {
        float y_local = 0;
        for (int j = j0; j < j1; ++j)
        {
            y_local += float(x[j]) * ((w[j * w_stride + k + offset * N * M] * r[j + offset * N] + o[j + offset * N]));
        }
        atomicAdd(reinterpret_cast<float *>(&y[k]), *reinterpret_cast<float *>(&y_local));
    }
}

void cudac_mm8_one(int N, int M,
                   float *x,
                   uint8_t *w, int w_stride,
                   float *y,
                   float *r,
                   float *o,
                   uint64_t offset)
{
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride, y, r, o, offset);
}
__global__ void addx(
    const int emb,
    double *__restrict__ const a,
    double *__restrict__ const b)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i] += double(a[i]);
    }}
}

__global__ void mixffn(
    const int emb,
    double *__restrict__ const rc,
    double *__restrict__ const ddd,
    double *__restrict__ const mixk,
    double *__restrict__ const mixr,
    double *__restrict__ const outk,
    double *__restrict__ const outr, int64_t offset

)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        outk[i] = mixk[i + offset * emb] * rc[i] + (1.0 - mixk[i + offset * emb]) * ddd[i + offset * emb];
        outr[i] = mixr[i + offset * emb] * rc[i] + (1.0 - mixr[i + offset * emb]) * ddd[i + offset * emb];
    }}
}

__global__ void mixatt(
    const int emb,
    double *__restrict__ const rc,
    double *__restrict__ const ddd,
    double *__restrict__ const mixk,
    double *__restrict__ const mixv,
    double *__restrict__ const mixr,
    float *__restrict__ const outkvr,
    int64_t offset,
    float* a,
    float* b,
    float* cc

)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        outkvr[i] = mixk[i + offset * emb] * rc[i] + (1.0 - mixk[i + offset * emb]) * ddd[i + offset * emb];
        outkvr[i+emb] = mixv[i + offset * emb] * rc[i] + (1.0 - mixv[i + offset * emb]) * ddd[i + offset * emb];
        outkvr[i+emb*2] = mixr[i + offset * emb] * rc[i] + (1.0 - mixr[i + offset * emb]) * ddd[i + offset * emb];
        ddd[i+offset*emb] = rc[i];
        a[i] = 0.0;
        b[i] = 0.0;
        cc[i] = 0.0;
    }}
}

__global__ void blockout(
    const int emb,
    double *__restrict__ const x,
    float *__restrict__ const rcm,
    float *__restrict__ const ddd)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        x[i] = x[i] + rcm[i] * ddd[i];
    }}
}
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <functional>
#include <cmath>

/*
 * @struct varianceshifteop
 * @brief a unary function that shifts input data
 * by their mean and computes the squares of them
 */
struct varianceshifteop
    : std::unary_function<float, float>
{
    varianceshifteop(float m)
        : mean(m)
    { /* no-op */ }

    const float mean;

    __device__ float operator()(float data) const
    {
        return ::pow(data - mean, 2.0f);
    }
};

void getOutput(int64_t n_embed, int64_t n_layers, float* logitsin, double* statexyin, double* stateaain, double* statebbin, double* stateppin, double* stateddin, 
                                                float* logitsout, double* statexyout, double* stateaaout, double* statebbout, double* stateppout, double* stateddout){
    // copy gpu tensor in to cpu tensor out
    cudaMemcpy(logitsout, logitsin, 50277*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(statexyout, statexyin, n_embed*n_layers*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(stateaaout, stateaain, n_embed*n_layers*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(statebbout, statebbin, n_embed*n_layers*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(stateppout, stateppin, n_embed*n_layers*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(stateddout, stateddin, n_embed*n_layers*sizeof(double), cudaMemcpyDeviceToHost);
    
};

void setState(int64_t n_embed, int64_t n_layers,
    double* stateaa, double* statebb, double* statecc, double* statedd, double* stateee,
    double* instateaa, double* instatebb, double* instatecc, double* instatedd, double* instateee){
    
    // copy cpu tensor to gpu
    cudaMemcpy(stateaa, instateaa, n_embed*n_layers*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(statebb, instatebb, n_embed*n_layers*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(statecc, instatecc, n_embed*n_layers*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(statedd, instatedd, n_embed*n_layers*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(stateee, instateee, n_embed*n_layers*sizeof(double), cudaMemcpyHostToDevice);
};

void cuda_rwkv(int64_t n_layers, int64_t n_emb, int64_t token, double *x,
               float *embed, double *layernorms,
               double *statexy, double *stateaa, double *statebb, double *statepp, double *statedd,
               double *buffer1, float *buffer2, float *buffer3, float *buffer4,
               double *mixk, double *mixv, double *mixr,
               uint8_t *km, uint8_t *vm, uint8_t *rm,
               float *kr, float *vr, float *rr,
               float *o1, float *o2, float *o3,
               uint8_t *attout, float *attoutr, float *attouto,
               double *ffnmixk, double *ffnmixv,
               uint8_t *ffnk, uint8_t *ffnv, uint8_t *ffnr,
               float *ffnkr, float *ffnvr, float *ffnrr,
               float *ffnko, float *ffnvo, float *ffnro,
               double *ffnkbuffer, double *ffnvbuffer, float *ffnrbuffer,
               double *decay, double *bonus,
               uint8_t *head, float *headr, float *heado)
{
    setxf<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, embed, buffer1, token);
    
    thrust::device_ptr<double> mp(buffer1);
    float ccmean = thrust::reduce(
        mp,
        mp+n_emb,
        0.0f,
        thrust::plus<float>()) / n_emb;
    float ccvariance = thrust::transform_reduce(
        mp,
        mp+n_emb,
        varianceshifteop(ccmean),
        0.0f,
        thrust::plus<float>()) / (n_emb - 1);

    cuda_layernorm<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer1, layernorms, 0,ccmean,ccvariance, x);
    thrust::device_ptr<double> xp(x);

    for (int64_t i = 0; i < n_layers; i++)
    {
        // xy = ln(x)
        // kmix, vmix, rmix = mix(xy, statexy[n_emb*y], mixkvr)
        // k, v, r = matmul(kmix, km), matmul(vmix, vm), matmul(rmix, rm)
        float camean = thrust::reduce(
            xp,
            xp+n_emb,
            0.0f,
            thrust::plus<float>()) / n_emb;
        float cavariance = thrust::transform_reduce(
            xp,
            xp+n_emb,
            varianceshifteop(camean),
            0.0f,
            thrust::plus<float>()) / (n_emb - 1);
    
        cuda_layernorm<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, x, layernorms, 4 * (i) + 2,camean, cavariance, buffer1);
        // buffers to 0
        mixatt<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer1, statexy, mixk , mixv , mixr , ffnrbuffer, i,buffer2,buffer3,buffer4);
        cuda_mm8_threec(n_emb, ffnrbuffer, km, vm, rm, kr, vr, rr, o1, o2, o3, buffer2, buffer3, buffer4, i);
        // buffer2, 3, 4 = k,v,r

        cuda_wkvc_forward(n_emb, decay, bonus, buffer2, buffer3, buffer4, buffer1, stateaa, statebb, statepp, i);

        // buffer1 = rwkv
        setx<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, x, buffer2);
        cudac_mm8_one(n_emb, n_emb, buffer1, attout, n_emb, buffer2, attoutr, attouto, i);
        // buffer2 = attout(rwkv) + x
        setx<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer2, x);
        float zzmean = thrust::reduce(
            xp,
            xp+n_emb,
            0.0f,
            thrust::plus<float>()) / n_emb;
        float zzvariance = thrust::transform_reduce(
            xp,
            xp+n_emb,
            varianceshifteop(zzmean),
            0.0f,
            thrust::plus<float>()) / (n_emb - 1);   
        cuda_layernorm<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, x, layernorms, 4 * (i + 1),zzmean, zzvariance, buffer1);
        // buffer1 = ln(x)
        // ffnmixk, ffnmixv = mix(buffer1, statedd[n_emb*y], ffnmixkvr)
        mixffn<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer1, statedd, ffnmixk, ffnmixv, ffnkbuffer, ffnvbuffer, i);
        setx<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer1, statedd, i);
        // ffnkbuffer, ffnvbuffer = mixk, mixv
        cuda_memset<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer2, 0);
        cudac_mm8_one(n_emb, n_emb, ffnvbuffer, ffnr, n_emb, buffer2, ffnrr, ffnro, i);
        sigmoid<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer2, buffer4, ffnrbuffer);
        // ffnvbuffer = sigmoid(ffnrbuffer @ ffnr)
        cudac_mm8_one(n_emb, n_emb * 4, ffnkbuffer, ffnk, n_emb * 4, ffnrbuffer, ffnkr, ffnko, i);
        // ffnrbuffer = ffnkbuffer @ ffnk
        cuda_relusquared<<<(n_emb*4+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb * 4, ffnrbuffer,buffer3);
        cudac_mm8_one(n_emb * 4, n_emb, ffnrbuffer, ffnv, n_emb, buffer3, ffnvr, ffnvo, i);
        // buffer3 = ffnrbuffer @ ffnv
        blockout<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, x, buffer3, buffer4);

        // cuda_layernorm<<<1,1>>>(n_emb, x, layernorms,4*(i)+4, buffer1);
        // setx<<<1,1>>>(n_emb, buffer1, x);
    }

    float mean = thrust::reduce(
        xp,
        xp+n_emb,
        0.0f,
        thrust::plus<float>()) / n_emb;
    float variance = thrust::transform_reduce(
        xp,
        xp+n_emb,
        varianceshifteop(mean),
        0.0f,
        thrust::plus<float>()) / (n_emb - 1);
    cuda_layernorm<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, x, layernorms, 4 * (n_layers) + 2,mean,variance, buffer1);
    cuda_memset<<<(50277+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(50277, buffer2, 0);
    cudac_mm8_one(n_emb, 50277, buffer1, head, 50277, buffer2, headr, heado, 0);
}

// this lazy loads the model from disk


std::tuple<int,int> load (const std::string& filename, int** ptrs) {
    std::ifstream binfile(filename, std::ios::in | std::ios::binary);
    if (!binfile.is_open()) {
        std::cout << "Error opening file " << filename << std::endl;
        exit(1);
    }

    // get n_layers
    // get n_embed
    int64_t n_layers, n_embed;
    binfile.read((char*)&n_layers, sizeof(int64_t));
    binfile.read((char*)&n_embed, sizeof(int64_t));
    // print
    std::cout << "n_layers: " << n_layers << std::endl;
    std::cout << "n_embed: " << n_embed << std::endl;
  

    for(int64_t i = 0; i < 46; i++) {
        int64_t size = getSize(i, n_layers, n_embed);
        if(Mtypes(i) == sizeof(double)){
            ptrs[i] = (int*)(new double[size]);
        } else if(Mtypes(i) == sizeof(float)) {
            ptrs[i] = (int*)(new float[size]);
        } else if(Mtypes(i) == sizeof(uint8_t)) {
            ptrs[i] = (int*)(new uint8_t[size]);
        } else {
            std::cout << "Error: size not supported" << std::endl;
            exit(1);
        }
        std::cout << "loading: " << getName(i) << "\n";
        binfile.read((char*)(ptrs[i]), size*Mtypes(i));

        if(Mtypes(i) == sizeof(float)){
            float first = ((float*)ptrs[i])[0];
            float last = ((float*)ptrs[i])[getSize(i,n_layers,n_embed)-1];
            printf("float %d: %f %f %d\n", int(i), first, last, int(getSize(i,n_layers,n_embed)));
            float* cuda_mem;
            cudaMalloc(&cuda_mem, getSize(i,n_layers,n_embed) * Mtypes(i));
            cudaMemcpy(cuda_mem, (float*)ptrs[i], getSize(i,n_layers,n_embed) * Mtypes(i), cudaMemcpyHostToDevice);
            // sync
            cudaDeviceSynchronize();
            free(ptrs[i]);
            ptrs[i] = (int*)cuda_mem;
        }
        else if(Mtypes(i) == sizeof(double)){
            double firstd = ((double*)ptrs[i])[0];
            double lastd = ((double*)ptrs[i])[getSize(i,n_layers,n_embed)-1];
            printf("double %d: %f %f %d\n",  int(i), firstd, lastd, int(getSize(i,n_layers,n_embed)));
            double* cuda_mem;
            cudaMalloc(&cuda_mem, getSize(i,n_layers,n_embed) * Mtypes(i));
            cudaMemcpy(cuda_mem, (double*)ptrs[i], getSize(i,n_layers,n_embed) * Mtypes(i), cudaMemcpyHostToDevice);
            // sync
            cudaDeviceSynchronize();
            free(ptrs[i]);
            ptrs[i] = (int*)cuda_mem;
        }
        else if(Mtypes(i) == sizeof(uint8_t)){
            uint8_t firstu = ((uint8_t*)ptrs[i])[0];
            uint8_t lastu = ((uint8_t*)ptrs[i])[getSize(i,n_layers,n_embed)-1];
            printf("uint8_t %d: %d %d %d\n",  int(i), int(firstu), int(lastu), int(getSize(i,n_layers,n_embed)));
            uint8_t* cuda_mem;
            cudaMalloc(&cuda_mem, getSize(i,n_layers,n_embed) * Mtypes(i));
            cudaMemcpy(cuda_mem, (uint8_t*)ptrs[i], getSize(i,n_layers,n_embed) * Mtypes(i), cudaMemcpyHostToDevice);
            // sync
            cudaDeviceSynchronize();
            free(ptrs[i]);
            ptrs[i] = (int*)cuda_mem;
        }
    }
    
    binfile.close();

    //   // return an array of pointers

    // return (n_layers, n_embed)
    return std::make_tuple((int)n_layers, (int)n_embed);

}
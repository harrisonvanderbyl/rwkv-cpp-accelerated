#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <fstream>
#define MM8_ONE_JSPLIT 16
#define MM8_ONE_TILE 1024
#define EMBSPLIT 512
#define EMBBLOCK 16


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

__global__ void cuda_layernorm(unsigned long long n_emb, const double *__restrict__ const x, const double *__restrict__ const weight, unsigned long long offset, float const inmean, float const instd, double *__restrict__ const out)
    {
    double xmean = inmean;
    double x2 = sqrt(instd);

    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
        if(i < n_emb){
            out[i] = weight[n_emb * offset + i] * ((x[i] - xmean) / x2) + weight[n_emb * (offset + 1) + i];
    }}
}
__global__ void kernel_mm8_threec(
    const unsigned long long N,
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
    unsigned long long offset)
{

    const unsigned long long k = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long long j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const unsigned long long j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < N)
    {
        float y_local = 0;
        float y1_local = 0;
        float y2_local = 0;
        for (long long j = j0; j < j1; ++j)
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
                     unsigned long long offset = 0

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
    const unsigned long long emb,
    const float *__restrict__ const a,
    double *__restrict__ const b,
    unsigned long long offset = 0)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = double(a[i]);
    }}
}
__global__ void setx(
    const unsigned long long emb,
    double *__restrict__ const a,
    double *__restrict__ const b,
    unsigned long long offset = 0)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = double(a[i]);
    }}
}
__global__ void setx(
    const unsigned long long emb,
    double *__restrict__ const a,
    float *__restrict__ const b,
    unsigned long long offset = 0)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = float(a[i]);
    }}
}
__global__ void setx(
    const unsigned long long emb,
    float *__restrict__ const a,
    float *__restrict__ const b,
    unsigned long long offset = 0)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = float(a[i]);
    }}
}
__global__ void setxf(
    const unsigned long long emb,
    float *__restrict__ const a,
    double *__restrict__ const b,
    unsigned long long offset = 0)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i] = double(a[i+ offset * emb]);
    }}
}
__global__ void cuda_memset(
    const unsigned long long emb,
    double *__restrict__ const a,
    double b)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        a[i] = b;
    }}
}

__global__ void cuda_memset(
    const unsigned long long emb,
    float *__restrict__ const a,
    float b)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        a[i] = b;
    }}
}
__global__ void cuda_relusquared(
    const unsigned long long emb,
    float *__restrict__ const a,
    float *__restrict__ const b
)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        a[i] = a[i] * (a[i] > 0);
        a[i] = a[i] * a[i];
        if(i%4 == 0){
            b[i/4] = 0.0f;
        }
    }}
}

__global__ void sigmoid(
    const unsigned long long emb,
    float *__restrict__ const a,
    float *__restrict__ const out,
    float *__restrict__ const d
)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        out[i] = float(1.0 / (1.0 + exp(-double(a[i]))));
        d[i*4] = 0.0;
        d[i*4+1] = 0.0;
        d[i*4+2] = 0.0;
        d[i*4+3] = 0.0;
    }}
}
__global__ void kernel_wkvc_forward(const unsigned long long C,
                                    const double *__restrict__ const w, const double *__restrict__ const u, const float *__restrict__ const k, const float *__restrict__ const v,
                                    const float *__restrict__ const r, double *__restrict__ const y, double *__restrict__ const _aa, double *__restrict__ const _bb, double *__restrict__ const _pp,
                                    unsigned long long offset)
{

    long long i = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        long long ii = i + threadIdx.x*EMBBLOCK + c;
    
    if(ii < C){
        double aa = _aa[ii + C * offset];
        double bb = _bb[ii + C * offset];
        double pp = _pp[ii + C * offset];

        const double vv = v[ii];
        const double wr1 = aa + exp(u[ii + C * offset] + w[ii + C * offset] + k[ii]) * vv;
        const double wr2 = bb + exp(u[ii + C * offset] + w[ii + C * offset] + k[ii]);
        y[ii] = (wr1) / wr2;
        y[ii] = (1.0 / (1.0 + exp(-r[ii]))) * y[ii];
        aa = (aa + exp(double(k[ii])) * vv) * exp(w[ii + C * offset]);
        bb = (bb + exp(double(k[ii]))) * exp(w[ii + C * offset]);
        _aa[ii + C * offset] = aa;
        _bb[ii + C * offset] = bb;
        _pp[ii + C * offset] = pp;
    }}
}

void cuda_wkvc_forward(unsigned long long B, double *w, double *u, float *k, float *v, float *r, double *y, double *aa, double *bb, double *pp, unsigned long long offset)
{

    kernel_wkvc_forward<<<(B+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(B, w, u, k, v, r, y, aa, bb, pp, offset);
}
__global__ void kernelc_mm8_one(
    const unsigned long long N, const unsigned long long M,
    const double *__restrict__ const x,
    const uint8_t *__restrict__ const w, const unsigned long long w_stride,
    float *__restrict__ const y,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    const unsigned long long offset)
{

    const unsigned long long k = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long long j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const unsigned long long j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M)
    {
        float y_local = 0;
        for (unsigned long long j = j0; j < j1; ++j)
        {
            y_local += float(x[j]) * ((w[j * w_stride + k + offset * N * M] * r[j + offset * N] + o[j + offset * N]));
        }
        atomicAdd(reinterpret_cast<float *>(&y[k]), *reinterpret_cast<float *>(&y_local));
    }
}

void cudac_mm8_one(unsigned long long N, unsigned long long M,
                   double *x,
                   uint8_t *w, unsigned long long w_stride,
                   float *y,
                   float *r,
                   float *o,
                   unsigned long long offset)
{
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride, y, r, o, offset);
}

__global__ void kernelc_mm8_one(
    const unsigned long long N, const unsigned long long M,
    const float *__restrict__ const x,
    const uint8_t *__restrict__ const w, const unsigned long long w_stride,
    float *__restrict__ const y,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    const unsigned long long offset)
{

    const unsigned long long k = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long long j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const unsigned long long j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M)
    {
        float y_local = 0;
        for (unsigned long long j = j0; j < j1; ++j)
        {
            y_local += float(x[j]) * ((w[j * w_stride + k + offset * N * M] * r[j + offset * N] + o[j + offset * N]));
        }
        atomicAdd(reinterpret_cast<float *>(&y[k]), *reinterpret_cast<float *>(&y_local));
    }
}

void cudac_mm8_one(unsigned long long  N, unsigned long long M,
                   float *x,
                   uint8_t *w, unsigned long long w_stride,
                   float *y,
                   float *r,
                   float *o,
                   unsigned long long offset)
{
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride, y, r, o, offset);
}
__global__ void addx(
    const unsigned long long emb,
    double *__restrict__ const a,
    double *__restrict__ const b)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i] += double(a[i]);
    }}
}

__global__ void mixffn(
    const unsigned long long emb,
    double *__restrict__ const rc,
    double *__restrict__ const ddd,
    double *__restrict__ const mixk,
    double *__restrict__ const mixr,
    double *__restrict__ const outk,
    double *__restrict__ const outr, unsigned long long offset

)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        outk[i] = mixk[i + offset * emb] * rc[i] + (1.0 - mixk[i + offset * emb]) * ddd[i + offset * emb];
        outr[i] = mixr[i + offset * emb] * rc[i] + (1.0 - mixr[i + offset * emb]) * ddd[i + offset * emb];
    }}
}

__global__ void mixatt(
    const unsigned long long emb,
    double *__restrict__ const rc,
    double *__restrict__ const ddd,
    double *__restrict__ const mixk,
    double *__restrict__ const mixv,
    double *__restrict__ const mixr,
    float *__restrict__ const outkvr,
    unsigned long long offset,
    float* a,
    float* b,
    float* cc

)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
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
    const unsigned long long emb,
    double *__restrict__ const x,
    float *__restrict__ const rcm,
    float *__restrict__ const ddd)
{
    unsigned long long ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(unsigned long long c = 0; c < EMBBLOCK; c++){
        unsigned long long i = ii + threadIdx.x*EMBBLOCK + c;
    
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

void getOutput(unsigned long long n_embed, unsigned long long n_layers, float* logitsin, double* statexyin, double* stateaain, double* statebbin, double* stateppin, double* stateddin, 
                                                float* logitsout, double* statexyout, double* stateaaout, double* statebbout, double* stateppout, double* stateddout){
    // copy gpu tensor in to cpu tensor out
    cudaMemcpy(logitsout, logitsin, 50277*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(statexyout, statexyin, n_embed*n_layers*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(stateaaout, stateaain, n_embed*n_layers*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(statebbout, statebbin, n_embed*n_layers*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(stateppout, stateppin, n_embed*n_layers*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(stateddout, stateddin, n_embed*n_layers*sizeof(double), cudaMemcpyDeviceToHost);
    
};

void setState(unsigned long long n_embed, unsigned long long n_layers,
    double* stateaa, double* statebb, double* statecc, double* statedd, double* stateee,
    double* instateaa, double* instatebb, double* instatecc, double* instatedd, double* instateee){
    
    // copy cpu tensor to gpu
    cudaMemcpy(stateaa, instateaa, n_embed*n_layers*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(statebb, instatebb, n_embed*n_layers*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(statecc, instatecc, n_embed*n_layers*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(statedd, instatedd, n_embed*n_layers*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(stateee, instateee, n_embed*n_layers*sizeof(double), cudaMemcpyHostToDevice);
};

void cuda_rwkv(unsigned long long n_layers, unsigned long long n_emb, unsigned long long token, double *x,
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
    // copy the embedding table to the gpu buffer3, using token as the index
    cudaMemcpy(buffer3, embed + token * n_emb, n_emb * sizeof(float), cudaMemcpyHostToDevice);
    // copy buffer3 to buffer1
    setx<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer3, buffer1);
    
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

    for (unsigned long long i = 0; i < n_layers; i++)
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
    cudaDeviceSynchronize();
}

// this lazy loads the model from disk

unsigned long long Mtypes(unsigned long long i);
unsigned long long getSize(unsigned long long i, unsigned long long n_layers, unsigned long long n_embed);
char* getName(unsigned long long i);
// ptrs, n_layers, n_embed



std::tuple<unsigned long long,unsigned long long> load (const std::string& filename, int** ptrs) {
    std::ifstream binfile(filename, std::ios::in | std::ios::binary);
    if (!binfile.is_open()) {
        std::cout << "Error opening file " << filename << std::endl;
        exit(1);
    }

    // get n_layers
    // get n_embed
    unsigned long long n_layers, n_embed;
    binfile.read((char*)&n_layers, sizeof(unsigned long long));
    binfile.read((char*)&n_embed, sizeof(unsigned long long));
    // print
    std::cout << "n_layers: " << n_layers << std::endl;
    std::cout << "n_embed: " << n_embed << std::endl;

    
  

    for(unsigned long long i = 0; i < 46; i++) {
        unsigned long long size = getSize(i, n_layers, n_embed);

        // malloc
        ptrs[i] = (int*)malloc(size*Mtypes(i));

        std::cout << "loading: " << getName(i) << "\n";

        binfile.read(ptrs[i], size*Mtypes(i));

        if( i == 1) // embedding table, stays on cpu
            continue;

        void* cuda_mem;
        cudaMalloc(&cuda_mem, size * Mtypes(i));
        cudaMemcpy(cuda_mem, ptrs[i], size * Mtypes(i), cudaMemcpyHostToDevice);
        // sync
        cudaDeviceSynchronize();
        free(ptrs[i]);
        ptrs[i] = (int*)cuda_mem;
        
    }
    
    binfile.close();

    //   // return an array of pointers

    // return (n_layers, n_embed)
    return std::make_tuple(n_layers, n_embed);

}

void freeTensors(int** ptrs) {
    for(unsigned long long i = 0; i < 46; i++) {
        if (i == 1) // embedding table, stays on cpu
        {
            delete[] ptrs[i];
            continue;
        }
        cudaFree(ptrs[i]);
    }
}
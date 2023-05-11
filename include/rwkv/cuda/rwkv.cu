#ifndef __NVCC__
// Allow HIP/amd to compile this file
#include "hip/hip_runtime.h"
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemset hipMemset
#endif

#include "rwkv/enums/enum.h"
#include <tuple>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <fstream>
#include <iostream>
#define MM8_ONE_JSPLIT 16
#define MM8_ONE_TILE 256
#define EMBSPLIT 256
#define EMBBLOCK 16

// log either float or double
template <typename DTYPE>
void logData(DTYPE* input, int N = 1, std::string logname = "log"){
    DTYPE * outbuf2;
    // copy the output back to the host
    outbuf2 = (DTYPE *)malloc(N * sizeof(DTYPE));
    cudaMemcpy(outbuf2, input, N * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    std::cout << logname << std::endl;
    for (int i = 0; i < N; i++)
    {
        std::cout << outbuf2[i] << std::endl;
    }
}

__global__ void cuda_layernorm(unsigned long long n_emb, const double *__restrict__ const x, const double *__restrict__ const weight, unsigned long long offset, float *const inmean, float *const instd, double *__restrict__ const out, unsigned long long tokenlength)
{
    for(unsigned long long token = 0; token < tokenlength; token++){
        double xmean = double(inmean[token]) / n_emb;
        double x2 = sqrt(instd[token] / (n_emb - 1));

        unsigned long long ii = blockIdx.x * (blockDim.x * EMBBLOCK);
        for (unsigned long long c = 0; c < EMBBLOCK; c++)
        {
            unsigned long long i = ii + threadIdx.x * EMBBLOCK + c;

            if (i < n_emb)
            {
                out[i+n_emb*token] = weight[n_emb * offset + i] * ((x[i + n_emb*token] - xmean) / x2) + weight[n_emb * (offset + 1) + i];
            }
        }
    }
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
    unsigned long long offset,
    unsigned long long tokenlength)
{

    for (unsigned long long token = 0; token < tokenlength; token++)
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
                y_local += xy[j + token*N*3] * ((w[j * N + k + offset * N * N] * r[j + offset * N]) + o1[j + offset * N]);
                y1_local += xy[j + N + token*N*3] * ((w1[j * N + k + offset * N * N] * r1[j + offset * N]) + o2[j + offset * N]);
                y2_local += xy[j + N * 2 + token*N*3] * ((w2[j * N + k + offset * N * N] * r2[j + offset * N]) + o3[j + offset * N]);
            }
            atomicAdd(reinterpret_cast<float *>(&y[k + token*N]), *reinterpret_cast<float *>(&y_local));
            atomicAdd(reinterpret_cast<float *>(&y1[k + token*N]), *reinterpret_cast<float *>(&y1_local));
            atomicAdd(reinterpret_cast<float *>(&y2[k + token*N]), *reinterpret_cast<float *>(&y2_local));
        }
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
    unsigned long long offset,
       unsigned long long tokenlength

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
offset,
tokenlength
);
}

template <typename T_SRC, typename T_DST>
__global__ void setx(
    const unsigned long long emb,
    const T_SRC *__restrict__ const a,
    T_DST *__restrict__ const b,
    unsigned long long offset = 0)
{
    unsigned long long ii = blockIdx.x * (blockDim.x * EMBBLOCK);
    for (unsigned long long c = 0; c < EMBBLOCK; c++)
    {
        unsigned long long i = ii + threadIdx.x * EMBBLOCK + c;

        if (i < emb)
            b[i + offset * emb] = T_DST(a[i]);
    }
}

template <typename DTYPE, typename FDTYPE>
__global__ void cuda_memset(
    const unsigned long long emb,
    DTYPE *__restrict__ const a,
    FDTYPE b)
{
    unsigned long long ii = blockIdx.x * (blockDim.x * EMBBLOCK);
    for (unsigned long long c = 0; c < EMBBLOCK; c++)
    {
        unsigned long long i = ii + threadIdx.x * EMBBLOCK + c;

        if (i < emb)
            a[i] = DTYPE(b);
    }
}

__global__ void cuda_relusquared(
    const unsigned long long emb,
    float *__restrict__ const a,
    float *__restrict__ const b)
{
    unsigned long long ii = blockIdx.x * (blockDim.x * EMBBLOCK);
    for (unsigned long long c = 0; c < EMBBLOCK; c++)
    {
        unsigned long long i = ii + threadIdx.x * EMBBLOCK + c;

        if (i < emb)
        {
            a[i] = a[i] * (a[i] > 0);
            a[i] = a[i] * a[i];
            if (i % 4 == 0)
            {
                b[i / 4] = 0.0f;
            }
        }
    }
}

__global__ void sigmoid(
    const unsigned long long emb,
    float *__restrict__ const a,
    float *__restrict__ const out,
    float *__restrict__ const d)
{
    unsigned long long ii = blockIdx.x * (blockDim.x * EMBBLOCK);
    for (unsigned long long c = 0; c < EMBBLOCK; c++)
    {
        unsigned long long i = ii + threadIdx.x * EMBBLOCK + c;

        if (i < emb)
        {
            out[i] = float(1.0 / (1.0 + exp(-double(a[i]))));
            d[i * 4] = 0.0;
            d[i * 4 + 1] = 0.0;
            d[i * 4 + 2] = 0.0;
            d[i * 4 + 3] = 0.0;
        }
    }
}
__global__ void kernel_wkvc_forward(const unsigned long long C,
                                    const double *__restrict__ const w, const double *__restrict__ const u, const float *__restrict__ const k, const float *__restrict__ const v,
                                    const float *__restrict__ const r, double *__restrict__ const y, double *__restrict__ const _aa, double *__restrict__ const _bb, double *__restrict__ const _pp,
                                    unsigned long long offset, unsigned long long layers, unsigned long long tokenlength, MODE mode = PARRALEL)
{

    for( unsigned long long token = 0; token < tokenlength; token ++){

        long long i = blockIdx.x * (blockDim.x * EMBBLOCK);
        for (unsigned long long c = 0; c < EMBBLOCK; c++)
        {
            long long ii = i + threadIdx.x * EMBBLOCK + c;

            if (ii < C)
            {
                unsigned long long stateoffset = ii + offset * C;

                if(mode == PARRALEL){
                    stateoffset = ii + offset * C + token * C * layers;
                }

                double aa = _aa[stateoffset];
                double bb = _bb[stateoffset];
                double pp = _pp[stateoffset];

                const double vv = v[ii + token * C];
                const double wr1 = aa + exp(u[ii + C * offset] + w[ii + C * offset] + k[ii + token * C]) * vv;
                const double wr2 = bb + exp(u[ii + C * offset] + w[ii + C * offset] + k[ii + token * C]);
                y[ii + token * C] = (wr1) / wr2;
                y[ii + token * C] = (1.0 / (1.0 + exp(-r[ii + token * C]))) * y[ii + token * C];
                aa = (aa + exp(double(k[ii + token * C])) * vv) * exp(w[ii + C * offset]);
                bb = (bb + exp(double(k[ii + token * C]))) * exp(w[ii + C * offset]);
                _aa[stateoffset] = aa;
                _bb[stateoffset] = bb;
                _pp[stateoffset] = pp;
            }
        }
    }
}

void cuda_wkvc_forward(unsigned long long B, double *w, double *u, float *k, float *v, float *r, double *y, double *aa, double *bb, double *pp, unsigned long long offset, unsigned long long layers, unsigned long long tokenlength, MODE mode = PARRALEL)
{

    kernel_wkvc_forward<<<(B + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(B, w, u, k, v, r, y, aa, bb, pp, offset, layers, tokenlength, mode);
}

template <typename DTYPE>
__global__ void kernelc_mm8_one(
    const unsigned long long N, const unsigned long long M,
    const DTYPE *__restrict__ const x,
    const uint8_t *__restrict__ const w, const unsigned long long w_stride,
    float *__restrict__ const y,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    const unsigned long long offset,
    unsigned long long tokenlength)
{

    for (unsigned long long token = 0; token < tokenlength; token++)
    {
        const unsigned long long k = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned long long j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
        const unsigned long long j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

        if (k < M)
        {
            float y_local = 0;
            for (unsigned long long j = j0; j < j1; ++j)
            {
                y_local += float(x[j + N * token]) * ((w[j * w_stride + k + offset * N * M] * r[j + offset * N] + o[j + offset * N]));
            }
            atomicAdd(reinterpret_cast<float *>(&y[k + M * token]), *reinterpret_cast<float *>(&y_local));
        }
    }
}

template <typename DTYPE>
void cudac_mm8_one(unsigned long long N, unsigned long long M,
                   DTYPE *x,
                   uint8_t *w, unsigned long long w_stride,
                   float *y,
                   float *r,
                   float *o,
                   unsigned long long offset,
                     unsigned long long tokenlength)
{
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride, y, r, o, offset, tokenlength);
}

__global__ void mixffn(
    const unsigned long long emb,
    double *__restrict__ const rc,
    double *__restrict__ const ddd,
    double *__restrict__ const mixk,
    double *__restrict__ const mixr,
    double *__restrict__ const outk,
    double *__restrict__ const outr, unsigned long long offset,
    unsigned long long layers,
    unsigned long long tokenlength,
    MODE mode = PARRALEL

)
{

    for(unsigned long long token = 0; token < tokenlength; token ++){

    
        unsigned long long ii = blockIdx.x * (blockDim.x * EMBBLOCK);
        for (unsigned long long c = 0; c < EMBBLOCK; c++)
        {
            unsigned long long i = ii + threadIdx.x * EMBBLOCK + c;

            const unsigned long long stateoffset = i + offset * emb + int(mode == PARRALEL)*token * emb * layers;

      
            if (i < emb)
            {
                outk[i+token*emb] = mixk[i + offset * emb] * rc[i+token*emb] + (1.0 - mixk[i + offset * emb]) * ddd[stateoffset];
                outr[i+token*emb] = mixr[i + offset * emb] * rc[i+token*emb] + (1.0 - mixr[i + offset * emb]) * ddd[stateoffset];

                ddd[stateoffset] = rc[i+token*emb];

            }
        }
    }
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
    float *a,
    float *b,
    float *cc,
    unsigned long long layers,
    unsigned long long tokenlength,
    MODE mode = PARRALEL
)
{
    for(unsigned long long token = 0; token < tokenlength; token++ ){
        unsigned long long ii = blockIdx.x * (blockDim.x * EMBBLOCK);
        for (unsigned long long c = 0; c < EMBBLOCK; c++)
        {
            unsigned long long i = ii + threadIdx.x * EMBBLOCK + c;

            if (i < emb)
            {
                unsigned long long stateoffset = i + offset * emb;

                if(mode == PARRALEL){
                    stateoffset = i + offset * emb + token * emb * layers;
                }

                outkvr[i + token*emb*3] = mixk[i + offset * emb] * rc[i + token*emb] + (1.0 - mixk[i + offset * emb]) * ddd[stateoffset];
                outkvr[i + emb + token*emb*3] = mixv[i + offset * emb] * rc[i + token*emb] + (1.0 - mixv[i + offset * emb]) * ddd[stateoffset];
                outkvr[i + emb * 2 + token*emb*3] = mixr[i + offset * emb] * rc[i + token*emb] + (1.0 - mixr[i + offset * emb]) * ddd[stateoffset];
                ddd[stateoffset] = rc[i + token*emb];
                a[i + token*emb] = 0.0;
                b[i + token*emb] = 0.0;
                cc[i + token*emb] = 0.0;
            }
        }
    }
}

__global__ void blockout(
    const unsigned long long emb,
    double *__restrict__ const x,
    float *__restrict__ const rcm,
    float *__restrict__ const ddd)
{
    unsigned long long ii = blockIdx.x * (blockDim.x * EMBBLOCK);
    for (unsigned long long c = 0; c < EMBBLOCK; c++)
    {
        unsigned long long i = ii + threadIdx.x * EMBBLOCK + c;

        if (i < emb)
        {
            x[i] = x[i] + rcm[i] * ddd[i];
        }
    }
}

__global__ void addall(const unsigned long long emb, float *acc, double *a, unsigned long long tokenlength)
{
    unsigned long long ii = blockIdx.x * (blockDim.x * EMBBLOCK);
    for (unsigned long long token = 0; token < tokenlength; token++)
    {
        
        float accmini = 0;
        for (unsigned long long c = 0; c < EMBBLOCK; c++)
        {
            unsigned long long i = ii + threadIdx.x * EMBBLOCK + c;

            if (i < emb)
            {
                accmini += a[i + token * emb];
            }
        }
        atomicAdd(&acc[token], accmini);
    }
}

__global__ void variance(const unsigned long long emb, float *acc, double *a, float *mean, unsigned long long tokenlength)
{
    unsigned long long ii = blockIdx.x * (blockDim.x * EMBBLOCK);
    for (unsigned long long token = 0; token < tokenlength; token++)
    {
        float accmini = 0;
        for (unsigned long long c = 0; c < EMBBLOCK; c++)
        {
            unsigned long long i = ii + threadIdx.x * EMBBLOCK + c;

            if (i < emb)
            {
                accmini += (a[i+emb*token] - mean[token] / emb) * (a[i+emb*token] - mean[token] / emb);
            }
        }

        atomicAdd(&acc[token], accmini);
    }
}

std::tuple<float *, float *> meanvar(unsigned long long emb, double *a, float *buffer, unsigned long long tokenlength)
{
    float *acc = &buffer[0];
    cudaMemset(acc, 0, sizeof(float)*tokenlength);
    addall<<<(emb + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(emb, acc, a, tokenlength);
    float *mean = &buffer[tokenlength];
    cudaMemcpy(mean, acc, sizeof(float)*tokenlength, cudaMemcpyDeviceToDevice);
    cudaMemset(acc, 0, sizeof(float)*tokenlength);
    variance<<<(emb + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(emb, acc, a, mean, tokenlength);
    float *var = &buffer[2*tokenlength];
    cudaMemcpy(var, acc, sizeof(float)*tokenlength, cudaMemcpyDeviceToDevice);

    return std::make_tuple(mean, var);
}

void getOutput(unsigned long long n_embed, unsigned long long n_layers, float *logitsin, double *statexyin, double *stateaain, double *statebbin, double *stateppin, double *stateddin,
               float *logitsout, double *statexyout, double *stateaaout, double *statebbout, double *stateppout, double *stateddout, unsigned long long tokenlength)
{
    // copy gpu tensor in to cpu tensor out
    cudaMemcpy(logitsout, logitsin, 50277 * sizeof(float) * tokenlength, cudaMemcpyDeviceToHost);
    cudaMemcpy(statexyout, statexyin, n_embed * n_layers * sizeof(double)* tokenlength, cudaMemcpyDeviceToHost);
    cudaMemcpy(stateaaout, stateaain, n_embed * n_layers * sizeof(double)* tokenlength, cudaMemcpyDeviceToHost);
    cudaMemcpy(statebbout, statebbin, n_embed * n_layers * sizeof(double)* tokenlength, cudaMemcpyDeviceToHost);
    cudaMemcpy(stateppout, stateppin, n_embed * n_layers * sizeof(double)* tokenlength, cudaMemcpyDeviceToHost);
    cudaMemcpy(stateddout, stateddin, n_embed * n_layers * sizeof(double)* tokenlength, cudaMemcpyDeviceToHost);
};

void setState(unsigned long long n_embed, unsigned long long n_layers,
              double *stateaa, double *statebb, double *statecc, double *statedd, double *stateee,
              double *instateaa, double *instatebb, double *instatecc, double *instatedd, double *instateee, unsigned long long tokenlength)
{
    
    // copy cpu tensor to gpu
    cudaMemcpy(stateaa, instateaa, n_embed * n_layers * sizeof(double)* tokenlength, cudaMemcpyHostToDevice);
    cudaMemcpy(statebb, instatebb, n_embed * n_layers * sizeof(double)* tokenlength, cudaMemcpyHostToDevice);
    cudaMemcpy(statecc, instatecc, n_embed * n_layers * sizeof(double)* tokenlength, cudaMemcpyHostToDevice);
    cudaMemcpy(statedd, instatedd, n_embed * n_layers * sizeof(double)* tokenlength, cudaMemcpyHostToDevice);
    cudaMemcpy(stateee, instateee, n_embed * n_layers * sizeof(double)* tokenlength, cudaMemcpyHostToDevice);
};


void cuda_rwkv_parralel(unsigned long long n_layers, unsigned long long n_emb, unsigned long long *token, double *x,
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
    uint8_t *head, float *headr, float *heado, 
    unsigned long long tokenlength, MODE mode = PARRALEL)
{
   
        // copy the embedding table to the gpu buffer3, using token as the index
        for(unsigned long long i = 0; i < tokenlength; i++){
            cudaMemcpy(&buffer3[i * n_emb], &embed[token[i] * n_emb], n_emb * sizeof(float), cudaMemcpyHostToDevice);
        }
    
    // copy buffer3 to buffer1
    setx<<<(n_emb*tokenlength + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb*tokenlength, buffer3, buffer1);
    
    float *ccmean;
    float *ccvariance;
    std::tie(ccmean, ccvariance) = meanvar(n_emb, buffer1, ffnrbuffer, tokenlength);
    
    cuda_layernorm<<<(n_emb + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb, buffer1, layernorms, 0, ccmean, ccvariance, x, tokenlength);
    
    
    
    for (unsigned long long i = 0; i < n_layers; i++)
    {
        // xy = ln(x)
        // kmix, vmix, rmix = mix(xy, statexy[n_emb*y], mixkvr)
        // k, v, r = matmul(kmix, km), matmul(vmix, vm), matmul(rmix, rm)
        float *camean;
        float *cavariance;
        std::tie(camean, cavariance) = meanvar(n_emb, x, ffnrbuffer, tokenlength);

        cuda_layernorm<<<(n_emb + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb, x, layernorms, 4 * (i) + 2, camean, cavariance, buffer1, tokenlength);
        // buffers to 0
        
        mixatt<<<(n_emb + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb, buffer1, statexy, mixk, mixv, mixr, ffnrbuffer, i, buffer2, buffer3, buffer4,n_layers, tokenlength, mode);
        
        cuda_mm8_threec(n_emb, ffnrbuffer, km, vm, rm, kr, vr, rr, o1, o2, o3, buffer2, buffer3, buffer4, i, tokenlength);
        // buffer2, 3, 4 = k,v,r
        
        cuda_wkvc_forward(n_emb, decay, bonus, buffer2, buffer3, buffer4, buffer1, stateaa, statebb, statepp, i, n_layers, tokenlength, mode);
        
        // buffer1 = rwkv
        setx<<<(n_emb*tokenlength + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb*tokenlength, x, buffer2);
        
        cudac_mm8_one(n_emb, n_emb, buffer1, attout, n_emb, buffer2, attoutr, attouto, i, tokenlength);
        logData(buffer2,2,"\nattout");
        // buffer2 = attout(rwkv) + x
        setx<<<(n_emb*tokenlength + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb*tokenlength, buffer2, x);
        
        float *zzmean;
        float *zzvariance;
        std::tie(zzmean, zzvariance) = meanvar(n_emb, x, ffnrbuffer, tokenlength);
        cuda_layernorm<<<(n_emb + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb, x, layernorms, 4 * (i + 1), zzmean, zzvariance, buffer1, tokenlength);
        
        // buffer1 = ln(x)
        // ffnmixk, ffnmixv = mix(buffer1, statedd[n_emb*y], ffnmixkvr)
        mixffn<<<(n_emb + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb, buffer1, statedd, ffnmixk, ffnmixv, ffnkbuffer, ffnvbuffer, i, n_layers, tokenlength, mode);
        // ffnkbuffer, ffnvbuffer = mixk, mixv
        // logData(ffnkbuffer,2,"mixk");
        // logData(ffnvbuffer,2,"mixr");
        cuda_memset<<<(n_emb*tokenlength + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb*tokenlength, buffer2, 0);
        cudac_mm8_one(n_emb, n_emb, ffnvbuffer, ffnr, n_emb, buffer2, ffnrr, ffnro, i, tokenlength);
        sigmoid<<<(n_emb*tokenlength + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb*tokenlength, buffer2, buffer4, ffnrbuffer);
        // ffnvbuffer = sigmoid(ffnrbuffer @ ffnr)
        cudac_mm8_one(n_emb, n_emb * 4, ffnkbuffer, ffnk, n_emb * 4, ffnrbuffer, ffnkr, ffnko, i, tokenlength);
        // logData(ffnrbuffer,2,"ffnr");
        // ffnrbuffer = ffnkbuffer @ ffnk
        cuda_relusquared<<<(tokenlength * n_emb * 4 + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(tokenlength*n_emb * 4, ffnrbuffer, buffer3);
        cudac_mm8_one(n_emb * 4, n_emb, ffnrbuffer, ffnv, n_emb, buffer3, ffnvr, ffnvo, i, tokenlength);
        // logData(buffer3,2,"buffer3");
        // buffer3 = ffnrbuffer @ ffnv
        blockout<<<(tokenlength*n_emb + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(tokenlength*n_emb, x, buffer3, buffer4);

       
        
    }

    float *mean;
    float *variance;
    std::tie(mean, variance) = meanvar(n_emb, x, ffnrbuffer, tokenlength);
    cuda_layernorm<<<(n_emb + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(n_emb, x, layernorms, 4 * (n_layers) + 2, mean, variance, buffer1, tokenlength);
    
    cuda_memset<<<(tokenlength*50277 + EMBSPLIT - 1) / EMBSPLIT, EMBSPLIT / EMBBLOCK>>>(tokenlength*50277, buffer2, 0);
    cudac_mm8_one(n_emb, 50277, buffer1, head, 50277, buffer2, headr, heado, 0, tokenlength);
    cudaDeviceSynchronize();

    
}

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
    uint8_t *head, float *headr, float *heado){
        cuda_rwkv_parralel(n_layers, n_emb, &token, x,
            embed, layernorms,
            statexy, stateaa, statebb, statepp, statedd,
            buffer1, buffer2, buffer3, buffer4,
            mixk, mixv, mixr,
            km, vm, rm,
            kr, vr, rr,
            o1, o2, o3,
            attout, attoutr, attouto,
            ffnmixk, ffnmixv,
            ffnk, ffnv, ffnr,
            ffnkr, ffnvr, ffnrr,
            ffnko, ffnvo, ffnro,
            ffnkbuffer, ffnvbuffer, ffnrbuffer,
            decay, bonus,
            head, headr, heado, 1, GPT);
        
    }


// this lazy loads the model from disk

unsigned long long Mtypes(unsigned long long i);
unsigned long long getSize(unsigned long long i, unsigned long long n_layers, unsigned long long n_embed);
const char *getName(unsigned long long i);
// ptrs, n_layers, n_embed

std::tuple<unsigned long long, unsigned long long> load(const std::string &filename, int **ptrs, unsigned long long maxGPT = 1)
{
    std::ifstream binfile(filename, std::ios::in | std::ios::binary);
    if (!binfile.is_open())
    {
        std::cout << "Error opening file " << filename << std::endl;
        exit(1);
    }

    // get n_layers
    // get n_embed
    unsigned long long n_layers, n_embed;
    binfile.read((char *)&n_layers, sizeof(unsigned long long));
    binfile.read((char *)&n_embed, sizeof(unsigned long long));
    // print
    std::cout << "n_layers: " << n_layers << std::endl;
    std::cout << "n_embed: " << n_embed << std::endl;

    const int buffers[13] = {
        X,
        STATEXY,
        STATEAA,
        STATEBB,
        STATEPP,
        STATEDD,
        BUFFER1,
        BUFFER2,
        BUFFER3,
        BUFFER4,
        FFNKBUFFER,
        FFNVBUFFER,
        FFNRBUFFER
    };

    for (unsigned long long i = 0; i < 46; i++)
    {
        unsigned long long size = getSize(i, n_layers, n_embed);

        // malloc
        ptrs[i] = (int *)malloc(size * Mtypes(i));

        std::cout << "loading: " << getName(i) << "\n";

        binfile.read((char *)ptrs[i], size * Mtypes(i));

        if (i == 1) // embedding table, stays on cpu
            continue;

        void *cuda_mem;
        bool isBuffer = false;

        for (unsigned long long j = 0; j < 13; j++)
        {
            if (i == buffers[j])
            {
                isBuffer = true;
                break;
            }
        }

        if (!isBuffer){
            cudaMalloc(&cuda_mem, size * Mtypes(i));
            cudaMemcpy(cuda_mem, ptrs[i], size * Mtypes(i), cudaMemcpyHostToDevice);
        }else{
            cudaMalloc(&cuda_mem, size * Mtypes(i) * maxGPT);
            for(unsigned long long j = 0; j < maxGPT; j++){
                cudaMemcpy(&((int *)cuda_mem)[j*size], ptrs[i], size * Mtypes(i), cudaMemcpyHostToDevice);
            }
        }

        free(ptrs[i]);
        ptrs[i] = (int *)cuda_mem;
        
    }

    binfile.close();

 
    return std::make_tuple(n_layers, n_embed);
}

void freeTensors(int **ptrs)
{
    for (unsigned long long i = 0; i < 46; i++)
    {
        if (i == 1) // embedding table, stays on cpu
        {
            delete[] ptrs[i];
            continue;
        }
        cudaFree(ptrs[i]);
    }
}
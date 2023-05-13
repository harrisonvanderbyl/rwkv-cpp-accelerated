#include "rwkv/enums/enum.h"
#include <tuple>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <fstream>
#include <iostream>

#include "rwkv/vulkan/ops/layernorm/layernorm.h"
#include "rwkv/vulkan/ops/mix/mix.h"
#include "rwkv/vulkan/ops/matmul/matmul.h"
#include "rwkv/vulkan/ops/wkv/wkv.h"
#include "rwkv/vulkan/ops/sigmoid/sigmoid.h"
#define MM8_ONE_JSPLIT 32
#define MM8_ONE_TILE 256
#define EMBSPLIT 16
#define EMBBLOCK 16



// log either float or double
template <typename DTYPE>
void logData(DTYPE* input, int N = 1, std::string logname = "log"){
    DTYPE * outbuf2;
    // copy the output back to the host
    std::cout << logname << std::endl;
    for (int i = 0; i < N; i++)
    {
        std::cout << input[i] << std::endl;
    }
}

template<typename T>
void memCopy(T* a, T*b, long long unsigned number){
    for (long long unsigned i = 0; i < number; i++)
    {
        b[i] = a[i];
    }
} 

template<typename T, typename U>
void setx(T* a, U* b, long long unsigned number){
    for (long long unsigned i = 0; i < number; i++)
    {
        a[i] = T(b[i]);
    }
}

template<typename T, typename U>
void setx(T* a, U b, long long unsigned number){
    for (long long unsigned i = 0; i < number; i++)
    {
        a[i] = T(b);
    }
}
void cuda_layernorm(unsigned long long n_emb, const double * const x, const double * const weight, unsigned long long offset, double * const out, unsigned long long tokenlength)
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


void getOutput(unsigned long long n_embed, unsigned long long n_layers, float *logitsin, double *statexyin, double *stateaain, double *statebbin, double *stateppin, double *stateddin,
               float *logitsout, double *statexyout, double *stateaaout, double *statebbout, double *stateppout, double *stateddout, unsigned long long tokenlength)
{
    memCopy(logitsout, logitsin, 50277 * sizeof(float) * tokenlength);
    memCopy(statexyout, statexyin, n_embed * n_layers * sizeof(double)* tokenlength);
    memCopy(stateaaout, stateaain, n_embed * n_layers * sizeof(double)* tokenlength);
    memCopy(statebbout, statebbin, n_embed * n_layers * sizeof(double)* tokenlength);
    memCopy(stateppout, stateppin, n_embed * n_layers * sizeof(double)* tokenlength);
    memCopy(stateddout, stateddin, n_embed * n_layers * sizeof(double)* tokenlength);
};

void setState(unsigned long long n_embed, unsigned long long n_layers,
              double *stateaa, double *statebb, double *statecc, double *statedd, double *stateee,
              double *instateaa, double *instatebb, double *instatecc, double *instatedd, double *instateee, unsigned long long tokenlength)
{
    memCopy(stateaa, instateaa, n_embed * n_layers * sizeof(double)* tokenlength);
    memCopy(statebb, instatebb, n_embed * n_layers * sizeof(double)* tokenlength);
    memCopy(statecc, instatecc, n_embed * n_layers * sizeof(double)* tokenlength);
    memCopy(statedd, instatedd, n_embed * n_layers * sizeof(double)* tokenlength);
    memCopy(stateee, instateee, n_embed * n_layers * sizeof(double)* tokenlength);
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
  
    // copy the embedding table to its own buffer at the end, so the data is contiguous
    for(unsigned long long i = 0; i < tokenlength; i++){
        for(int m = 0; m < n_emb; m++){
            embed[50277*n_emb + i * n_emb + m] = embed[token[i] * n_emb + m];
        }
    }
    memCopy(buffer3, &embed[50277*n_emb], n_emb * sizeof(float)* tokenlength);
    setx(buffer1, buffer3, n_emb*tokenlength);
    layernorm(EMBSPLIT,EMBBLOCK, n_emb, buffer1, layernorms, n_layers, 0, tokenlength, x, ffnrbuffer);
    for (unsigned long long i = 0; i < n_layers; i++)
    { 
        layernorm(EMBSPLIT,EMBBLOCK, n_emb, x, layernorms, n_layers, 4*i + 2, tokenlength, buffer1, ffnrbuffer);
        mixatt(EMBSPLIT,n_emb, buffer1, statexy, mixk, mixv, mixr, ffnrbuffer, i, buffer2, buffer3, buffer4,n_layers, tokenlength, mode);
        cuda_mm8_threec(n_emb, ffnrbuffer, km, vm, rm, kr, vr, rr, o1, o2, o3, buffer2, buffer3, buffer4, i, tokenlength, MM8_ONE_JSPLIT);
        kernel_wkvc_forward(EMBSPLIT, n_emb, decay, bonus, buffer2, buffer3, buffer4, buffer1, stateaa, statebb, statepp, i, n_layers, tokenlength, mode);
        setx(buffer2, x, n_emb*tokenlength);
        cuda_mm8_one(n_emb, n_emb, buffer1, attout, attoutr, attouto, buffer2, i, tokenlength, MM8_ONE_JSPLIT);
        setx(x, buffer2, n_emb*tokenlength);
        layernorm(EMBSPLIT,EMBBLOCK, n_emb, x, layernorms, n_layers, 4 * (i + 1), tokenlength, buffer1, ffnrbuffer);
        mixffn(EMBSPLIT, n_emb, buffer1, statedd, ffnmixk, ffnmixv, ffnkbuffer, ffnvbuffer, i, n_layers, tokenlength, mode);
        setx(buffer2, 0.0, n_emb*tokenlength);
        cuda_mm8_one(n_emb, n_emb, ffnvbuffer, ffnr, ffnrr, ffnro, buffer2, i, tokenlength, MM8_ONE_JSPLIT);
        setx(ffnrbuffer, 0.0, 4*n_emb*tokenlength);
        cuda_mm8_one(n_emb, n_emb * 4, ffnkbuffer, ffnk, ffnkr, ffnko, ffnrbuffer, i, tokenlength, MM8_ONE_JSPLIT);
        sigmoidrelusquared(buffer2, ffnrbuffer, n_emb);
        setx(buffer3, 0.0, n_emb*tokenlength);
        cuda_mm8_one(n_emb * 4, n_emb, ffnrbuffer, ffnv, ffnvr, ffnvo, buffer3, i, tokenlength, MM8_ONE_JSPLIT);
        blockout(x, buffer3, buffer2, n_emb);   
        
    }

    layernorm(EMBSPLIT,EMBBLOCK, n_emb, x, layernorms, n_layers,  4 * (n_layers) + 2, tokenlength, buffer1, ffnrbuffer);
    setx(buffer2, 0.0, tokenlength*50277);
    cuda_mm8_one(n_emb, 50277, buffer1, head, headr, heado, buffer2, 0, tokenlength, MM8_ONE_JSPLIT);
    
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
    cudaSetDevice(0);
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
        if(i == 1){
            // add some extra to the end of the embedding table
            // vulkan needs this as you cant partially copy a buffer to the gpu
            ptrs[i] = (int *)malloc((size + maxGPT*n_embed) * Mtypes(i));
        }else{
            ptrs[i] = (int *)malloc(size * Mtypes(i));
        }

        std::cout << "loading: " << getName(i) << "\n";

        binfile.read((char *)ptrs[i], size * Mtypes(i));

        
    }

    binfile.close();

 
    return std::make_tuple(n_layers, n_embed);
}

void freeTensors(int **ptrs)
{
    for (unsigned long long i = 0; i < 46; i++)
    {
        delete[] ptrs[i]
        
    }
}


// #include "rwkv.h"
// int main(void)
// {
//     // assign a device to the thread
//     // allocate memory on the device

//     cudaSetDevice(0);
//     float* test;
//     cudaMalloc((void**)&test, 100*sizeof(float));
//     setx(test, 1.0, 100, 16, 16);
//     logData(test, 100, "test");
//     // int* ptrs[46];
//     // unsigned long long n_layers, n_embed;
//     // std::tie(n_layers, n_embed) = load("model.bin", ptrs, 1);

//     // unsigned long long tokens[1] = {4118};
//     // cuda_rwkv_parralel(n_layers, n_embed, tokens, (double *)(ptrs[X]),
//     //               (float *)(ptrs[EMBED]), (double *)(ptrs[LAYERNORMS]),
//     //               (double *)(ptrs[STATEXY]), (double *)(ptrs[STATEAA]), (double *)(ptrs[STATEBB]), (double *)(ptrs[STATEPP]), (double *)(ptrs[STATEDD]),
//     //               (double *)(ptrs[BUFFER1]), (float *)(ptrs[BUFFER2]), (float *)(ptrs[BUFFER3]), (float *)(ptrs[BUFFER4]),
//     //               (double *)(ptrs[MIXK]), (double *)(ptrs[MIXV]), (double *)(ptrs[MIXR]),
//     //               (uint8_t *)(ptrs[KM]), (uint8_t *)(ptrs[VM]), (uint8_t *)(ptrs[RM]),
//     //               (float *)(ptrs[KR]), (float *)(ptrs[VR]), (float *)(ptrs[RR]),
//     //               (float *)(ptrs[O1]), (float *)(ptrs[O2]), (float *)(ptrs[O3]),
//     //               (uint8_t *)(ptrs[ATTOUT]), (float *)(ptrs[ATTOUTR]), (float *)(ptrs[ATTOUTO]),
//     //               (double *)(ptrs[FFNMIXK]), (double *)(ptrs[FFNMIXV]),
//     //               (uint8_t *)(ptrs[FFNK]), (uint8_t *)(ptrs[FFNV]), (uint8_t *)(ptrs[FFNR]),
//     //               (float *)(ptrs[FFNKR]), (float *)(ptrs[FFNVR]), (float *)(ptrs[FFNRR]),
//     //               (float *)(ptrs[FFNKO]), (float *)(ptrs[FFNVO]), (float *)(ptrs[FFNRO]),
//     //               (double *)(ptrs[FFNKBUFFER]), (double *)(ptrs[FFNVBUFFER]), (float *)(ptrs[FFNRBUFFER]),
//     //               (double *)(ptrs[DECAY]), (double *)(ptrs[BONUS]),
//     //               (uint8_t *)(ptrs[HEAD]), (float *)(ptrs[HEADR]), (float *)(ptrs[HEADO]), 1, GPT);

// }
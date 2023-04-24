#if !defined(RWKV_H)
#define RWKV_H
#include <cstdint>
#include <string>
#include <iostream>
#include <tuple>
enum {X,EMBED,LAYERNORMS,STATEXY,STATEAA,STATEBB,STATEPP,STATEDD,BUFFER1,BUFFER2,BUFFER3,BUFFER4,MIXK,MIXV,MIXR,KM,VM,RM,KR,VR,RR,O1,O2,O3,ATTOUT,ATTOUTR,ATTOUTO,FFNMIXK,FFNMIXV,FFNK,FFNV,FFNR,FFNKR,FFNVR,FFNRR,FFNKO,FFNVO,FFNRO,FFNKBUFFER,FFNVBUFFER,FFNRBUFFER,DECAY,BONUS,HEAD,HEADR,HEADO};
char* names[46] = {
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

// cuda hooks 

// load from file
std::tuple<int64_t,int64_t> load (const std::string& filename, int** ptrs);
void setState(int64_t n_embed, int64_t n_layers,
    double* stateaa, double* statebb, double* statecc, double* statedd, double* stateee,
    double* instateaa, double* instatebb, double* instatecc, double* instatedd, double* instateee);

// copy the state into the gpu
void setState(int64_t n_embed, int64_t n_layers,
    double* stateaa, double* statebb, double* statecc, double* statedd, double* stateee,
    double* instateaa, double* instatebb, double* instatecc, double* instatedd, double* instateee);

// copy the logits to the output
void getOutput(int64_t n_embed, int64_t n_layers, float* logitsin, double* statexyin, double* stateaain, double* statebbin, double* stateppin, double* stateddin, 
        float* logitsout, double* statexyout, double* stateaaout, double* statebbout, double* stateppout, double* stateddout);

const unsigned long f = sizeof(float);
const unsigned long d = sizeof(double);
const unsigned long g = sizeof(uint8_t);

// the byte sizes of the tensors
const unsigned long types[46] = {d,f,d,d,d,d,d,d,d,f,f,f,d,d,d,g,g,g,f,f,f,f,f,f,g,f,f,d,d,g,g,g,f,f,f,f,f,f,d,d,f,d,d,g,f,f};

// forward pass
void cuda_rwkv(int64_t n_layers, int64_t n_emb, int64_t token, double* x, 
    float* embed, double* layernorms, 
    double* statexy, double* stateaa, double* statebb, double* statepp, double* statedd,
    double* buffer1, float* buffer2, float* buffer3, float* buffer4,
    double* mixk, double* mixv, double* mixr,
    uint8_t* km, uint8_t* vm, uint8_t* rm,
    float* kr, float* vr, float* rr,
    float* o1, float* o2, float* o3,
    uint8_t* attout, float* attoutr, float* attouto,
    double* ffnmixk, double* ffnmixv,
    uint8_t* ffnk, uint8_t* ffnv, uint8_t* ffnr, 
    float* ffnkr, float* ffnvr, float* ffnrr, 
    float* ffnko, float* ffnvo, float* ffnro,
    double* ffnkbuffer, double* ffnvbuffer, float* ffnrbuffer,
    double* decay, double* bonus,
    uint8_t* head, float* headr, float* heado
    );


int64_t getSize(int64_t i, int64_t a, int64_t b) {
    int64_t sizes[46] = {b,50277*b,4*(a+1)*b, a*b,a*b,a*b,a*b,a*b, b, 50277,b,b,a*b,a*b,a*b,a*b*b,a*b*b,a*b*b,a*b,a*b,a*b,a*b,a*b,a*b,a*b*b,a*b,a*b,a*b,a*b,a*b*b*4,a*b*b*4,a*b*b,a*b,a*b*4,a*b,a*b,a*b*4,a*b,b,b,b*4,a*b,a*b,50277*b,b,b};
        return sizes[i];
    }

unsigned long Mtypes(int64_t i) {
    return types[i];
}

char* getName(int64_t i) {
    return names[i];
}

class RWKV {
    public:
        // Tensor pointers
        int** tensors = new int*[46];

        // Number of layers in model
        int num_layers = 0;

        // Number of elements per embedding
        int num_embed = 0;

        // Cpu tensor for reading logits
        float* out = new float[50277];

        // Cpu state tensors
        double* statexy;
        double* stateaa;
        double* statebb;
        double* statepp;
        double* statedd;

        RWKV() {
            
        };

        // Load from .bin file
        void loadFile(const std::string& filename) {
            std::tie(num_layers, num_embed) = load(filename, tensors);
            statexy = new double[num_layers*num_embed];
            stateaa = new double[num_layers*num_embed];
            statebb = new double[num_layers*num_embed];
            statepp = new double[num_layers*num_embed];
            statedd = new double[num_layers*num_embed];
            for (int i = 0; i < num_layers*num_embed; i++) {
                statexy[i] = 0;
                stateaa[i] = 0;
                statebb[i] = 0;
                statepp[i] = 0;
                statedd[i] = 0;
            }
            for (int i = 0; i < 50277; i++) {
                out[i] = 0;
            }
        };

        // Get number of elements in a tensor
        int64_t getTensorSize(int64_t i) {
            return getSize(i, num_layers, num_embed);
        };

        // Get the bytesize of a tensor
        int64_t getTensorTypes(int64_t i) {
            return types[i];
        };

        

        float* forward(int64_t token){

            // copy the state into the gpu
            setState(num_layers, num_embed, (double*)(tensors[STATEXY]), (double*)(tensors[STATEAA]), (double*)(tensors[STATEBB]), (double*)(tensors[STATEPP]), (double*)(tensors[STATEDD]), statexy, stateaa, statebb, statepp, statedd);

            cuda_rwkv(num_layers, num_embed, token, (double*)(tensors[X]),
            (float*)(tensors[EMBED]), (double*)(tensors[LAYERNORMS]),
            (double*)(tensors[STATEXY]), (double*)(tensors[STATEAA]), (double*)(tensors[STATEBB]), (double*)(tensors[STATEPP]), (double*)(tensors[STATEDD]),
            (double*)(tensors[BUFFER1]), (float*)(tensors[BUFFER2]), (float*)(tensors[BUFFER3]), (float*)(tensors[BUFFER4]),
            (double*)(tensors[MIXK]), (double*)(tensors[MIXV]), (double*)(tensors[MIXR]),
            (uint8_t*)(tensors[KM]), (uint8_t*)(tensors[VM]), (uint8_t*)(tensors[RM]),
            (float*)(tensors[KR]), (float*)(tensors[VR]), (float*)(tensors[RR]),
            (float*)(tensors[O1]), (float*)(tensors[O2]), (float*)(tensors[O3]),
            (uint8_t*)(tensors[ATTOUT]), (float*)(tensors[ATTOUTR]), (float*)(tensors[ATTOUTO]),
            (double*)(tensors[FFNMIXK]), (double*)(tensors[FFNMIXV]),
            (uint8_t*)(tensors[FFNK]), (uint8_t*)(tensors[FFNV]), (uint8_t*)(tensors[FFNR]), 
            (float*)(tensors[FFNKR]), (float*)(tensors[FFNVR]), (float*)(tensors[FFNRR]), 
            (float*)(tensors[FFNKO]), (float*)(tensors[FFNVO]), (float*)(tensors[FFNRO]), 
            (double*)(tensors[FFNKBUFFER]), (double*)(tensors[FFNVBUFFER]), (float*)(tensors[FFNRBUFFER]),
            (double*)(tensors[DECAY]), (double*)(tensors[BONUS]),
            (uint8_t*)(tensors[HEAD]), (float*)(tensors[HEADR]), (float*)(tensors[HEADO])
            );

            
            getOutput(num_layers, num_embed, (float*)(tensors[BUFFER2]), (double*)(tensors[STATEXY]), (double*)(tensors[STATEAA]), (double*)(tensors[STATEBB]), (double*)(tensors[STATEPP]), (double*)(tensors[STATEDD]),
                        out, statexy, stateaa, statebb, statepp, statedd
            );

            return out;
        }

};

#endif

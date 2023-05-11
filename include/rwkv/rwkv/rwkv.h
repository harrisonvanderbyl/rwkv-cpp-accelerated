#if !defined(RWKV_H)
#define RWKV_H
#include <cstdint>
#include <string>
#include <iostream>
#include <tuple>
#include <vector>
#include "rwkv/enums/enum.h"

std::string names[46] = {
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
    "head_o"};

// cuda hooks



// load from file
std::tuple<unsigned long long, unsigned long long> load(const std::string &filename, int **ptrs, unsigned long long maxGPT);
void setState(unsigned long long n_embed, unsigned long long n_layers,
              double *stateaa, double *statebb, double *statecc, double *statedd, double *stateee,
              double *instateaa, double *instatebb, double *instatecc, double *instatedd, double *instateee, unsigned long long tokenlength);

// copy the state into the gpu
void setState(unsigned long long n_embed, unsigned long long n_layers,
              double *stateaa, double *statebb, double *statecc, double *statedd, double *stateee,
              double *instateaa, double *instatebb, double *instatecc, double *instatedd, double *instateee, unsigned long long tokenlength);

// copy the logits to the output
void getOutput(unsigned long long n_embed, unsigned long long n_layers, float *logitsin, double *statexyin, double *stateaain, double *statebbin, double *stateppin, double *stateddin,
               float *logitsout, double *statexyout, double *stateaaout, double *statebbout, double *stateppout, double *stateddout, unsigned long long tokenlength);

void freeTensors(int **ptrs);

const unsigned long f = sizeof(float);
const unsigned long d = sizeof(double);
const unsigned long g = sizeof(uint8_t);

// the byte sizes of the tensors
const unsigned long types[46] = {d, f, d, d, d, d, d, d, d, f, f, f, d, d, d, g, g, g, f, f, f, f, f, f, g, f, f, d, d, g, g, g, f, f, f, f, f, f, d, d, f, d, d, g, f, f};

// forward pass
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
               uint8_t *head, float *headr, float *heado);

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
               unsigned long long tokenlength, 
               MODE mode
               );

unsigned long long getSize(unsigned long long i, unsigned long long a, unsigned long long b)
{
    unsigned long long sizes[46] = {b, 50277 * b, 4 * (a + 1) * b, a * b, a * b, a * b, a * b, a * b, b, 50277, b, b, a * b, a * b, a * b, a * b * b, a * b * b, a * b * b, a * b, a * b, a * b, a * b, a * b, a * b, a * b * b, a * b, a * b, a * b, a * b, a * b * b * 4, a * b * b * 4, a * b * b, a * b, a * b * 4, a * b, a * b, a * b * 4, a * b, b, b, b * 4, a * b, a * b, 50277 * b, b, b};
    return sizes[i];
}

unsigned long long Mtypes(unsigned long long i)
{
    return types[i];
}

const char *getName(unsigned long long i)
{
    return names[i].c_str();
}

class RWKVState
{
    public:
    double *statexy;
    double *stateaa;
    double *statebb;
    double *statepp;
    double *statedd;
    unsigned long long num_layers;
    unsigned long long num_embed;
    unsigned long long stateSize;

    RWKVState(unsigned long long num_layers, unsigned long long num_embed, unsigned long long stateSize)
    {
        this->num_layers = num_layers;
        this->num_embed = num_embed;
        this->stateSize = stateSize;
        statexy = new double[num_layers * num_embed * stateSize];
        stateaa = new double[num_layers * num_embed * stateSize];
        statebb = new double[num_layers * num_embed * stateSize];
        statepp = new double[num_layers * num_embed * stateSize];
        statedd = new double[num_layers * num_embed * stateSize];

        for (unsigned long long i = 0; i < num_layers * num_embed * stateSize; i++)
        {
            statexy[i] = 0;
            stateaa[i] = 0;
            statebb[i] = 0;
            statepp[i] = 0;
            statedd[i] = 0;
        }
    }

    RWKVState(RWKVState &other){
        this->num_layers = other.num_layers;
        this->num_embed = other.num_embed;
        this->stateSize = other.stateSize;
        statexy = new double[num_layers * num_embed * stateSize];
        stateaa = new double[num_layers * num_embed * stateSize];
        statebb = new double[num_layers * num_embed * stateSize];
        statepp = new double[num_layers * num_embed * stateSize];
        statedd = new double[num_layers * num_embed * stateSize];

        for (unsigned long long i = 0; i < num_layers * num_embed * stateSize; i++)
        {
            statexy[i] = other.statexy[i];
            stateaa[i] = other.stateaa[i];
            statebb[i] = other.statebb[i];
            statepp[i] = other.statepp[i];
            statedd[i] = other.statedd[i];
        }
    }

    RWKVState(RWKVState &other, unsigned long long offset){
        this->num_layers = other.num_layers;
        this->num_embed = other.num_embed;
        this->stateSize = 1;
        statexy = new double[num_layers * num_embed * stateSize];
        stateaa = new double[num_layers * num_embed * stateSize];
        statebb = new double[num_layers * num_embed * stateSize];
        statepp = new double[num_layers * num_embed * stateSize];
        statedd = new double[num_layers * num_embed * stateSize];

        for (unsigned long long i = 0; i < num_layers * num_embed * stateSize; i++)
        {
            statexy[i] = other.statexy[i + offset];
            stateaa[i] = other.stateaa[i + offset];
            statebb[i] = other.statebb[i + offset];
            statepp[i] = other.statepp[i + offset];
            statedd[i] = other.statedd[i + offset];
        }
    }

    ~RWKVState()
    {
        delete[] statexy;
        delete[] stateaa;
        delete[] statebb;
        delete[] statepp;
        delete[] statedd;
    }

    // Get a substate
    RWKVState getSubState(unsigned long long offset = 0){
        if (offset >= stateSize){
            throw std::runtime_error("State get offset out of bounds, max offset is " + std::to_string(stateSize));
        }
        return RWKVState(*this, offset);
    }

    // Set a substate
    void setSubState(RWKVState &other, unsigned long long offset = 0){
        for (unsigned long long i = 0; i < num_layers * num_embed; i++)
        {
            statexy[i + offset * num_layers * num_embed] = other.statexy[i];
            stateaa[i + offset * num_layers * num_embed] = other.stateaa[i];
            statebb[i + offset * num_layers * num_embed] = other.statebb[i];
            statepp[i + offset * num_layers * num_embed] = other.statepp[i];
            statedd[i + offset * num_layers * num_embed] = other.statedd[i];
        }
    }

};

#include "rwkv/tokenizer/tokenizer.h"
class RWKV
{
public:
    // Tensor pointers
    int **tensors = new int *[46];

    // Number of layers in model
    unsigned long long num_layers = 0;

    // Number of elements per embedding
    unsigned long long num_embed = 0;

    // Cpu tensor for reading logits
    float *out;

    unsigned long long maxContext = 1;

    // Cpu state tensors
    RWKVState *state;

    GPT2Tokenizer *tokenizer;

    bool ready = false;

    // Compatibility
    double* statexy;
    double* stateaa;
    double* statebb;
    double* statepp;
    double* statedd;

    RWKV(){

    };

    // Load from .bin file
    void loadFile(const std::string &filename, unsigned long long maxGPT = 1)
    {
        if (ready)
        {
            throw std::runtime_error("RWKV already loaded");
        }

        std::tie(num_layers, num_embed) = load(filename, tensors, maxGPT);
        state = new RWKVState(num_layers, num_embed, maxGPT);

        // Deprecated, compatibility layer
        statexy = state->statexy;
        stateaa = state->stateaa;
        statebb = state->statebb;
        statepp = state->statepp;
        statedd = state->statedd;

        out = new float[50277 * maxGPT];
        
        for (unsigned long long i = 0; i < 50277 * maxGPT; i++)
        {
            out[i] = 0;
        }



        maxContext = maxGPT;

        ready = true;
    };

    void loadTokenizer(std::string vocabPath){
        auto _tokenizer = GPT2Tokenizer::load(vocabPath+"/vocab.json", vocabPath+"/merges.txt");
        if (!_tokenizer.has_value()) {
            std::cerr << "Failed to load tokenizer" << std::endl;
            return;
        };
        tokenizer = new GPT2Tokenizer(_tokenizer.value());
    }


    

    // Get number of elements in a tensor
    unsigned long long getTensorSize(unsigned long long i)
    {
        return getSize(i, num_layers, num_embed);
    };

    // Get the bytesize of a tensor
    unsigned long long getTensorTypes(unsigned long long i)
    {
        return types[i];
    };

    


    float *forward(std::vector<unsigned long long> token, MODE mode)
    {
        
        if (!ready)
        {
            throw std::runtime_error("RWKV not loaded");
        }

        if (token.size() > maxContext)
        {
            throw std::runtime_error("Context too large, max context is " + std::to_string(maxContext));
        }

        // copy the state into the gpu
        setState(num_layers, num_embed, (double *)(tensors[STATEXY]), (double *)(tensors[STATEAA]), (double *)(tensors[STATEBB]), (double *)(tensors[STATEPP]), (double *)(tensors[STATEDD]), state->statexy, state->stateaa, state->statebb, state->statepp, state->statedd, token.size());

        cuda_rwkv_parralel(num_layers, num_embed, token.data(), (double *)(tensors[X]),
                  (float *)(tensors[EMBED]), (double *)(tensors[LAYERNORMS]),
                  (double *)(tensors[STATEXY]), (double *)(tensors[STATEAA]), (double *)(tensors[STATEBB]), (double *)(tensors[STATEPP]), (double *)(tensors[STATEDD]),
                  (double *)(tensors[BUFFER1]), (float *)(tensors[BUFFER2]), (float *)(tensors[BUFFER3]), (float *)(tensors[BUFFER4]),
                  (double *)(tensors[MIXK]), (double *)(tensors[MIXV]), (double *)(tensors[MIXR]),
                  (uint8_t *)(tensors[KM]), (uint8_t *)(tensors[VM]), (uint8_t *)(tensors[RM]),
                  (float *)(tensors[KR]), (float *)(tensors[VR]), (float *)(tensors[RR]),
                  (float *)(tensors[O1]), (float *)(tensors[O2]), (float *)(tensors[O3]),
                  (uint8_t *)(tensors[ATTOUT]), (float *)(tensors[ATTOUTR]), (float *)(tensors[ATTOUTO]),
                  (double *)(tensors[FFNMIXK]), (double *)(tensors[FFNMIXV]),
                  (uint8_t *)(tensors[FFNK]), (uint8_t *)(tensors[FFNV]), (uint8_t *)(tensors[FFNR]),
                  (float *)(tensors[FFNKR]), (float *)(tensors[FFNVR]), (float *)(tensors[FFNRR]),
                  (float *)(tensors[FFNKO]), (float *)(tensors[FFNVO]), (float *)(tensors[FFNRO]),
                  (double *)(tensors[FFNKBUFFER]), (double *)(tensors[FFNVBUFFER]), (float *)(tensors[FFNRBUFFER]),
                  (double *)(tensors[DECAY]), (double *)(tensors[BONUS]),
                  (uint8_t *)(tensors[HEAD]), (float *)(tensors[HEADR]), (float *)(tensors[HEADO]), token.size(), mode);

        getOutput(num_layers, num_embed, (float *)(tensors[BUFFER2]), (double *)(tensors[STATEXY]), (double *)(tensors[STATEAA]), (double *)(tensors[STATEBB]), (double *)(tensors[STATEPP]), (double *)(tensors[STATEDD]),
                  out, state->statexy, state->stateaa, state->statebb, state->statepp, state->statedd, token.size());

        return out;
    }

    float* forward(unsigned long long token){
        return forward(std::vector<unsigned long long>{token}, GPT);
    }

    float *forward(std::vector<long long> token, MODE mode){
        std::vector<unsigned long long> token2;
        for(long long i : token){
            token2.push_back(i);
        }
        return forward(token2, mode);
    }

    RWKVState emptyState(){
        // This is simultaniously the most fucked and the most useful thing
        return {num_layers, num_embed, 1};
    }

    long long loadContext(std::string input, bool progress = false){
        std::vector<long long> initial = tokenizer->encode(input);auto tokens = tokenizer->encode(input);
        std::cout << initial[0] << ":token";
        for(int i = 0; i < initial.size(); i+=maxContext)
        {
        
            // load initial as .subvector(i -> i+maxContext)
            auto mvec = std::vector<unsigned long long>(initial.begin()+i, initial.begin()+(std::min((size_t)(i+maxContext), initial.size())));
            forward(mvec, GPT);
            // delete last progress
            if(progress){
                std::cout << "\r";
                std::cout << int(float(i)/initial.size()*100) << "%";
                std::flush(std::cout);
            }
        }

        return initial[initial.size()-1];
    }

    // destructor
    ~RWKV()
    {
        delete[] out;

        if (!ready)
        {
            return;
        }

        freeTensors(tensors);
        delete[] tensors;
        delete state;
    };
};

#endif

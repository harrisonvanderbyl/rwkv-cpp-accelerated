#include <torch/extension.h>
#include "ATen/ATen.h"
#include "../include/rwkv/enums/enum.h"
#include <iostream>
#include <fstream>
// generic T either float or fp16 or fp64

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
const unsigned long f = sizeof(float);
const unsigned long d = sizeof(double);
const unsigned long g = sizeof(uint8_t);

// the byte sizes of the tensors
const unsigned long types[46] = {d, f, d, d, d, d, d, d, d, f, f, f, d, d, d, g, g, g, f, f, f, f, f, f, g, f, f, d, d, g, g, g, f, f, f, f, f, f, d, d, f, d, d, g, f, f};
unsigned long long getSize(unsigned long long i, unsigned long long a, unsigned long long b, unsigned long long vocab)
{
    unsigned long long sizes[46] = {b, vocab * b, 4 * (a + 1) * b, a * b, a * b, a * b, a * b, a * b, b, vocab, b, b, a * b, a * b, a * b, a * b * b, a * b * b, a * b * b, a * b, a * b, a * b, a * b, a * b, a * b, a * b * b, a * b, a * b, a * b, a * b, a * b * b * 4, a * b * b * 4, a * b * b, a * b, a * b * 4, a * b, a * b, a * b * 4, a * b, b, b, b * 4, a * b, a * b, vocab * b, b, b};
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
void save(std::string outfile,int64_t n_layers, int64_t n_emb, int64_t vocab, torch::Tensor &x, 
    torch::Tensor &embed, torch::Tensor &layernorms, 
    torch::Tensor &statexy, torch::Tensor &stateaa, torch::Tensor &statebb, torch::Tensor &statepp, torch::Tensor &statedd,
    torch::Tensor &buffer1, torch::Tensor &buffer2, torch::Tensor &buffer3, torch::Tensor &buffer4,
    torch::Tensor &mixk, torch::Tensor &mixv, torch::Tensor &mixr,
    torch::Tensor &km, torch::Tensor &vm, torch::Tensor &rm,
    torch::Tensor &kr, torch::Tensor &vr, torch::Tensor &rr,
    torch::Tensor &o1, torch::Tensor &o2, torch::Tensor &o3,
    torch::Tensor &attout, torch::Tensor &attoutr, torch::Tensor &attouto,
    torch::Tensor &ffnmixk, torch::Tensor &ffnmixv,
    torch::Tensor &ffnk, torch::Tensor &ffnv, torch::Tensor &ffnr, 
    torch::Tensor &ffnkr, torch::Tensor &ffnvr, torch::Tensor &ffnrr, 
    torch::Tensor &ffnko, torch::Tensor &ffnvo, torch::Tensor &ffnro, 
    torch::Tensor &ffnkbuffer, torch::Tensor &ffnvbuffer, torch::Tensor &ffnrbuffer,

    torch::Tensor &decay, torch::Tensor &bonus,
    torch::Tensor &head, torch::Tensor &headr, torch::Tensor &heado
    ) {
    
    
    std::cout << "vocab: " << vocab << "\n";
    if(vocab < 10000){
        std::cout << "Warning: vocab is less than 10000, this is probably not what you want\n";
        throw "error";
    }
    torch::Tensor* tensors[46] = {
        &x,
        &embed,
        &layernorms,
        &statexy,
        &stateaa,
        &statebb,
        &statepp,
        &statedd,
        &buffer1,
        &buffer2,
        &buffer3,
        &buffer4,
        &mixk,
        &mixv,
        &mixr,
        &km,
        &vm,
        &rm,
        &kr,
        &vr,
        &rr,
        &o1,
        &o2,
        &o3,
        &attout,
        &attoutr,
        &attouto,
        &ffnmixk,
        &ffnmixv,
        &ffnk,
        &ffnv,
        &ffnr,
        &ffnkr,
        &ffnvr,
        &ffnrr,
        &ffnko,
        &ffnvo,
        &ffnro,
        &ffnkbuffer,
        &ffnvbuffer,
        &ffnrbuffer,
        &decay,
        &bonus,
        &head,
        &headr,
        &heado
    };
    std::ofstream binfile;
    binfile.open(outfile, std::ios::out | std::ios::binary);
    binfile.write((char*)&n_layers, sizeof(int64_t));
    binfile.write((char*)&n_emb, sizeof(int64_t));
    for(int i = 0; i < 46; i++) {
        std::cout << "saving: " << names[i] << "\n";
        assert(getSize(i,n_layers,n_emb, vocab) == tensors[i]->numel());
        if(types[i] == sizeof(double)){
            assert(tensors[i]->scalar_type() == torch::kDouble);
        }
        else if (types[i] == sizeof(float)){
            assert(tensors[i]->scalar_type() == torch::kFloat);
        }
        else if (types[i] == sizeof(uint8_t)){
            assert(tensors[i]->scalar_type() == torch::kByte);
        }
        // to cpu
        binfile.write((char*)tensors[i]->data_ptr(), (unsigned long long)tensors[i]->numel() * (unsigned long long)types[i]);
        // delete cputensor;
    }
    
    binfile.close();

    }

void freePtr (torch::Tensor &t){
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("save", &save, "save");
    m.def("freePtr", &freePtr, "freePtr");
}

TORCH_LIBRARY(rwkv, m) {
    m.def("save", save);
    m.def("freePtr", freePtr);
}


#include <torch/extension.h>
#include "ATen/ATen.h"
#include "rwkv.h"
#include <iostream>
// generic T either float or fp16 or fp64


void save(std::string outfile,int64_t n_layers, int64_t n_emb, torch::Tensor &x, 
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
        assert(getSize(i,n_layers,n_emb) == tensors[i]->numel());
        if(types[i] == sizeof(double)){
            assert(tensors[i]->scalar_type() == torch::kDouble);
        }
        else if (types[i] == sizeof(float)){
            assert(tensors[i]->scalar_type() == torch::kFloat);
        }
        else if (types[i] == sizeof(uint8_t)){
            assert(tensors[i]->scalar_type() == torch::kByte);
        }
        binfile.write((char*)tensors[i]->data_ptr(), (unsigned long long)tensors[i]->numel() * (unsigned long long)types[i]);
        
    }
    
    binfile.close();

    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("save", &save, "save");
}

TORCH_LIBRARY(rwkv, m) {
    m.def("save", save);
}


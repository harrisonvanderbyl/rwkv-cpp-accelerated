from tkinter import filedialog
import tkinter
import torch
from tqdm import tqdm

def OptRWKV(path):
    from torch.utils.cpp_extension import load_inline
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))
    load_inline(
        name=f"wkv_cuda_export",
        cpp_sources="""#include <torch/extension.h>
#include "ATen/ATen.h"
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
        binfile.write((char*)tensors[i]->data_ptr(), tensors[i]->numel() * types[i]);
        
    }
    
    binfile.close();

    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("save", &save, "save");
}

TORCH_LIBRARY(rwkv, m) {
    m.def("save", save);
}
""",
        )
    

    class myRWKV(torch.nn.Module):
        def QuantizeMatrix(self, xx):
                x = xx
                rang = 255
                mini = x.min(0)[0].double()
                out = x-mini
                ran = out.max(0)[0].double()/rang
                out = out/ran
                fractmat = out.frac().double()
                fractmat = fractmat.mean(0)
                mini = mini.double() + fractmat*ran.double()
                
                return [out.t().to(torch.uint8).contiguous(),ran.to(torch.float32).clone(), mini.to(torch.float32).clone()]
        def __init__(self,w,dims,layers):
            super(myRWKV, self).__init__()
            
            self.emptyState = layers * [[[0]*dims]*4+[[-1e30]*dims]]
            vustatea = torch.tensor([self.emptyState[i][0] for i in range(layers)]).double().contiguous()
            vustateb = torch.tensor([self.emptyState[i][2] for i in range(layers)]).double().contiguous()
            vustatec = torch.tensor([self.emptyState[i][3] for i in range(layers)]).double().contiguous()
            vustated = torch.tensor([self.emptyState[i][4] for i in range(layers)]).double().contiguous()
            vustatee = torch.tensor([self.emptyState[i][1] for i in range(layers)]).double().contiguous()
            self.emptyState = [vustatea,vustateb,vustatec,vustated,vustatee]

            self.emb =  w["emb.weight"].float().contiguous()
            
            
            sn = ["blocks.0.ln0.weight","blocks.0.ln0.bias"]
            for i in range(layers):
                sn.append(f"blocks.{i}.ln1.weight")
                sn.append(f"blocks.{i}.ln1.bias")
                sn.append(f"blocks.{i}.ln2.weight")
                sn.append(f"blocks.{i}.ln2.bias")
            sn += [
                "ln_out.weight",
                "ln_out.bias"
            ]
            
            self.cudalnin = torch.stack (
                [w[x] for x in sn]).double().contiguous()
            self.mixk = [w[f"blocks.{i}.att.time_mix_k"].squeeze() for i in range(layers)]
            self.mixk = torch.stack(self.mixk).double().contiguous()
            self.mixv = [w[f"blocks.{i}.att.time_mix_v"].squeeze() for i in range(layers)]
            self.mixv = torch.stack(self.mixv).double().contiguous()
            self.mixr = [w[f"blocks.{i}.att.time_mix_r"].squeeze() for i in range(layers)]
            self.mixr = torch.stack(self.mixr).double().contiguous()
            self.mixffnk = [w[f"blocks.{i}.ffn.time_mix_k"].squeeze() for i in range(layers)]
            self.mixffnk = torch.stack(self.mixffnk).double().contiguous()
            self.mixffnr = [w[f"blocks.{i}.ffn.time_mix_r"].squeeze() for i in range(layers)]
            self.mixffnr = torch.stack(self.mixffnr).double().contiguous()
            self.decay = [w[f"blocks.{i}.att.time_decay"].squeeze() for i in range(layers)]
            self.decay = torch.stack(self.decay).double().contiguous().exp().neg()
            self.bonus = [w[f"blocks.{i}.att.time_first"].squeeze() for i in range(layers)]
            self.bonus = torch.stack(self.bonus).double().contiguous()

            toQuantize = [
                "att.key.weight",
                "att.value.weight",
                "att.receptance.weight",
                "ffn.key.weight",
                "ffn.value.weight",
                "ffn.receptance.weight",
                "att.output.weight"
            ]
            for key in toQuantize:
                weights =[self.QuantizeMatrix(w[f"blocks.{i}.{key}"] ) for i in tqdm(range(layers), desc=f"Quantizing {key}")]
                print("stacking weights")
                keysp = key.split(".")[:2]
                keysp = "".join(keysp)                
                self.__dict__[keysp+"weights"] = torch.stack([x[0] for x in weights])
                self.__dict__[keysp+"ranges"] = torch.stack([x[1] for x in weights]).to(dtype=torch.float32, memory_format=torch.contiguous_format)
                self.__dict__[keysp+"zp"] = torch.stack([x[2] for x in weights]).to(dtype=torch.float32, memory_format=torch.contiguous_format)
                import gc
                for x in tqdm(range(layers), desc=f"Cleaning {key}"):
                    del w[f"blocks.{x}.{key}"]
                del weights 
                gc.collect()
                torch.cuda.empty_cache()
            
            
            self.cudahead, self.cudaheadr, self.cudaheadzp = self.QuantizeMatrix(w["head.weight"])
            self.cudahead = self.cudahead.to(dtype=torch.uint8, memory_format=torch.contiguous_format)
            del w

            self.dim = dims
            self.layers = layers
            
            self.rx = torch.arange((self.dim), dtype = torch.double)
            self.buffer0 = torch.arange(self.dim, dtype=torch.double)
            self.buffer1 = torch.arange(50277, dtype=torch.float)
            self.buffer2 = torch.arange(self.dim, dtype=torch.float)
            self.buffer3 = torch.arange(self.dim, dtype=torch.float)
            self.ffnvbuf = torch.arange(self.dim, dtype=torch.double)
            self.ffnkbuf = torch.arange(self.dim, dtype=torch.double)
            self.ffkeybuffer = torch.arange(self.dim*4, dtype=torch.float)       

            
        def forward(self):
            # save as .bin file
            # get currentpath ..
            outfile = current_path+"/model.bin"

            torch.ops.rwkv.save(
                outfile,
                self.layers,
                self.dim,
                self.rx,
                self.emb,
                self.cudalnin,
                self.emptyState[0],
                self.emptyState[1],
                self.emptyState[2],
                self.emptyState[3],
                self.emptyState[4],
                self.buffer0,
                self.buffer1,
                self.buffer2,
                self.buffer3,
                self.mixk,
                self.mixv,
                self.mixr,
                self.attkeyweights,
                self.attvalueweights,
                self.attreceptanceweights,
                self.attkeyranges,
                self.attvalueranges,
                self.attreceptanceranges,
                self.attkeyzp,
                self.attvaluezp,
                self.attreceptancezp,
                self.attoutputweights,
                self.attoutputranges,
                self.attoutputzp,
                self.mixffnk,
                self.mixffnr,
                self.ffnkeyweights,
                self.ffnvalueweights,
                self.ffnreceptanceweights,
                self.ffnkeyranges,
                self.ffnvalueranges,
                self.ffnreceptanceranges,
                self.ffnkeyzp,
                self.ffnvaluezp,
                self.ffnreceptancezp,
                self.ffnkbuf,
                self.ffnvbuf,
                self.ffkeybuffer,
                self.decay,
                self.bonus,
                self.cudahead,
                self.cudaheadr,
                self.cudaheadzp
            )

    w = torch.load(path, map_location="cpu")
    # detach weights

    dims = len(w["blocks.0.att.key.weight"])
    layers = len(
        list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))

    myrwkv = myRWKV(w,dims,layers)
    myrwkv.forward()
    exit()

# ui library for choosing file
def chooseFile():
    # only .pth files
    root = tkinter.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("PyTorch Weights", "*.pth")])
    return file_path

OptRWKV(chooseFile())
import os
import gc
import sys
import argparse

import torch

from torch.utils.cpp_extension import load as torch_load_cpp
from tqdm.auto import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))


class ConvertRWKV(torch.nn.Module):

    def __init__(self, w, dims, layers, vocabsize):
        super().__init__()

        self.emptyState = layers * [[[0]*dims]*4+[[-1e30]*dims]]
        vustatea = torch.tensor([self.emptyState[i][0] for i in range(layers)]).double().contiguous()
        vustateb = torch.tensor([self.emptyState[i][2] for i in range(layers)]).double().contiguous()
        vustatec = torch.tensor([self.emptyState[i][3] for i in range(layers)]).double().contiguous()
        vustated = torch.tensor([self.emptyState[i][4] for i in range(layers)]).double().contiguous()
        vustatee = torch.tensor([self.emptyState[i][1] for i in range(layers)]).double().contiguous()
        self.vocabsize = vocabsize
        self.emptyState = [vustatea,vustateb,vustatec,vustated,vustatee]

        self.emb =  w["emb.weight"].float().contiguous()


        sn = ["blocks.0.ln0.weight","blocks.0.ln0.bias"]
        for i in range(layers):
            sn.extend(
                (
                    f"blocks.{i}.ln1.weight",
                    f"blocks.{i}.ln1.bias",
                    f"blocks.{i}.ln2.weight",
                    f"blocks.{i}.ln2.bias",
                )
            )
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

        # tqdm creates nice progress bar around the conversion process
        for key in tqdm(toQuantize, desc="Quantizing"):
            weights =[self.quantize_matrix(w[f"blocks.{i}.{key}"] ) for i in tqdm(range(layers), desc=f"Quantizing {key}")]
            print("stacking weights")
            keysp = key.split(".")[:2]
            keysp = "".join(keysp)
            self.__dict__[f"{keysp}weights"] = torch.stack([x[0] for x in weights]).cpu()
            self.__dict__[f"{keysp}ranges"] = torch.stack(
                [x[1] for x in weights]
            ).to(dtype=torch.float32, memory_format=torch.contiguous_format)
            self.__dict__[f"{keysp}zp"] = torch.stack([x[2] for x in weights]).to(
                dtype=torch.float32, memory_format=torch.contiguous_format
            )
            for x in tqdm(range(layers), desc=f"Cleaning {key}"):
                w[f"blocks.{x}.{key}"] = None
            del weights
            gc.collect()
            torch.cuda.empty_cache()


        self.cudahead, self.cudaheadr, self.cudaheadzp = self.quantize_matrix(w["head.weight"])
        self.cudahead = self.cudahead.to(dtype=torch.uint8, memory_format=torch.contiguous_format)
        del w

        self.dim = dims
        self.layers = layers

        self.rx = torch.arange((self.dim), dtype = torch.double)
        self.buffer0 = torch.arange(self.dim, dtype=torch.double)
        self.buffer1 = torch.arange(vocabsize, dtype=torch.float)
        self.buffer2 = torch.arange(self.dim, dtype=torch.float)
        self.buffer3 = torch.arange(self.dim, dtype=torch.float)
        self.ffnvbuf = torch.arange(self.dim, dtype=torch.double)
        self.ffnkbuf = torch.arange(self.dim, dtype=torch.double)
        self.ffkeybuffer = torch.arange(self.dim*4, dtype=torch.float)       

    def quantize_matrix(self, xx:torch.Tensor):
        x = xx
        rang = 255
        mini = x.min(0)[0].double()
        out = x-mini
        ran = out.max(0)[0].double()/rang
        out = out/ran
        fractmat = out.frac().double()
        fractmat = fractmat.mean(0)
        mini = mini.double() + fractmat*ran.double()
        
        return [out.t().to(torch.uint8).contiguous(),ran.to(torch.float32).clone().cpu(), mini.to(torch.float32).clone().cpu()]


    def save_converted(self):
        # save as .bin file
        # get currentpath ..
        assert hasattr(torch.ops, "rwkv"), "Save tensor cpp code is not loaded!"

        outfile = "{0}/model.bin".format(current_path)

        torch.ops.rwkv.save(
            outfile,
            self.layers,
            self.dim,
            self.vocabsize,
            self.rx.cpu(),
            self.emb.cpu(),
            self.cudalnin.cpu(),
            self.emptyState[0].cpu(),
            self.emptyState[1].cpu(),
            self.emptyState[2].cpu(),
            self.emptyState[3].cpu(),
            self.emptyState[4].cpu(),
            self.buffer0.cpu(),
            self.buffer1.cpu(),
            self.buffer2.cpu(),
            self.buffer3.cpu(),
            self.mixk.cpu(),
            self.mixv.cpu(),
            self.mixr.cpu(),
            self.attkeyweights.cpu(),
            self.attvalueweights.cpu(),
            self.attreceptanceweights.cpu(),
            self.attkeyranges.cpu(),
            self.attvalueranges.cpu(),
            self.attreceptanceranges.cpu(),
            self.attkeyzp.cpu(),
            self.attvaluezp.cpu(),
            self.attreceptancezp.cpu(),
            self.attoutputweights.cpu(),
            self.attoutputranges.cpu(),
            self.attoutputzp.cpu(),
            self.mixffnk.cpu(),
            self.mixffnr.cpu(),
            self.ffnkeyweights.cpu(),
            self.ffnvalueweights.cpu(),
            self.ffnreceptanceweights.cpu(),
            self.ffnkeyranges.cpu(),
            self.ffnvalueranges.cpu(),
            self.ffnreceptanceranges.cpu(),
            self.ffnkeyzp.cpu(),
            self.ffnvaluezp.cpu(),
            self.ffnreceptancezp.cpu(),
            self.ffnkbuf.cpu(),
            self.ffnvbuf.cpu(),
            self.ffkeybuffer.cpu(),
            self.decay.cpu(),
            self.bonus.cpu(),
            self.cudahead.cpu(),
            self.cudaheadr.cpu(),
            self.cudaheadzp.cpu()
        )


def is_valid_weights_file(weights: dict) -> bool:
    required_keys = [
        "emb.weight",
        "ln_out.weight",
        "ln_out.bias",
        "blocks.0.ln0.weight",
        "blocks.0.ln0.bias",
        # Add other keys as needed
    ]
    return all(key in weights for key in required_keys)



def convert_model(path: str):
    
    torch_load_cpp(
        name="wkv_cuda_export",
        sources=[os.path.join(current_path, "cpp_save_tensor.cpp")],
        # add ../include to include path
        extra_include_paths=[os.path.join(current_path, "../include/rwkv/rwkv")],
        )

    w = torch.load(path, map_location="cuda:1")
    
    # Verify the file contents
    if not is_valid_weights_file(w):
        print("Invalid weights file structure. Please provide a valid .pth file.")
        sys.exit(1)

    dims = len(w["blocks.0.att.key.weight"])
    layers = len(
        list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))
    
    vocabsize = w["emb.weight"].shape[0]
    print("Converting model...")
    print(f"Dimensions: {dims}")
    print(f"Layers: {layers}")
    print(f"Vocab size: {vocabsize}")

    convert_model_instance = ConvertRWKV(w, dims, layers, vocabsize)
    convert_model_instance.save_converted()



def main():
    parser = argparse.ArgumentParser(description="Convert a model to a binary format.", add_help=False)
    parser.add_argument("path", nargs='?', type=argparse.FileType('rb'), help="Path to the model file (.pth).")
    args, unknown = parser.parse_known_args()

    if args.path:
        model_path = args.path.name
    else:
        try:
            import tkinter
            from tkinter import filedialog

            def chooseFile():
                root = tkinter.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    filetypes=[("PyTorch Weights", "*.pth")])
                return file_path

            model_path = chooseFile()
        except ImportError:
            print("tkinter not found, falling back to command line argument")
            parser.print_help()
            sys.exit(1)

    if model_path:
        convert_model(model_path)
    else:
        print("No model file selected. Exiting.")

if __name__ == "__main__":
    main()
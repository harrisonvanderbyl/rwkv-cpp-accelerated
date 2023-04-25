import os

import torch

from typing import List

from torch.utils.cpp_extension import load as load_cpp_extension
from tqdm import tqdm


class RwkvCppWrapper:


    def __init__(self):
        assert hasattr(torch.ops, "rwkv"), "RWKV CPP wrapper is not loaded!"

        self.output = torch.zeros(50277,dtype=torch.float32)
        self.state = torch.ops.rwkv.attachState(self.output)
        self.emptyState = [
            self.state[i].clone() for i in range(5)
        ]
        

    def forward(self, x, state: List[torch.Tensor]):
        for i,o in enumerate(state):
            # copy values in without changing the pointer
            self.state[i].copy_(o)
        
        torch.ops.rwkv.rwkvc(x[-1].item())
        torch.cuda.synchronize()
        
        return self.output, [o.clone() for o in self.state]
     
def load_cpp_bindings(
        *,
        torch_binding_cpp_path: str, 
        rwkv_cuda_path: str,
        headers_include_path: str,
        extension_name="wkv_cuda",
        ):

    load_cpp_extension(
        name=extension_name,
        sources=[torch_binding_cpp_path, rwkv_cuda_path,],
        include=[headers_include_path]
        )


def load_cpp_rwkv(*, path: str) -> RwkvCppWrapper:
    layers, embed = torch.ops.rwkv.load(path)

    return RwkvCPPWrapper()

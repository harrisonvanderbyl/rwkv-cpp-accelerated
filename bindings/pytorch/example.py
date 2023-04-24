import os

import torch

from typing import List

from torch.utils.cpp_extension import load as load_cpp_extension
from tqdm import tqdm


current_path = os.path.dirname(os.path.abspath(__file__))

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
     
def load_cpp_rwkv(path, **kwargs) -> RwkvCppWrapper:

    load_cpp_extension(
        name=f"wkv_cuda",
        sources=[f"{current_path}/torchbind.cpp",
                f"{current_path}/rwkv.cu",
                ],
        )

    layers, embed = torch.ops.rwkv.load(path)
    print(layers,embed)
    
    return RwkvCPPWrapper()

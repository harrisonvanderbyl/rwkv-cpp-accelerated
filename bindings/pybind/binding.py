import os
import copy

import torch
import numpy as np

from typing import List

from tqdm import tqdm

import rwkv


class RwkvCppWrapper:


    def __init__(self, model_path: str):

        layers, embed = rwkv.load(model_path)
        self.output = rwkv.initOutput()
        self.state = rwkv.initState()
        print(self.state)
        print(rwkv.getRwkvState())
        self.emptyState = [
            np.copy(state) for state in self.state
        ]
        

    def forward(self, token: int, state: List[np.array]):
        print("forward")
        for i,o in enumerate(state):
            # copy values in without changing the pointer
            self.state[i] = copy.deepcopy(o)

        rwkv.rwkvc(token)
        torch.cuda.synchronize()
        
        return self.output, [np.copy(state) for state in self.state]
     

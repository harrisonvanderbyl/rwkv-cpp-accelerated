import os
import copy

import rwkv

class RwkvCppWrapper:


    def __init__(self, model_path: str):
        self.cpp_rwkv = rwkv.initRwkv()
        layers, embed = rwkv.loadModel(self.cpp_rwkv, model_path)
        print(layers)
        print(embed)

        rwkv.initOutput(self.cpp_rwkv)
        rwkv.initState(self.cpp_rwkv)


    def forward(self, token: int):
        rwkv.modelForward(self.cpp_rwkv, token)
        
        return rwkv.getOutput(self.cpp_rwkv), rwkv.getState(self.cpp_rwkv)
     

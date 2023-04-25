import argparse
import sys

from bindings.pytorch import torch_binding


torch_binding_cpp_path = "./bindings/pytorch/tochbind.cpp"
rwkv_cuda_path = "./include/rwkv/cuda/rwkv.cu"
headers_include_path = "./include/"




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Simple chat with rwkv cpp cuda'
            )
    parser.add_argument(
            '--torch_binding_cpp_path', 
            help='Path to torchbind.cpp file containing bindings for torch', 
            required=False,
            default=torch_binding_cpp_path,
            )

    parser.add_argument(
            '--rwkv_cuda_path', 
            help='Path to rwkv.cu containg cuda implementations of common funcs.', 
            required=False,
            default=rwkv_cuda_path,
            )

    parser.add_argument(
            '--include_path', 
            help='Path to include folder containing headers of NumCpp, rwkv.h, etc.', 
            required=False,
            default=headers_include_path,
            )



    args = vars(parser.parse_args())

    # load cpp bindings into torch.ops namespace, so that underlying
    # RwkvCppWrapper class could use the binding of torchbind.cpp and rwkv.cu 
    # via torch.ops.rwkv.<func_name>
    torch_binding.load_cpp_bindings(
        torch_binding_cpp_path=torch_binding_cpp_path,
        rwkv_cuda_path=rwkv_cuda_path,
        headers_include_path=headers_include_path,
        )


    torch_binding.load_cpp_rwkv()

nvcc -O3 -Xcompiler -fPIC -shared ./c_binding.cpp ../../include/rwkv/cuda/rwkv.cu -I../../include $(python3 -m pybind11 --includes) -o rwkv.so


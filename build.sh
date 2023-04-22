nvcc ./rwkv.cu -c
g++ ./rwkv.o ./chat.cpp -lcudart_static -L/usr/local/cuda/lib64
rm ./rwkv.o
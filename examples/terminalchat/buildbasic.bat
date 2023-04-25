@REM The purpose of this file is to show you can compile without visual studio and stuff, just with nvcc and cuda toolkit
nvcc chat.cpp ../../include/rwkv/cuda/rwkv.cu -I../../include --std c++17 -o build/chat
mkdir .\release
mkdir .\release\vocab
copy .\build\chat.exe release\chat.exe
copy ..\..\include\rwkv\tokenizer\vocab\ .\release\vocab\
nvcc chat.cpp ../../include/rwkv/cuda/rwkv.cu -I../../include --std c++17 -o build/chat
mkdir .\release
mkdir .\release\vocab
cp .\build\chat.exe release\chat.exe
cp ..\..\include\rwkv\tokenizer\vocab\ .\release\vocab\
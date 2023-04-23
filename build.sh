# simplistic build script equivalent to the following commands
# nvcc ./rwkv.cu -c
# g++ ./rwkv.o ./chat.cpp -lcudart_static -L/usr/local/cuda/lib64
# rm ./rwkv.o
# mv ./a.out ./release/chat


echo "compiling"
mkdir build
cd build
cmake ..
cmake --build . --config release
cd ..
# windows
mv ./build/rwkv.exe ./release/chat.exe
# linux
mv ./build/rwkv ./release/chat



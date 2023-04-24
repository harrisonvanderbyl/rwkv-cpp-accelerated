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
# linux
mv ./build/rwkv ./release/chat
# copy directory ../../include/rwkv/tokenizer/vocab to ./release
cp -r ../../include/rwkv/tokenizer/vocab ./release
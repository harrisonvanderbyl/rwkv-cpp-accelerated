# simplistic build script equivalent to the following commands
# nvcc ./rwkv.cu -c
# g++ ./rwkv.o ./chat.cpp -lcudart_static -L/usr/local/cuda/lib64
# rm ./rwkv.o
# mv ./a.out ./release/chat


VERSION="release"

echo "compiling"
mkdir build
mkdir -p ${VERSION}/vocab
cd build
cmake ..
cmake --build . --config ${VERSION}
cd ..
# linux
mv ./build/rwkv ./${VERSION}/chat
# copy directory ../../include/rwkv/tokenizer/vocab to ./release
cp -r ../../include/rwkv/tokenizer/vocab ./${VERSION}

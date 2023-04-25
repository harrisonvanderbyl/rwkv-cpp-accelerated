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

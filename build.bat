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



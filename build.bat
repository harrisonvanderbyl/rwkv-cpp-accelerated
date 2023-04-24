echo "compiling"
mkdir build
cd build
cmake ..
cmake --build . --config release
cd ..
copy .\build\rwkv.exe .\release\chat.exe



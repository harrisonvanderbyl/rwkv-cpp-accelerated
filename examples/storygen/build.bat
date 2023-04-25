echo "compiling"
mkdir build
cd build
cmake ..
cmake --build . --config release
cd ..
mkdir .\release
mkdir .\release\vocab
copy .\build\Release\rwkv.exe .\release\storygen.exe
copy ..\..\include\rwkv\tokenizer\vocab\ .\release\vocab\

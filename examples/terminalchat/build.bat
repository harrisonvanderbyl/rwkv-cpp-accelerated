echo "compiling"
mkdir build
cd build
cmake ..
cmake --build . --config release
cd ..
copy .\build\Release\rwkv.exe .\release\chat.exe
copy ..\..\include\rwkv\tokenizer\vocab .\release\vocab /s /e /y

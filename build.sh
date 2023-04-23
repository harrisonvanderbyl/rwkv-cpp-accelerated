# if linux
if $OSTYPE == "linux-gnu"; then
    echo "compiling for linux"
    nvcc ./rwkv.cu -c
    g++ ./rwkv.o ./chat.cpp -lcudart_static -L/usr/local/cuda/lib64
    rm ./rwkv.o
    mv ./a.out ./release/chat
else
    echo "compiling for windows"
    mkdir build
    cd build
    cmake ..
    cmake --build . --config release
    cd ..
    mv ./build/release/chat.exe ./release/chat.exe
fi

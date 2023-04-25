# RWKV Cuda

This is a super simple c++/cuda implementation of rwkv with no pytorch/libtorch dependencies.

included is a simple example of how to use in both c++ and python.

### Features

* Direct Disk -> Gpu loading ( practically no ram needed )
* Uint8 by default
* Incredibly fast
* No dependencies
* Simple to use
* Simple to build
* Optional Python binding using pytorch tensors as wrappers
* Native tokenizer!
* Windows Support!
* Distributable programs! (check actions for the prebuilt example apps)

### Roadmap

* Optimize .pth converter (currently uses a lot of ram)
* Better uint8 support ( currently only uses Q8_0 algorythm)
* Fully fleshed out demos
* Godot module

## Run example app
1) go to the actions tab
2) find a green checkmark for your platform
3) download the executable
4) download or convert a model (download links pending)
5) place the model.bin file in the same place as the executable
6) run the executable

## Build Instructions

### Build on Linux
```
$./build.sh
```

### Build on Windows

```
> mkdir build
> cd build
> cmake ..
> cmake --build . --config release
```

You can find executable at build/release/rwkv.exe

Make sure you already installed CUDA Toolkit and Visual Studio 2022.

## Convert the model into the format

Make sure you have python, torch installed
```
> cd export
> python3 export.py
```

* On Windows, please run the above commands in "x64 Native Tools Command Prompt for VS 2022" terminal.



C++ tokenizer came from this project:
https://github.com/gf712/gpt2-cpp/
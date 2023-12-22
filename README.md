# RWKV Cuda [![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3DUnexplored_Horizons%26type%3Dpatrons&style=flat)](https://patreon.com/Unexplored_Horizons)
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
* HIP(AMD) GPU support!
* Vulkan(All) Support!
* Distributable programs! (check actions for the prebuilt example apps)
* [Godot module](https://github.com/harrisonvanderbyl/godot-rwkv)

### Roadmap

* Optimize .pth converter (currently uses a lot of ram)
* Better uint8 support ( currently only uses Q8_0 algorythm)
* Fully fleshed out demos

## Run example app
1) go to the actions tab
2) find a green checkmark for your platform
3) download the executable
4) download or convert a model ([downloads here](https://huggingface.co/nenkoru/rwkv-cuda-cpp/tree/main))
5) place the model.bin file in the same place as the executable
6) run the executable

## Build Instructions

### Build librwkv_cuda.a 

In the top of the source directory
```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Build example storygen on Linux/windows
Make sure you already installed CUDA Toolkit / HIP development tools / Vulkan development tools

```
# in example/storygen
build.sh # Linux/nvidia
build.bat # Windows/nvidia
amd.sh # Linux/Amd
vulkan.sh # Linux/Vulkan(all)
```

You can find executable at build/storygen[.exe] that can be run from the build directory. It expects a 'model.bin' file at the converter folder. See the following note on downloading and converting the RWKV 4 models. 

```
$ cd build 
$ ./storygen
```

## Convert the model into the format

You can download the weights of the model here:
https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main

For conversion to a .bin model you can choose between 2 options:

### GUI option

Make sure you have python + torch, tkinter, tqdm and Ninja packages installed.
```
> cd converter
> python3 convert_model.py
```

### CLI option

Make sure you have python + torch, tqdm and Ninja packages installed.
```
> cd converter
> python3 convert_model.py your_downloaded_model.pth
```

* On Windows, please run the above commands in "x64 Native Tools Command Prompt for VS 2022" terminal.

C++ tokenizer came from this project:
https://github.com/gf712/gpt2-cpp/

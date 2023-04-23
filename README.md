# RWKV Cuda

This is a super simple c++/cuda implementation of rwkv with no pytorch/libtorch dependencies.

included is a simple example of how to use in both c++ and python.

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
> python
  >>> from export import *
  >>> OptRWKV("<Model_path>")
```

* On Windows, please run the above commands in "x64 Native Tools Command Prompt for VS 2022" terminal.

## Features

* Direct Disk -> Gpu loading ( practically no ram needed )
* Uint8 by default
* Incredibly fast
* No dependencies
* Simple to use
* Simple to build
* Optional Python binding using pytorch tensors as wrappers
* Native tokenizer!

### TODO

* Add support for windows ( pretty sure all you need to do is change the path to the cuda lib in build.sh )
* Optimize .pth converter (currently uses a lot of ram)
* Better uint8 support ( currently only uses Q8_0 algorythm)
* Fully fleshed out demos
* better sampler
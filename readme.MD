# RWKV Cuda

This is a super simple c++/cuda implementation of rwkv with no pytorch/libtorch dependencies.

included is a simple example of how to use in both c++ and python.

use ./build.sh to build for linux

### Features

* Direct Disk -> Gpu loading ( practically no ram needed )
* Uint8 by default
* Incredibly fast
* No dependencies
* Simple to use
* Simple to build

### TODO

* Add support for windows ( pretty sure all you need to do is change the path to the cuda lib in build.sh )
* Add tokenizer
* Optimize .pth converter (currently uses a lot of ram)
* Better uint8 support ( currently only uses Q8_0 algorythm)


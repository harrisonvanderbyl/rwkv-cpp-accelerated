VERSION="release"

   
echo "compiling"
mkdir build
mkdir -p ${VERSION}/vocab
cp -r ../../include/rwkv/tokenizer/vocab ./${VERSION}

# get input
echo "Choose your build type:"
echo "c) CUDA(nvidia)"
echo "h) HIP(amd)"
echo "v) Vulkan(all)"
echo "Default: c"
read buildtype

#get name of folder
NAME=${PWD##*/}

if [ "$buildtype" = "h" ]; then
    hipcc --std=c++17 ./${NAME}.cpp ../../include/rwkv/cuda/rwkv.cu -I../../include -o ./release/${NAME}-rocm #--offload-arch=gfx700,gfx701,gfx702,gfx703,gfx704,gfx705,gfx801,gfx802,gfx803,gfx805,gfx810,gfx900,gfx902,gfx904,gfx906,gfx908,gfx909,gfx1010,gfx1011,gfx1012,gfx1030
    exit 0
elif [ "$buildtype" = "v" ]; then
    g++ --std=c++17 ./${NAME}.cpp ../../include/rwkv/vulkan/rwkv.cpp -I../../include -o ./release/${NAME}-vulkan -lvulkan
    files=$(ls ../../include/rwkv/vulkan/ops/**/*.comp)

    for file in $files; do
        echo "Compiling $file"
        # get only the filename
        filename=$(basename -- "$file")
        glslc $file -o ./release/${filename%.*}.spv
    done 
    exit 0
else
    cd build
    cmake ..
    cmake --build . --config ${VERSION}
    cd ..
    # linux
    mv ./build/rwkv ./${VERSION}/${NAME}

    exit 0
fi

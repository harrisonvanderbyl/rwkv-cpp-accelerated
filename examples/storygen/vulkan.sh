g++ --std=c++17 ./storygen.cpp ../../include/rwkv/vulkan/rwkv.cpp -I../../include -o ./release/storygen-vulkan -lvulkan
files=$(ls ../../include/rwkv/vulkan/ops/**/*.comp)

for file in $files; do
    echo "Compiling $file"
    # get only the filename
    filename=$(basename -- "$file")
    glslc $file -o ./release/${filename%.*}.spv
done 

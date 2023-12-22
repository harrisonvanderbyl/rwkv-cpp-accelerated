VERSION="Release"

mkdir build
cd build
cmake ..
cmake --build . --config ${VERSION}


#!/bin/bash
# Linux build script for cpp_pipeline
set -e

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

echo "Binary: ./build/cpp_pipeline"

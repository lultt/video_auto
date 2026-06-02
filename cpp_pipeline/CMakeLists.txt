cmake_minimum_required(VERSION 3.16)
project(cpp_pipeline VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Performance flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2 /fp:fast")
    add_compile_options(/bigobj)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx2 -ffast-math")
endif()

# Header-only library
add_library(pipeline_headers INTERFACE)
target_include_directories(pipeline_headers INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)

# Main executable
add_executable(cpp_pipeline
    src/main.cpp
)
target_link_libraries(cpp_pipeline PRIVATE pipeline_headers)

# Link Windows socket libraries if needed
if(WIN32)
    target_link_libraries(cpp_pipeline PRIVATE psapi)
endif()

# Benchmark executable (future: separate benchmark with more metrics)
add_executable(cpp_benchmark
    src/main.cpp
)
target_link_libraries(cpp_benchmark PRIVATE pipeline_headers)
if(WIN32)
    target_link_libraries(cpp_benchmark PRIVATE psapi)
endif()
target_compile_definitions(cpp_benchmark PRIVATE BENCHMARK_MODE=1)

install(TARGETS cpp_pipeline RUNTIME DESTINATION bin)

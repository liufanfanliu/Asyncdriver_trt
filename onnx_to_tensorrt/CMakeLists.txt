cmake_minimum_required(VERSION 3.10)
project(ONNX_to_TensorRT_v1)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

# Add extra optimization compilation options
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native")

# Set TensorRT and CUDA paths
set(TENSORRT_DIR $ENV{HOME}/TensorRT-10.9.0.34)
set(CUDA_ROOT "/usr/local/cuda")

# Add header file include paths
include_directories(${TENSORRT_DIR}/include)
include_directories(/usr/local/cuda/include)

# Add library search paths
link_directories(${TENSORRT_DIR}/lib)
link_directories(/usr/local/cuda/lib64)

# Add source files
add_executable(onnx_to_tensorrt src/ONNX_to_TensorRT.cpp)

set_target_properties(onnx_to_tensorrt PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib:$ENV{HOME}/TensorRT-10.9.0.34/lib"
)

target_link_libraries(onnx_to_tensorrt
    nvinfer
    nvonnxparser
    cudart
    pthread
)

# Ensure RPATH is set correctly
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

cmake_minimum_required(VERSION 3.10)
project(ONNX_to_TensorRT_v1)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

# 添加额外的优化编译选项

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native")
# 设置 TensorRT 和 CUDA 路径
set(TENSORRT_DIR $ENV{HOME}/TensorRT-10.9.0.34)
set(CUDA_ROOT "/usr/local/cuda")

# 添加头文件搜索路径
include_directories(${TENSORRT_DIR}/include)
include_directories(/usr/local/cuda/include)

# 添加库文件搜索路径
link_directories(${TENSORRT_DIR}/lib)
link_directories(/usr/local/cuda/lib64)

# 添加源文件
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

# 确保 RPATH 被设置正确
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
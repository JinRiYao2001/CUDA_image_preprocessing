# 设置项目名称和最低 CMake 版本要求
cmake_minimum_required(VERSION 3.29.0)
set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_PATH}")
set(CMAKE_CXX_COMPILER "$ENV{MinGW}/bin/g++.exe")
set(TENSORRT "$ENV{TENSORRT_PATH}")
set(OPENCV_ROOT "$ENV{OPENCV_ROOT}")
set(OPENCV_WORLD490D "$ENV{OPENCV_WORLD490D}")

set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")


set(TENSORRT_INCLUDE_DIRS "${TENSORRT}/include")
set(TENSORRT_LIBRARIES "${TENSORRT}/lib")


project(inference LANGUAGES CXX CUDA)

# 添加可执行文件
add_executable(inference inference.cpp image_preprocessing.cu image_preprocessing.h)


# 链接 CUDA 库
target_include_directories(inference PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_directories(inference PUBLIC ${CUDA_LIBRARIES})
target_link_libraries(inference PUBLIC ${CUDA_LIBRARIES})
target_include_directories(inference PUBLIC ${TENSORRT_INCLUDE_DIRS})
target_link_directories(inference PUBLIC ${TENSORRT_LIBRARIES})
target_link_libraries(inference PUBLIC ${TENSORRT_LIBRARIES})
target_link_libraries(inference PUBLIC "${TENSORRT}/lib/nvinfer.lib")
target_link_libraries(inference PUBLIC "${TENSORRT}/lib/nvonnxparser.lib")
target_include_directories(inference PUBLIC "${OPENCV_ROOT}/opencv-4.9.0/include")
target_include_directories(inference PUBLIC "${OPENCV_ROOT}/opencv/build/include")

target_link_libraries(inference PUBLIC "${OPENCV_WORLD490D}")
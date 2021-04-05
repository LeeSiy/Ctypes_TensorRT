cmake_minimum_required(VERSION 2.6)

project(crnn_copy)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()


find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#add_executable(crnn_copy ${PROJECT_SOURCE_DIR}/crnn_copy.cpp, crnn_wrapper_header.cpp, main.cpp)
#add_executable(crnn_copy ${PROJECT_SOURCE_DIR}/crnn_copy.cpp crnn_wrapper_header.cpp main.cpp)
add_library(crnn_copy SHARED crnn_copy.cpp crnn_wrapper_header.cpp image.cpp)
find_package(PythonInterp REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_subdirectory(pybind11)
#pybind11_add_module(crnn_copy crnn_copy.cpp)

set(OUTPUT_SHARED_LIB "${CMAKE_PROJECT_NAME}")

target_link_libraries(crnn_copy nvinfer)
target_link_libraries(crnn_copy cudart)
target_link_libraries(crnn_copy ${OpenCV_LIBS})

add_definitions(-O2 -pthread)


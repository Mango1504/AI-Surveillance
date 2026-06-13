cmake_minimum_required(VERSION 3.18)
project(metropolis_cpp LANGUAGES CXX CUDA)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Source files
set(SOURCES
    src/preprocess.cu
    src/nms.cu
    src/risk_engine.cpp
    src/bindings.cpp
)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create shared library target
pybind11_add_module(metropolis_cpp ${SOURCES})

# Link libraries
target_link_libraries(metropolis_cpp PRIVATE
    CUDA::cudart
    pybind11::module
)

# Set target properties
set_target_properties(metropolis_cpp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

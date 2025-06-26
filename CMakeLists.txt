cmake_minimum_required(VERSION 3.16)
project(gpgpu_examples)

# COMPILER FLAGS AND DEFINITIONS
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_86)

find_package(CUDA REQUIRED)

cuda_add_executable(add_vector_gpu add.cu)
add_executable(add_vector_cpu add.c)


cmake_minimum_required(VERSION 3.16)
project(gpgpu_examples LANGUAGES C CUDA)

# COMPILER FLAGS AND DEFINITIONS
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86")

add_executable(add_vector_gpu add.cu)
set_target_properties(add_vector_gpu PROPERTIES CUDA_ARCHITECTURES 86)
add_executable(add_vector_cpu add.c)


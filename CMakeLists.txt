cmake_minimum_required(VERSION 3.22)
project(gpgpu_examples LANGUAGES C CXX CUDA)
enable_language(CUDA)

add_subdirectory(01_introduction)
add_subdirectory(02_communication_patterns)
add_subdirectory(03_parallel_primitives)
add_subdirectory(04_Unit)

add_executable(reduce 01_reduce.cu)
set_target_properties(reduce PROPERTIES C_STANDARD 11 CUDA_ARCHITECTURES 86)

add_executable(histogram 03_histogram.cu)
set_target_properties(histogram PROPERTIES C_STANDARD 11 CUDA_ARCHITECTURES 86)

add_subdirectory(problem_set_3)
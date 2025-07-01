

# vector addtion GPU 
add_executable(add_vector_gpu 01_add.cu)
set_target_properties(add_vector_gpu PROPERTIES CUDA_ARCHITECTURES 86)


# vector addtion CPU
add_executable(add_vector_cpu 01_add.c)
set_target_properties(add_vector_cpu PROPERTIES C_STANDARD 11) 



# problem set 1
add_subdirectory(problem_set_1)
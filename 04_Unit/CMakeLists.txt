# Matrix transpose GPU 
add_executable(matrix_transpose matrix_transpose.cu)
set_target_properties(matrix_transpose PROPERTIES CUDA_ARCHITECTURES 86)
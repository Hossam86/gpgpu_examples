# Matrix transpose GPU 
add_executable(matrix_transpose matrix_transpose.cu)
set_target_properties(matrix_transpose PROPERTIES CUDA_ARCHITECTURES 86)


add_executable(matrix_multiplication matrix_multiplication.cu)
set_target_properties(matrix_multiplication PROPERTIES CUDA_ARCHITECTURES 86)
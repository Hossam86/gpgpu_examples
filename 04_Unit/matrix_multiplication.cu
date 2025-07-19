// refernce https://github.com/Dhanush295/Cuda_program/blob/main/Matrix_Multiply_sharedMem.cu

#include <iostream>

const int TILE_SIZE = 16;

__global__ void MatrixMultiShared(float* A, float* B, float* C, int N)
{

	__shared__ float tile_A[TILE_SIZE][TILE_SIZE];
	__shared__ float tile_B[TILE_SIZE][TILE_SIZE];

	int row = threadIdx.y + blockIdx.y * TILE_SIZE;
	int col = threadIdx.x + blockIdx.x * TILE_SIZE;
};

int main()
{
	int N = 1024;
	int size = N * N * sizeof(float);

	float* h_a = (float*)malloc(size);
	float* h_b = (float*)malloc(size);
	float* h_c = (float*)malloc(size);

	for (int i = 0; i < N * N; i++)
	{
		h_a[i] = 1.0f;
		h_b[i] = 1.0f;
	}

	float *d_a, *d_b, *d_c;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice);

	dim3 blockdim(TILE_SIZE, TILE_SIZE);
	dim3 griddim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
	MatrixMultiShared<<<griddim, blockdim>>>(d_a, d_b, d_c, N);

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(h_c);
}
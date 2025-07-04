// reference https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

#include <stdio.h>
#include <assert.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

// Check errors and print GB/s
void postprocess(const float* ref, const float* res, int n, float ms)
{
	bool passed = true;
	for (int i = 0; i < n; i++)
		if (res[i] != ref[i])
		{
			printf("%d %f %f\n", i, res[i], ref[i]);
			printf("%25s\n", "*** FAILED ***");
			passed = false;
			break;
		}
	if (passed)
		printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
}

__global__ void
copy(float* odata, const float* idata)
{
	int x = threadIdx.x + TILE_DIM * blockIdx.x;
	int y = threadIdx.y + TILE_DIM * blockIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		odata[((y + i) * width + x)] = idata[(y + i) * width + x];
	}
}

int main(int argc, char** argv)
{
	const int nx = 1024;
	const int ny = 1024;
	const int mem_size = nx * ny * sizeof(float);

	dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

	int devId = 0;
	if (argc > 1)
		devId = atoi(argv[1]);

	cudaDeviceProp prop;
	checkCuda(cudaGetDeviceProperties(&prop, devId));
	printf("\nDevice : %s\n", prop.name);
	printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
		nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
	printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
		dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

	checkCuda(cudaSetDevice(devId));

	float* h_idata = (float*)malloc(mem_size);
	float* h_cdata = (float*)malloc(mem_size);
	float* h_tdata = (float*)malloc(mem_size);
	float* gold = (float*)malloc(mem_size);

	float *d_idata, *d_cdata, *d_tdata;
	checkCuda(cudaMalloc(&d_idata, mem_size));
	checkCuda(cudaMalloc(&d_cdata, mem_size));
	checkCuda(cudaMalloc(&d_tdata, mem_size));

	// check parameters and calculate execution configuration
	if (nx % TILE_DIM || ny % TILE_DIM)
	{
		printf("nx and ny must be a multiple of TILE_DIM\n");
		goto error_exit;
	}

	if (TILE_DIM % BLOCK_ROWS)
	{
		printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
		goto error_exit;
	}

	// host
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			h_idata[j * nx + i] = j * nx + i;

	// correct result for error checking
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			gold[j * nx + i] = h_idata[i * nx + j];

	// device
	checkCuda(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

	// events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// ------------
	// time kernels
	// ------------
	printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

	// ----
	// copy
	// ----
	printf("%25s", "copy");
	checkCuda(cudaMemset(d_cdata, 0, mem_size));
	// warm up
	copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
	checkCuda(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < NUM_REPS; i++)
		copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
	postprocess(h_idata, h_cdata, nx * ny, ms);

error_exit:
	// cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	checkCuda(cudaFree(d_tdata));
	checkCuda(cudaFree(d_cdata));
	checkCuda(cudaFree(d_idata));
	free(h_idata);
	free(h_tdata);
	free(h_cdata);
	free(gold);
}
#include <stdio.h>

// reduce serial implementation
void reduce(float *dout, float *din, int n) {

  float sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += din[i];
  }
  *dout = sum;
}

// reduce on gpu (global memory)
__global__ void globalMem_reduce_kernel(float *d_out, float *d_in) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // do reduction in global mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      d_in[myId] += d_in[myId + s];
    }
    __syncthreads(); // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0) {
    d_out[blockIdx.x] = d_in[myId];
  }
}

__global__ void sharedMem_reduce_kernel(float *d_out, const float *d_in) {

  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId];
  __syncthreads(); // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads(); // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

void reduce(float *d_out, float *d_intermediate, float *d_in, int size,
            bool usesSharedMemory) {

  // assume that size is not greater than maxThreadsPerBlock^2
  // and that size is a multiple of maxThreadsPerBlock
  const int maxThreadPerBlock = 1024;

  int threads = maxThreadPerBlock;
  int blocks = size / maxThreadPerBlock;

  if (usesSharedMemory) {
    sharedMem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_intermediate, d_in);
  } else {
    globalMem_reduce_kernel<<<blocks, threads>>>(d_intermediate, d_in);
  }

  // now we're down to one block left, so reduce it
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;

  if (usesSharedMemory) {
    sharedMem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_intermediate, d_in);
  } else {

    globalMem_reduce_kernel<<<blocks, threads>>>(d_out, d_intermediate);
  }
}

int main() {
  const int ARRAYSIZE = 1 << 20;
  const int ARRAYBYTES = ARRAYSIZE * sizeof(float);

  // allocate host arrays
  float *h_in = (float *)std::malloc(ARRAYBYTES);
  for (int i = 0; i < ARRAYSIZE; ++i) {
    h_in[i] = i;
  }
  float h_out;

  // run reduce (serial) in host
  reduce(&h_out, h_in, ARRAYSIZE);
  printf("reduce calculated on host: %f\n", h_out);

  // reset
  h_out = 0;
  // declare gpu memo pointers
  float *d_in, *d_out, *d_intermediate;
  cudaMalloc((void **)&d_in, ARRAYBYTES);
  cudaMalloc((void **)&d_intermediate, ARRAYBYTES);

  cudaMalloc((void **)&d_out, sizeof(float));

  // copy data to gpu;
  cudaMemcpy(d_in, h_in, ARRAYSIZE, cudaMemcpyHostToDevice);

  // reduce
  reduce(d_out, d_intermediate, d_in, ARRAYSIZE, false);

  // copy back to host
  cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  printf("reduce calculated on Gpu: %f\n", h_out);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_intermediate);
}
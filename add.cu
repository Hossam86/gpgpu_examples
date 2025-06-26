#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    c[idx] = a[idx] + b[idx];
}

#define N 2048 * 2048
#define M 512

int main() {

  // host data
  int *a, *b, *c;

  int bytecount = sizeof(int) * N;

  // allocate data on host

  a = (int *)malloc(bytecount);
  b = (int *)malloc(bytecount);
  c = (int *)malloc(bytecount);
  for (int i = 0; i < N; ++i) {

    a[i] = i;
    b[i] = i;
  }

  // device copies
  int *da, *db, *dc;

  cudaMalloc((void **)&da, bytecount);
  cudaMalloc((void **)&db, bytecount);
  cudaMalloc((void **)&dc, bytecount);

  // copy from host to device
  cudaMemcpy(da, a, bytecount, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, bytecount, cudaMemcpyHostToDevice);

  // lunch kernel on gpu
  add<<<(N + M - 1) / M, M>>>(da, db, dc, N);

  // copy results back to host
  cudaMemcpy(c, dc, bytecount, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < N; ++i) {
  //   printf("c[%d]: %d\n", i, c[i]);
  // }
  free(a);
  free(b);
  free(c);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return 0;
}
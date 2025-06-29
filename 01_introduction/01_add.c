
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048 * 2048

void add(int *a, int *b, int *c, int n) {

  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }
}

int main() {

  int bytecount = N * sizeof(int);

  int *a = (int *)malloc(bytecount);
  int *b = (int *)malloc(bytecount);
  int *c = (int *)malloc(bytecount);
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i;
  }

  clock_t start = clock();

  add(a, b, c, N);

  clock_t end = clock();

  double duration = (double)(end - start) / CLOCKS_PER_SEC;
  free(a);
  free(b);
  free(c);

  printf("elpased time %f", duration);
}
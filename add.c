
#include <stdio.h>
#include <stdlib.h>

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

  add(a, b, c, N);

  //   for (int i = 0; i < N; ++i) {
  //     printf("c[%d]: %d\n", i, c[i]);
  //   }

  free(a);
  free(b);
  free(c);
}
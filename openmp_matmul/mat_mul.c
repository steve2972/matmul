#include "mat_mul.h"
#include <omp.h>
#include <stdlib.h>

void init(float *C, int M, int N) {
  # pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    float * c = &C[i * N];
    for (int j = 0; j < N; ++j) {
        *(c+j) = 0;
    }
  }
}


void mat_mul(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
  omp_set_num_threads(num_threads);
  int i, j, k;
  
  init(C, M, N);
  # pragma omp parallel for private(i, j, k)
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      float * ptr_c = &C[i * N];
      float * ptr_b = &B[k * N];
      float * ptr_a = &A[i * K + k];
      for (j = 0; j < N; ++j) {
        *(ptr_c+j) += *ptr_a * *(ptr_b + j);
      }
      
    }
  }
}
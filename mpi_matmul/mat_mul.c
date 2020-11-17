#include "mat_mul.h"
#include <mpi.h>
#include "util.h"
#include <omp.h>

static float* myA;
static float* myB;
static float* myC;
static int mpi_rank, mpi_size;

void mat_mul_init(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  alloc_mat(&myA, M, K);
  alloc_mat(&myB, K, N);
  alloc_mat(&myC, M, N);
}

void mat_mul_finalize() {
  free_mat(myA);
  free_mat(myB);
  free_mat(myC);
}

void init(float *C, int M, int N) {
  # pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    float * c = &C[i * N];
    for (int j = 0; j < N; ++j) {
        *(c+j) = 0;
    }
  }
}


void mat_mul(float *A, float *B, float *C, int M, int N, int K) {
  int i, j, k;
  int num_workers = mpi_size-1;
  int rows;
  int nrow = M / num_workers;
  int extra = M % num_workers;
  int offset = 0;

  if (mpi_rank == 0){
    for (int i=1; i < mpi_size; i++){
      rows = (i <= extra) ? nrow + 1 : nrow;
      MPI_Send(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
      MPI_Send(A+(offset*K), rows*K, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
      MPI_Send(B, K*N, MPI_FLOAT, i, 3, MPI_COMM_WORLD);
      offset += rows;
    }
  } else {
    MPI_Recv(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
    MPI_Recv(&rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, NULL);
    MPI_Recv(myA+(offset*K), rows*K, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, NULL);
    MPI_Recv(myB, K*N, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, NULL);
  }

  omp_set_num_threads(32);
  init(myC, M, N);
  #pragma omp parallel for private(i, j, k)
  for (i = offset; i < offset + rows; ++i) {
    for (k = 0; k < K; ++k) {
      float * ptr_a = &myA[i * K + k];
      float * ptr_b = &myB[k * N];
      float * ptr_c = &myC[i * N];
      for (j = 0; j < N; ++j) {
        *(ptr_c + j) += *ptr_a * *(ptr_b + j);
      }
    }
  }

  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size; i++) {
      MPI_Recv(&offset, 1, MPI_INT, i, 4, MPI_COMM_WORLD, NULL);
      MPI_Recv(&rows, 1, MPI_INT, i, 5, MPI_COMM_WORLD, NULL);
      MPI_Recv(C+(N*offset), rows*N, MPI_FLOAT, i, 6, MPI_COMM_WORLD, NULL);
    }
  } else {
    MPI_Send(&offset, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
    MPI_Send(myC+(N*offset), rows*N, MPI_FLOAT, 0, 6, MPI_COMM_WORLD);
  }  
}
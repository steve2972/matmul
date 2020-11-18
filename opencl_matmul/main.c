#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>

#include "timer.h"
#include "util.h"
#include "mat_mul.h"
#include "mat_mul_ref.h"

#ifdef USE_MPI

#include <mpi.h>
#define EXIT(status) \
  do { \
    MPI_Finalize(); \
    exit(status); \
  } while (0);
#define PRINTF_WITH_RANK(rank, fmt, ...) \
  do { \
    printf("[rank %d] " fmt "\n", rank, ##__VA_ARGS__); \
  } while (0);
#define BARRIER() \
  do { \
    MPI_Barrier(MPI_COMM_WORLD); \
  } while (0);

#else

#define EXIT(status) \
  do { \
    exit(status); \
  } while (0);
#define PRINTF_WITH_RANK(rank, fmt, ...) \
  do { \
    printf(fmt "\n", ##__VA_ARGS__); \
  } while (0);
#define BARRIER()

#endif

static bool print_matrix = false;
static bool validation = false;
static int M = 16, N = 16, K = 16;
static int num_threads = 1;
static int num_iterations = 1;
static int num_warmup = 0;
static int mpi_rank = 0, mpi_size = 1;

static void print_help(const char* prog_name) {
  if (mpi_rank == 0) {
    PRINTF_WITH_RANK(mpi_rank, "Usage: %s [-pvh] [-t num_threads] [-n num_iterations] [-w num_warmup] M N K", prog_name);
    PRINTF_WITH_RANK(mpi_rank, "Options:");
    PRINTF_WITH_RANK(mpi_rank, "  -p : print matrix data. (default: off)");
    PRINTF_WITH_RANK(mpi_rank, "  -v : validate matrix multiplication. (default: off)");
    PRINTF_WITH_RANK(mpi_rank, "  -h : print this page.");
    PRINTF_WITH_RANK(mpi_rank, "  -t : number of threads (default: 1)");
    PRINTF_WITH_RANK(mpi_rank, "  -n : number of iterations (default: 1)");
    PRINTF_WITH_RANK(mpi_rank, "  -w : number of warmup iteration. (default: 0)");
    PRINTF_WITH_RANK(mpi_rank, "   M : number of rows of matrix A and C. multiple of 16. (default: 16)");
    PRINTF_WITH_RANK(mpi_rank, "   N : number of columns of matrix B and C. multiple of 16. (default: 16)");
    PRINTF_WITH_RANK(mpi_rank, "   K : number of columns of matrix A and rows of B. multiple of 16. (default: 16)");
  }
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:w:")) != -1) {
    switch (c) {
      case 'p':
        print_matrix = true;
        break;
      case 'v':
        validation = true;
        break;
      case 't':
        num_threads = atoi(optarg);
        break;
      case 'n':
        num_iterations = atoi(optarg);
        break;
      case 'w':
        num_warmup = atoi(optarg);
        break;
      case 'h':
      default:
        print_help(argv[0]);
        EXIT(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0:
        M = atoi(argv[i]);
        if (M % 16 != 0) {
          PRINTF_WITH_RANK(mpi_rank, "M should be multiple of 16.");
          EXIT(0);
        }
        break;
      case 1:
        N = atoi(argv[i]);
        if (N % 16 != 0) {
          PRINTF_WITH_RANK(mpi_rank, "N should be multiple of 16.");
          EXIT(0);
        }
        break;
      case 2:
        K = atoi(argv[i]);
        if (K % 16 != 0) {
          PRINTF_WITH_RANK(mpi_rank, "K should be multiple of 16.");
          EXIT(0);
        }
        break;
      default:
        break;
    }
  }
  if (mpi_rank == 0) {
    PRINTF_WITH_RANK(mpi_rank, "Options:");
    PRINTF_WITH_RANK(mpi_rank, "  Problem size: M = %d, N = %d, K = %d", M, N, K);
    PRINTF_WITH_RANK(mpi_rank, "  Number of threads: %d", num_threads);
    PRINTF_WITH_RANK(mpi_rank, "  Number of iterations: %d", num_iterations);
    PRINTF_WITH_RANK(mpi_rank, "  Number of warmup iterations: %d", num_warmup);
    PRINTF_WITH_RANK(mpi_rank, "  Print matrix: %s", print_matrix ? "on" : "off");
    PRINTF_WITH_RANK(mpi_rank, "  Validation: %s", validation ? "on" : "off");
    PRINTF_WITH_RANK(mpi_rank, "  Comm size: %d", mpi_size);
    PRINTF_WITH_RANK(mpi_rank, "");
  }
}

int main(int argc, char **argv) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  timer_init(9);
  parse_opt(argc, argv);

  float *A, *B, *C;
  if (mpi_rank == 0) {
    PRINTF_WITH_RANK(mpi_rank, "Initializing matrices...");
    alloc_mat(&A, M, K);
    alloc_mat(&B, K, N);
    alloc_mat(&C, M, N);
    rand_mat(A, M, K);
    rand_mat(B, K, N);
    PRINTF_WITH_RANK(mpi_rank, "Initializing matrices done!");
  }

  BARRIER();

  PRINTF_WITH_RANK(mpi_rank, "Initializing...");
  mat_mul_init(M, N, K);
  PRINTF_WITH_RANK(mpi_rank, "Initializing done!");

  BARRIER();

  double elapsed_time_sum = 0;
  for (int i = -num_warmup; i < num_iterations; ++i) {
    if (i < 0) {
      PRINTF_WITH_RANK(mpi_rank, "Warming up...");
    } else {
      PRINTF_WITH_RANK(mpi_rank, "Calculating...(iter=%d)", i);
    }
    timer_reset(0);
    BARRIER();
    timer_start(0);
    mat_mul(A, B, C, M, N, K);
    BARRIER();
    timer_stop(0);
    double elapsed_time = timer_read(0);
    if (i < 0) {
      PRINTF_WITH_RANK(mpi_rank, "Warming up done!: %f sec", elapsed_time);
    } else {
      PRINTF_WITH_RANK(mpi_rank, "Calculating done!(iter=%d): %f sec", i, elapsed_time);
    }
    if (i >= 0) {
      elapsed_time_sum += elapsed_time;
    }
  }

  if (mpi_rank == 0) {
    if (print_matrix) {
      PRINTF_WITH_RANK(mpi_rank, "MATRIX A:"); print_mat(A, M, K);
      PRINTF_WITH_RANK(mpi_rank, "MATRIX B:"); print_mat(B, K, N);
      PRINTF_WITH_RANK(mpi_rank, "MATRIX C:"); print_mat(C, M, N);
    }
  }

  if (mpi_rank == 0) {
    if (validation) {
      float *C_ref;
      alloc_mat(&C_ref, M, N);
      timer_reset(0);
      timer_start(0);
      mat_mul_ref(A, B, C_ref, M, N, K, 16);
      timer_stop(0);
      double elapsed_time = timer_read(0);
      check_mat_mul(C, C_ref, M, N, K);
      free_mat(C_ref);
      PRINTF_WITH_RANK(mpi_rank, "Reference time: %f sec", elapsed_time);
      PRINTF_WITH_RANK(mpi_rank, "Reference throughput: %f GFLOPS", 2.0 * M * N * K / elapsed_time / 1e9);
    }
  }

  if (mpi_rank == 0) {
    free_mat(A);
    free_mat(B);
    free_mat(C);
  }

  if (mpi_rank == 0) {
    double elapsed_time_avg = elapsed_time_sum / num_iterations;
    PRINTF_WITH_RANK(mpi_rank, "Your Avg. time: %f sec", elapsed_time_avg);
    PRINTF_WITH_RANK(mpi_rank, "Your Avg. throughput: %f GFLOPS", 2.0 * M * N * K / elapsed_time_avg / 1e9);
  }

  PRINTF_WITH_RANK(mpi_rank, "Finalizing...");
  mat_mul_finalize();
  PRINTF_WITH_RANK(mpi_rank, "Finalizing done!");

  timer_finalize();

#ifdef USE_MPI
  MPI_Finalize();
#endif

  return 0;
}

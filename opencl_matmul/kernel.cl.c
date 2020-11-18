__kernel void mat_mul(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  if (i >= M || j >= N) return;

  float s = 0;
  for (int k = 0; k < K; ++k) {
    s += A[i * K + k] * B[k * N + j];
  }
  C[i * N + j] = s;
}

  /*
  int i = get_global_id(0);
  int j = get_global_id(1);
  if (i >= M || j >= N) return;

  const int TS = 128;

  const int global_row = TS * get_group_id(0) + i;
  const int global_col = TS * get_group_id(1) + j;
  
  __local float Asub[128][128];
  __local float Bsub[128][128];

  float value = 0.0f;

  const int numTiles = K/TS;
  for (int t = 0; t < numTiles; t++) {
    const int tiledRow = TS*t + i;
    const int tiledCol = TS*t + j;
    Asub[j][i] = A[tiledCol*M + global_row];
    Bsub[j][i] = B[global_col*K + tiledRow];


    for (unsigned int k = 0; k < K; ++k) {
      value += Asub[k][i] * Bsub[j][k];
    }


  
  }
  C[global_col*M + global_row] = value;

*/

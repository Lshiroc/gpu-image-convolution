#include "support.h"

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void gaussianConvolution(Matrix N, Matrix P) {

  __shared__ float shared_mem_tile[BLOCK_SIZE][BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row_i = by * TILE_SIZE - FILTER_SIZE / 2 + ty;
  int col_i = bx * TILE_SIZE - FILTER_SIZE / 2 + tx;

  if (row_i >= 0 && col_i >= 0 &&
      row_i < N.height && col_i < N.width) {
    shared_mem_tile[ty][tx] =
        N.elements[row_i * N.pitch + col_i];
  } else {
    shared_mem_tile[ty][tx] = 0.0f;
  }

  __syncthreads();

  if (ty < TILE_SIZE && tx < TILE_SIZE) {
    float sum = 0.0f;

    for (int i = 0; i < FILTER_SIZE; i++) {
      for (int j = 0; j < FILTER_SIZE; j++) {
        sum += M_c[i][j] * shared_mem_tile[ty + i][tx + j];
      }
    }

    int row_o = by * TILE_SIZE + ty;
    int col_o = bx * TILE_SIZE + tx;

    if (row_o < P.height && col_o < P.width) {
      P.elements[row_o * P.pitch + col_o] = sum;
    }
  }
}

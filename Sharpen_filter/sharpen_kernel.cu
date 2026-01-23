#include "sharpen_kernel.cuh"
#include <cuda_runtime.h>

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Laplacian sharpen:
// out = in + amount * (4*in - left - right - up - down)
__global__ void sharpen_laplacian(const std::uint8_t* in,
                                  std::uint8_t* out,
                                  int w, int h,
                                  float amount) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  int xm1 = clampi(x - 1, 0, w - 1);
  int xp1 = clampi(x + 1, 0, w - 1);
  int ym1 = clampi(y - 1, 0, h - 1);
  int yp1 = clampi(y + 1, 0, h - 1);

  int idx  = (y * w + x) * 3;
  int idxL = (y * w + xm1) * 3;
  int idxR = (y * w + xp1) * 3;
  int idxU = (ym1 * w + x) * 3;
  int idxD = (yp1 * w + x) * 3;

  #pragma unroll
  for (int c = 0; c < 3; ++c) {
    int center = (int)in[idx + c];
    int lap = 4 * center
              - (int)in[idxL + c] - (int)in[idxR + c]
              - (int)in[idxU + c] - (int)in[idxD + c];

    float vf = (float)center + amount * (float)lap;
    int v = (int)(vf + 0.5f);
    out[idx + c] = (std::uint8_t)clampi(v, 0, 255);
  }
}

void sharpen_rgb_u8(const std::uint8_t* d_in,
                    std::uint8_t* d_out,
                    int width,
                    int height,
                    float amount) {
  dim3 block(16,16);
  dim3 grid((width + 15) / 16, (height + 15) / 16);
  sharpen_laplacian<<<grid, block>>>(d_in, d_out, width, height, amount);
}

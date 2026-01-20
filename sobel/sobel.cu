#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ================= stb imports =================
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void sobelFilterShared(unsigned char *input,
                                  unsigned char *output,
                                  int width, int height)
{
    __shared__ unsigned char tile[18][18];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    if (x < width && y < height)
    {
        tile[ty][tx] = input[y * width + x];

        if (threadIdx.x == 0 && x > 0)
            tile[ty][0] = input[y * width + (x - 1)];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1)
            tile[ty][tx + 1] = input[y * width + (x + 1)];
        if (threadIdx.y == 0 && y > 0)
            tile[0][tx] = input[(y - 1) * width + x];
        if (threadIdx.y == blockDim.y - 1 && y < height - 1)
            tile[ty + 1][tx] = input[(y + 1) * width + x];
    }

    __syncthreads();

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)
    {
        if (x < width && y < height)
            output[y * width + x] = 0;
        return;
    }

    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    int sumX = 0, sumY = 0;

    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++)
        {
            int pixel = tile[ty + ky][tx + kx];
            sumX += pixel * Gx[ky + 1][kx + 1];
            sumY += pixel * Gy[ky + 1][kx + 1];
        }

    int mag = (int)sqrtf((float)(sumX * sumX + sumY * sumY));
    output[y * width + x] = (mag > 255) ? 255 : mag;
}

// Sobel for CPU
void sobelCPU(unsigned char *input,
              unsigned char *output,
              int width, int height)
{
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int sumX = 0, sumY = 0;

            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++)
                {
                    int pixel = input[(y + ky) * width + (x + kx)];
                    sumX += pixel * Gx[ky + 1][kx + 1];
                    sumY += pixel * Gy[ky + 1][kx + 1];
                }

            int mag = (int)sqrtf((float)(sumX * sumX + sumY * sumY));
            output[y * width + x] = (mag > 255) ? 255 : mag;
        }
    }
}

//
#define CUDA_CHECK(call)                        \
    do                                          \
    {                                           \
        cudaError_t err = call;                 \
        if (err != cudaSuccess)                 \
        {                                       \
            fprintf(stderr, "CUDA error: %s\n", \
                    cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                 \
        }                                       \
    } while (0)

// main function
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s input_image output_image\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *h_input =
        stbi_load(argv[1], &width, &height, &channels, 1);

    if (!h_input)
    {
        printf("Failed to load image: %s\n", argv[1]);
        return 1;
    }

    size_t imageSize = width * height;

    unsigned char *h_output_gpu =
        (unsigned char *)malloc(imageSize);
    unsigned char *h_output_cpu =
        (unsigned char *)malloc(imageSize);

    // GPU memory
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));

    CUDA_CHECK(cudaMemcpy(d_input, h_input,
                          imageSize,
                          cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Timing for GPU
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    sobelFilterShared<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    sobelFilterShared<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTimeMs, start, stop));

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output,
                          imageSize,
                          cudaMemcpyDeviceToHost));

    // Timing for CPU
    clock_t cpuStart = clock();
    sobelCPU(h_input, h_output_cpu, width, height);
    clock_t cpuEnd = clock();

    float cpuTimeMs =
        1000.0f * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    // Save output image
    stbi_write_png(argv[2],
                   width,
                   height,
                   1,
                   h_output_gpu,
                   width);

    // output
    printf("Image size: %d x %d\n", width, height);
    printf("GPU Sobel time: %.3f ms\n", gpuTimeMs);
    printf("CPU Sobel time: %.3f ms\n", cpuTimeMs);
    printf("Speedup: %.2fx\n", cpuTimeMs / gpuTimeMs);
    printf("Output written to %s\n", argv[2]);

    // cleanup
    stbi_image_free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}

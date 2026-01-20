#include "kernel.cu"
#include "support.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

Timer timer;

/* ===== JPEG RGB loader ===== */
void readJPEG_RGB(const char* filename,
                  Matrix& R, Matrix& G, Matrix& B,
                  int& width, int& height) {

    int channels;
    startTime(&timer);
    unsigned char* img = stbi_load(filename, &width, &height, &channels, 3);
    stopTime(&timer);
    printf("Image load time: %.6f s\n", elapsedTime(timer));

    if (!img) {
        printf("Failed to load image %s\n", filename);
        exit(1);
    }

    startTime(&timer);
    R = allocateMatrix(height, width);
    G = allocateMatrix(height, width);
    B = allocateMatrix(height, width);
    stopTime(&timer);
    printf("Host matrix allocation time: %.6f s\n", elapsedTime(timer));

    startTime(&timer);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            R.elements[y * R.pitch + x] = img[idx];
            G.elements[y * G.pitch + x] = img[idx + 1];
            B.elements[y * B.pitch + x] = img[idx + 2];
        }
    }
    stopTime(&timer);
    printf("Image to host matrices copy time: %.6f s\n", elapsedTime(timer));

    stbi_image_free(img);
}

/* ===== JPEG RGB writer ===== */
void writeJPEG_RGB(const char* filename,
                   Matrix R, Matrix G, Matrix B) {

    int width = R.width;
    int height = R.height;

    startTime(&timer);
    unsigned char* out = (unsigned char*)malloc(width * height * 3);
    stopTime(&timer);
    printf("Output buffer allocation time: %.6f s\n", elapsedTime(timer));

    startTime(&timer);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            float r = R.elements[y * R.pitch + x];
            float g = G.elements[y * G.pitch + x];
            float b = B.elements[y * B.pitch + x];

            out[idx]     = (unsigned char)fminf(fmaxf(r, 0.f), 255.f);
            out[idx + 1] = (unsigned char)fminf(fmaxf(g, 0.f), 255.f);
            out[idx + 2] = (unsigned char)fminf(fmaxf(b, 0.f), 255.f);
        }
    }
    stopTime(&timer);
    printf("Host matrices to output buffer copy time: %.6f s\n", elapsedTime(timer));

    startTime(&timer);
    stbi_write_jpg(filename, width, height, 3, out, 95);
    stopTime(&timer);
    printf("Image save time: %.6f s\n", elapsedTime(timer));

    free(out);
}

/* ===== Gaussian kernel ===== */
void generateGaussianKernel(Matrix M, float sigma) {
    int r = FILTER_SIZE / 2;
    float sum = 0.0f;

    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            float v = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            M.elements[(y + r) * M.pitch + (x + r)] = v;
            sum += v;
        }
    }

    for (int i = 0; i < FILTER_SIZE; i++)
        for (int j = 0; j < FILTER_SIZE; j++)
            M.elements[i * M.pitch + j] /= sum;
}

/* ===== Main ===== */
int main(int argc, char* argv[]) {

    if (argc != 3) {
        printf("Usage: %s input.jpg output.jpg\n", argv[0]);
        return 0;
    }

    int width, height;
    Matrix R_h, G_h, B_h;
    Matrix R_blur_h, G_blur_h, B_blur_h;

    /* ===== Read input image ===== */
    readJPEG_RGB(argv[1], R_h, G_h, B_h, width, height);

    /* ===== Allocate output host matrices ===== */
    startTime(&timer);
    R_blur_h = allocateMatrix(height, width);
    G_blur_h = allocateMatrix(height, width);
    B_blur_h = allocateMatrix(height, width);
    stopTime(&timer);
    printf("Host output allocation time: %.6f s\n", elapsedTime(timer));

    /* ===== Gaussian kernel ===== */
    Matrix M_h = allocateMatrix(FILTER_SIZE, FILTER_SIZE);
    generateGaussianKernel(M_h, 5.0f);
    cudaMemcpyToSymbol(M_c, M_h.elements,
                       FILTER_SIZE * FILTER_SIZE * sizeof(float));

    /* ===== Allocate device matrices ===== */
    startTime(&timer);
    Matrix R_d = allocateDeviceMatrix(height, width);
    Matrix G_d = allocateDeviceMatrix(height, width);
    Matrix B_d = allocateDeviceMatrix(height, width);

    Matrix R_blur_d = allocateDeviceMatrix(height, width);
    Matrix G_blur_d = allocateDeviceMatrix(height, width);
    Matrix B_blur_d = allocateDeviceMatrix(height, width);
    stopTime(&timer);
    printf("Device memory allocation time: %.6f s\n", elapsedTime(timer));

    /* ===== Copy input to device ===== */
    startTime(&timer);
    copyToDeviceMatrix(R_d, R_h);
    copyToDeviceMatrix(G_d, G_h);
    copyToDeviceMatrix(B_d, B_h);
    stopTime(&timer);
    printf("Host to device copy time: %.6f s\n", elapsedTime(timer));

    /* ===== Kernel launch ===== */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
              (height + TILE_SIZE - 1) / TILE_SIZE);

    printf("Launching kernel...\n");
    fflush(stdout);

    startTime(&timer);
    gaussianConvolution<<<grid, block>>>(R_d, R_blur_d);
    gaussianConvolution<<<grid, block>>>(G_d, G_blur_d);
    gaussianConvolution<<<grid, block>>>(B_d, B_blur_d);

    cudaError_t cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
        FATAL("Unable to launch/execute kernel");

    stopTime(&timer);
    printf("Kernel execution time: %.6f s\n", elapsedTime(timer));

    /* ===== Copy output back ===== */
    startTime(&timer);
    copyFromDeviceMatrix(R_blur_h, R_blur_d);
    copyFromDeviceMatrix(G_blur_h, G_blur_d);
    copyFromDeviceMatrix(B_blur_h, B_blur_d);
    stopTime(&timer);
    printf("Device to host copy time: %.6f s\n", elapsedTime(timer));

    /* ===== Save output image ===== */
    writeJPEG_RGB(argv[2], R_blur_h, G_blur_h, B_blur_h);

    /* ===== Cleanup ===== */
    freeMatrix(R_h); freeMatrix(G_h); freeMatrix(B_h);
    freeMatrix(R_blur_h); freeMatrix(G_blur_h); freeMatrix(B_blur_h);
    freeMatrix(M_h);

    freeDeviceMatrix(R_d); freeDeviceMatrix(G_d); freeDeviceMatrix(B_d);
    freeDeviceMatrix(R_blur_d); freeDeviceMatrix(G_blur_d); freeDeviceMatrix(B_blur_d);

    return 0;
}

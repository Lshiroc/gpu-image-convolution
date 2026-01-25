#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

void generateGaussianKernel(vector<float>& kernel, int radius, float sigma) {
    float sum = 0.0f;
    int size = 2 * radius + 1;
    kernel.resize(size * size);
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float exponent = -(x * x + y * y) / (2 * sigma * sigma);
            float value = exp(exponent) / (2 * M_PI * sigma * sigma);
            kernel[(y + radius) * size + (x + radius)] = value;
            sum += value;
        }
    }
    for (int i = 0; i < size * size; ++i) kernel[i] /= sum;
}


void cpuGaussianBlur(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius, float sigma) {
    vector<float> kernel;
    generateGaussianKernel(kernel, radius, sigma);
    int kernelSize = 2 * radius + 1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int ky = -radius; ky <= radius; ++ky) {
                    for (int kx = -radius; kx <= radius; ++kx) {
                        int nX = min(max(x + kx, 0), width - 1);
                        int nY = min(max(y + ky, 0), height - 1);
                        
                        float kVal = kernel[(ky + radius) * kernelSize + (kx + radius)];
                        unsigned char pVal = input[(nY * width + nX) * channels + c];
                        sum += kVal * pVal;
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)sum;
            }
        }
    }
}

void cpuGaussianSeparable(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius, float sigma) {
    int kernelSize = 2 * radius + 1;
    vector<float> kernel(kernelSize);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        float val = exp(-(i * i) / (2 * sigma * sigma));
        kernel[i + radius] = val;
        sum += val;
    }
    for (int i = 0; i < kernelSize; ++i) kernel[i] /= sum;

    unsigned char* temp = (unsigned char*)malloc(width * height * channels);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float pixelSum = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int nx = min(max(x + k, 0), width - 1);
                    pixelSum += input[(y * width + nx) * channels + c] * kernel[k + radius];
                }
                temp[(y * width + x) * channels + c] = (unsigned char)pixelSum;
            }
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float pixelSum = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int ny = min(max(y + k, 0), height - 1); \
                    pixelSum += temp[(ny * width + x) * channels + c] * kernel[k + radius];
                }
                output[(y * width + x) * channels + c] = (unsigned char)pixelSum;
            }
        }
    }

    free(temp);
}

int main(int argc, char** argv) {
    const char* inputFile = (argc >= 2) ? argv[1] : "image.jpg";
    int width, height, channels;

    unsigned char* img = stbi_load(inputFile, &width, &height, &channels, 0);
    if (img == NULL) {
        cout << "Error: Could not load " << inputFile << endl;
        return -1;
    }
    cout << "Loaded: " << width << "x" << height << " channels: " << channels << endl;

    unsigned char* outImg = (unsigned char*)malloc(width * height * channels);

    cout << "--- Starting CPU Benchmark ---" << endl;
    
    auto startCPU = std::chrono::high_resolution_clock::now();

    cpuGaussianBlur(img, outImg, width, height, channels, 10, 5.0f);
    
    auto endCPU = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsedCPU = endCPU - startCPU;
    cout << "CPU Time: " << elapsedCPU.count() << " ms" << endl;


    cout << "--- Starting Optimized CPU ---" << endl;
    
    auto startOptimizedCPU = std::chrono::high_resolution_clock::now();

    cpuGaussianSeparable(img, outImg, width, height, channels, 10, 5.0f); // Takes ~20s
    
    auto endOptimizedCPU = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsedOptimizedCPU = endOptimizedCPU - startOptimizedCPU;
    cout << "Optimized CPU Time: " << elapsedOptimizedCPU.count() << " ms" << endl;


    stbi_write_png("cpu_result.png", width, height, channels, outImg, width * channels);
    cout << "Saved to cpu_result.png" << endl;


    
    stbi_image_free(img);
    free(outImg);

    return 0;
}
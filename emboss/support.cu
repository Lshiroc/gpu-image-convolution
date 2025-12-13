#include <iostream>
#include "support.h"

Image allocateImageDevice(int width, int height) {
	Image img;
	img.width = width;
	img.height = height;
	cudaError_t cudaErr = cudaMalloc((void**)&img.elements, width * height * 3 * sizeof(unsigned char));
	if(cudaErr != cudaSuccess)
		std::cout << "\nCould not allocate image" << std::endl;

	return img;
}

Matrix allocateMatrix(int width, int height) {
	Matrix mat;
	mat.width = width;
	mat.height = height;
	mat.elements = (int *)malloc(width * height * sizeof(int));
	if (mat.elements == NULL) 
		std::cout << "Could not allocate Matrix" << std::endl;

	return mat;
}

void copyFromHostToDevice(Image dst, const uchar *src) {
	cudaError_t cudaErr = cudaMemcpy(dst.elements, src, dst.width * dst.height * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
	if(cudaErr != cudaSuccess)
		std::cout << "\nCould not copy from host to device" << std::endl;
}

void copyFromDeviceToHost(uchar *dst, const Image src) {
	cudaError_t cudaErr = cudaMemcpy(dst, src.elements, src.width * src.height * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);
	if(cudaErr != cudaSuccess)
		std::cout << "\nCould not copy from device to host" << std::endl;
}


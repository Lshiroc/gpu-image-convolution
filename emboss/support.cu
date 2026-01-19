#include "support.h"
#include <cstdlib>
#include <iostream>
#include <string>

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

std::string substr(const char *str, size_t start, size_t end) {
	size_t length = len(str);

	if (start >= length || start >= end) {
		return "";
	}

	if (end > length) {
		end = length;
	}

	return std::string(str + start, end - start);
}

int parseInt(std::string str) {
	int parsedInt{};
	if ((parsedInt = std::stoi(str)) == 0) {
		if (str != "0") {
			return -1;
		}
	}

	return parsedInt;
}

size_t len(const char *str) {
	size_t length = 0;
	char const *p = str;
	while (*(p++) != '\0') {
		length++;
	}

	return length;
}


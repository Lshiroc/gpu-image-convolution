#include "support.h"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

Image allocateImageDevice(int width, int height) {
	Image img;
	img.width = width;
	img.height = height;
	cudaError_t cudaErr = cudaMalloc((void**)&img.elements, width * height * 3 * sizeof(unsigned char));
	if(cudaErr != cudaSuccess)
		std::cout << "\nCould not allocate image\n";

	return img;
}

Image allocateImage(int width, int height) {
	Image img;
	img.width = width;
	img.height = height;
	img.elements = (uchar *)malloc(width * height * 3 * sizeof(unsigned char));
	if (img.elements == NULL) 
		std::cout << "\nCould not allocate image\n"; 

	return img;
}

Matrix allocateMatrix(int width, int height) {
	Matrix mat;
	mat.width = width;
	mat.height = height;
	mat.elements = (float *)malloc(width * height * sizeof(float));
	if (mat.elements == NULL) 
		std::cout << "\nCould not allocate Matrix\n"; 
	return mat;
}

void copyFromHostToDevice(Image dst, const uchar *src) {
	cudaError_t cudaErr = cudaMemcpy(dst.elements, src, dst.width * dst.height * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
	if(cudaErr != cudaSuccess)
		std::cout << "\nCould not copy from host to device\n";
}

void copyFromDeviceToHost(uchar *dst, const Image src) {
	cudaError_t cudaErr = cudaMemcpy(dst, src.elements, src.width * src.height * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);
	if(cudaErr != cudaSuccess)
		std::cout << "\nCould not copy from device to host\n";
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

Matrix initializeFilter(int width, int height, float angle) {
	Matrix filter = allocateMatrix(width, height);
	std::vector<std::vector<float>> Gx(width, std::vector<float>(width));
	std::vector<std::vector<float>> Gy(width, std::vector<float>(width));

	int center = width / 2;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			int x = j - center;
			int y = i - center;

			Gx[i][j] = (float)x;
			Gy[i][j] = (float)y;
		}
	}

	float rad = angle * M_PI / 180.0f;
	float cosTheta = std::cos(rad);
	float sinTheta = std::sin(rad);

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			filter.elements[i*width + j] = cosTheta*Gx[i][j] + sinTheta*Gy[i][j];
		}
	}

	return filter;
}

void startTime(Timer *timer) {
	gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer *timer) {
	gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
	return ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec) 
		+ (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

void help() {
	std::cout << "usage: emboss <filepath> [--grayscale] [--depth] [--angle] [--help]\n\n";
	std::cout << "\t<filepath>\tThe file path for the target image\n";
	std::cout << "\t--grayscale\tApply grayscale filter on top of emboss filter\n";
	std::cout << "\t--depth\t\tDepth of the emboss filter defined by floating point number between 0 and 1\n";
	std::cout << "\t--angle\t\tSets an angle to apply the emboss filter on a specific angle\n";
	std::cout << "\t--help\t\tPrints information on parameters\n";
}


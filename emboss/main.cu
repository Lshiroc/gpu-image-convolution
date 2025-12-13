#include <iostream>
#include "support.h"
#include "kernel.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
	cv::Mat img_h;
	Image img_d;
	Matrix M_h;

	if (argc == 1) {
		cout << "No image given\n";
		return 0;
	} else if (argc == 2) {
		string img_path = cv::samples::findFile("../img2.png");
		img_h = cv::imread(img_path, cv::IMREAD_COLOR);
	} else {
		cout << "Extra arguments provided, only 1 required\n";
		return 0;
	}

	img_d = allocateImageDevice(img_h.cols, img_h.rows);
	M_h = allocateMatrix(FILTER_SIZE, FILTER_SIZE);

	// Initialize Filter
	M_h.elements[0] = -2;
	M_h.elements[1] = -1;
	M_h.elements[2] = 0;
	M_h.elements[3] = -1;
	M_h.elements[4] = 1;
	M_h.elements[5] = 1;
	M_h.elements[6] = 0;
	M_h.elements[7] = 1;
	M_h.elements[8] = 2;

/*	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			if (j == 0)
				M_h.elements[i*FILTER_SIZE + j] = -1;
			else if (j == FILTER_SIZE - 1)
				M_h.elements[i*FILTER_SIZE + j] = 1;
		}
	}*/

	copyFromHostToDevice(img_d, img_h.data);
	cudaError_t cudaSymbolErr = cudaMemcpyToSymbol(M_c, M_h.elements,
								FILTER_SIZE * FILTER_SIZE * sizeof(int));

	if (cudaSymbolErr != cudaSuccess)
		cout << "Error during cudaMemcpyToSymbol: " << cudaGetErrorString(cudaSymbolErr) << endl;
	cudaDeviceSynchronize();

	// Launch kernel
	cudaError_t cudaErr = cudaDeviceSynchronize();
	if (cudaErr != cudaSuccess) {
		cout << "Could not launch kernel" << endl;
		// Create your own FATAL()
		return 0;
	}

	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((img_h.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, 
			   (img_h.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
	apply_emboss<<<dim_grid, dim_block>>>(img_d);
	cudaDeviceSynchronize();

	copyFromDeviceToHost(img_h.data, img_d);
	cv::imshow("Emboss filtered image", img_h);
	waitKey(0);

//	cudaFree(img_d);
//	free(M_h);

	return 0;
}


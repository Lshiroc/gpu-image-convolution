#include "support.h"
#include "parser.h"
#include "kernel.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char const *argv[]) {
	cv::Mat img_h;
	Image img_in_d;
	Image img_out_d;
	Matrix M_h;
	Options opts;

	opts = parser(argc, argv);
	img_h = cv::imread(opts.imgPath, cv::IMREAD_COLOR);
	if (!img_h.isContinuous()) {
		img_h = img_h.clone();
	}

	img_in_d = allocateImageDevice(img_h.cols, img_h.rows);
	img_out_d = allocateImageDevice(img_h.cols, img_h.rows);
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

	copyFromHostToDevice(img_in_d, img_h.data);
	cudaError_t cudaSymbolErr = cudaMemcpyToSymbol(M_c, M_h.elements,
								FILTER_SIZE * FILTER_SIZE * sizeof(int));

	if (cudaSymbolErr != cudaSuccess)
		std::cout << "Error during cudaMemcpyToSymbol: " << cudaGetErrorString(cudaSymbolErr) << '\n';
	cudaDeviceSynchronize();

	// Launch kernel
	cudaError_t cudaErr = cudaDeviceSynchronize();
	if (cudaErr != cudaSuccess) {
		std::cout << "Could not launch kernel" << '\n';
		// Create your own FATAL()
		return 0;
	}

	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((img_h.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, 
			   (img_h.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (opts.grayscale) {
		apply_grayscale<<<dim_grid, dim_block>>>(img_in_d);
	}
	apply_emboss<<<dim_grid, dim_block>>>(img_in_d, img_out_d, opts);
	cudaDeviceSynchronize();

	copyFromDeviceToHost(img_h.data, img_out_d);
	cv::imshow("Emboss filtered image", img_h);
	cv::waitKey(0);
	cv::imwrite("linux_gray_embossed.jpg", img_h);

	cudaFree(img_in_d.elements);
	cudaFree(img_out_d.elements);
    free(M_h.elements);

	return 0;
}


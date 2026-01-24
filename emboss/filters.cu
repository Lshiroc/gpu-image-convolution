#include "filters.h"
#include "support.h"
#include "parser.h"
#include "kernel.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

void emboss_gpu(cv::Mat *img, Options opts) {
	Matrix M_h;
	Image img_in_d;
	Image img_out_d;
	Timer timer;

	// Initializing the variables
	std::cout << "Initializing variables\n";
	startTime(&timer);

	img_in_d = allocateImageDevice(img->cols, img->rows);
	img_out_d = allocateImageDevice(img->cols, img->rows);
	M_h = initializeFilter(FILTER_SIZE, FILTER_SIZE, opts.angle);

	stopTime(&timer);
	std::cout << elapsedTime(timer) << "s\n";

	// Copying data from host to device
	std::cout << "Copying data from host to device\n";
	startTime(&timer);

	copyFromHostToDevice(img_in_d, img->data);
	cudaError_t cudaSymbolErr = cudaMemcpyToSymbol(M_c, M_h.elements,
								FILTER_SIZE * FILTER_SIZE * sizeof(int));
	if (cudaSymbolErr != cudaSuccess) {
		std::cerr << "Error during cudaMemcpyToSymbol: " << cudaGetErrorString(cudaSymbolErr) << '\n';
	}
	cudaDeviceSynchronize();

	stopTime(&timer);
	std::cout << elapsedTime(timer) << "s\n";

	// Launch kernel
	std::cout << "Launching kernel\n";
	startTime(&timer);

	cudaError_t cudaErr = cudaDeviceSynchronize();
	if (cudaErr != cudaSuccess) {
		std::cerr << "Could not launch kernel" << '\n';
		exit(1);
	}
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((img->cols + BLOCK_SIZE - 1) / BLOCK_SIZE, 
			   (img->rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
	apply_emboss<<<dim_grid, dim_block>>>(img_in_d, img_out_d, opts);
	cudaDeviceSynchronize();

	stopTime(&timer);
	std::cout << elapsedTime(timer) << "s\n";

	// Copying data from device to host
	std::cout << "Copying data from device to host\n";
	startTime(&timer);

	copyFromDeviceToHost(img->data, img_out_d);

	stopTime(&timer);
	std::cout << elapsedTime(timer) << "s\n";

	// Free memory
	cudaFree(img_in_d.elements);
	cudaFree(img_out_d.elements);
    free(M_h.elements);
}

void emboss_cpu(cv::Mat *img, Options opts) {
	Matrix M_h;
	Image img_in;
	Image img_out;
	Timer timer;

	// Initializing the variables
	std::cout << "Initializing variables\n";
	startTime(&timer);

	img_in = allocateImage(img->cols, img->rows);
	img_out = allocateImage(img->cols, img->rows);
	memcpy(img_in.elements, img->data,
       img->cols * img->rows * img->elemSize());
	M_h = initializeFilter(FILTER_SIZE, FILTER_SIZE, opts.angle);

	stopTime(&timer);
	std::cout << elapsedTime(timer) << "s\n";

	// Applying filter
	std::cout << "Applying filter\n";
	startTime(&timer);

	emboss_cpu_impl(img_in, &img_out, M_h, opts);

	stopTime(&timer);
	std::cout << elapsedTime(timer) << "s\n";

	memcpy(img->data, img_out.elements,
		img_out.width * img_out.height * 3 * sizeof(uchar));

	// Free memory
	free(img_in.elements);
	free(img_out.elements);
    free(M_h.elements);
}

void emboss_cpu_impl(Image img_in, Image *img_out, Matrix M_h, Options opts) {
	for (int row = 0; row < img_in.height; row++) {
		for (int col = 0; col < img_in.width; col++) {
			if (row >= img_in.height || col >= img_in.width) {
				return;
			}

			int red_sum = 0;
			int green_sum = 0;
			int blue_sum = 0;
			for (int i = 0; i < FILTER_SIZE; i++) {
				for (int j = 0; j < FILTER_SIZE; j++) {
					int inner_row = row - FILTER_SIZE / 2 + i;
					int inner_col = col - FILTER_SIZE / 2 + j;

					if (inner_row < img_in.height && inner_row >= 0 && inner_col < img_in.width && inner_col >= 0) {
						int loc = inner_row*img_in.width + inner_col;
						int filter_idx = i*FILTER_SIZE + j;

						red_sum += (int)img_in.elements[loc*3]*M_h.elements[filter_idx];
						green_sum += (int)img_in.elements[loc*3 + 1]*M_h.elements[filter_idx];		
						blue_sum += (int)img_in.elements[loc*3 + 2]*M_h.elements[filter_idx];
					}
				}
			}

			int outLoc = row*img_in.width + col;
			float r = opts.depth * red_sum;
			float g = opts.depth * green_sum;
			float b = opts.depth * blue_sum;
			if (opts.grayscale) {
				r = g = b = 0.299f*r + 0.587f*g + 0.114f*b;
				r += 128;
				g += 128;
				b += 128;
			} else {
				r += img_in.elements[outLoc*3];
				g += img_in.elements[outLoc*3 + 1];
				b += img_in.elements[outLoc*3 + 2];
			}

			img_out->elements[outLoc*3] = (uchar)max(0, min(255, (int)r));
			img_out->elements[outLoc*3 + 1] = (uchar)max(0, min(255, (int)g));
			img_out->elements[outLoc*3 + 2] = (uchar)max(0, min(255, (int)b));
		}
	}
}


#include "support.h"
#include "parser.h"
#include <stdio.h>

__constant__ int M_c[FILTER_SIZE * FILTER_SIZE];

__global__ void apply_emboss(Image img_in, Image img_out, Options opts) {
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

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

				red_sum += (int)img_in.elements[loc*3]*M_c[filter_idx];
				green_sum += (int)img_in.elements[loc*3 + 1]*M_c[filter_idx];		
				blue_sum += (int)img_in.elements[loc*3 + 2]*M_c[filter_idx];
			}
		}
	}

	int outLoc = row*img_in.width + col;
	img_out.elements[outLoc*3] = (uchar) max(0, min(255, red_sum));
	img_out.elements[outLoc*3 + 1] = (uchar) max(0, min(255, green_sum));
	img_out.elements[outLoc*3 + 2] = (uchar) max(0, min(255, blue_sum));
}

__global__ void apply_grayscale(Image img_in) {
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if (row >= img_in.height || col >= img_in.width) {
		return;
	}

	int loc = row*img_in.width + col;
	int gray = 0.2126*img_in.elements[loc*3] + 0.7152*img_in.elements[loc*3 + 1] + 0.0722*img_in.elements[loc*3 + 2];
	img_in.elements[loc*3] = gray;
	img_in.elements[loc*3 + 1] = gray;
	img_in.elements[loc*3 + 2] = gray;
}


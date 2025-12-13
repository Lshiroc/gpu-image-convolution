#include <stdio.h>
#include "support.h"

__constant__ int M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void apply_emboss(Image img) {
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;	

	int rSum = 0;
	int gSum = 0;
	int bSum = 0;
	for (int i = 0; i < FILTER_SIZE; i++)	{
		for (int j = 0; j < FILTER_SIZE; j++) {
			int iRow = row - FILTER_SIZE / 2 + i;
			int iCol = col - FILTER_SIZE / 2 + j;
			int loc = iRow*img.width + iCol;

			if (iRow < img.height && loc >= 0 && iCol < img.width && iCol >= 0) {
				rSum += (int)img.elements[loc*3]*M_c[i][j];
				gSum += (int)img.elements[loc*3 + 1]*M_c[i][j];		
				bSum += (int)img.elements[loc*3 + 2]*M_c[i][j];
			} else {
				rSum += (int)0;
				gSum += (int)0;
				bSum += (int)0;
			}
		}
	}
	//printf("\n%d %d: r: %d, g: %d, b: %d\n", row, col, rSum, gSum, bSum);
	int outLoc = (row*img.width + col)*3;
	img.elements[outLoc] = (uchar)max(0, min(255, rSum + 128)); // maybe add + 128
	img.elements[outLoc + 1] = (uchar)max(0, min(255, gSum + 128));
	img.elements[outLoc + 2] = (uchar)max(0, min(255, bSum + 128));
}


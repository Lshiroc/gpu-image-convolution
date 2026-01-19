#ifndef KERNEL_H
#define KERNEL_H

#include "support.h"
#include "parser.h"

__global__ void apply_emboss(Image img_in, Image img_out, Options opts);
__global__ void apply_grayscale(Image img_in);
extern __constant__ int M_c[FILTER_SIZE * FILTER_SIZE];

#endif


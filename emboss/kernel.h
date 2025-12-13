#include "support.h"

__global__ void apply_emboss(Image img);
extern __constant__ int M_c[FILTER_SIZE][FILTER_SIZE];


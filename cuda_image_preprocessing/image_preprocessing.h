#include "cuda_runtime.h"
#include <stdio.h>

__global__ void image_resize(const unsigned char* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight);
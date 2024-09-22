#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void resizeImage(const unsigned char* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight);

void resize2GPU(const unsigned char* src, int w, int h, float* dst, int dstWidth, int dstHeight);
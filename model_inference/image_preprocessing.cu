#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "image_preprocessing.h"

using namespace std;

__global__ void resizeImage(const unsigned char* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight)
{

    int dstX = blockIdx.x;  // 目标图像的 x 坐标
    int dstY = blockIdx.y;  // 目标图像的 y 坐标

    if (dstX < dstWidth && dstY < dstHeight)
    {
        // 计算目标图像中的线性索引
        int dstIndex0 = 0 * dstHeight * dstWidth + dstY * dstWidth + dstX;
        int dstIndex1 = 1 * dstHeight * dstWidth + dstY * dstWidth + dstX;
        int dstIndex2 = 2 * dstHeight * dstWidth + dstY * dstWidth + dstX;

        // 计算源图像中对应的位置
        float srcX = (float)dstX * srcWidth / dstWidth;
        float srcY = (float)dstY * srcHeight / dstHeight;

        // 计算源图像中的线性索引
        int srcIndex0 = 0 + (int)srcY * srcWidth * 3 + (int)srcX * 3;
        int srcIndex1 = 1 + (int)srcY * srcWidth * 3 + (int)srcX * 3;
        int srcIndex2 = 2 + (int)srcY * srcWidth * 3 + (int)srcX * 3;

        // 将像素从源图像复制到目标图像
        dst[dstIndex0] = (float)src[srcIndex2] / 255.0;
        dst[dstIndex1] = (float)src[srcIndex1] / 255.0;
        dst[dstIndex2] = (float)src[srcIndex0] / 255.0;


    }
}
void resize2GPU(const unsigned char* src, int w, int h, float* dst, int dstWidth, int dstHeight)
{
    dim3 blocks((16 * dstWidth + 15) / 16, (16 * dstHeight + 15) / 16);
    dim3 threads(1);
    resizeImage << < blocks, threads >> > (src, w, h, dst, dstWidth, dstHeight);
    cudaDeviceSynchronize();

}
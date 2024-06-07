#include <iostream>
#include <fstream>
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include <cuda_runtime_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include "device_functions.h"
#include "image_preprocessing.h"

using namespace cv;
using namespace cv::dnn;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;


// const char* classNames[] = {'person'}; // 类别标签列表
const char* classNames[] = { "reflective_clothes", "other_clothes", "hat", "person" }; // 类别标签列表

class Logger : public ILogger
{
    //void log(Severity severity, const char* msg) override
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;


int main() {


    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout <<"can not open camera"<< std::endl;
        return 0;
    }
    cv::Mat img;
    cap.read(img);
    int h = img.rows;//获取图片的长
    int w = img.cols;//获取图片的宽

    // 反序列化
    IRuntime* runtime = createInferRuntime(gLogger);
    std::string cached_path = "engine.trt";
    std::ifstream trtModelFile(cached_path, std::ios_base::in | std::ios_base::binary);
    trtModelFile.seekg(0, ios::end);
    int size = trtModelFile.tellg();
    trtModelFile.seekg(0, ios::beg);

    char* buff = new char[size];
    trtModelFile.read(buff, size);
    trtModelFile.close();
    ICudaEngine* re_engine = runtime->deserializeCudaEngine((void*)buff, size, NULL);
    delete[] buff;


    //创建buffers 指向输入输出流
    float* buffers[2];
    //void** buffers = new void* [2];
    int inputIndex=1;
    int outputIndex=1;
    int numInputs = 0;
    int numOutputs = 0;
    int numBindings = re_engine->getNbBindings();
    for (int bi = 0; bi < numBindings; bi++)
    {
        if (re_engine->bindingIsInput(bi) == true) {
            inputIndex = bi;
            numInputs++;
        }
        else {

            numOutputs++;
            outputIndex = bi;
        }
    }
    int input_size = 1;
    int dimSize = 1;
    const nvinfer1::Dims inputDims = re_engine->getBindingDimensions(inputIndex);
    int numDims = inputDims.nbDims; // 获取维度数量
    for (int i = 0; i < numDims; i++) {
        dimSize = inputDims.d[i]; // 获取每个维度的大小
        input_size = input_size * dimSize;
        std::cout << "Dimension " << i << ": " << dimSize << std::endl;
    }
    int input_shape1 = inputDims.d[numDims - 2];
    int input_shape2 = inputDims.d[numDims - 1];

    cout << endl;
    cout << "intput_size:" << input_size << endl;
    cout << endl;
    int output_size = 1;
    const nvinfer1::Dims outputDims = re_engine->getBindingDimensions(outputIndex);
    numDims = outputDims.nbDims; // 获取维度数量
    for (int i = 0; i < numDims; i++) {
        dimSize = outputDims.d[i]; // 获取每个维度的大小
        output_size = output_size * dimSize;
        std::cout << "Dimension " << i << ": " << dimSize << std::endl;
    }
    cout << endl;
    cout << "output_size:" << output_size << endl;

    
    uchar* devSrc;
    // 分配buffers空间
    cudaMalloc(&devSrc, w * h * 3 * sizeof(uchar));
    cudaMalloc(&buffers[inputIndex], input_size * sizeof(float));
    cudaMalloc(&buffers[outputIndex], output_size * sizeof(float));
    //创建cuda流 cuda网络 cuda块
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);
    dim3 threads(16, 16);


    // 设置过滤阈值
    float threshold = 0.5;
    float* result_array = new float[output_size];

    //创建context
    IExecutionContext* context = re_engine->createExecutionContext();
    // 图片处理
    while (1)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cap.read(img);
        
        //cap >> img;
        cudaMemcpy(devSrc, img.ptr<uchar>(), w * h * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        //Mat inputBlob = blobFromImage(img, 1.0 /255.0, Size(input_shape1, input_shape2), Scalar(0, 0, 0), true, false, CV_32FC1);
        resize2GPU(devSrc, w, h, buffers[inputIndex], input_shape1, input_shape2);
        //复制图片数据到GPU

        //cudaMemcpyAsync(buffers[inputIndex], inputBlob.ptr<float>(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream);

        //执行推理
        context->enqueueV2((void* const*)buffers, stream, nullptr);
        
        // 将GPU数据拷贝回CPU

        cudaMemcpyAsync(result_array, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
        
        for (int j = 0; j < output_size; j = j + 9)
        {

            float* data = result_array + j;
            float* class_scores = data + 5;
            auto max_it = std::max_element(class_scores, class_scores + 4);
            int classId = std::distance(class_scores, max_it);
            float score = data[4] * data[5 + classId];
            if (score > threshold)
            {
                int left = (int)(((data[0] - (data[2] / 2)) / 640) * w);
                int top = (int)(((data[1] - (data[3] / 2)) / 640) * h);
                int width1 = (int)(((data[2]) / 640) * w);
                int height1 = (int)(((data[3]) / 640) * h);
                Rect box(left, top, width1, height1);
                String label = classNames[classId];
                putText(img, label, Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
                rectangle(img, box, Scalar(0, 255, 0), 2);
            }
        }
        // 计算循环时间
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // 在图像右上角绘制帧数信息
        double fps = 1000.0 / duration.count(); // 每秒的帧数
        std::ostringstream ss;
        ss << "FPS: " << fps;
        cv::putText(img, ss.str(), cv::Point(img.cols - 150, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        
        
        imshow("Detection Results", img);
        waitKey(1);
    }
    //释放资源
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();
    re_engine->destroy();
    runtime->destroy();
    cap.release();
    return 0;
}

#ifndef __OPERATION_H
#define __OPERATION_H
//#include <pybind11/pybind11.h>
#include <iostream>
#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

////////////////////////////
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
//#include<pybind11/pybind11.h>
//#include<pybind11/numpy.h>
//#include<pybind11/stl.h>
////////////////////////////

using namespace nvinfer1;

std::string strDecode(std::vector<int>& preds, bool raw);

std::map<std::string, Weights> loadWeights(const std::string file);

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);

ILayer* convRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int i, bool use_bn);

ILayer* addLSTM(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int nHidden, std::string lname);

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);


class TensorRT
{
    private:
        char *trtModelStream{nullptr};
        size_t size{0};

        IRuntime* runtime;
        ICudaEngine* engine;
        IExecutionContext* context;

        void* buffers[2];
        int inputIndex;     //const
        int outputIndex;    //const



        // Create stream
        cudaStream_t stream;
    public:
        TensorRT();
	//~TensorRT();
	void memory_free();
        int run();
};

void splitLstmWeights(std::map<std::string, Weights>& weightMap, std::string lname);

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize); 

int run();
#endif


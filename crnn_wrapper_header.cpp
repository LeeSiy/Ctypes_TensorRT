#include "crnn_header.h"
#include "crnn_wrapper_header.h"
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

//using namespace nvinfer1;

extern "C" {

	std::string main_strDecode(std::vector<int>& preds, bool raw){
		return strDecode(preds, raw);
	}
	
	std::map<std::string, Weights> main_loadWeights(const std::string file){
		return loadWeights(file);
	}

        IScaleLayer* main_addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps){
	        return addBatchNorm2d(network, weightMap, input, lname, eps);	
        }

        ILayer* main_convRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int i, bool use_bn){
		return convRelu(network, weightMap, input, i, use_bn = false);
	}

        ILayer* main_addLSTM(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int nHidden, std::string lname){
		return addLSTM(network, weightMap, input, nHidden, lname);
	}

        ICudaEngine* main_createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt){
                return createEngine(maxBatchSize, builder, config, dt);
        }		
       	
        TensorRT* newTensorRT() {
                return new TensorRT();
        } 
        void TensorRT_TensorRT(TensorRT* tr) {
                TensorRT();
		tr->~TensorRT();
	}

        int TensorRT_run(TensorRT* tr) {
                return tr->run();
        }
        
	void TensorRT_memory_free(TensorRT* tr){
		tr->memory_free();
	}

        int main_run() {
                return run();
        }
        void main_splitLstmWeights(std::map<std::string, Weights>& weightMap, std::string lname){
		splitLstmWeights( weightMap, lname);
        }

	void main_APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream){
                APIToModel(maxBatchSize, modelStream);
	}

        void main_doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
                doInference(context, stream, buffers, input, output, batchSize);
        }

}

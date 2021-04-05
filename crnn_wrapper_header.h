#ifndef __MYWRAPPER_H
#define __MYWRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

std::string main_strDecode(std::vector<int>& preds, bool raw);

std::map<std::string, Weights> main_loadWeights(const std::string file);

IScaleLayer* main_addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);

ILayer* main_convRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int i, bool use_bn);

ILayer* main_addLSTM(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int nHidden, std::string lname);	

ICudaEngine* main_createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);

typedef struct TensorRT TensorRT;
TensorRT* newTensorRT();
void TensorRT_TensorRT(TensorRT* tr);
int TensorRT_run(TensorRT* tr);
void TensorRT_memory_free(TensorRT* tr);
int main_run();
void main_splitLstmWeights(std::map<std::string, Weights>& weightMap, std::string lname);
void main_APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);
void main_doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);

#ifdef __cplusplus
}
#endif
#endif


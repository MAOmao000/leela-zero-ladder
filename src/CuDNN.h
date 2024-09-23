/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Henrik Forsten

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CUDNN_H_INCLUDED
#define CUDNN_H_INCLUDED

#include "config.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <cudnn.h>
#include <cublas_v2.h>
#if defined(USE_CUDNN_GRAPH)
#include <cudnn_frontend.h>
#endif

#if defined(USE_TENSOR_RT)
#include <stdlib.h>
#include <fstream>
#include <ostream>
#include <iostream>
#include <new>
#include <string>
#include <numeric>
#include <type_traits>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <map>
#include <iterator>
#include <filesystem>
#include <stdarg.h>

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferRuntimeBase.h"
#include "NvInferSafeRuntime.h"
#include "NvInferConsistency.h"

#include "sha2.h"
#endif

template <typename net_t> class CuDNN_Network;

#if defined(USE_CUDNN_GRAPH)
#define checkCUDNNFE(expression)                                  \
    {                                                             \
        cudnn_frontend::error_t err = (expression);               \
        if (err.is_bad()) {                                       \
            myprintf_error("Error on %s(%d): %s\n",               \
                __FILE__, __LINE__, err.get_message().c_str());   \
            throw std::runtime_error("cuDNN FrontEnd error");     \
        }                                                         \
    }
#endif

#if defined(USE_TENSOR_RT)
#define ASSERT(condition)                                         \
    do {                                                          \
        if (!(condition)) {                                       \
            myprintf_error("Assertion failure %s(%d): %s\n",      \
                __FILE__, __LINE__, #condition);                  \
            throw std::runtime_error("TensorRT error");           \
        }                                                         \
    } while (0)
#endif

#define checkCUDNN(expression)                                    \
    {                                                             \
        cudnnStatus_t status = (expression);                      \
        if (status != CUDNN_STATUS_SUCCESS) {                     \
            myprintf_error("Error on %s(%d): %s\n",               \
                __FILE__, __LINE__, cudnnGetErrorString(status)); \
            throw std::runtime_error("cuDNN error");              \
        }                                                         \
    }

#define checkCUDA(error)                                          \
    {                                                             \
        if (error != cudaSuccess) {                               \
            myprintf_error("Error on %s(%d): %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(error));   \
            throw std::runtime_error("CUDA error");               \
        }                                                         \
    }

#define checkCUBLAS(status)                                         \
    {                                                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                      \
            myprintf_error("Error on %s(%d): %s\n",                 \
                __FILE__, __LINE__, cublasGetStatusString(status)); \
            throw std::runtime_error("cuBlas error");               \
        }                                                           \
    }

void global_average_pooling_float(
    const float *input,
    float *output,
    const int batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void global_average_pooling_float_NHWC(
    const float *input,
    float *output,
    const int batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void global_average_pooling_half(
    const __half *input,
    __half *output,
    const int batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void global_average_pooling_half_NHWC(
    const __half *input,
    __half *output,
    const int batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void add_bias_float(
    float *buf,
    const float *biases,
    const int batch_size,
    const int channels,
    const bool relu,
    cudaStream_t stream
    );

void add_bias_half(
    __half *buf,
    const __half *biases,
    const int batch_size,
    const int channels,
    const bool relu,
    cudaStream_t stream
    );

void se_scale_float(
    float *outbuf,
    const float *buf,
    const float *biases,
    const float *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void se_scale_float_NHWC(
    float *outbuf,
    const float *buf,
    const float *biases,
    const float *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void se_scale_half(
    __half *outbuf,
    const __half *buf,
    const __half *biases,
    const __half *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void se_scale_half_NHWC(
    __half *outbuf,
    const __half *buf,
    const __half *biases,
    const __half *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

struct conv_descriptor {
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnTensorDescriptor_t bias_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnActivationDescriptor_t activation_descriptor;
    cudnnActivationDescriptor_t activation_identity_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    size_t workspace_size{0};

#if defined(USE_CUDNN_GRAPH)
    cudnn_frontend::graph::Graph graph;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> X;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> W;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> B;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Y;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Z;
#endif
};

class CuDNN_Layer {
    template <typename> friend class CuDNN_Network;
    template <typename> friend class CuDNNScheduler;
private:
    unsigned int channels{0};
    unsigned int outputs{0};
    unsigned int filter_size{0};
    bool is_input_convolution{false};
    bool is_residual_block{false};
    bool is_se_block{false};
    bool is_value{false};
    bool is_policy{false};
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    std::vector<std::shared_ptr<conv_descriptor>> conv_desc_single;
    std::vector<std::shared_ptr<conv_descriptor>> conv_desc_multi;
    std::vector<std::shared_ptr<conv_descriptor>> conv_no_relu_desc_single;
    std::vector<std::shared_ptr<conv_descriptor>> conv_no_relu_desc_multi;
    std::vector<std::shared_ptr<conv_descriptor>> conv_add_relu_desc_single;
    std::vector<std::shared_ptr<conv_descriptor>> conv_add_relu_desc_multi;
#endif
    float scale_1{1.0f};
    float scale_2{1.0f};
    float scale_3{1.0f};
    std::vector<void *> weights;
#if defined(USE_TENSOR_RT)
    std::vector<int64_t> weights_size;
    std::string name;
#endif
};

#if defined(USE_TENSOR_RT)
static std::string vformat(const char *fmt, va_list ap) {
    // Allocate a buffer on the stack that's big enough for us almost
    // all the time.  Be prepared to allocate dynamically if it doesn't fit.
    size_t size = 4096;
    char stackbuf[4096];
    std::vector<char> dynamicbuf;
    char *buf = &stackbuf[0];

    int needed;
    while (true) {
        // Try to vsnprintf into our buffer.
        needed = vsnprintf(buf, size, fmt, ap);
        // NB. C99 (which modern Linux and OS X follow) says vsnprintf
        // failure returns the length it would have needed.  But older
        // glibc and current Windows return -1 for failure, i.e., not
        // telling us how much was needed.

        if (needed <= (int)size && needed >= 0)
            break;

        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So try again using a dynamic buffer.  This
        // doesn't happen very often if we chose our initial size well.
        size = (needed > 0) ? (needed+1) : (size*2);
        dynamicbuf.resize(size+1);
        buf = &dynamicbuf[0];
    }
    return std::string(buf, (size_t)needed);
}

inline std::string strprintf(const char* fmt, ...) {
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat(fmt, ap);
    va_end (ap);
    return buf;
}

inline std::string readFileBinary(
    const std::string& filename) {
    std::ifstream ifs;
    ifs.open(filename, std::ios::binary);
    std::string str((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    return str;
}

struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;
#endif

class CuDNNContext {
    template <typename> friend class CuDNN_Network;
    template <typename> friend class CuDNNScheduler;
private:
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    void *m_workspace{nullptr};
    void *m_InBuffer{nullptr};
    void *m_OutBuffer{nullptr};
    void *m_IdentityOutBuffer{nullptr};
    void *m_PoolBuffer{nullptr};
    void *m_TempBuffer{nullptr};
    void *m_alpha_16{nullptr};
    void *m_alpha_32{nullptr};
    void *m_beta_16{nullptr};
    void *m_beta_32{nullptr};
#endif
#if defined(USE_TENSOR_RT)
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext_n{nullptr};
    std::map<std::string, void*> mBuffers;
#endif
    bool m_buffers_allocated{false};
};

#if defined(USE_CUDNN_GRAPH)
namespace fe = cudnn_frontend;
#endif

template <typename net_t>
class CuDNN_Network {
    template <typename> friend class CuDNNScheduler;

public:
    CuDNN_Network(
        const int gpu,
        const bool silent = false
    );

    void initialize(
        const int channels,
        const int batch_size,
        const int net_type,
        const int num_worker_threads,
        const std::string &model_hash = ""
    );

    void push_input_convolution(
        const unsigned int filter_size,
        unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases,
        const float scale
    );

    void push_residual(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights_1,
        const std::vector<float>& biases_1,
        const std::vector<float>& weights_2,
        const std::vector<float>& biases_2,
        const float scale_1,
        const float scale_2,
        const float scale_3
    );

    void push_residual_se(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights_1,
        const std::vector<float>& biases_1,
        const std::vector<float>& weights_2,
        const std::vector<float>& biases_2,
        const std::vector<float>& se_fc1_w,
        const std::vector<float>& se_fc1_b,
        const std::vector<float>& se_fc2_w,
        const std::vector<float>& se_fc2_b,
        const float scale_1,
        const float scale_2,
        const float scale_3
    );

    void push_convolve(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases,
        const std::vector<float>& stddevs,
        const std::vector<float>& ip1_w,
        const std::vector<float>& ip1_b,
        const std::vector<float>& ip2_w,
        const std::vector<float>& ip2_b
    );

    size_t get_layer_count() const {
        return m_layers.size();
    }

    void forward(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val,
        const int tid,
        const int batch_size = 1
    );

    void forward_activations(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val,
        CuDNNContext& cudnn_context,
        const int tid,
        const int batch_size = 1
    );

    bool has_fp16_compute() {
        return m_fp16_compute;
    }

    bool has_tensor_cores() {
        return m_tensorcore;
    }

    std::vector<CuDNN_Layer> m_layers;
    std::vector<std::unique_ptr<CuDNNContext>> m_context;

#if defined(USE_TENSOR_RT)
    std::vector<std::unique_ptr<nvinfer1::IRuntime>> mRuntime;
    std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> mEngine;

protected:
    std::map<std::string, nvinfer1::Weights> mWeightMap;
#endif

private:
#if defined(USE_CUDNN)
    void convolve(
        const int tid,
        const void *bufferIn,
        void *bufferOut,
        const void *weights,
        void *workspace,
        const std::shared_ptr<conv_descriptor>& conv_desc,
        const float alpha
    );

    void convolveActivation(
        const int tid,
        const void *bufferIn,
        void *bufferOut,
        const void *weights,
        void *residualBuffer,
        const void *biases,
        void *workspace,
        const std::shared_ptr<conv_descriptor>& conv_desc,
        const float alpha,
        const float alpha2 = 1.0f
    );

    void convolveIdentityActivation(
        const int tid,
        const void *bufferIn,
        void *bufferOut,
        const void *weights,
        void *residualBuffer,
        const void *biases,
        void *workspace,
        const std::shared_ptr<conv_descriptor>& conv_desc,
        const float alpha,
        const float alpha2 = 1.0f
    );

    std::shared_ptr<conv_descriptor> convolve_init(
        cudnnHandle_t handle,
        const int channels,
        const int outputs,
        const int filter_size,
        const int batch_size = 1
    );
#endif

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    void squeeze_excitation_float(
        cublasHandle_t cublas_handle,
        cudaStream_t stream,
        const CuDNNContext& cudnn_context,
        const void *bufferIn1,
        const void *bufferIn2,
        void *bufferTemp,
        const void *fc1_weights,
        const void *fc1_biases,
        const void *fc2_weights,
        const void *fc2_biases,
        void *bufferOut,
        void *bufferPool,
        const int batch_size,
        const int channels,
        const int spatial
    );

    void squeeze_excitation_half(
        cublasHandle_t cublas_handle,
        cudaStream_t stream,
        const CuDNNContext& cudnn_context,
        const void *bufferIn1,
        const void *bufferIn2,
        void *bufferTemp,
        const void *fc1_weights,
        const void *fc1_biases,
        const void *fc2_weights,
        const void *fc2_biases,
        void *bufferOut,
        void *bufferPool,
        const int batch_size,
        const int channels,
        const int spatial
    );

#if defined(USE_CUDNN_GRAPH)
    std::shared_ptr<conv_descriptor> convolve_fe_init(
        cudnnHandle_t handle,
        const int channels,
        const int outputs,
        const int filter_size,
        const int batch_size = 1
    );

    std::shared_ptr<conv_descriptor> convolve_fe_no_relu_init(
        cudnnHandle_t handle,
        const int channels,
        const int outputs,
        const int filter_size,
        const int batch_size = 1
    );

    std::shared_ptr<conv_descriptor> convolve_fe_add_relu_init(
        cudnnHandle_t handle,
        const int channels,
        const int outputs,
        const int batch_size = 1
    );

    std::shared_ptr<conv_descriptor> convolve_fe_head_init(
        cudnnHandle_t handle,
        const int channels,
        const int outputs,
        const int filter_size,
        const int batch_size = 1
    );
#endif

    void push_weights(
        const size_t layer,
        const std::vector<float>& weights
    );
    void push_weights_col_major(
        const size_t layer,
        const std::vector<float>& weights,
        const int row,
        const int column
    );
#endif

#if defined(USE_TENSOR_RT)
    void push_weights_trt(
        const size_t layer,
        const std::vector<float>& weights
    );

    void push_weights_trt_col_major(
        const size_t layer,
        const std::vector<float>& weights,
        const int row,
        const int column,
        const int channels = 1
    );

    // Builds the network engine
    bool build(
        const int num_worker_threads,
        const int batch_size //,
    );

    // Create full model using the TensorRT network definition API and build the engine.
    void constructNetwork(
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        nvinfer1::IOptimizationProfile* profile,
        nvinfer1::IOptimizationProfile* profile_n,
        const int batch_size
    );

    nvinfer1::ITensor* initInputs(
        char const *inputName,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        nvinfer1::IOptimizationProfile* profile,
        nvinfer1::IOptimizationProfile* profile_n,
        const int channels,
        const int rows,
        const int cols,
        const int batch_size
    );

    nvinfer1::ILayer* buildConvLayer(
        nvinfer1::ITensor* input,
        unsigned int filter_size,
        int64_t weights_size,
        void* weights,
        int64_t biases_size,
        void* biases,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name,
        unsigned int outputs
    );

    nvinfer1::ILayer* buildActivationLayer(
        nvinfer1::ITensor* input,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name,
        nvinfer1::ActivationType act_type
    );

    nvinfer1::ILayer* applyGPoolLayer(
        nvinfer1::ITensor* input,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name
    );

    std::string mTuneDesc; // Serves as a hash of the network architecture specific to tuning
#endif

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    std::vector<cudnnHandle_t> m_handle;
    std::vector<cublasHandle_t> m_cublas_handles;
#endif
    std::vector<cudaStream_t> m_streams;

    int m_net_type{0};
    int m_num_worker_threads{1};
    cudaDeviceProp m_device_prop;
    bool m_fp16_compute{false};
    bool m_tensorcore{false};
    std::string m_model_hash{""};

#if defined(USE_CUDNN_GRAPH)
    using graph_and_tensors1 =
        std::tuple<fe::graph::Graph,
            std::shared_ptr<fe::graph::Tensor_attributes>,  // X
            std::shared_ptr<fe::graph::Tensor_attributes>,  // W
            std::shared_ptr<fe::graph::Tensor_attributes>,  // B
            std::shared_ptr<fe::graph::Tensor_attributes>   // Y
        >;
    using graph_and_tensors2 =
        std::tuple<fe::graph::Graph,
            std::shared_ptr<fe::graph::Tensor_attributes>,  // X
            std::shared_ptr<fe::graph::Tensor_attributes>,  // W
            std::shared_ptr<fe::graph::Tensor_attributes>,  // B
            std::shared_ptr<fe::graph::Tensor_attributes>   // Y
        >;
    using graph_and_tensors3 =
        std::tuple<fe::graph::Graph,
            std::shared_ptr<fe::graph::Tensor_attributes>,  // X
            std::shared_ptr<fe::graph::Tensor_attributes>,  // Z
            std::shared_ptr<fe::graph::Tensor_attributes>   // Y
        >;
    using graph_and_tensors4 =
        std::tuple<fe::graph::Graph,
            std::shared_ptr<fe::graph::Tensor_attributes>,  // X
            std::shared_ptr<fe::graph::Tensor_attributes>,  // W
            std::shared_ptr<fe::graph::Tensor_attributes>   // Y
        >;
    std::unordered_map<std::size_t, graph_and_tensors1> m_maintained_cache1;
    std::unordered_map<std::size_t, graph_and_tensors2> m_maintained_cache2;
    std::unordered_map<std::size_t, graph_and_tensors3> m_maintained_cache3;
    std::unordered_map<std::size_t, graph_and_tensors4> m_maintained_cache4;
#endif
};
#endif

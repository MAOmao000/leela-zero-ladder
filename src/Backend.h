/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Henrik Forsten
    Copyright (C) 2024 MAOmao000

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

#ifndef BACKEND_H_INCLUDED
#define BACKEND_H_INCLUDED

#include "config.h"

#if defined(USE_TENSOR_RT) || defined(USE_CUDNN)
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdlib.h>
#include <fstream>
#include <ostream>
#include <iostream>
#include <new>
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
#include <cudnn.h>
#include <cublas_v2.h>
#include <cudnn_frontend.h>

#if defined(USE_TENSOR_RT)
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferRuntimeBase.h"
#include "NvInferSafeRuntime.h"
#include "NvInferConsistency.h"
#include "sha2.h"
#endif

#include "Utils.h"

using namespace Utils;

template <typename net_t> class BackendCuDNN;
template <typename net_t> class BackendGraph;
#if defined(USE_TENSOR_RT)
template <typename net_t> class BackendTRT;
#endif

namespace fe = cudnn_frontend;

#define checkCUDNNFE(expression)                                    \
    {                                                               \
        fe::error_t err = (expression);                             \
        if (err.is_bad()) {                                         \
            myprintf_error("Error on %s(%d): %s\n",                 \
                __FILE__, __LINE__, err.get_message().c_str());     \
            throw std::runtime_error("cuDNN FrontEnd error");       \
        }                                                           \
    }

#define ASSERT(condition)                                           \
    {                                                               \
        if (!(condition)) {                                         \
            myprintf_error("Assertion failure %s(%d): %s\n",        \
                __FILE__, __LINE__, #condition);                    \
            throw std::runtime_error("TensorRT error");             \
        }                                                           \
    }

#define checkCUDNN(expression)                                      \
    {                                                               \
        cudnnStatus_t status = (expression);                        \
        if (status != CUDNN_STATUS_SUCCESS) {                       \
            myprintf_error("Error on %s(%d): %s\n",                 \
                __FILE__, __LINE__, cudnnGetErrorString(status));   \
            throw std::runtime_error("cuDNN error");                \
        }                                                           \
    }

#define checkCUDA(error)                                            \
    {                                                               \
        if (error != cudaSuccess) {                                 \
            myprintf_error("Error on %s(%d): %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(error));     \
            throw std::runtime_error("CUDA error");                 \
        }                                                           \
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
    const size_t batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void global_average_pooling_float_NHWC(
    const float *input,
    float *output,
    const size_t batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void global_average_pooling_half(
    const __half *input,
    __half *output,
    const size_t batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void global_average_pooling_half_NHWC(
    const __half *input,
    __half *output,
    const size_t batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void add_bias_float(
    float *buf,
    const float *biases,
    const size_t batch_size,
    const int channels,
    const bool relu,
    cudaStream_t stream
    );

void add_bias_half(
    __half *buf,
    const __half *biases,
    const size_t batch_size,
    const int channels,
    const bool relu,
    cudaStream_t stream
    );

void se_scale_float(
    float *outbuf,
    const float *buf,
    const float *biases,
    const float *bufferIn,
    const size_t batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void se_scale_float_NHWC(
    float *outbuf,
    const float *buf,
    const float *biases,
    const float *bufferIn,
    const size_t batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void se_scale_half(
    __half *outbuf,
    const __half *buf,
    const __half *biases,
    const __half *bufferIn,
    const size_t batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

void se_scale_half_NHWC(
    __half *outbuf,
    const __half *buf,
    const __half *biases,
    const __half *bufferIn,
    const size_t batch_size,
    const int channels,
    const int spatial,
    cudaStream_t stream
    );

class BackendContext {
public:
    bool m_buffers_allocated{false};
    // Only CUDNNGRAPH backend are used.
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
#if defined(USE_TENSOR_RT)
    // Only TENSORRT backend are used.
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
    std::map<std::string, void*> mBuffers;
#endif
};

// Only CUDNN and CUDNNGRAPH backend are used.
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
    // Only CUDNNGRAPH backend are used.
    cudnn_frontend::graph::Graph graph;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> X;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> W;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> B;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Y;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Z;
};

class BackendLayer {
public:
    unsigned int channels{0};
    unsigned int outputs{0};
    unsigned int filter_size{0};
    std::vector<void *> weights;
    bool is_input_convolution{false};
    bool is_residual_block{false};
    bool is_se_block{false};
    bool is_value{false};
    bool is_policy{false};
    // Only CUDNN and CUDNNGRAPH backend are used.
    std::vector<std::shared_ptr<conv_descriptor>> conv_desc_multi;
    std::vector<std::shared_ptr<conv_descriptor>> conv_no_relu_desc_multi;
    std::vector<std::shared_ptr<conv_descriptor>> conv_add_relu_desc_multi;
    float scale_1{1.0f};
    float scale_2{1.0f};
    float scale_3{1.0f};
#if defined(USE_TENSOR_RT)
    // Only TENSORRT backend are used.
    std::vector<int64_t> weights_size;
    std::string name;
#endif
};

// Filter layout KRSC: output, rows, columns, inputs
//   K: number of output feature maps
//   R: number of rows per filter
//   S: number of columns per filter
//   C: number of input feature maps
//  CUDNN_TENSOR_NCHW = KCRS
//  CUDNN_TENSOR_NHWC = KRSC

namespace BE {
template <typename T>
std::vector<float> NHWC_to_NCHW(const std::vector<T>& x,
                                unsigned int N,
                                unsigned int H,
                                unsigned int W,
                                unsigned int C) {

    std::vector<float> x_out(N * H * W * C);

    for (auto n = size_t{0}; n < N; n++) {
        for (auto h = size_t{0}; h < H; h++) {
            for (auto w = size_t{0}; w < W; w++) {
                for (auto c = size_t{0}; c < C; c++) {
                    x_out[n * H * W * C + c * H * W + h * W + w] =
                    static_cast<float>(x[n * H * W * C + h * W * C + w * C + c]);
                }
            }
        }
    }
    return x_out;
}

template <typename net_t>
std::vector<net_t> NCHW_to_NHWC(const std::vector<float> &x,
                                unsigned int N,
                                unsigned int H,
                                unsigned int W,
                                unsigned int C) {

    std::vector<net_t> x_out(N * H * W * C);

    for (auto n = size_t{0}; n < N; n++) {
        for (auto h = size_t{0}; h < H; h++) {
            for (auto w = size_t{0}; w < W; w++) {
                for (auto c = size_t{0}; c < C; c++) {
                    x_out[n * H * W * C + h * W * C + w * C + c] =
                    static_cast<net_t>(x[n * H * W * C + c * H * W + h * W + w]);
                }
            }
        }
    }
    return x_out;
}

void squeeze_excitation_float(
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const BackendContext& cudnn_context,
    const void *bufferIn1,   // residual input(before convolve)
    const void *bufferIn2,   // residual output
    void *TempBuffer,
    const void *fc1_weights,
    const void *fc1_biases,
    const void *fc2_weights,
    const void *fc2_biases,
    void *bufferOut,
    void *bufferPool,
    const size_t batch_size,
    const int channels,
    const int spatial,
    const bool isNCHW,
    const bool isTensorCore);

void squeeze_excitation_half(
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const BackendContext& cudnn_context,
    const void *bufferIn1,   // residual input(before convolve)
    const void *bufferIn2,   // residual output
    void *TempBuffer,
    const void *fc1_weights,
    const void *fc1_biases,
    const void *fc2_weights,
    const void *fc2_biases,
    void *bufferOut,
    void *bufferPool,
    const size_t batch_size,
    const int channels,
    const int spatial,
    const bool isNCHW,
    const bool isTensorCore);
}

#if defined(USE_TENSOR_RT)
struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;
#endif

template <typename net_t>
class Backend {
public:
    Backend() {}
    Backend(
        const int gpu,
        const bool silent = false
    );

    virtual ~Backend() = default;

    void initialize(
        const int channels,
        const size_t batch_size,
        const NetworkType net_type,
        const size_t num_worker_threads,
        const std::string &model_hash = ""
    );

    virtual void push_input_convolution(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases,
        const float scale
    ) = 0;

    virtual void push_residual(
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
    ) = 0;

    virtual void push_residual_se(
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
    ) = 0;

    virtual void push_convolve(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases
    ) = 0;

    virtual void forward_activations(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val,
        BackendContext& cudnn_context,
        const int tid,
        const size_t batch_size = 1
    ) = 0;

    void forward(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val,
        const int tid,
        const size_t batch_size = 1
    );

    virtual size_t get_layer_count() const {
        return m_layers.size();
    }

    virtual bool has_fp16_compute() const {
        return m_fp16_compute;
    }

    virtual bool has_tensor_cores() const {
        return m_tensorcore;
    }

    std::vector<BackendLayer> m_layers;
    std::vector<std::unique_ptr<BackendContext>> m_context;

protected:
    bool m_fp16_compute{false};
    bool m_tensorcore{false};
    int m_num_worker_threads{1};
    cudaDeviceProp m_device_prop;
    std::vector<cudnnHandle_t> m_handle;
    std::vector<cublasHandle_t> m_cublas_handles;
    std::string m_model_hash{""};
    NetworkType m_net_type{NetworkType::LEELA_ZERO};
};

#endif
#endif

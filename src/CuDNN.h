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
#include <cudnn.h>
#include <cublas_v2.h>

auto constexpr CONV_DESC_INPUT = 0;
auto constexpr CONV_DESC_RESIDUAL = 1;
auto constexpr CONV_DESC_VALUE = 2;
auto constexpr CONV_DESC_POLICY = 3;
auto constexpr SINGLE_BATCH = 0;
auto constexpr MULTIPLE_BATCHES = 1;

template <typename net_t> class CuDNN;
template <typename net_t> class CuDNN_Network;

#define checkCUDNN(expression)                                    \
    {                                                             \
        cudnnStatus_t status = (expression);                      \
        if (status != CUDNN_STATUS_SUCCESS) {                     \
            std::cerr << "Error on " << __FILE__ << "("           \
                <<  __LINE__ << "): "                             \
                << cudnnGetErrorString(status) << std::endl;      \
            throw std::runtime_error("CuDNN error");              \
        }                                                         \
    }

#define checkCUDA(error)                                          \
    {                                                             \
        if (error != cudaSuccess) {                               \
            std::cerr << "Error on " << __FILE__ << "("           \
                <<  __LINE__ << "): "                             \
                << cudaGetErrorString(error) << std::endl;        \
            throw std::runtime_error("CUDA error");               \
        }                                                         \
    }

#define checkCUBLAS(status)                                       \
    {                                                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                    \
            std::cerr << "Error on " << __FILE__ << "("           \
                <<  __LINE__ << "): "                             \
                << cublasGetStatusString(status) << std::endl;    \
            throw std::runtime_error("cuBlas error");             \
        }                                                         \
    }

void global_average_pooling_float(
    const float *input,
    float *output,
    const int batch_size,
    const int channels,
    const int spatial
    );

void global_average_pooling_float_NHWC(
    const float *input,
    float *output,
    const int batch_size,
    const int channels,
    const int spatial
    );

void global_average_pooling_half(
    const __half *input,
    __half *output,
    const int batch_size,
    const int channels,
    const int spatial
    );

void global_average_pooling_half_NHWC(
    const __half *input,
    __half *output,
    const int batch_size,
    const int channels,
    const int spatial
    );

void add_bias_float(
    float *buf,
    const float *biases,
    const int batch_size,
    const int channels,
    const bool relu
    );

void add_bias_half(
    __half *buf,
    const __half *biases,
    const int batch_size,
    const int channels,
    const bool relu
    );

void se_scale_float(
    float *outbuf,
    const float *buf,
    const float *biases,
    const float *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial
    );

void se_scale_float_NHWC(
    float *outbuf,
    const float *buf,
    const float *biases,
    const float *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial
    );

void se_scale_half(
    __half *outbuf,
    const __half *buf,
    const __half *biases,
    const __half *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial
    );

void se_scale_half_NHWC(
    __half *outbuf,
    const __half *buf,
    const __half *biases,
    const __half *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial
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
    cudnnConvolutionFwdAlgo_t convolution_identity_algorithm;
    size_t workspace_size{0};
    size_t workspace_identity_size{0};
    bool is_initialized{false};
};

class CuDNN_Layer {
    template <typename> friend class CuDNN_Network;
private:
    unsigned int channels{0};
    unsigned int outputs{0};
    unsigned int filter_size{0};
    bool is_input_convolution{false};
    bool is_residual_block{false};
    bool is_se_block{false};
    bool is_convolve1{false};
    conv_descriptor conv_desc[2];
    float scale_1{1.0f};
    float scale_2{1.0f};
    float scale_3{1.0f};
    std::vector<void*> weights;
};

class CuDNNContext {
    template <typename> friend class CuDNN;
    template <typename> friend class CuDNN_Network;
private:
    void *m_workspace;
    void *m_InBuffer;
    void *m_OutBuffer;
    void *m_IdentityOutBuffer;
    void *m_PoolBuffer;
    void *m_TempBuffer;
    bool m_is_initialized{false};
    bool m_buffers_allocated{false};
};

template <typename net_t>
class CuDNN_Network {

public:
    CuDNN_Network(CuDNN<net_t>& cudnn) : m_cudnn(cudnn) {}
    CuDNN<net_t>& getCuDNN() {
        return m_cudnn;
    }

    void push_input_convolution(const unsigned int filter_size,
                                unsigned int channels,
                                const unsigned int outputs,
                                const std::vector<float>& weights,
                                const std::vector<float>& biases,
                                const float scale);

    void push_residual(const unsigned int filter_size,
                       const unsigned int channels,
                       const unsigned int outputs,
                       const std::vector<float>& weights_1,
                       const std::vector<float>& biases_1,
                       const std::vector<float>& weights_2,
                       const std::vector<float>& biases_2,
                       const float scale_1,
                       const float scale_2,
                       const float scale_3);

    void push_residual_se(const unsigned int filter_size,
                          const unsigned int channels,
                          const unsigned int outputs,
                          const std::vector<float>& weights_1,
                          const std::vector<float>& biases_1,
                          const std::vector<float>& weights_2,
                          const std::vector<float>& biases_2,
                          const std::vector<float>& se_fc1_w,
                          const std::vector<float>& se_fc1_b,
                          const std::vector<float>& se_fc2_w,
                          const std::vector<float>& se_fc2_b, // );
                          const float scale_1,
                          const float scale_2,
                          const float scale_3);

    void push_convolve(const unsigned int filter_size,
                       const unsigned int channels,
                       const unsigned int outputs,
                       const std::vector<float>& weights);

    size_t get_layer_count() const {
        return m_layers.size();
    }

    void forward(const std::vector<float>& input,
                 std::vector<float>& output_pol,
                 std::vector<float>& output_val,
                 CuDNNContext& cudnn_context,
                 const int batch_size = 1);

    void forward_activations(const std::vector<float>& input,
                             std::vector<float>& output_pol,
                             std::vector<float>& output_val,
                             CuDNNContext& cudnn_context,
                             const int batch_size = 1);

private:
    void push_weights(const size_t layer, const std::vector<float>& weights);
    void push_weights_col_major(const size_t layer,
                                const std::vector<float>& weights,
                                const int row,
                                const int column);

    CuDNN<net_t>& m_cudnn;
    std::vector<CuDNN_Layer> m_layers;
    conv_descriptor m_conv_desc[4][2];

};

template <typename net_t>
class CuDNN {
    friend class CuDNN_Network<net_t>;
    friend class CuDNN_Layer;
public:
    CuDNN(const int gpu, const bool silent = false);

    void initialize(const int channels, const int batch_size, const int net_type);
    bool has_fp16_compute();
    bool has_tensor_cores();

    int m_batch_size = 1;

private:
    void convolve(const void *bufferIn,
                  void *bufferOut,
                  const void *weights,
                  void *workspace,
                  const conv_descriptor& conv_desc,
                  const float alpha);

    void convolveActivation(const void *bufferIn,
                            void *bufferOut,
                            const void *weights,
                            void *residualBuffer,
                            const void *biases,
                            void *workspace,
                            const conv_descriptor& conv_desc,
                            const float alpha,
                            const float alpha2 = 1.0f);

    void convolveIdentityActivation(const void *bufferIn,
                                    void *bufferOut,
                                    const void *weights,
                                    void *residualBuffer,
                                    const void *biases,
                                    void *workspace,
                                    const conv_descriptor& conv_desc,
                                    const float alpha,
                                    const float alpha2 = 1.0f);

    void squeeze_excitation_float(const void *bufferIn1,
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
                                  const int spatial);

    void squeeze_excitation_half(const void *bufferIn1,
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
                                 const int spatial);

    void convolve_init(const int channels,
                       const int outputs,
                       const int filter_size,
                       conv_descriptor& conv_desc,
                       const int batch_size = 1);

    cudnnHandle_t m_handle;
    cublasHandle_t m_cublas_handles;
    bool m_fp16_compute{false};
    bool m_tensorcore{false};
    bool m_init_ok{false};
    int m_net_type{0};
};
#endif

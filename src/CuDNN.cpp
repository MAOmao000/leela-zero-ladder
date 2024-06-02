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

#include "config.h"

#ifdef USE_CUDNN
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "GTP.h"
#include "Utils.h"
#include "CuDNN.h"

using namespace Utils;

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define checkCUDA(error)                                     \
  {                                                          \
    if (error != cudaSuccess) {                              \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudaGetErrorString(error) << std::endl;   \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

template <typename net_t>
CuDNN<net_t>::CuDNN(const int gpu, const bool silent) {
    auto best_bandwidth = 0.0;
    auto found_device = false;
    auto nDevices = 0;
    auto best_device_id = 0;
    cudaDeviceProp best_device;

    cudaGetDeviceCount(&nDevices);

    if (!silent) {
        myprintf("Detected %d CUDA devices.\n", nDevices);
    }

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        auto bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        if (!silent) {
            myprintf("Device Number: %d\n", i);
            myprintf("  Device name: %s\n", prop.name);
            myprintf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            myprintf("  Peak Memory Bandwidth (GB/s): %.1f\n\n", bandwidth);
        }

        bool preferred = (gpu == i);

        if (bandwidth > best_bandwidth || preferred) {
            best_bandwidth = bandwidth;
            best_device = prop;
            best_device_id = i;
            if (preferred) {
                best_bandwidth = std::numeric_limits<decltype(best_bandwidth)>::max();
            } else {
                best_bandwidth = bandwidth;
            }
            found_device = true;
        }
    }

    if (!found_device) {
        throw std::runtime_error("No suitable CUDA device found.");
    }

    myprintf("Selected device: %s\n", best_device.name);
    myprintf("with compute capability %d.%d.\n", best_device.major, best_device.minor);

    if (best_device.major >= 7) {
        m_tensorcore = true;
    } else if (best_device.major >= 6) {
        m_fp16_compute = true;
    }

    cudaSetDevice(best_device_id);
}

template <typename net_t>
void CuDNN<net_t>::initialize(const int channels, int batch_size, int net_type) {

    /* For compatibility with OpenCL implementation */
    (void)channels;
    (void)net_type;

    m_batch_size = batch_size;

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    m_handle = cudnn;
    m_init_ok = true;
}

template <typename net_t>
void CuDNN<net_t>::convolve(void *bufferIn,
                            void *bufferOut,
                            void *weights,
                            void *workspace,
                            const conv_descriptor& conv_desc,
                            float alpha) {

    const float beta = 0.0f;

    checkCUDNN(cudnnConvolutionForward(m_handle,
                                       &alpha,
                                       conv_desc.input_descriptor,
                                       bufferIn,
                                       conv_desc.filter_descriptor,
                                       weights,
                                       conv_desc.convolution_descriptor,
                                       conv_desc.convolution_algorithm,
                                       workspace,
                                       conv_desc.workspace_size,
                                       &beta,
                                       conv_desc.output_descriptor,
                                       bufferOut));
}

template <typename net_t>
void CuDNN<net_t>::convolveActivation(void *bufferIn,
                                      void *bufferOut,
                                      void *weights,
                                      void *residualBuffer,
                                      void *biases,
                                      void *workspace,
                                      const conv_descriptor& conv_desc,
                                      float alpha1,
                                      float alpha2) {

    void *residual = bufferOut;

    float _alpha2 = 0.0f;
    if (residualBuffer != nullptr) {
        _alpha2 = alpha2;
        residual = residualBuffer;
    }

    checkCUDNN(cudnnConvolutionBiasActivationForward(
        /* handle */m_handle,
        /* alpha1 */&alpha1,
        /* xDesc */conv_desc.input_descriptor,
        /* x */bufferIn,
        /* wDesc */conv_desc.filter_descriptor,
        /* w */weights,
        /* convDesc */conv_desc.convolution_descriptor,
        /* algo */conv_desc.convolution_algorithm,
        /* workSpace */workspace,
        /* workSpaceSize */conv_desc.workspace_size,
        /* alpha2 */&_alpha2,
        /* zDesc */conv_desc.output_descriptor,
        /* z */residual,
        /* biasDesc */conv_desc.bias_descriptor,
        /* bias */biases,
        /* activationDesc */conv_desc.activation_descriptor,
        /* yDesc */conv_desc.output_descriptor,
        /* y */bufferOut));
}

template <typename net_t>
void CuDNN<net_t>::convolve_init(int channels, int outputs, int filter_size,
                                 conv_descriptor& conv_desc, int batch_size) {
    cudnnDataType_t data_type;
    cudnnDataType_t bias_type;
    cudnnDataType_t compute_type;
    cudnnTensorFormat_t tensor_format;

    if (typeid(net_t) == typeid(float)) {
        // Convolve layers Value and Policy are calculated in single precision when using int8.
        data_type = CUDNN_DATA_FLOAT;
        bias_type = CUDNN_DATA_FLOAT;
        compute_type = CUDNN_DATA_FLOAT;
        tensor_format = CUDNN_TENSOR_NCHW;
    } else { // typeid: half_float::half
        data_type = CUDNN_DATA_HALF;
        bias_type = CUDNN_DATA_HALF;
        /* Use half computation if supported */
        compute_type = CUDNN_DATA_HALF;
        tensor_format = CUDNN_TENSOR_NCHW;
    }

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv_desc.input_descriptor,
                                          /*format=*/tensor_format,
                                          /*dataType=*/data_type,
                                          /*batch_size=*/batch_size,
                                          /*channels=*/channels,
                                          /*image_height=*/BOARD_SIZE,
                                          /*image_width=*/BOARD_SIZE));

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv_desc.output_descriptor,
                                          /*format=*/tensor_format,
                                          /*dataType=*/data_type,
                                          /*batch_size=*/batch_size,
                                          /*channels=*/outputs,
                                          /*image_height=*/BOARD_SIZE,
                                          /*image_width=*/BOARD_SIZE));

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.bias_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv_desc.bias_descriptor,
                                          /*format=*/tensor_format,
                                          /*dataType=*/bias_type,
                                          /*number of images=*/1,
                                          /*channels=*/outputs,
                                          /*image_height=*/1,
                                          /*image_width=*/1));

    checkCUDNN(cudnnCreateFilterDescriptor(&conv_desc.filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(conv_desc.filter_descriptor,
                                          /*dataType=*/data_type,
                                          /*format=*/tensor_format,
                                          /*out_channels=*/outputs,
                                          /*in_channels=*/channels,
                                          /*filter_height=*/filter_size,
                                          /*filter_width=*/filter_size));

    checkCUDNN(cudnnCreateActivationDescriptor(&conv_desc.activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(conv_desc.activation_descriptor,
                                            /*mode=*/CUDNN_ACTIVATION_RELU,
                                            /*reluNanOpt=*/CUDNN_NOT_PROPAGATE_NAN,
                                            /*coef=*/0.));

    auto pad_size = 0;

    if (filter_size == 1) {
        pad_size = 0;
    } else if (filter_size == 3) {
        pad_size = 1;
    }

    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc.convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc.convolution_descriptor,
                                               /*zero-padding height=*/pad_size,
                                               /*zero-padding width=*/pad_size,
                                               /*vertical filter stride=*/1,
                                               /*horizontal filter stride=*/1,
                                               /*filter height dilation=*/1,
                                               /*filter width dilation=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/compute_type));
    checkCUDNN(cudnnSetConvolutionGroupCount(conv_desc.convolution_descriptor, 128));
    checkCUDNN(cudnnSetConvolutionMathType(conv_desc.convolution_descriptor,
                                           CUDNN_TENSOR_OP_MATH));

    using perf_t = cudnnConvolutionFwdAlgoPerf_t;
    int num_algos = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(m_handle, &num_algos));
    int returned_algo_count = 0;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(m_handle, 
                                                      conv_desc.input_descriptor,
                                                      conv_desc.filter_descriptor,
                                                      conv_desc.convolution_descriptor,
                                                      conv_desc.output_descriptor,
                                                      num_algos,
                                                      &returned_algo_count,
                                                      perf_results.get()))
    conv_desc.convolution_algorithm = perf_results[0].algo;

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_handle,
                            /*cudnnTensorDescriptor_t*/conv_desc.input_descriptor,
                            /*cudnnFilterDescriptor_t*/conv_desc.filter_descriptor,
                       /*cudnnConvolutionDescriptor_t*/conv_desc.convolution_descriptor,
                            /*cudnnTensorDescriptor_t*/conv_desc.output_descriptor,
                          /*cudnnConvolutionFwdAlgo_t*/conv_desc.convolution_algorithm,
                                          /*size_t * */&conv_desc.workspace_size));
}

template <typename net_t>
void CuDNN_Network<net_t>::push_weights(size_t layer, const std::vector<float>& weights) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(CuDNN_Layer());
    }

    if (typeid(net_t) == typeid(float)) {
        auto weightSize = weights.size() * sizeof(float);

        void *device_mem;
        checkCUDA(cudaMalloc((void**)&device_mem, weightSize));
        checkCUDA(cudaMemcpy(device_mem, (net_t*)&weights[0], weightSize, cudaMemcpyHostToDevice));
        m_layers.back().weights.emplace_back(device_mem);

    } else {
        auto converted_weights = std::vector<net_t>();
        for(auto i = size_t{0}; i < weights.size(); i++) {
            converted_weights.emplace_back(weights[i]);
        }

        auto weightSize = weights.size() * sizeof(net_t);

        void *device_mem;
        checkCUDA(cudaMalloc((void**)&device_mem, weightSize));
        checkCUDA(cudaMemcpy(device_mem, (net_t*)&converted_weights[0], weightSize, cudaMemcpyHostToDevice));
        m_layers.back().weights.emplace_back(device_mem);
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::push_input_convolution(unsigned int filter_size,
                                                  unsigned int channels,
                                                  unsigned int outputs,
                                                  const std::vector<float>& weights,
                                                  const std::vector<float>& biases,
                                                  float scale) {

    size_t layer = get_layer_count();
    push_weights(layer, weights);
    push_weights(layer, biases);
    m_layers[layer].is_input_convolution = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;
    m_layers[layer].scale_1 = 1.0f / scale;
    m_layers[layer].scale_2 = 1.0f / scale;
    m_layers[layer].scale_3 = 1.0f;

    if (!m_conv_desc[CONV_DESC_INPUT][0].is_initialized) {
        m_cudnn.convolve_init(channels, outputs, filter_size, m_conv_desc[CONV_DESC_INPUT][0]);
        m_cudnn.convolve_init(channels, outputs, filter_size, m_conv_desc[CONV_DESC_INPUT][1], cfg_batch_size);
    }
    m_layers[layer].conv_desc[0] = m_conv_desc[CONV_DESC_INPUT][0];
    m_layers[layer].conv_desc[1] = m_conv_desc[CONV_DESC_INPUT][1];
}

template <typename net_t>
void CuDNN_Network<net_t>::push_residual(unsigned int filter_size,
                                         unsigned int channels,
                                         unsigned int outputs,
                                         const std::vector<float>& weights_1,
                                         const std::vector<float>& biases_1,
                                         const std::vector<float>& weights_2,
                                         const std::vector<float>& biases_2,
                                         float scale_1,
                                         float scale_2,
                                         float scale_3) {

    size_t layer = get_layer_count();
    push_weights(layer, weights_1);
    push_weights(layer, biases_1);
    push_weights(layer, weights_2);
    push_weights(layer, biases_2);
    m_layers[layer].is_residual_block = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;
    m_layers[layer].scale_1 = 1.0f / scale_1;
    m_layers[layer].scale_2 = 1.0f / scale_2;
    m_layers[layer].scale_3 = 1.0f / scale_3;

    if (!m_conv_desc[CONV_DESC_RESIDUAL][0].is_initialized) {
        m_cudnn.convolve_init(channels, outputs, filter_size, m_conv_desc[CONV_DESC_RESIDUAL][0]);
        m_cudnn.convolve_init(channels, outputs, filter_size, m_conv_desc[CONV_DESC_RESIDUAL][1], cfg_batch_size);
    }

    m_layers[layer].conv_desc[0] = m_conv_desc[CONV_DESC_RESIDUAL][0];
    m_layers[layer].conv_desc[1] = m_conv_desc[CONV_DESC_RESIDUAL][1];
}

template <typename net_t>
void CuDNN_Network<net_t>::push_convolve(unsigned int filter_size,
                                         unsigned int channels,
                                         unsigned int outputs,
                                         const std::vector<float>& weights) {

    size_t layer = get_layer_count();

    push_weights(layer, weights);
    m_layers[layer].is_convolve1 = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].channels = channels;
    m_layers[layer].filter_size = filter_size;

    if (outputs == 1) {
        if (!m_conv_desc[CONV_DESC_VALUE][0].is_initialized) {
            m_cudnn.convolve_init(channels, outputs, filter_size, m_conv_desc[CONV_DESC_VALUE][0]);
            m_cudnn.convolve_init(channels, outputs, filter_size, m_conv_desc[CONV_DESC_VALUE][1], cfg_batch_size);
        }
        m_layers[layer].conv_desc[0] = m_conv_desc[CONV_DESC_VALUE][0];
        m_layers[layer].conv_desc[1] = m_conv_desc[CONV_DESC_VALUE][1];
    } else {
        if (!m_conv_desc[CONV_DESC_POLICY][0].is_initialized) {
            m_cudnn.convolve_init(channels, outputs, filter_size, m_conv_desc[CONV_DESC_POLICY][0]);
            m_cudnn.convolve_init(channels, outputs, filter_size, m_conv_desc[CONV_DESC_POLICY][1], cfg_batch_size);
        }
        m_layers[layer].conv_desc[0] = m_conv_desc[CONV_DESC_POLICY][0];
        m_layers[layer].conv_desc[1] = m_conv_desc[CONV_DESC_POLICY][1];
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::forward_activations(std::vector<float>& input,
                                               std::vector<float>& output_pol,
                                               std::vector<float>& output_val,
                                               CuDNNContext& cudnn_context,
                                               const int batch_size) {
    int conv_desc_idx = 0;
    if (batch_size > 1) {
        conv_desc_idx = 1;
    }

    /* Always allocates enough space for floats */
    constexpr auto one_plane = NUM_INTERSECTIONS * sizeof(float);
    const auto pol_elements = batch_size * m_layers[m_layers.size() - 2].outputs * NUM_INTERSECTIONS;
    const auto val_elements = batch_size * m_layers.back().outputs * NUM_INTERSECTIONS;

    auto pol_net_t = std::vector<net_t>(pol_elements);
    auto val_net_t = std::vector<net_t>(val_elements);

    if (!cudnn_context.m_buffers_allocated) {
        auto max_wsize = size_t{0};
        auto max_channels = unsigned{0};
        for (const auto& layer : m_layers) {
            max_wsize = std::max(max_wsize, layer.conv_desc[conv_desc_idx].workspace_size);
            max_channels = std::max(max_channels,
                                std::max(layer.channels, layer.outputs));
        }
        auto alloc_insize = batch_size * max_channels * one_plane;

        void *d_workspace;
        checkCUDA(cudaMalloc((void**)&d_workspace, max_wsize));

        void *d_InBuffer;
        checkCUDA(cudaMalloc((void**)&d_InBuffer, alloc_insize));

        void *d_OutBuffer;
        checkCUDA(cudaMalloc((void**)&d_OutBuffer, alloc_insize));

        cudnn_context.m_workspace = d_workspace;
        cudnn_context.m_InBuffer = d_InBuffer;
        cudnn_context.m_OutBuffer = d_OutBuffer;
        cudnn_context.m_buffers_allocated = true;
    }

    auto workspace = cudnn_context.m_workspace;
    auto InBuffer = cudnn_context.m_InBuffer;
    auto OutBuffer = cudnn_context.m_OutBuffer;

    const auto inSize = batch_size * sizeof(net_t) * m_layers[0].channels * NUM_INTERSECTIONS;
    auto input_net_t = std::vector<net_t>(batch_size * m_layers[0].channels * NUM_INTERSECTIONS);

    auto output_t_size = sizeof(net_t);
    if (typeid(net_t) == typeid(half_float::half)) {
        auto itr_in = input.begin();
        auto itr_out = input_net_t.begin();
        for ( ; itr_in != input.end(); itr_in++, itr_out++) {
            *itr_out = static_cast<net_t>(*itr_in);
        }
    } else {
        std::copy(input.begin(), input.end(), input_net_t.begin());
    }

    checkCUDA(cudaMemcpy(InBuffer, (net_t*)&input_net_t[0], inSize, cudaMemcpyHostToDevice));

    for (auto iter = std::begin(m_layers); iter != std::end(m_layers); iter++) {
        const auto& layer = *iter;
        const auto niter = std::next(iter);

        if (layer.is_input_convolution) {
            assert(niter != std::end(m_layers));
            auto conv_weights = begin(layer.weights);
            auto conv_biases = begin(layer.weights) + 1;
            m_cudnn.convolveActivation(InBuffer,
                                       OutBuffer,
                                       conv_weights[0],
                                       nullptr,
                                       conv_biases[0],
                                       workspace,
                                       layer.conv_desc[conv_desc_idx],
                                       layer.scale_1,
                                       1.0f);

        } else if (layer.is_residual_block) {
            assert(layer.channels == layer.outputs);
            assert(niter != std::end(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases = begin(layer.weights) + 3;

            m_cudnn.convolveActivation(OutBuffer,
                                       InBuffer,
                                       conv1_weights[0],
                                       nullptr,
                                       conv1_biases[0],
                                       workspace,
                                       layer.conv_desc[conv_desc_idx],
                                       layer.scale_1,
                                       1.0f);

            m_cudnn.convolveActivation(InBuffer,
                                       OutBuffer,
                                       conv2_weights[0],
                                       OutBuffer,
                                       conv2_biases[0],
                                       workspace,
                                       layer.conv_desc[conv_desc_idx],
                                       layer.scale_2,
                                       layer.scale_3);

        } else {
            assert(layer.is_convolve1);
            m_cudnn.convolve(OutBuffer,
                             InBuffer,
                             layer.weights[0],
                             workspace,
                             layer.conv_desc[conv_desc_idx],
                             layer.scale_1);

            if (niter == std::end(m_layers)) {
                /* Value */
                checkCUDA(cudaMemcpy(&val_net_t[0], InBuffer, val_elements * output_t_size, cudaMemcpyDeviceToHost));
            } else {
                /* Policy */
                checkCUDA(cudaMemcpy(&pol_net_t[0], InBuffer, pol_elements * output_t_size, cudaMemcpyDeviceToHost));
            }
        }
    }

    if (typeid(net_t) == typeid(half_float::half)) {
        auto itr_in = val_net_t.begin();
        auto itr_out = output_val.begin();
        for ( ; itr_in != val_net_t.end(); itr_in++, itr_out++) {
            *itr_out = static_cast<float>(*itr_in);
        }
        itr_in = pol_net_t.begin();
        itr_out = output_pol.begin();
        for ( ; itr_in != pol_net_t.end(); itr_in++, itr_out++) {
            *itr_out = static_cast<float>(*itr_in);
        }
    } else {
        std::copy(val_net_t.begin(), val_net_t.end(), output_val.begin());
        std::copy(pol_net_t.begin(), pol_net_t.end(), output_pol.begin());
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::forward(std::vector<float>& input,
                                   std::vector<float>& output_pol,
                                   std::vector<float>& output_val,
                                   CuDNNContext& cudnn_context,
                                   const int batch_size) {

    forward_activations(input, output_pol, output_val, cudnn_context, batch_size);
}

template <typename net_t>
bool CuDNN<net_t>::has_fp16_compute() {
    return m_fp16_compute;
}

template <typename net_t>
bool CuDNN<net_t>::has_tensor_cores() {
    return m_tensorcore;
}

template class CuDNN<float>;
template class CuDNN_Network<float>;
#ifdef USE_HALF
template class CuDNN<half_float::half>;
template class CuDNN_Network<half_float::half>;
#endif

#endif

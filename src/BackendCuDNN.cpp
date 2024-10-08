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

#include "config.h"

#if defined(USE_CUDNN)
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "GTP.h"
#include "Utils.h"
#include "BackendCuDNN.h"

using namespace Utils;

template <typename net_t>
void BackendCuDNN<net_t>::convolve(
    const int tid,
    const void *bufferIn,
    void *bufferOut,
    const void *weights,
    void *workspace,
    const std::shared_ptr<conv_descriptor>& conv_desc,
    const float alpha) {

    const float beta = 0.0f;
    // dstValue = alpha[0] * result + beta[0] * priorDstValue
    checkCUDNN(cudnnConvolutionForward(
        /* handle               */m_handle[tid],
        /* *alpha               */&alpha,
        /* xDesc                */conv_desc->input_descriptor,
        /* *x                   */bufferIn,
        /* wDesc                */conv_desc->filter_descriptor,
        /* *w                   */weights,
        /* convDesc             */conv_desc->convolution_descriptor,
        /* algo                 */conv_desc->convolution_algorithm,
        /* *workSpace           */workspace,
        /* workSpaceSizeInBytes */conv_desc->workspace_size,
        /* *beta                */&beta,
        /* yDesc                */conv_desc->output_descriptor,
        /* *y                   */bufferOut));
}

template <typename net_t>
void BackendCuDNN<net_t>::convolveActivation(
    const int tid,
    const void *bufferIn,
    void *bufferOut,
    const void *weights,
    void *residualBuffer,
    const void *biases,
    void *workspace,
    const std::shared_ptr<conv_descriptor>& conv_desc,
    const float alpha1,
    const float alpha2) {

    void *residual = bufferOut;
    float _alpha2 = 0.0f;
    if (residualBuffer != nullptr) {
        _alpha2 = alpha2;
        residual = residualBuffer;
    }
    // y = act (alpha1 * conv(x) + alpha2 * z + bias)
    checkCUDNN(cudnnConvolutionBiasActivationForward(
        /* handle         */m_handle[tid],
        /* *alpha1        */&alpha1,
        /* xDesc          */conv_desc->input_descriptor,
        /* *x             */bufferIn,
        /* wDesc          */conv_desc->filter_descriptor,
        /* *w             */weights,
        /* convDesc       */conv_desc->convolution_descriptor,
        /* algo           */conv_desc->convolution_algorithm,
        /* *workSpace     */workspace,
        /* workSpaceSize  */conv_desc->workspace_size,
        /* *alpha2        */&_alpha2,
        /* zDesc          */conv_desc->output_descriptor,
        /* *z             */residual,
        /* biasDesc       */conv_desc->bias_descriptor,
        /* *bias          */biases,
        /* activationDesc */conv_desc->activation_descriptor,
        /* yDesc          */conv_desc->output_descriptor,
        /* *y             */bufferOut));
}

template <typename net_t>
void BackendCuDNN<net_t>::convolveIdentityActivation(
    const int tid,
    const void *bufferIn,
    void *bufferOut,
    const void *weights,
    void *residualBuffer,
    const void *biases,
    void *workspace,
    const std::shared_ptr<conv_descriptor>& conv_desc,
    const float alpha1,
    const float alpha2) {

    void *residual = bufferOut;

    float _alpha2 = 0.0f;
    if (residualBuffer != nullptr) {
        _alpha2 = alpha2;
        residual = residualBuffer;
    }

    // y = act (alpha1 * conv(x) + alpha2 * z + bias)
    checkCUDNN(cudnnConvolutionBiasActivationForward(
        /* handle         */m_handle[tid],
        /* *alpha1        */&alpha1,
        /* xDesc          */conv_desc->input_descriptor,
        /* *x             */bufferIn,
        /* wDesc          */conv_desc->filter_descriptor,
        /* *w             */weights,
        /* convDesc       */conv_desc->convolution_descriptor,
        /* algo           */conv_desc->convolution_algorithm,
        /* *workSpace     */workspace,
        /* workSpaceSize  */conv_desc->workspace_size,
        /* *alpha2        */&_alpha2,
        /* zDesc          */conv_desc->output_descriptor,
        /* *z             */residual,
        /* biasDesc       */conv_desc->bias_descriptor,
        /* *bias          */biases,
        /* activationDesc */conv_desc->activation_identity_descriptor,
        /* yDesc          */conv_desc->output_descriptor,
        /* *y             */bufferOut));
}

template <typename net_t>
std::shared_ptr<conv_descriptor> BackendCuDNN<net_t>::convolve_init(
    cudnnHandle_t handle,
    const int channels,
    const int outputs,
    const int filter_size,
    const int batch_size) {

    cudnnDataType_t data_type;
    cudnnDataType_t compute_type;
    cudnnTensorFormat_t tensor_format;
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = CUDNN_DATA_FLOAT;
        compute_type = CUDNN_DATA_FLOAT;
        tensor_format = cfg_NCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    } else {
        data_type = CUDNN_DATA_HALF;
        compute_type = CUDNN_DATA_HALF;
        tensor_format = cfg_NCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    }

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc->input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
                      /* tensorDesc     */conv_desc->input_descriptor,
                      /* format         */tensor_format,
                      /* dataType       */data_type,
                      /* N batch_size   */batch_size,
                      /* C channels     */channels,
                      /* H image_height */BOARD_SIZE,
                      /* W image_width  */BOARD_SIZE));

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc->output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
                      /* tensorDesc     */conv_desc->output_descriptor,
                      /* format         */tensor_format,
                      /* dataType       */data_type,
                      /* N batch_size   */batch_size,
                      /* C channels     */outputs,
                      /* H image_height */BOARD_SIZE,
                      /* W image_width  */BOARD_SIZE));

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc->bias_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
                      /* tensorDesc     */conv_desc->bias_descriptor,
                      /* format         */tensor_format,
                      /* dataType       */data_type,
                      /* N number of images=*/1, // Not the batch_size
                      /* C channels         */outputs,
                      /* H image_height     */1,
                      /* W image_width      */1));

    checkCUDNN(cudnnCreateFilterDescriptor(&conv_desc->filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
                      /* filterDesc     */conv_desc->filter_descriptor,
                      /* dataType       */data_type,
                      /* format         */tensor_format,
                      /* K Number of output feature maps */outputs,
                      /* C Number of input feature maps  */channels,
                      /* R Number of rows per filter     */filter_size,
                      /* S Number of columns per filter  */filter_size));

    checkCUDNN(cudnnCreateActivationDescriptor(&conv_desc->activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(
                      /* activationDesc   */conv_desc->activation_descriptor,
                      /* mode             */CUDNN_ACTIVATION_RELU,
                      /* reluNanOpt       */CUDNN_NOT_PROPAGATE_NAN,
                      /* coef             */0.));

    auto pad_size = filter_size / 2;

    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc->convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
                 /* convDesc                 */conv_desc->convolution_descriptor,
                 /* zero-padding height      */pad_size,
                 /* zero-padding width       */pad_size,
                 /* vertical filter stride   */1,
                 /* horizontal filter stride */1,
                 /* filter height dilation   */1,
                 /* filter width dilation    */1,
                 /* mode                     */CUDNN_CROSS_CORRELATION,
                 /* computeType              */compute_type));

    checkCUDNN(cudnnSetConvolutionGroupCount(conv_desc->convolution_descriptor, 8));
    checkCUDNN(cudnnSetConvolutionMathType(
                             /* convDesc */conv_desc->convolution_descriptor,
                             /* mathType */CUDNN_TENSOR_OP_MATH));

    using perf_t = cudnnConvolutionFwdAlgoPerf_t;
    int num_algos = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &num_algos));
    int returned_algo_count = 0;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
                              /* handle             */handle,
                              /* xDesc              */conv_desc->input_descriptor,
                              /* wDesc              */conv_desc->filter_descriptor,
                              /* convDesc           */conv_desc->convolution_descriptor,
                              /* yDesc              */conv_desc->output_descriptor,
                              /* requestedAlgoCount */num_algos,
                              /* *returnedAlgoCount */&returned_algo_count,
                              /* *perfResults       */perf_results.get()));

    conv_desc->convolution_algorithm = perf_results[0].algo;

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                                     /* handle       */handle,
                                     /* xDesc        */conv_desc->input_descriptor,
                                     /* wDesc        */conv_desc->filter_descriptor,
                                     /* convDesc     */conv_desc->convolution_descriptor,
                                     /* yDesc        */conv_desc->output_descriptor,
                                     /* algo         */conv_desc->convolution_algorithm,
                                     /* *sizeInBytes */&conv_desc->workspace_size));

    if (m_net_type == NetworkType::MINIGO_SE) {
        checkCUDNN(cudnnCreateActivationDescriptor(&conv_desc->activation_identity_descriptor));
        checkCUDNN(cudnnSetActivationDescriptor(
                            /* activationDesc */conv_desc->activation_identity_descriptor,
                            /* mode           */CUDNN_ACTIVATION_IDENTITY,
                            /* reluNanOpt     */CUDNN_NOT_PROPAGATE_NAN,
                            /* coef           */0.));

    }
    return conv_desc;
}

template <typename net_t>
void BackendCuDNN<net_t>::push_weights(
    const size_t layer,
    const std::vector<float>& weights) {

    if (layer >= m_layers.size()) {
        m_layers.emplace_back(BackendLayer());
    }
    auto weights_net_t = std::vector<net_t>(weights.size());
    std::copy(weights.begin(), weights.end(), weights_net_t.begin());
    void *device_mem;
    checkCUDA(cudaMalloc((void**)&device_mem, weights.size() * sizeof(net_t)));
    checkCUDA(cudaMemcpy(device_mem,
                         (net_t *)&weights_net_t[0],
                         weights.size() * sizeof(net_t),
                         cudaMemcpyHostToDevice));
    m_layers.back().weights.emplace_back(device_mem);
}

template <typename net_t>
void BackendCuDNN<net_t>::push_weights_col_major(
    const size_t layer,
    const std::vector<float>& weights,
    const int row,
    const int column) {

    if (layer >= m_layers.size()) {
        m_layers.emplace_back(BackendLayer());
    }

    // Transpose from model's CK to cublas's KC
    auto weightSize = weights.size() * sizeof(net_t);
    auto transposed_weights = std::vector<net_t>(weights.size());
    for (int i = 0; i < column; i++) {
        for (int j = 0; j < row; j++) {
            transposed_weights[i * row + j] = (net_t)weights[i + j * column];
        }
    }
    void *device_mem;
    checkCUDA(cudaMalloc((void**)&device_mem, weightSize));
    checkCUDA(cudaMemcpy(device_mem,
                         (net_t *)&transposed_weights[0],
                         weightSize,
                         cudaMemcpyHostToDevice));
    m_layers.back().weights.emplace_back(device_mem);
}

template <typename net_t>
void BackendCuDNN<net_t>::push_input_convolution(
    const unsigned int filter_size,
    unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    const float scale) {

    size_t layer = get_layer_count();

    if (cfg_NCHW) {
        push_weights(layer, weights); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases);  // Here it is still float(Convert precision with push_weights)
    } else {
        auto weights_convert = BE::NCHW_to_NHWC<float>(weights, outputs, filter_size, filter_size, channels);
        push_weights(layer, weights_convert); // Convert precision with push_weights
        push_weights(layer, biases);          // Convert precision with push_weights
    }
    m_layers[layer].is_input_convolution = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;

    m_layers[layer].scale_1 = 1.0f / scale;
    m_layers[layer].scale_2 = 1.0f / scale;
    m_layers[layer].scale_3 = 1.0f;

    for (auto i = 0; i < m_num_worker_threads; i++) {
        auto conv_desc_single
            = convolve_init(m_handle[i], channels, outputs, filter_size);
        m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
        auto conv_desc_multi
            = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
        m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
    }
}

template <typename net_t>
void BackendCuDNN<net_t>::push_residual(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights_1,
    const std::vector<float>& biases_1,
    const std::vector<float>& weights_2,
    const std::vector<float>& biases_2,
    const float scale_1,
    const float scale_2,
    const float scale_3) {

    size_t layer = get_layer_count();

    if (cfg_NCHW) {
        push_weights(layer, weights_1); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
        push_weights(layer, weights_2); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
    } else {
        auto weights_convert_1 = BE::NCHW_to_NHWC<float>(
            weights_1, outputs, filter_size, filter_size, channels);
        auto weights_convert_2 = BE::NCHW_to_NHWC<float>(
            weights_2, outputs, filter_size, filter_size, channels);
        push_weights(layer, weights_convert_1); // Convert precision with push_weights
        push_weights(layer, biases_1);          // Convert precision with push_weights
        push_weights(layer, weights_convert_2); // Convert precision with push_weights
        push_weights(layer, biases_2);          // Convert precision with push_weights
    }
    m_layers[layer].is_residual_block = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;

    m_layers[layer].scale_1 = 1.0f / scale_1;
    m_layers[layer].scale_2 = 1.0f / scale_2;
    m_layers[layer].scale_3 = 1.0f / scale_3;

    if (layer == 1) {
        for (auto i = 0; i < m_num_worker_threads; i++) {
            auto conv_desc_single
                = convolve_init(m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
            auto conv_desk_multi
                = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_desc_multi.emplace_back(conv_desk_multi);
        }
    }
}

template <typename net_t>
void BackendCuDNN<net_t>::push_residual_se(
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
    const float scale_3) {

    size_t layer = get_layer_count();

    if (cfg_NCHW) {
        push_weights(layer, weights_1); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
        push_weights(layer, weights_2); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
        push_weights_col_major(layer, se_fc1_w, channels / 2, channels);
        push_weights(layer, se_fc1_b);
        push_weights_col_major(layer, se_fc2_w, channels * 2, channels / 2);
        push_weights(layer, se_fc2_b);
    } else {
        auto weights_convert_1 = BE::NCHW_to_NHWC<float>(
            weights_1, outputs, filter_size, filter_size, channels);
        auto weights_convert_2 = BE::NCHW_to_NHWC<float>(
            weights_2, outputs, filter_size, filter_size, channels);
        push_weights(layer, weights_convert_1); // Convert precision with push_weights
        push_weights(layer, biases_1);          // Convert precision with push_weights
        push_weights(layer, weights_convert_2); // Convert precision with push_weights
        push_weights(layer, biases_2);          // Convert precision with push_weights
        push_weights_col_major(layer, se_fc1_w, channels / 2, channels);
        push_weights(layer, se_fc1_b);
        push_weights_col_major(layer, se_fc2_w, channels * 2, channels / 2);
        push_weights(layer, se_fc2_b);
    }
    m_layers[layer].is_residual_block = true;
    m_layers[layer].is_se_block = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;

    m_layers[layer].scale_1 = 1.0f / scale_1;
    m_layers[layer].scale_2 = 1.0f / scale_2;
    m_layers[layer].scale_3 = 1.0f / scale_3;

    if (layer == 1) {
        for (auto i = 0; i < m_num_worker_threads; i++) {
            auto conv_desc_single
                = convolve_init(m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
            auto conv_desc_multi
                = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
        }
    }
}

template <typename net_t>
void BackendCuDNN<net_t>::push_convolve(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    const std::vector<float>& stddevs,
    const std::vector<float>& ip1_w,
    const std::vector<float>& ip1_b,
    const std::vector<float>& ip2_w,
    const std::vector<float>& ip2_b) {

    size_t layer = get_layer_count();

    (void) biases;
    (void) stddevs;
    (void) ip1_w;
    (void) ip1_b;
    (void) ip2_w;
    (void) ip2_b;

    if (cfg_NCHW) {
        push_weights(layer, weights); // Here it is still float(Convert precision with push_weights)
    } else {
        auto weights_convert = BE::NCHW_to_NHWC<float>(
            weights, outputs, filter_size, filter_size, channels);
        push_weights(layer, weights_convert); // Convert precision with push_weights
    }
    m_layers[layer].outputs = outputs;
    m_layers[layer].channels = channels;
    m_layers[layer].filter_size = filter_size;

    if (outputs == Network::OUTPUTS_VALUE) {
        m_layers[layer].is_value = true;
        for (auto i = 0; i < m_num_worker_threads; i++) {
            auto conv_desc_single
                = convolve_init(m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
            auto conv_desc_multi
                = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
        }
    } else {
        m_layers[layer].is_policy = true;
        for (auto i = 0; i < m_num_worker_threads; i++) {
            std::shared_ptr<conv_descriptor> conv_desc_single
                = convolve_init(m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
            std::shared_ptr<conv_descriptor> conv_desc_multi
                = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
        }
    }
}

template <typename net_t>
void BackendCuDNN<net_t>::forward_activations(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val,
    BackendContext& cudnn_context,
    const int tid,
    const int batch_size) {

    const auto inSize = batch_size * sizeof(net_t) * m_layers[0].channels * NUM_INTERSECTIONS;
    const auto pol_elements
        = batch_size * m_layers[m_layers.size() - 2].outputs * NUM_INTERSECTIONS;
    const auto val_elements
        = batch_size * m_layers.back().outputs * NUM_INTERSECTIONS;

    // input: input(float) 18 chanels * (BOARD_SIZE * BOARD_SIZE)
    auto pol_net_t = std::vector<net_t>(pol_elements);
    auto val_net_t = std::vector<net_t>(val_elements);
    // Always allocates enough space for floats
    constexpr auto one_plane = NUM_INTERSECTIONS * sizeof(float);

    if (!cudnn_context.m_buffers_allocated) {
        auto max_wsize = size_t{0};
        auto max_channels = unsigned{0};
        int layer_i = 0;
        for (const auto& layer : m_layers) {
            for (auto i = 0; i < m_num_worker_threads; i++) {
                if (!layer.is_residual_block || layer_i == 1) {
                    max_wsize = std::max(max_wsize,
                                         layer.conv_desc_single[i]->workspace_size);
                    max_wsize = std::max(max_wsize,
                                         layer.conv_desc_multi[i]->workspace_size);
                    if (cfg_backend == backend_t::CUDNNGRAPH) {
                        if (layer.conv_no_relu_desc_single.size() > 0) {
                            max_wsize = std::max(max_wsize,
                                                 layer.conv_no_relu_desc_single[i]->workspace_size);
                            max_wsize = std::max(max_wsize,
                                                 layer.conv_no_relu_desc_multi[i]->workspace_size);
                        }
                        if (layer.conv_add_relu_desc_single.size() > 0) {
                            max_wsize = std::max(max_wsize,
                                                 layer.conv_add_relu_desc_single[i]->workspace_size);
                            max_wsize = std::max(max_wsize,
                                                 layer.conv_add_relu_desc_multi[i]->workspace_size);
                        }
                    }
                    max_channels = std::max(max_channels,
                                            std::max(layer.channels, layer.outputs));
                }
            }
            layer_i++;
        }
        auto alloc_insize = cfg_batch_size * max_channels * one_plane;

        void *d_workspace;
        checkCUDA(cudaMalloc((void**)&d_workspace, max_wsize));

        void *d_InBuffer;
        checkCUDA(cudaMalloc((void**)&d_InBuffer, alloc_insize));

        void *d_OutBuffer;
        checkCUDA(cudaMalloc((void**)&d_OutBuffer, alloc_insize));

        void *d_TempBuffer;
        checkCUDA(cudaMalloc((void**)&d_TempBuffer, alloc_insize));

        cudnn_context.m_workspace = d_workspace;
        cudnn_context.m_InBuffer = d_InBuffer;
        cudnn_context.m_OutBuffer = d_OutBuffer;
        cudnn_context.m_TempBuffer = d_TempBuffer;
        cudnn_context.m_buffers_allocated = true;

        if (m_net_type == NetworkType::MINIGO_SE) {
            void *d_IdentityOutBuffer;
            checkCUDA(cudaMalloc((void**)&d_IdentityOutBuffer, alloc_insize));

            void *d_PoolBuffer;
            checkCUDA(cudaMalloc((void**)&d_PoolBuffer,
                                 cfg_batch_size * max_channels * sizeof(net_t)));

            cudnn_context.m_IdentityOutBuffer = d_IdentityOutBuffer;
            cudnn_context.m_PoolBuffer = d_PoolBuffer;

            const __half alpha_16 = __float2half(1.0f);
            const float alpha_32 = 1.0f;
            const __half beta_16 = __float2half(0.0f);
            const float beta_32 = 0.0f;
            void *d_m_alpha_16;
            checkCUDA(cudaMalloc((void**)&d_m_alpha_16, sizeof(alpha_16)));
            checkCUDA(cudaMemcpyAsync(d_m_alpha_16, (__half*)&alpha_16, sizeof(alpha_16),
                      cudaMemcpyHostToDevice, m_streams[tid]));
            void *d_m_alpha_32;
            checkCUDA(cudaMalloc((void**)&d_m_alpha_32, sizeof(alpha_32)));
            checkCUDA(cudaMemcpyAsync(d_m_alpha_32, (float*)&alpha_32, sizeof(alpha_32),
                      cudaMemcpyHostToDevice, m_streams[tid]));
            void *d_m_beta_16;
            checkCUDA(cudaMalloc((void**)&d_m_beta_16, sizeof(beta_16)));
            checkCUDA(cudaMemcpyAsync(d_m_beta_16, (__half*)&beta_16, sizeof(beta_16),
                      cudaMemcpyHostToDevice, m_streams[tid]));
            void *d_m_beta_32;
            checkCUDA(cudaMalloc((void**)&d_m_beta_32, sizeof(beta_32)));
            checkCUDA(cudaMemcpyAsync(d_m_beta_32, (float*)&beta_32, sizeof(beta_32),
                      cudaMemcpyHostToDevice, m_streams[tid]));
            cudaStreamSynchronize(m_streams[tid]);
            cudnn_context.m_alpha_32 = d_m_alpha_32;
            cudnn_context.m_alpha_16 = d_m_alpha_16;
            cudnn_context.m_beta_32 = d_m_beta_32;
            cudnn_context.m_beta_16 = d_m_beta_16;
        }
    }

    auto workspace = cudnn_context.m_workspace;
    auto InBuffer = cudnn_context.m_InBuffer;
    auto OutBuffer = cudnn_context.m_OutBuffer;
    auto IdentityOutBuffer = cudnn_context.m_IdentityOutBuffer;
    auto PoolBuffer = cudnn_context.m_PoolBuffer;
    auto TempBuffer = cudnn_context.m_TempBuffer;

    if (typeid(net_t) == typeid(float) && cfg_NCHW) {
        checkCUDA(cudaMemcpy(InBuffer, (net_t*)&input[0], inSize, cudaMemcpyHostToDevice));
    } else if (typeid(net_t) == typeid(__half) && cfg_NCHW) {
        auto input_net_t = std::vector<net_t>(batch_size * m_layers[0].channels * NUM_INTERSECTIONS);
        std::copy(input.begin(), input.end(), input_net_t.begin());
        checkCUDA(cudaMemcpy(InBuffer, (net_t*)&input_net_t[0], inSize, cudaMemcpyHostToDevice));
    } else {
        auto input_net_t = std::vector<net_t>(batch_size * m_layers[0].channels * NUM_INTERSECTIONS);
        input_net_t = BE::NCHW_to_NHWC<net_t>(
            input, batch_size, BOARD_SIZE, BOARD_SIZE, m_layers[0].channels);
        checkCUDA(cudaMemcpy(InBuffer, (net_t*)&input_net_t[0], inSize, cudaMemcpyHostToDevice));
    }

    for (auto iter = std::begin(m_layers); iter != std::end(m_layers); iter++) {
        const auto& layer = *iter;
        const auto niter = std::next(iter);

        if (layer.is_input_convolution) {
            // input: InBuffer
            assert(niter != std::end(m_layers));
            auto conv_weights = begin(layer.weights);
            auto conv_biases = begin(layer.weights) + 1;

            if (batch_size == 1) {
                convolveActivation(tid,
                                   InBuffer,
                                   OutBuffer,
                                   conv_weights[0],
                                   nullptr,
                                   conv_biases[0],
                                   workspace,
                                   layer.conv_desc_single[tid],
                                   layer.scale_1,
                                   1.0f);
            } else {
                convolveActivation(tid,
                                   InBuffer,
                                   OutBuffer,
                                   conv_weights[0],
                                   nullptr,
                                   conv_biases[0],
                                   workspace,
                                   layer.conv_desc_multi[tid],
                                   layer.scale_1,
                                   1.0f);
            }
            // output: OutBuffer
        } else if (layer.is_residual_block && !layer.is_se_block) {
            // input: OutBuffer
            assert(layer.channels == layer.outputs);
            assert(niter != std::end(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases  = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases  = begin(layer.weights) + 3;

            if (batch_size == 1) {
                convolveActivation(tid,
                                   OutBuffer,
                                   InBuffer,
                                   conv1_weights[0],
                                   nullptr,
                                   conv1_biases[0],
                                   workspace,
                                   m_layers[1].conv_desc_single[tid],
                                   layer.scale_1,
                                   1.0f);

                convolveActivation(tid,
                                   InBuffer,
                                   OutBuffer,
                                   conv2_weights[0],
                                   OutBuffer,          // *residualBuffer: first input
                                   conv2_biases[0],
                                   workspace,
                                   m_layers[1].conv_desc_single[tid],
                                   layer.scale_2,
                                   layer.scale_3);
            } else {
                convolveActivation(tid,
                                   OutBuffer,
                                   InBuffer,
                                   conv1_weights[0],
                                   nullptr,
                                   conv1_biases[0],
                                   workspace,
                                   m_layers[1].conv_desc_multi[tid],
                                   layer.scale_1,
                                   1.0f);

                convolveActivation(tid,
                                   InBuffer,
                                   OutBuffer,
                                   conv2_weights[0],
                                   OutBuffer,          // *residualBuffer: first input
                                   conv2_biases[0],
                                   workspace,
                                   m_layers[1].conv_desc_multi[tid],
                                   layer.scale_2,
                                   layer.scale_3);
            }
            // output: OutBuffer
        } else if (layer.is_residual_block && layer.is_se_block) {
            // input: OutBuffer
            assert(layer.channels == layer.outputs);
            assert(niter != std::end(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases  = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases  = begin(layer.weights) + 3;
            auto fc1_weights   = begin(layer.weights) + 4;
            auto fc1_biases    = begin(layer.weights) + 5;
            auto fc2_weights   = begin(layer.weights) + 6;
            auto fc2_biases    = begin(layer.weights) + 7;

            if (batch_size == 1) {
                convolveActivation(tid,
                                   OutBuffer,        // *bufferIn
                                   InBuffer,         // *bufferOut
                                   conv1_weights[0],
                                   nullptr,
                                   conv1_biases[0],
                                   workspace,
                                   m_layers[1].conv_desc_single[tid],
                                   layer.scale_1,
                                   1.0f);

                convolveIdentityActivation(tid,
                                           InBuffer,          // *bufferIn
                                           IdentityOutBuffer, // *bufferOut
                                           conv2_weights[0],
                                           nullptr,
                                           conv2_biases[0],
                                           workspace,
                                           m_layers[1].conv_desc_single[tid],
                                           layer.scale_2,
                                           layer.scale_3);
            } else {
                convolveActivation(tid,
                                   OutBuffer,        // *bufferIn
                                   InBuffer,         // *bufferOut
                                   conv1_weights[0],
                                   nullptr,
                                   conv1_biases[0],
                                   workspace,
                                   m_layers[1].conv_desc_multi[tid],
                                   layer.scale_1,
                                   1.0f);

                convolveIdentityActivation(tid,
                                           InBuffer,          // *bufferIn
                                           IdentityOutBuffer, // *bufferOut
                                           conv2_weights[0],
                                           nullptr,
                                           conv2_biases[0],
                                           workspace,
                                           m_layers[1].conv_desc_multi[tid],
                                           layer.scale_2,
                                           layer.scale_3);
            }
            if (typeid(net_t) == typeid(float)) {
                BE::squeeze_excitation_float(m_cublas_handles[tid],
                                             m_streams[tid],
                                             cudnn_context,
                                             OutBuffer,         // *bufferIn1: first input
                                             IdentityOutBuffer, // *bufferIn2: second output
                                             TempBuffer,
                                             fc1_weights[0],
                                             fc1_biases[0],
                                             fc2_weights[0],
                                             fc2_biases[0],
                                             InBuffer,          // *bufferOut
                                             PoolBuffer,
                                             batch_size,
                                             layer.outputs,
                                             NUM_INTERSECTIONS,
                                             cfg_NCHW,
                                             has_tensor_cores());
            } else {
                BE::squeeze_excitation_half(m_cublas_handles[tid],
                                            m_streams[tid],
                                            cudnn_context,
                                            OutBuffer,         // *bufferIn1: first input
                                            IdentityOutBuffer, // *bufferIn2: second output
                                            TempBuffer,
                                            fc1_weights[0],
                                            fc1_biases[0],
                                            fc2_weights[0],
                                            fc2_biases[0],
                                            InBuffer,          // *bufferOut
                                            PoolBuffer,
                                            batch_size,
                                            layer.outputs,
                                            NUM_INTERSECTIONS,
                                            cfg_NCHW,
                                            has_tensor_cores());
            }
            std::swap(InBuffer, OutBuffer);
            // output: OutBuffer
        } else {
            auto conv_weights = begin(layer.weights);
            // input: OutBuffer(net_t is float or __half)
            if (batch_size == 1) {
                convolve(tid,
                         OutBuffer,
                         InBuffer,
                         conv_weights[0],
                         workspace,
                         layer.conv_desc_single[tid],
                         layer.scale_1);
            } else {
                convolve(tid,
                         OutBuffer,
                         InBuffer,
                         conv_weights[0],
                         workspace,
                         layer.conv_desc_multi[tid],
                         layer.scale_1);
            }
            if (niter == std::end(m_layers)) {
                // Value input: InBuffer
                checkCUDA(cudaMemcpy(&val_net_t[0], InBuffer,
                                     val_elements * sizeof(net_t),
                                     cudaMemcpyDeviceToHost));
                // output: val_net_t
            } else {
                // Policy input: InBuffer
                checkCUDA(cudaMemcpy(&pol_net_t[0], InBuffer,
                                     pol_elements * sizeof(net_t),
                                     cudaMemcpyDeviceToHost));
                // output: pol_net_t
            }
        }
    }

    // input: val_net_t(net_t), pol_net_t(net_t)
    if (cfg_NCHW) {
        std::copy(val_net_t.begin(), val_net_t.end(), output_val.begin()); 
        std::copy(pol_net_t.begin(), pol_net_t.end(), output_pol.begin());
    } else {
        output_val = BE::NHWC_to_NCHW<net_t>(
            val_net_t, batch_size, BOARD_SIZE, BOARD_SIZE, Network::OUTPUTS_VALUE);
        output_pol = BE::NHWC_to_NCHW<net_t>(
            pol_net_t, batch_size, BOARD_SIZE, BOARD_SIZE, Network::OUTPUTS_POLICY);
    }
    // output: output_val(float) 1 chanels * (BOARD_SIZE * BOARD_SIZE)
    // output: output_pol(float) 2 chanels * (BOARD_SIZE * BOARD_SIZE)
}

template class BackendCuDNN<float>;
template class BackendCuDNN<half_float::half>;

#endif

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

auto constexpr CONV_DESC_INPUT = 0;
auto constexpr CONV_DESC_RESIDUAL = 1;
auto constexpr CONV_DESC_VALUE = 2;
auto constexpr CONV_DESC_POLICY = 3;

template <typename net_t> class CuDNN;
template <typename net_t> class CuDNN_Network;

struct conv_descriptor {
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnTensorDescriptor_t bias_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnActivationDescriptor_t activation_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    size_t workspace_size{0};
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

    void push_input_convolution(unsigned int filter_size,
                                unsigned int channels,
                                unsigned int outputs,
                                const std::vector<float>& weights,
                                const std::vector<float>& biases,
                                float scale);

    void push_residual(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights_1,
                       const std::vector<float>& biases_1,
                       const std::vector<float>& weights_2,
                       const std::vector<float>& biases_2,
                       float scale_1,
                       float scale_2,
                       float scale_3);

    void push_convolve(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights);

    size_t get_layer_count() const {
        return m_layers.size();
    }

    void forward(std::vector<float>& input,
                 std::vector<float>& output_pol,
                 std::vector<float>& output_val,
                 CuDNNContext& cudnn_context,
                 const int batch_size = 1);

    void forward_activations(std::vector<float>& input,
                             std::vector<float>& output_pol,
                             std::vector<float>& output_val,
                             CuDNNContext& cudnn_context,
                             const int batch_size = 1);

private:

    void push_weights(size_t layer, const std::vector<float>& weights);

    CuDNN<net_t>& m_cudnn;
    std::vector<CuDNN_Layer> m_layers;
    conv_descriptor m_conv_desc[4][2];

};

template <typename net_t>
class CuDNN {
    friend class CuDNN_Network<net_t>;
    friend class CuDNN_Layer;
public:
    CuDNN(int gpu, bool silent = false);

    void initialize(const int channels, int batch_size, int net_type);
    bool has_fp16_compute();
    bool has_tensor_cores();

    int m_batch_size = 1;

private:
    void convolve(void *bufferIn,
                  void *bufferOut,
                  void *weights,
                  void *workspace,
                  const conv_descriptor& conv_desc,
                  float alpha);

    void convolveActivation(void *bufferIn,
                            void *bufferOut,
                            void *weights,
                            void *residualBuffer,
                            void *biases,
                            void *workspace,
                            const conv_descriptor& conv_desc,
                            float alpha,
                            float alpha2 = 1.0f);

    void convolve_init(int channels, int outputs, int filter_size,
                       conv_descriptor& conv_desc, int batch_size = 1);

    cudnnHandle_t m_handle;
    bool m_fp16_compute{false};
    bool m_tensorcore{false};
    bool m_init_ok{false};
    int m_net_type{0};
};
#endif

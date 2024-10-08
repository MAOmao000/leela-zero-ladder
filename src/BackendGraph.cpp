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
#include "BackendGraph.h"

using namespace Utils;
namespace fe = cudnn_frontend;

// Y = ReLU(Convolve(X, W) + B)
template <typename net_t>
std::shared_ptr<conv_descriptor> BackendGraph<net_t>::convolve_init(
    cudnnHandle_t handle,
    const int channels,
    const int outputs,
    const int filter_size,
    const int batch_size) {

    int64_t n = batch_size;
    int64_t c = channels;
    int64_t h = BOARD_SIZE;
    int64_t w = BOARD_SIZE;
    int64_t k = outputs;
    int64_t r = filter_size;
    int64_t s = filter_size;
    fe::DataType_t data_type;
    fe::DataType_t compute_type;
    fe::DataType_t conv_compute_type;
    fe::DataType_t intermediate_type;
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = fe::DataType_t::FLOAT;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::FLOAT;
        intermediate_type = fe::DataType_t::FLOAT;
    } else {
        data_type = fe::DataType_t::HALF;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::HALF;
        intermediate_type = fe::DataType_t::HALF;
    }
    auto pad_size = filter_size / 2;
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = fe::graph::Graph();
        graph.set_io_data_type(data_type)
              .set_intermediate_data_type(intermediate_type)
              .set_compute_data_type(compute_type);
        std::shared_ptr<fe::graph::Tensor_attributes> X;
        std::shared_ptr<fe::graph::Tensor_attributes> W;
        std::shared_ptr<fe::graph::Tensor_attributes> B;
        std::shared_ptr<fe::graph::Tensor_attributes> Y;

        X = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({ n, c, h, w })
            .set_stride({ c * h * w, 1, c * w, c }));
        W = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("filter")
            .set_dim({ k, c, r, s })
            .set_stride({ c * r * s, 1, c * s, c }));
        auto conv_options = fe::graph::Conv_fprop_attributes()
                            .set_compute_data_type(conv_compute_type)
                            .set_padding({ pad_size, pad_size })
                            .set_stride({ 1, 1 })
                            .set_dilation({ 1, 1 });
        auto conv_output = graph.conv_fprop(X, W, conv_options);
        B = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("bias")
            .set_dim({ 1, k, 1, 1 })
            .set_stride({ k, 1, 1, 1 }));
        auto bias_options = fe::graph::Pointwise_attributes()
                            .set_mode(fe::PointwiseMode_t::ADD);
        auto bias_output = graph.pointwise(conv_output, B, bias_options);
        auto relu_options = fe::graph::Pointwise_attributes()
                            .set_mode(fe::PointwiseMode_t::RELU_FWD);
        Y = graph.pointwise(bias_output, relu_options);
        Y->set_output(true);

        checkCUDNNFE(graph.validate());
        checkCUDNNFE(graph.build_operation_graph(handle));
        checkCUDNNFE(graph.create_execution_plans({ fe::HeurMode_t::A }));
        checkCUDNNFE(graph.check_support(handle));
        checkCUDNNFE(graph.build_plans(handle));
        return std::make_tuple(graph, X, W, B, Y);
    };

    auto [graph, X, W, B, Y] = build_new_graph(handle);
    conv_desc->graph = graph;
    conv_desc->X = X;
    conv_desc->W = W;
    conv_desc->B = B;
    conv_desc->Y = Y;
    conv_desc->workspace_size = graph.get_workspace_size();
    return conv_desc;
}

// Y = Convolve(X, W) + B
template <typename net_t>
std::shared_ptr<conv_descriptor> BackendGraph<net_t>::convolve_no_relu_init(
    cudnnHandle_t handle,
    const int channels,
    const int outputs,
    const int filter_size,
    const int batch_size) {

    int64_t n = batch_size;
    int64_t c = channels;
    int64_t h = BOARD_SIZE;
    int64_t w = BOARD_SIZE;
    int64_t k = outputs;
    int64_t r = filter_size;
    int64_t s = filter_size;
    fe::DataType_t data_type;
    fe::DataType_t compute_type;
    fe::DataType_t conv_compute_type;
    fe::DataType_t intermediate_type;
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = fe::DataType_t::FLOAT;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::FLOAT;
        intermediate_type = fe::DataType_t::FLOAT;
    } else {
        data_type = fe::DataType_t::HALF;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::HALF;
        intermediate_type = fe::DataType_t::HALF;
    }
    auto pad_size = filter_size / 2;
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = fe::graph::Graph();
        graph.set_io_data_type(data_type)
              .set_intermediate_data_type(intermediate_type)
              .set_compute_data_type(compute_type);
        std::shared_ptr<fe::graph::Tensor_attributes> X;
        std::shared_ptr<fe::graph::Tensor_attributes> W;
        std::shared_ptr<fe::graph::Tensor_attributes> B;
        std::shared_ptr<fe::graph::Tensor_attributes> Y;

        X = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({ n, c, h, w })
            .set_stride({ c * h * w, 1, c * w, c }));
        W = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("filter")
            .set_dim({ k, c, r, s })
            .set_stride({ c * r * s, 1, c * s, c }));
        auto conv_options = fe::graph::Conv_fprop_attributes()
                            .set_compute_data_type(conv_compute_type)
                            .set_padding({ pad_size, pad_size })
                            .set_stride({ 1, 1 })
                            .set_dilation({ 1, 1 });
        auto conv_output = graph.conv_fprop(X, W, conv_options);
        B = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("bias")
            .set_dim({ 1, k, 1, 1 })
            .set_stride({ k, 1, 1, 1 }));
        auto bias_options = fe::graph::Pointwise_attributes()
                            .set_mode(fe::PointwiseMode_t::ADD);
        Y = graph.pointwise(conv_output, B, bias_options);
        Y->set_output(true);

        checkCUDNNFE(graph.validate());
        checkCUDNNFE(graph.build_operation_graph(handle));
        checkCUDNNFE(graph.create_execution_plans({ fe::HeurMode_t::A }));
        checkCUDNNFE(graph.check_support(handle));
        checkCUDNNFE(graph.build_plans(handle));
        return std::make_tuple(graph, X, W, B, Y);
    };

    auto [graph, X, W, B, Y] = build_new_graph(handle);
    conv_desc->graph = graph;
    conv_desc->X = X;
    conv_desc->W = W;
    conv_desc->B = B;
    conv_desc->Y = Y;
    conv_desc->workspace_size = graph.get_workspace_size();
    return conv_desc;
}

// Y = ReLU(X + Z)
template <typename net_t>
std::shared_ptr<conv_descriptor> BackendGraph<net_t>::convolve_add_relu_init(
    cudnnHandle_t handle,
    const int channels,
    const int outputs,
    const int batch_size) {

    int64_t n = batch_size;
    int64_t c = channels;
    int64_t h = BOARD_SIZE;
    int64_t w = BOARD_SIZE;
    int64_t k = outputs;
    fe::DataType_t data_type;
    fe::DataType_t compute_type;
    fe::DataType_t intermediate_type;
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = fe::DataType_t::FLOAT;
        compute_type = fe::DataType_t::FLOAT;
        intermediate_type = fe::DataType_t::FLOAT;
    } else {
        data_type = fe::DataType_t::HALF;
        compute_type = fe::DataType_t::FLOAT;
        intermediate_type = fe::DataType_t::HALF;
    }
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = fe::graph::Graph();
        graph.set_io_data_type(data_type)
              .set_intermediate_data_type(intermediate_type)
              .set_compute_data_type(compute_type);
        std::shared_ptr<fe::graph::Tensor_attributes> X;
        std::shared_ptr<fe::graph::Tensor_attributes> Z;
        std::shared_ptr<fe::graph::Tensor_attributes> Y;

        X = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({ n, c, h, w })
            .set_stride({ c * h * w, 1, c * w, c }));
        Z = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("feature")
            .set_dim({ n, c, h, w })
            .set_stride({ c * h * w, 1, c * w, c }));  // Should be p,q
        auto add_options = fe::graph::Pointwise_attributes()
                           .set_mode(fe::PointwiseMode_t::ADD);
        auto add_output = graph.pointwise(X, Z, add_options);
        auto relu_options = fe::graph::Pointwise_attributes()
                            .set_mode(fe::PointwiseMode_t::RELU_FWD);
        Y = graph.pointwise(add_output, relu_options);
        Y->set_output(true).set_stride({ k * h * w, 1, k * w, k });

        checkCUDNNFE(graph.validate());
        checkCUDNNFE(graph.build_operation_graph(handle));
        checkCUDNNFE(graph.create_execution_plans({ fe::HeurMode_t::A }));
        checkCUDNNFE(graph.check_support(handle));
        checkCUDNNFE(graph.build_plans(handle));
        return std::make_tuple(graph, X, Z, Y);
    };

    auto [graph, X, Z, Y] = build_new_graph(handle);
    conv_desc->graph = graph;
    conv_desc->X = X;
    conv_desc->Z = Z;
    conv_desc->Y = Y;
    conv_desc->workspace_size = graph.get_workspace_size();
    return conv_desc;
}

// Y = Convolve(X, W)
template <typename net_t>
std::shared_ptr<conv_descriptor> BackendGraph<net_t>::convolve_head_init(
    cudnnHandle_t handle,
    const int channels,
    const int outputs,
    const int filter_size,
    const int batch_size) {

    int64_t n = batch_size;
    int64_t c = channels;
    int64_t h = BOARD_SIZE;
    int64_t w = BOARD_SIZE;
    int64_t k = outputs;
    int64_t r = filter_size;
    int64_t s = filter_size;
    fe::DataType_t data_type;
    fe::DataType_t compute_type;
    fe::DataType_t conv_compute_type;
    fe::DataType_t intermediate_type;
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = fe::DataType_t::FLOAT;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::FLOAT;
        intermediate_type = fe::DataType_t::FLOAT;
    } else {
        data_type = fe::DataType_t::HALF;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::HALF;
        intermediate_type = fe::DataType_t::HALF;
    }
    auto pad_size = filter_size / 2;
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = fe::graph::Graph();
        graph.set_io_data_type(data_type)
              .set_intermediate_data_type(intermediate_type)
              .set_compute_data_type(compute_type);
        std::shared_ptr<fe::graph::Tensor_attributes> X;
        std::shared_ptr<fe::graph::Tensor_attributes> W;
        std::shared_ptr<fe::graph::Tensor_attributes> Y;

        X = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({ n, c, h, w })
            .set_stride({ c * h * w, 1, c * w, c }));
        W = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("filter")
            .set_dim({ k, c, r, s })
            .set_stride({ c * r * s, 1, c * s, c }));
        auto conv_options = fe::graph::Conv_fprop_attributes()
                            .set_compute_data_type(conv_compute_type)
                            .set_padding({ pad_size, pad_size })
                            .set_stride({ 1, 1 })
                            .set_dilation({ 1, 1 });
        Y = graph.conv_fprop(X, W, conv_options);
        Y->set_output(true);

        checkCUDNNFE(graph.validate());
        checkCUDNNFE(graph.build_operation_graph(handle));
        checkCUDNNFE(graph.create_execution_plans({ fe::HeurMode_t::A }));
        checkCUDNNFE(graph.check_support(handle));
        checkCUDNNFE(graph.build_plans(handle));
        return std::make_tuple(graph, X, W, Y);
    };

    auto [graph, X, W, Y] = build_new_graph(handle);
    //std::cout << graph << std::endl;
    conv_desc->graph = graph;
    conv_desc->X = X;
    conv_desc->W = W;
    conv_desc->Y = Y;
    conv_desc->workspace_size = graph.get_workspace_size();
    return conv_desc;
}

template <typename net_t>
void BackendGraph<net_t>::push_weights(
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
void BackendGraph<net_t>::push_weights_col_major(
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
void BackendGraph<net_t>::push_input_convolution(
    const unsigned int filter_size,
    const unsigned int channels,
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
void BackendGraph<net_t>::push_residual(
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
            auto conv_desc_multi
                = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
            auto conv_desc_no_relu_single
                = convolve_no_relu_init(m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_no_relu_desc_single.emplace_back(conv_desc_no_relu_single);
            auto conv_desc_no_relu_multi
                = convolve_no_relu_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_no_relu_desc_multi.emplace_back(conv_desc_no_relu_multi);
            auto conv_desc_add_relu_single
                = convolve_add_relu_init(m_handle[i], channels, outputs);
            m_layers[layer].conv_add_relu_desc_single.emplace_back(conv_desc_add_relu_single);
            auto conv_desc_add_relu_multi
                = convolve_add_relu_init(m_handle[i], channels, outputs, cfg_batch_size);
            m_layers[layer].conv_add_relu_desc_multi.emplace_back(conv_desc_add_relu_multi);
        }
    }
}

template <typename net_t>
void BackendGraph<net_t>::push_residual_se(
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
            auto conv_desc_no_relu_single
                = convolve_no_relu_init(m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_no_relu_desc_single.emplace_back(conv_desc_no_relu_single);
            auto conv_desc_no_relu_multi
                = convolve_no_relu_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_no_relu_desc_multi.emplace_back(conv_desc_no_relu_multi);
        }
    }
}

template <typename net_t>
void BackendGraph<net_t>::push_convolve(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases,  // Dummy arguments for inheritance usage
    const std::vector<float>& stddevs, // Dummy arguments for inheritance usage
    const std::vector<float>& ip1_w,   // Dummy arguments for inheritance usage
    const std::vector<float>& ip1_b,   // Dummy arguments for inheritance usage
    const std::vector<float>& ip2_w,   // Dummy arguments for inheritance usage
    const std::vector<float>& ip2_b) { // Dummy arguments for inheritance usage

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
                = convolve_head_init(m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
            auto conv_desc_multi
                = convolve_head_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
        }
    } else {
        m_layers[layer].is_policy = true;
        for (auto i = 0; i < m_num_worker_threads; i++) {
            auto conv_desc_single
                = convolve_head_init(m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
            auto conv_desc_multi
                = convolve_head_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
        }
    }
}

template <typename net_t>
void BackendGraph<net_t>::forward_activations(
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
                // Y = ReLU(Convolve(X, W) + B)
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                    {layer.conv_desc_single[tid]->X, InBuffer},
                    {layer.conv_desc_single[tid]->W, conv_weights[0]},
                    {layer.conv_desc_single[tid]->B, conv_biases[0]},
                    {layer.conv_desc_single[tid]->Y, OutBuffer} };
                checkCUDNNFE(layer.conv_desc_single[tid]->graph.execute(m_handle[tid],
                                                                        variant_pack, workspace));
            } else {
                // Y = ReLU(Convolve(X, W) + B)
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                    {layer.conv_desc_multi[tid]->X, InBuffer},
                    {layer.conv_desc_multi[tid]->W, conv_weights[0]},
                    {layer.conv_desc_multi[tid]->B, conv_biases[0]},
                    {layer.conv_desc_multi[tid]->Y, OutBuffer} };
                checkCUDNNFE(layer.conv_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                       variant_pack, workspace));
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
                // Y = ReLU(Convolve(X, W) + B)
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                    {m_layers[1].conv_desc_single[tid]->X, OutBuffer},
                    {m_layers[1].conv_desc_single[tid]->W, conv1_weights[0]},
                    {m_layers[1].conv_desc_single[tid]->B, conv1_biases[0]},
                    {m_layers[1].conv_desc_single[tid]->Y, InBuffer} };
                checkCUDNNFE(m_layers[1].conv_desc_single[tid]->graph.execute(m_handle[tid],
                                                                              variant_pack1, workspace));
                // Y = Convolve(X, W) + B
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                    {m_layers[1].conv_no_relu_desc_single[tid]->X, InBuffer},
                    {m_layers[1].conv_no_relu_desc_single[tid]->W, conv2_weights[0]},
                    {m_layers[1].conv_no_relu_desc_single[tid]->B, conv2_biases[0]},
                    {m_layers[1].conv_no_relu_desc_single[tid]->Y, TempBuffer} };
                checkCUDNNFE(m_layers[1].conv_no_relu_desc_single[tid]->graph.execute(m_handle[tid],
                                                                                      variant_pack2, workspace));
                // Y = ReLU(X + Z)
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack3 = {
                    {m_layers[1].conv_add_relu_desc_single[tid]->X, TempBuffer},
                    {m_layers[1].conv_add_relu_desc_single[tid]->Z, OutBuffer},
                    {m_layers[1].conv_add_relu_desc_single[tid]->Y, InBuffer} };
                checkCUDNNFE(m_layers[1].conv_add_relu_desc_single[tid]->graph.execute(m_handle[tid],
                                                                                       variant_pack3, workspace));
            } else {
                // Y = ReLU(Convolve(X, W) + B)
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                    {m_layers[1].conv_desc_multi[tid]->X, OutBuffer},
                    {m_layers[1].conv_desc_multi[tid]->W, conv1_weights[0]},
                    {m_layers[1].conv_desc_multi[tid]->B, conv1_biases[0]},
                    {m_layers[1].conv_desc_multi[tid]->Y, InBuffer} };
                checkCUDNNFE(m_layers[1].conv_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                             variant_pack1, workspace));
                // Y = Convolve(X, W) + B
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                    {m_layers[1].conv_no_relu_desc_multi[tid]->X, InBuffer},
                    {m_layers[1].conv_no_relu_desc_multi[tid]->W, conv2_weights[0]},
                    {m_layers[1].conv_no_relu_desc_multi[tid]->B, conv2_biases[0]},
                    {m_layers[1].conv_no_relu_desc_multi[tid]->Y, TempBuffer} };
                checkCUDNNFE(m_layers[1].conv_no_relu_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                                     variant_pack2, workspace));
                // Y = ReLU(X + Z)
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack3 = {
                    {m_layers[1].conv_add_relu_desc_multi[tid]->X, TempBuffer},
                    {m_layers[1].conv_add_relu_desc_multi[tid]->Z, OutBuffer},
                    {m_layers[1].conv_add_relu_desc_multi[tid]->Y, InBuffer} };
                checkCUDNNFE(m_layers[1].conv_add_relu_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                                      variant_pack3, workspace));
            }
            std::swap(InBuffer, OutBuffer);
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
                // Y = ReLU(Convolve(X, W) + B)
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                    {m_layers[1].conv_desc_single[tid]->X, OutBuffer},
                    {m_layers[1].conv_desc_single[tid]->W, conv1_weights[0]},
                    {m_layers[1].conv_desc_single[tid]->B, conv1_biases[0]},
                    {m_layers[1].conv_desc_single[tid]->Y, InBuffer} };
                checkCUDNNFE(m_layers[1].conv_desc_single[tid]->graph.execute(m_handle[tid],
                                                                              variant_pack1, workspace));
                // Y = Convolve(X, W) + B
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                    {m_layers[1].conv_no_relu_desc_single[tid]->X, InBuffer},
                    {m_layers[1].conv_no_relu_desc_single[tid]->W, conv2_weights[0]},
                    {m_layers[1].conv_no_relu_desc_single[tid]->B, conv2_biases[0]},
                    {m_layers[1].conv_no_relu_desc_single[tid]->Y, TempBuffer} };
                checkCUDNNFE(m_layers[1].conv_no_relu_desc_single[tid]->graph.execute(m_handle[tid],
                                                                                      variant_pack2, workspace));
            } else {
                // Y = ReLU(Convolve(X, W) + B)
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                    {m_layers[1].conv_desc_multi[tid]->X, OutBuffer},
                    {m_layers[1].conv_desc_multi[tid]->W, conv1_weights[0]},
                    {m_layers[1].conv_desc_multi[tid]->B, conv1_biases[0]},
                    {m_layers[1].conv_desc_multi[tid]->Y, InBuffer} };
                checkCUDNNFE(m_layers[1].conv_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                             variant_pack1, workspace));
                // Y = Convolve(X, W) + B
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                    {m_layers[1].conv_no_relu_desc_multi[tid]->X, InBuffer},
                    {m_layers[1].conv_no_relu_desc_multi[tid]->W, conv2_weights[0]},
                    {m_layers[1].conv_no_relu_desc_multi[tid]->B, conv2_biases[0]},
                    {m_layers[1].conv_no_relu_desc_multi[tid]->Y, TempBuffer} };
                checkCUDNNFE(m_layers[1].conv_no_relu_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                                     variant_pack2, workspace));
            }
            std::swap(TempBuffer, IdentityOutBuffer);
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
                // Y = Convolve(X, W)
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                    {layer.conv_desc_single[tid]->X, OutBuffer},
                    {layer.conv_desc_single[tid]->W, conv_weights[0]},
                    {layer.conv_desc_single[tid]->Y, InBuffer} };
                checkCUDNNFE(layer.conv_desc_single[tid]->graph.execute(m_handle[tid],
                                                                        variant_pack, workspace));
            } else {
                // Y = Convolve(X, W)
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                    {layer.conv_desc_multi[tid]->X, OutBuffer},
                    {layer.conv_desc_multi[tid]->W, conv_weights[0]},
                    {layer.conv_desc_multi[tid]->Y, InBuffer} };
                checkCUDNNFE(layer.conv_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                       variant_pack, workspace));
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

template class BackendGraph<float>;
template class BackendGraph<half_float::half>;

#endif

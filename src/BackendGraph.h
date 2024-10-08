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

#ifndef BACKENDGRAPH_H_INCLUDED
#define BACKENDGRAPH_H_INCLUDED

#include "Backend.h"

class BackendContext;
struct conv_descriptor;

template <typename net_t>
class BackendGraph : public Backend<net_t> {
public:
    BackendGraph() : Backend<net_t>() {}
    BackendGraph(
        const int gpu,
        const bool silent = false)
        : Backend<net_t>(gpu, silent) {}

    void push_input_convolution(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases,
        const float scale
    ) override;

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
    ) override;

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
    ) override;

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
    ) override;

private:
    void forward_activations(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val,
        BackendContext& cudnn_context,
        const int tid,
        const int batch_size = 1
    ) override;

    std::shared_ptr<conv_descriptor> convolve_init(
        cudnnHandle_t handle,
        const int channels,
        const int outputs,
        const int filter_size,
        const int batch_size = 1
    );

    std::shared_ptr<conv_descriptor> convolve_no_relu_init(
        cudnnHandle_t handle,
        const int channels,
        const int outputs,
        const int filter_size,
        const int batch_size = 1
    );

    std::shared_ptr<conv_descriptor> convolve_add_relu_init(
        cudnnHandle_t handle,
        const int channels,
        const int outputs,
        const int batch_size = 1
    );

    std::shared_ptr<conv_descriptor> convolve_head_init(
        cudnnHandle_t handle,
        const int channels,
        const int outputs,
        const int filter_size,
        const int batch_size = 1
    );

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

    size_t get_layer_count() const override {
        return this->m_layers.size();
    }

    bool has_tensor_cores() const override {
        return this->m_tensorcore;
    }
};
#endif

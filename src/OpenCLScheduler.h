/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Junhee Yoo and contributors
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

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef OPENCLSCHEDULER_H_INCLUDED
#define OPENCLSCHEDULER_H_INCLUDED
#include "config.h"

#include <list>
#include <thread>
#include <vector>

#include "GPUScheduler.h"
#include "OpenCL.h"
#include "GTP.h"
#include "SMP.h"

template <typename net_t>
class OpenCLScheduler : public GPUScheduler<net_t> {
public:
    OpenCLScheduler();
    ~OpenCLScheduler() = default;

    void initialize(
        const int channels,
        const NetworkType net_type,
        const std::string &model_hash = nullptr
    ) override;
    bool needs_autodetect() override;

private:
    void push_input_convolution(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const size_t weight_index,
        const std::shared_ptr<const ForwardPipe::ForwardPipeWeights> weights
    ) override;
    void push_residual(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const size_t weight_index,
        const std::shared_ptr<const ForwardPipe::ForwardPipeWeights> weights
    ) override;
    void push_residual_se(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const size_t weight_index,
        const std::shared_ptr<const ForwardPipe::ForwardPipeWeights> weights
    ) override;
    void push_convolve(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::shared_ptr<const ForwardPipe::ForwardPipeWeights> weights
    ) override;

    std::vector<std::unique_ptr<OpenCL<net_t>>> m_opencl;
};
#endif

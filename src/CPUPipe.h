/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Junhee Yoo and contributors

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

#ifndef CPUPIPE_H_INCLUDED
#define CPUPIPE_H_INCLUDED
#include "config.h"

#include <cassert>
#include <vector>
#include <cmath>

#if defined(USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(USE_DNNL)
#include <dnnl.hpp>
#elif defined(USE_OPENBLAS)
#include <cblas.h>
#else
#ifdef NDEBUG
#define EIGEN_NO_DEBUG // Disable assertions in your code．
#endif
#include <Eigen/Dense>
#endif

#include "ForwardPipe.h"

class CPUPipe : public ForwardPipe {
public:
    static void blas_initialize();
    void initialize(const int channels,
                    const NetworkType net_type,
                    const std::string &model_hash = "") override;
    bool forward(const std::vector<float>& input,
                 std::vector<float>& output_pol,
                 std::vector<float>& output_val) override;

    void push_weights(const unsigned int filter_size,
                      const unsigned int channels,
                      const unsigned int outputs,
                      const std::shared_ptr<const ForwardPipeWeights> weights) override;

    template <size_t spatial_size>
    static void batchnorm(const size_t channels,
                          std::vector<float>& data,
                          const float* const means,
                          const float* const stddevs,
                          const float* const eltwise = nullptr)
    {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto mean = means[c];
            const auto scale_stddev = stddevs[c];
            const auto arr = &data[c * spatial_size];

            if (eltwise == nullptr) {
                // Classical BN
                for (auto b = size_t{0}; b < spatial_size; b++) {
                    arr[b] = std::max(0.0f, scale_stddev * (arr[b] - mean));
                }
            } else {
                // BN + residual add
                const auto res = &eltwise[c * spatial_size];
                for (auto b = size_t{0}; b < spatial_size; b++) {
                    arr[b] =
                        std::max(0.0f, (scale_stddev * (arr[b] - mean)) + res[b]);
                }
            }
        }
    }

    template <unsigned int inputs, unsigned int outputs, bool ReLU>
    static std::vector<float> innerproduct_pub(const std::vector<float>& input,
                                               const std::vector<float>& weights,
                                               const std::vector<float>& biases)
    {
        std::vector<float> output(outputs);
#if defined(USE_EIGEN)
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> y(output.data(), outputs);
        y.noalias() =
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
                weights.data(), inputs, outputs).transpose()
            * Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>>(
                input.data(), inputs);
#else
#if defined(USE_DNNL)
        dnnl_sgemm('N', 'N',
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
#endif
            //  M  , N,   K
            outputs, 1, inputs,
            1.0f, &weights[0], inputs,
            &input[0], 1,
            0.0f, &output[0], 1);
#endif
        for (unsigned int o = 0; o < outputs; o++) {
            auto val = biases[o] + output[o];
            if (ReLU) {
                val = std::max(0.0f, val);
            }
            output[o] = val;
        }
        return output;
    }

private:
    void winograd_transform_in(const std::vector<float>& in,
                               std::vector<float>& V, int C);

    void winograd_sgemm(const std::vector<float>& U,
                        const std::vector<float>& V,
                        std::vector<float>& M, int C, int K);

    void winograd_transform_out(const std::vector<float>& M,
                                std::vector<float>& Y, int K);

    void winograd_convolve3(int outputs,
                            const std::vector<float>& input,
                            const std::vector<float>& U,
                            std::vector<float>& V,
                            std::vector<float>& M,
                            std::vector<float>& output);

    void innerproduct(const size_t inputs,
                      const size_t outputs,
                      const std::vector<float>& input,
                      const std::vector<float>& weights,
                      const std::vector<float>& biases,
                      std::vector<float>& output);

    int m_input_channels{};

    // Input + residual block tower + header
    std::shared_ptr<const ForwardPipeWeights> m_weights;

    NetworkType m_net_type{NetworkType::LEELA_ZERO};
};
#endif

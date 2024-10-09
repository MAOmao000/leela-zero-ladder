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
#include "config.h"

#ifdef USE_OPENCL

#include "OpenCLScheduler.h"
#include "OpenCL.h"
#include "Network.h"
#include "Random.h"
#include "Utils.h"

using Utils::ceilMultiple;
using Utils::myprintf;

class from_float {
public:
    from_float(const std::vector<float>& f) : m_f(f) {}

    operator const std::vector<float> &() {
        return m_f;
    }

    operator std::vector<half_float::half>() {
        auto ret = std::vector<half_float::half>(m_f.size());
        std::copy(cbegin(m_f), cend(m_f), begin(ret));
        return ret;
    }

private:
    const std::vector<float>& m_f;
};

template <typename T>
static std::vector<T> zeropad_U(
    const std::vector<float>& U,
    const int outputs,
    const int channels,
    const int outputs_pad,
    const int channels_pad)
{
    // Fill with zeroes
    auto Upad = std::vector<T>(WINOGRAD_TILE * outputs_pad * channels_pad);

    for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
        for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
            for (auto c = 0; c < channels; c++) {
                for (auto o = 0; o < outputs; o++) {
                    Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                         + nu * (outputs_pad * channels_pad) + c * outputs_pad
                         + o] =
                        U[xi * (WINOGRAD_ALPHA * outputs * channels)
                          + nu * (outputs * channels) + c * outputs + o];
                }
            }
        }
    }

    return Upad;
}

template <typename net_t>
OpenCLScheduler<net_t>::OpenCLScheduler()
{
    // multi-gpu?
    auto gpus = cfg_gpus;
    // An empty GPU list from the command line represents autodetect.
    // Put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }
    auto silent{false};

    this->m_out_pol_size = Network::OUTPUTS_POLICY * NUM_INTERSECTIONS;
    this->m_out_val_size = Network::OUTPUTS_VALUE * NUM_INTERSECTIONS;
    for (auto gpu : gpus) {
        auto opencl = std::make_unique<OpenCL<net_t>>(gpu, silent);
        auto net = std::make_unique<OpenCL_Network<net_t>>(*opencl);
        m_opencl.push_back(std::move(opencl));
        this->m_networks.push_back(std::move(net));
        // Starting next GPU, let's not dump full list of GPUs.
        silent = true;
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::initialize(
    const int channels,
    const NetworkType net_type,
    const std::string &model_hash)
{
    (void) model_hash;
    this->m_net_type = net_type;
    // Launch the worker threads.  Minimum 1 worker per GPU, but use enough
    // threads so that we can at least concurrently schedule something to the
    // GPU.
    size_t gpus_size;
    if (cfg_gpus.empty()) {
        gpus_size = 1;
    } else {
        gpus_size = cfg_gpus.size();
    }
    auto num_worker_threads =
        cfg_num_threads / cfg_batch_size / (gpus_size + 1) + 1;
    for (auto gnum = size_t{0}; gnum < gpus_size; gnum++) {
        m_opencl[gnum]->initialize(channels, cfg_batch_size, net_type);
        for (auto i = unsigned{0}; i < num_worker_threads; i++) {
            auto t =
                std::thread(&GPUScheduler<net_t>::batch_worker, this, gnum, i);
            this->m_worker_threads.push_back(std::move(t));
        }
    }
    // Exit immediately after tuning.  We should exit here because we skipped
    // initializing rest of the kernels due to some NVIDIA drivers crashing.
    if (cfg_tune_only) {
        exit(EXIT_SUCCESS);
    }
}

template <typename net_t>
bool OpenCLScheduler<net_t>::needs_autodetect()
{
    for (auto& opencl : m_opencl) {
        // If any card has no native fp16 compute, we'll have to benchmark.
        if (!opencl->has_fp16_compute() && !opencl->has_tensor_cores()) {
            return true;
        }
    }
    return false;
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_input_convolution(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipe::ForwardPipeWeights> weights)
{
    for (const auto& opencl_net : this->m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();
        const auto mwg = tuners[0];
        const auto kwg = tuners[2];
        const auto vwm = tuners[3];
        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto k_ceil = ceilMultiple(ceilMultiple(channels, kwg), vwm);
        const auto Upad = zeropad_U<net_t>(
            weights->m_conv_weights[weight_index],
            outputs,
            channels,
            m_ceil,
            k_ceil
        );
        opencl_net->push_input_convolution(
            filter_size,
            channels,
            outputs,
            Upad,
            from_float(weights->m_batchnorm_means[weight_index]),
            from_float(weights->m_batchnorm_stddevs[weight_index])
        );
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_residual(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipe::ForwardPipeWeights> weights)
{
    for (const auto& opencl_net : this->m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();
        const auto mwg = tuners[0];
        const auto vwm = tuners[3];
        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto Upad1 =
            zeropad_U<net_t>(weights->m_conv_weights[weight_index],
                outputs, outputs, m_ceil, m_ceil);
        const auto Upad2 =
            zeropad_U<net_t>(weights->m_conv_weights[weight_index + 1],
                outputs, outputs, m_ceil, m_ceil);
        opencl_net->push_residual(
            filter_size,
            channels,
            outputs,
            Upad1,
            from_float(weights->m_batchnorm_means[weight_index]),
            from_float(weights->m_batchnorm_stddevs[weight_index]),
            Upad2,
            from_float(weights->m_batchnorm_means[weight_index + 1]),
            from_float(weights->m_batchnorm_stddevs[weight_index + 1])
        );
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_residual_se(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipe::ForwardPipeWeights> weights)
{
    for (const auto& opencl_net : this->m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();
        const auto mwg = tuners[0];
        const auto vwm = tuners[3];
        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto Upad1 = zeropad_U<net_t>(
            weights->m_conv_weights[weight_index],
            outputs,
            outputs,
            m_ceil,
            m_ceil
        );
        const auto Upad2 = zeropad_U<net_t>(
            weights->m_conv_weights[weight_index + 1],
            outputs,
            outputs,
            m_ceil,
            m_ceil
        );
        opencl_net->push_residual_se(
            filter_size,
            channels,
            outputs,
            Upad1,
            from_float(weights->m_batchnorm_means[weight_index]),
            from_float(weights->m_batchnorm_stddevs[weight_index]),
            Upad2,
            from_float(weights->m_batchnorm_means[weight_index + 1]),
            from_float(weights->m_batchnorm_stddevs[weight_index + 1]),
            from_float(weights->m_se_weights[weight_index - 1]),
            from_float(weights->m_se_biases[weight_index - 1]),
            from_float(weights->m_se_weights[weight_index]),
            from_float(weights->m_se_biases[weight_index])
        );
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_convolve(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::shared_ptr<const ForwardPipe::ForwardPipeWeights> weights)
{
    for (const auto& opencl_net : this->m_networks) {
        if (outputs == Network::OUTPUTS_POLICY) {
            opencl_net->push_convolve(
                filter_size,
                channels,
                outputs,
                from_float(weights->m_conv_pol_w)
            );
        } else {
            opencl_net->push_convolve(
                filter_size,
                channels,
                outputs,
                from_float(weights->m_conv_val_w)
            );
        }
    }
}

template class OpenCLScheduler<float>;
#ifdef USE_HALF
template class OpenCLScheduler<half_float::half>;
#endif

#endif

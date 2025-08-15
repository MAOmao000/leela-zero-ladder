/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Junhee Yoo and contributors
    Copyright (C) 2025 MAOmao000

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

#if !defined(USE_CPU_ONLY)

#if defined(USE_CUDNN)
#include "BackendCuDNN.h"
#include "BackendGraph.h"
#if defined(USE_TENSOR_RT)
#include "BackendTensorRT.h"
#endif
#endif

#include "CPUPipe.h"
#include "GPUScheduler.h"
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
GPUScheduler<net_t>::GPUScheduler()
{
    m_waittime = cfg_batch_wait_time;
    // multi-gpu?
    auto gpus = cfg_gpus;
    // An empty GPU list from the command line represents autodetect.
    // Put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }

    auto silent{false};
    for (auto gpu : gpus) {
        if (cfg_backend == backend_t::OPENCL) {
            auto opencl = std::make_unique<OpenCL<net_t>>(gpu, silent);
            auto net = std::make_unique<OpenCL_Network<net_t>>(*opencl);
            m_opencl.push_back(std::move(opencl));
            m_networks.push_back(std::move(net));
#if defined(USE_CUDNN)
        } else if (cfg_backend == backend_t::CUDNN) {
            auto net = std::make_unique<BackendCuDNN<net_t>>(gpu, silent);
            m_backend.emplace_back(std::move(net));
        } else if (cfg_backend == backend_t::CUDNNGRAPH) {
            auto net = std::make_unique<BackendGraph<net_t>>(gpu, silent);
            m_backend.emplace_back(std::move(net));
#if defined(USE_TENSOR_RT)
        } else if (cfg_backend == backend_t::TENSORRT) {
            auto net = std::make_unique<BackendTRT>(gpu, silent);
            m_backend_trt.emplace_back(std::move(net));
#endif
#endif
        }
        // Starting next GPU, let's not dump full list of GPUs.
        silent = true;
    }
}

template <typename net_t>
void GPUScheduler<net_t>::initialize(
    const int channels,
    const NetworkType net_type,
    const std::string &model_hash)
{
#if !defined(USE_CUDNN)
    (void) model_hash;
#endif
    m_net_type = net_type;
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
#if defined(USE_CUDNN)
        if (cfg_backend == backend_t::OPENCL) {
            m_opencl[gnum]->initialize(channels, cfg_batch_size, net_type);
#if defined(USE_TENSOR_RT)
        } else if (cfg_backend == backend_t::TENSORRT) {
            m_backend_trt[gnum]->initialize(channels, cfg_batch_size, net_type, num_worker_threads, model_hash);
#endif
        } else {
            m_backend[gnum]->initialize(channels, cfg_batch_size, net_type, num_worker_threads, model_hash);
        }
#else
        m_opencl[gnum]->initialize(channels, cfg_batch_size, net_type);
#endif
        for (auto i = unsigned{0}; i < num_worker_threads; i++) {
            auto t =
                std::thread(&GPUScheduler<net_t>::batch_worker, this, gnum, i);
            m_worker_threads.push_back(std::move(t));
#if defined(USE_CUDNN)
            if (cfg_backend == backend_t::CUDNN || cfg_backend == backend_t::CUDNNGRAPH) {
                auto context = std::make_unique<BackendContext>();
                m_backend[gnum]->m_context.emplace_back(std::move(context));
            }
#endif
        }
    }
    // Exit immediately after tuning.  We should exit here because we skipped
    // initializing rest of the kernels due to some NVIDIA drivers crashing.
    if (cfg_tune_only) {
        exit(EXIT_SUCCESS);
    }
}

template <typename net_t>
GPUScheduler<net_t>::~GPUScheduler()
{
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_running = false;
    }
    m_cv.notify_all();
    for (auto& x : m_worker_threads) {
        x.join();
    }

#if defined(USE_CUDNN)
    if (cfg_backend == backend_t::TENSORRT) {
        for (const auto& backend : m_backend_trt) {
            for (auto iter = std::begin(backend->m_layers);
                iter != std::end(backend->m_layers);
                iter++)
            {
                const auto& layer = *iter;
                for (auto it = layer.weights.begin();
                    it != layer.weights.end();
                    ++it)
                {
                    void *w_mem;
                    cudaHostGetDevicePointer((void**)&w_mem, *it, 0);
                    if (w_mem) {
                        cudaFreeAsync(w_mem, cudaStreamDefault);
                    }
                    cudaFreeHost(*it);
                }
            }
        }
        for (const auto& backend : m_backend_trt) {
            for (const auto& context : backend->m_context) {
                if (context->m_buffers_allocated) {
#if defined(USE_TENSOR_RT)
                    for (auto ptr: context->mBuffers) {
                        cudaFreeAsync(ptr.second, cudaStreamDefault);
                    }
#endif
                }
            }
        }
        cudaStreamSynchronize(cudaStreamDefault);
        for (auto& backend : m_backend_trt) {
            backend.release();
        }
    } else {
        for (const auto& backend : m_backend) {
            for (auto iter = std::begin(backend->m_layers);
                iter != std::end(backend->m_layers);
                iter++)
            {
                const auto& layer = *iter;
                for (auto it = layer.weights.begin();
                    it != layer.weights.end();
                    ++it)
                {
                    cudaFreeAsync(*it, cudaStreamDefault);
                }
            }
        }
        for (const auto& backend : m_backend) {
            for (const auto& context : backend->m_context) {
                if (context->m_buffers_allocated) {
                    if (context->m_workspace)
                        cudaFreeAsync(context->m_workspace, cudaStreamDefault);
                    if (context->m_InBuffer)
                        cudaFreeAsync(context->m_InBuffer, cudaStreamDefault);
                    if (context->m_OutBuffer)
                        cudaFreeAsync(context->m_OutBuffer, cudaStreamDefault);
                    if (context->m_IdentityOutBuffer)
                        cudaFreeAsync(context->m_IdentityOutBuffer, cudaStreamDefault);
                    if (context->m_PoolBuffer)
                        cudaFreeAsync(context->m_PoolBuffer, cudaStreamDefault);
                    if (context->m_TempBuffer)
                        cudaFreeAsync(context->m_TempBuffer, cudaStreamDefault);
                    if (m_net_type == NetworkType::MINIGO_SE) {
                        if (context->m_alpha_16)
                            cudaFreeAsync(context->m_alpha_16, cudaStreamDefault);
                        if (context->m_alpha_32)
                            cudaFreeAsync(context->m_alpha_32, cudaStreamDefault);
                        if (context->m_beta_16)
                            cudaFreeAsync(context->m_beta_16, cudaStreamDefault);
                        if (context->m_beta_32)
                            cudaFreeAsync(context->m_beta_32, cudaStreamDefault);
                    }
                }
            }
        }
        cudaStreamSynchronize(cudaStreamDefault);
        for (auto& backend : m_backend) {
            backend.release();
        }
    }
#endif
}

template <typename net_t>
bool GPUScheduler<net_t>::needs_autodetect()
{
    if (cfg_backend == backend_t::OPENCL) {
        for (auto& opencl : m_opencl) {
            // If any card has no native fp16 compute, we'll have to benchmark.
            if (!opencl->has_fp16_compute() && !opencl->has_tensor_cores()) {
                return true;
            }
        }
#if defined(USE_CUDNN)
    } else if (cfg_backend != backend_t::TENSORRT) {
        for (auto& backend : m_backend) {
            // If any card has no native fp16 compute, we'll have to benchmark.
            if (!backend->has_fp16_compute() && !backend->has_tensor_cores()) {
                return true;
            }
        }
#endif
    }
    return false;
}

template <typename net_t>
void GPUScheduler<net_t>::push_input_convolution(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
#if defined(USE_CUDNN)
    if (cfg_backend != backend_t::OPENCL) {
#if defined(USE_TENSOR_RT)
        if (cfg_backend == backend_t::TENSORRT) {
            for (const auto& backend : m_backend_trt) {
                backend->push_input_convolution(
                    filter_size,
                    channels,
                    outputs,
                    weights->m_conv_weights[weight_index],
                    weights->m_batchnorm_means[weight_index]
                );
            }
        } else {
#endif
            for (const auto& backend : m_backend) {
                backend->push_input_convolution(
                    filter_size,
                    channels,
                    outputs,
                    weights->m_conv_weights[weight_index],
                    weights->m_batchnorm_means[weight_index]
                );
            }
#if defined(USE_TENSOR_RT)
        }
#endif
        return;
    }
#endif
    for (const auto& opencl_net : m_networks) {
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
void GPUScheduler<net_t>::push_residual(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
#if defined(USE_CUDNN)
    if (cfg_backend != backend_t::OPENCL) {
#if defined(USE_TENSOR_RT)
        if (cfg_backend == backend_t::TENSORRT) {
            for (const auto& backend : m_backend_trt) {
                backend->push_residual(
                    filter_size,
                    channels,
                    outputs,
                    weights->m_conv_weights[weight_index],
                    weights->m_batchnorm_means[weight_index],
                    weights->m_conv_weights[weight_index + 1],
                    weights->m_batchnorm_means[weight_index + 1]
                );
            }
        } else {
#endif
            for (const auto& backend : m_backend) {
                backend->push_residual(
                    filter_size,
                    channels,
                    outputs,
                    weights->m_conv_weights[weight_index],
                    weights->m_batchnorm_means[weight_index],
                    weights->m_conv_weights[weight_index + 1],
                    weights->m_batchnorm_means[weight_index + 1]
                );
            }
#if defined(USE_TENSOR_RT)
        }
#endif
        return;
    }
#endif
    for (const auto& opencl_net : m_networks) {
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
void GPUScheduler<net_t>::push_residual_se(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
#if defined(USE_CUDNN)
    if (cfg_backend != backend_t::OPENCL) {
#if defined(USE_TENSOR_RT)
        if (cfg_backend == backend_t::TENSORRT) {
            for (const auto& backend : m_backend_trt) {
                backend->push_residual_se(
                    filter_size,
                    channels,
                    outputs,
                    weights->m_conv_weights[weight_index],
                    weights->m_batchnorm_means[weight_index],
                    weights->m_conv_weights[weight_index + 1],
                    weights->m_batchnorm_means[weight_index + 1],
                    weights->m_se_weights[weight_index - 1],
                    weights->m_se_biases[weight_index - 1],
                    weights->m_se_weights[weight_index],
                    weights->m_se_biases[weight_index]
                );
            }
        } else {
#endif
            for (const auto& backend : m_backend) {
                backend->push_residual_se(
                    filter_size,
                    channels,
                    outputs,
                    weights->m_conv_weights[weight_index],
                    weights->m_batchnorm_means[weight_index],
                    weights->m_conv_weights[weight_index + 1],
                    weights->m_batchnorm_means[weight_index + 1],
                    weights->m_se_weights[weight_index - 1],
                    weights->m_se_biases[weight_index - 1],
                    weights->m_se_weights[weight_index],
                    weights->m_se_biases[weight_index]
                );
            }
#if defined(USE_TENSOR_RT)
        }
#endif
        return;
    }
#endif
    for (const auto& opencl_net : m_networks) {
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
void GPUScheduler<net_t>::push_convolve(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
#if defined(USE_CUDNN)
    if (cfg_backend != backend_t::OPENCL) {
#if defined(USE_TENSOR_RT)
        if (cfg_backend == backend_t::TENSORRT) {
            for (const auto& backend : m_backend_trt) {
                if (outputs == Network::OUTPUTS_POLICY) {
                    backend->push_convolve(
                        filter_size,
                        channels,
                        outputs,
                        weights->m_conv_pol_w,
                        weights->m_bn_pol_w1,
                        weights->m_ip_pol_w, 
                        weights->m_ip_pol_b,
                        weights->m_ip_pol_w, 
                        weights->m_ip_pol_b
                    );
                } else {
                    backend->push_convolve(
                        filter_size,
                        channels,
                        outputs,
                        weights->m_conv_val_w,
                        weights->m_bn_val_w1,
                        weights->m_ip1_val_w,
                        weights->m_ip1_val_b,
                        weights->m_ip2_val_w,
                        weights->m_ip2_val_b
                    );
                }
            }
        } else {
#endif
            for (const auto& backend : m_backend) {
                if (outputs == Network::OUTPUTS_POLICY) {
                    backend->push_convolve(
                        filter_size,
                        channels,
                        outputs,
                        weights->m_conv_pol_w,
                        weights->m_bn_pol_w1,
                        weights->m_ip_pol_w, 
                        weights->m_ip_pol_b,
                        weights->m_ip_pol_w, 
                        weights->m_ip_pol_b
                    );
                } else {
                    backend->push_convolve(
                        filter_size,
                        channels,
                        outputs,
                        weights->m_conv_val_w,
                        weights->m_bn_val_w1,
                        weights->m_ip1_val_w,
                        weights->m_ip1_val_b,
                        weights->m_ip2_val_w,
                        weights->m_ip2_val_b
                    );
                }
            }
#if defined(USE_TENSOR_RT)
        }
#endif
        return;
    }
#endif
    for (const auto& opencl_net : m_networks) {
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

template <typename net_t>
void GPUScheduler<net_t>::push_weights(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
    auto weight_index = size_t{0};
    // Winograd filter transformation changes filter size to 4x4
    push_input_convolution(
        filter_size,
        channels,
        outputs,
        weight_index,
        weights
    );
    weight_index++;
    if (m_net_type == NetworkType::LEELA_ZERO) {
        // residual blocks : except the first entry,
        // the second ~ last entry is all on residual topwer
        for (auto i = size_t{0}; i < weights->m_conv_weights.size() / 2; i++) {
            push_residual(
                filter_size,
                outputs,
                outputs,
                weight_index,
                weights
            );
            weight_index += 2;
        }
    } else if (m_net_type == NetworkType::MINIGO_SE) {
        // residual blocks : except the first entry,
        // the second ~ last entry is all on residual topwer
        for (auto i = size_t{0}; i < weights->m_conv_weights.size() / 2; i++) {
            push_residual_se(
                filter_size,
                outputs,
                outputs,
                weight_index,
                weights
            );
            weight_index += 2;
        }
    }
    // Output head convolutions
    push_convolve(
        1,
        outputs,
        Network::OUTPUTS_POLICY,
        weights
    );
    push_convolve(
        1,
        outputs,
        Network::OUTPUTS_VALUE,
        weights
    );
    if (cfg_backend != backend_t::TENSORRT) {
        if (cfg_backend == backend_t::OPENCL) {
            m_bn_pol_w1 = weights->m_bn_pol_w1;
            m_bn_pol_w2 = weights->m_bn_pol_w2;
        }
        m_ip_pol_w = weights->m_ip_pol_w;
        m_ip_pol_b = weights->m_ip_pol_b;
        if (cfg_backend == backend_t::OPENCL) {
            m_bn_val_w1 = weights->m_bn_val_w1;
            m_bn_val_w2 = weights->m_bn_val_w2;
        }
        m_ip1_val_w = weights->m_ip1_val_w;
        m_ip1_val_b = weights->m_ip1_val_b;
        m_ip2_val_w = weights->m_ip2_val_w;
        m_ip2_val_b = weights->m_ip2_val_b;
    }
#if defined(USE_CUDNN)
    if (cfg_backend != backend_t::OPENCL) {
        // Asynchronously cudaMemcpyAsync
        cudaStreamSynchronize(cudaStreamPerThread);
    }
#endif
}

template <typename net_t>
bool GPUScheduler<net_t>::forward(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val,
    const bool full_batch)
{
    if (m_draining.load()) {
        return false;
    }
#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        auto entry =
            std::make_shared<ForwardQueueEntry>(input, output_pol, output_val, full_batch);
        size_t queue_size = 0;
        std::unique_lock<std::mutex> lk(entry->mutex);
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            m_forward_queue.emplace_back(entry);
            queue_size = m_forward_queue.size();
        }
        if (!full_batch || queue_size >= cfg_batch_size) {
            m_cv.notify_one();
        }
        entry->cv.wait(lk);
        if (output_pol[0] == -1.0f) {
            return false;
        }
        return true;
    }
#endif
    std::vector<float> policy_data(Network::OUTPUTS_POLICY * NUM_INTERSECTIONS);
    std::vector<float> value_data(Network::OUTPUTS_VALUE * NUM_INTERSECTIONS);
    auto entry =
        std::make_shared<ForwardQueueEntry>(input, policy_data, value_data, full_batch);
    size_t queue_size = 0;
    std::unique_lock<std::mutex> lk(entry->mutex);
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.emplace_back(entry);
        queue_size = m_forward_queue.size();
        if (cfg_backend == backend_t::OPENCL &&
            cfg_batch_wait_time &&
            m_single_eval_in_progress.load()) {
            m_waittime += 2;
        }
    }
    if (!full_batch || queue_size >= cfg_batch_size) {
        m_cv.notify_one();
    }
    entry->cv.wait(lk);
    if (policy_data[0] == -1.0f) {
        return false;
    }
    // Get the moves
    if (cfg_backend == backend_t::OPENCL) {
        CPUPipe::batchnorm<NUM_INTERSECTIONS>(Network::OUTPUTS_POLICY, policy_data,
                                              m_bn_pol_w1.data(),
                                              m_bn_pol_w2.data());
    }
    const auto policy_out =
        CPUPipe::innerproduct_pub<Network::OUTPUTS_POLICY * NUM_INTERSECTIONS, POTENTIAL_MOVES, false>
            (policy_data, m_ip_pol_w, m_ip_pol_b);
    output_pol = Utils::softmax(policy_out, cfg_softmax_temp);

    // Now get the value
    if (cfg_backend == backend_t::OPENCL) {
        CPUPipe::batchnorm<NUM_INTERSECTIONS>(Network::OUTPUTS_VALUE, value_data,
                                              m_bn_val_w1.data(),
                                              m_bn_val_w2.data());
    }
    const auto winrate_data =
        CPUPipe::innerproduct_pub<Network::OUTPUTS_VALUE * NUM_INTERSECTIONS, Network::VALUE_LAYER, true>
            (value_data, m_ip1_val_w, m_ip1_val_b);
    const auto winrate_out =
        CPUPipe::innerproduct_pub<Network::VALUE_LAYER, 1, false>
            (winrate_data, m_ip2_val_w, m_ip2_val_b);

    output_val[0] = std::tanh(winrate_out[0]);

    return true;
}

#ifndef NDEBUG
struct batch_stats_t batch_stats;
#endif

template <typename net_t>
void GPUScheduler<net_t>::batch_worker(
    const size_t gnum,
    const size_t tid)
{
#if !defined(USE_CUDNN)
    (void) tid;
#endif
    constexpr auto in_size = Network::INPUT_CHANNELS * NUM_INTERSECTIONS;
    size_t out_pol_size{};
    size_t out_val_size{};
    if (cfg_backend == backend_t::TENSORRT) {
        out_pol_size = POTENTIAL_MOVES;
        out_val_size = 1;
    } else {
        out_pol_size = Network::OUTPUTS_POLICY * NUM_INTERSECTIONS;
        out_val_size = Network::OUTPUTS_VALUE * NUM_INTERSECTIONS;
    }
    OpenCLContext context;
    // batch scheduling heuristic.
    // Returns the batch picked up from the queue (m_forward_queue)
    // 1) Wait for m_waittime milliseconds for full batch
    // 2) if we don't have a full batch then just do a single eval
    //
    // The purpose of m_waittime is to prevent the system from deadlocking
    // because we were waiting for a job too long, while the job is never
    // going to come due to a control dependency (e.g., evals stuck on a
    // critical path).  To do so:
    //
    // 1) if we couldn't form a batch after waiting m_waittime ms, it means
    // that we hit the critical path and should do scalar evals.
    // Wait 1ms shorter next time.
    //
    // 2) if we picked up a single eval, but were getting additional evals
    // while that single eval was being processed, it means that we made
    // the wrong decision.  Wait 2ms longer next time.
    auto pickup_task = [this]() {
        std::list<std::shared_ptr<ForwardQueueEntry>> inputs;
        size_t count = 0;
        std::unique_lock<std::mutex> lk(m_mutex);
        while (true) {
            if (!m_running) {
                return inputs;
            }
            count = m_forward_queue.size();
            if (count >= cfg_batch_size) {
                count = cfg_batch_size;
                break;
            }
            bool timeout = !m_cv.wait_for(
                lk, std::chrono::milliseconds(m_waittime), [this]() {
                    return !m_running
                           || m_forward_queue.size() >= cfg_batch_size;
                }
            );
            if (!m_forward_queue.empty()) {
                if (cfg_backend == backend_t::OPENCL) {
                    if (timeout
                        && m_single_eval_in_progress.exchange(true) == false) {
                        // Waited long enough but couldn't form a batch.
                        // Check if there is any other single eval in progress,
                        // and if not, do one from this thread.
                        if (m_waittime > 1) {
                            m_waittime--;
                        }
                        count = 1;
                        break;
                    }
                } else {
                    if (timeout) {
                        count = std::min(static_cast<size_t>(cfg_batch_size), m_forward_queue.size());
                        break;
                    }
                }
            }
        }
        if (!m_running) {
            return inputs;
        }
        // Move 'count' evals from shared queue to local list.
        auto end = begin(m_forward_queue);
        std::advance(end, count);
        std::move(begin(m_forward_queue), end, std::back_inserter(inputs));
        m_forward_queue.erase(begin(m_forward_queue), end);
        return inputs;
    };
    // Returns the batch picked up from the queue (m_forward_queue)
    auto pickup_task_wait = [this]() {
        std::list<std::shared_ptr<ForwardQueueEntry>> inputs;
        std::unique_lock<std::mutex> lk(m_mutex);
        m_cv.wait(lk, [this] {
            return !m_running ||
                m_draining.load() ||
                m_forward_queue.size() >= cfg_batch_size ||
                (m_forward_queue.size() == 1 && !m_forward_queue.front()->full_batch);
        });
        if (!m_running) {
            return inputs;
        }
        auto count = m_forward_queue.size();
        if (!count) {
            return inputs;
        } else if (count >= static_cast<size_t>(cfg_batch_size)) {
            count = cfg_batch_size;
        } else if (!m_draining.load() &&
            m_forward_queue.front()->full_batch) {
            return inputs;
        }
        // Move 'count' evals from shared queue to local list.
        auto end = begin(m_forward_queue);
        std::advance(end, count);
        std::move(begin(m_forward_queue), end, std::back_inserter(inputs));
        m_forward_queue.erase(begin(m_forward_queue), end);
        return inputs;
    };
    auto batch_input = std::vector<float>(in_size * cfg_batch_size);
#if defined(USE_CUDNN)
    const auto dummy_input = std::vector<float>(in_size);
#endif
    auto batch_output_pol = std::vector<float>(out_pol_size * cfg_batch_size);
    auto batch_output_val = std::vector<float>(out_val_size * cfg_batch_size);
    while (true) {
        std::list<std::shared_ptr<ForwardQueueEntry>> inputs;
        if (cfg_batch_wait_time) {
            inputs = pickup_task();
        } else {
            inputs = pickup_task_wait();
        }
        if (!m_running) {
            return;
        }
        auto count = inputs.size();
        if (!count) {
            continue;
        }
#ifndef NDEBUG
        if (cfg_backend == backend_t::OPENCL && count < cfg_batch_size) {
            batch_stats.single_evals++;
        } else {
            batch_stats.batch_evals++;
        }
#endif
        if (cfg_backend == backend_t::TENSORRT || cfg_backend == backend_t::OPENCL) {
            // prepare input for forward() call
            batch_input.resize(in_size * count);
            batch_output_pol.resize(out_pol_size * count);
            batch_output_val.resize(out_val_size * count);
        }
        auto index = size_t{0};
        for (auto& x : inputs) {
            std::unique_lock<std::mutex> lk(x->mutex);
            std::copy(
                begin(x->in),
                end(x->in),
                begin(batch_input) + in_size * index
            );
            index++;
        }
#if defined(USE_CUDNN)
        if (cfg_backend == backend_t::CUDNN || cfg_backend == backend_t::CUDNNGRAPH) {
            for (auto i = index; i < cfg_batch_size; i++) {
                std::copy(
                    begin(dummy_input),
                    end(dummy_input),
                    begin(batch_input) + in_size * i
                );
            }
        }
#endif
        if (!m_draining.load()) {
            // run the NN evaluation
            if (cfg_backend == backend_t::OPENCL) {
                m_networks[gnum]->forward(
                    batch_input,
                    batch_output_pol,
                    batch_output_val,
                    context,
                    (const int)count
                );
#if defined(USE_CUDNN)
#if defined(USE_TENSOR_RT)
            } else if (cfg_backend == backend_t::TENSORRT) {
                m_backend_trt[gnum]->forward(
                    batch_input,
                    batch_output_pol,
                    batch_output_val,
                    static_cast<int>(tid),
                    static_cast<int>(count)
                );
#endif
            } else {
                m_backend[gnum]->forward(
                    batch_input,
                    batch_output_pol,
                    batch_output_val,
                    static_cast<int>(tid),
                    cfg_batch_size
                );
#endif
            }
        } else {
            for (size_t i = 0; i < index; i++) {
                batch_output_pol[out_pol_size * i] = -1.0f;
            }
        }
        // Get output and copy back
        index = 0;
        for (auto& x : inputs) {
            std::copy(
                begin(batch_output_pol) + out_pol_size * index,
                begin(batch_output_pol) + out_pol_size * (index + 1),
                begin(x->out_p)
            );
            std::copy(
                begin(batch_output_val) + out_val_size * index,
                begin(batch_output_val) + out_val_size * (index + 1),
                begin(x->out_v)
            );
            x->cv.notify_all();
            index++;
        }
        if (cfg_backend == backend_t::OPENCL &&
            cfg_batch_wait_time &&
            count == 1) {
            m_single_eval_in_progress.exchange(false);
        }
    }
}

template <typename net_t>
void GPUScheduler<net_t>::drain()
{
    // When signaled to drain requests, this method picks up all pending
    // requests and wakes them up.  Throws exception once the woken up request
    // sees m_draining.
    m_draining.exchange(true);
    m_cv.notify_all();
}

template <typename net_t>
void GPUScheduler<net_t>::resume()
{
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.clear();
    }
    // UCTNode::think() should wait for all child threads to complete before resuming.
    m_draining.exchange(false);
}

template class GPUScheduler<float>;
template class GPUScheduler<half_float::half>;

#endif

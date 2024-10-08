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
#include "GPUScheduler.h"
#if defined(USE_CUDNN)
#include "BackendCuDNN.h"
#endif
#if defined(USE_CUDNN)
#include "BackendGraph.h"
#endif
#if defined(USE_TENSOR_RT)
#include "BackendTensorRT.h"
#endif
#include "Network.h"
#include "Random.h"
#include "Utils.h"

template <typename net_t>
GPUScheduler<net_t>::GPUScheduler()
{
    if (cfg_backend == backend_t::OPENCL) {
        return;
    }
    // multi-gpu?
    auto gpus = cfg_gpus;
    // An empty GPU list from the command line represents autodetect.
    // Put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }
    if (cfg_backend == backend_t::TENSORRT) {
        m_out_pol_size = POTENTIAL_MOVES;
        m_out_val_size = Network::OUTPUTS_VALUE;
    } else {
        m_out_pol_size = Network::OUTPUTS_POLICY * NUM_INTERSECTIONS;
        m_out_val_size = Network::OUTPUTS_VALUE * NUM_INTERSECTIONS;
    }
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
    auto silent{false};
    for (auto gpu : gpus) {
        if (cfg_backend == backend_t::CUDNN) {
            auto net = std::make_unique<BackendCuDNN<net_t>>(gpu, silent);
            m_backend.emplace_back(std::move(net));
        } else if (cfg_backend == backend_t::CUDNNGRAPH) {
            auto net = std::make_unique<BackendGraph<net_t>>(gpu, silent);
            m_backend.emplace_back(std::move(net));
#if defined(USE_TENSOR_RT)
        } else if (cfg_backend == backend_t::TENSORRT) {
            auto net = std::make_unique<BackendTRT<net_t>>(gpu, silent);
            m_backend.emplace_back(std::move(net));
#endif
        } else {
            exit(EXIT_FAILURE);
        }
        // Starting next GPU, let's not dump full list of GPUs.
        silent = true;
    }
#endif
}

template <typename net_t>
void GPUScheduler<net_t>::initialize(
    const int channels,
    const NetworkType net_type,
    const std::string &model_hash)
{
    m_net_type = net_type;
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
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
    for (auto gnum = 0; gnum < gpus_size; gnum++) {
        m_backend[gnum]->initialize(channels, cfg_batch_size, net_type, num_worker_threads, model_hash);
        for (auto i = unsigned{0}; i < num_worker_threads; i++) {
            auto t =
                std::thread(&GPUScheduler<net_t>::batch_worker, this, gnum, i);
            m_worker_threads.push_back(std::move(t));
            if (cfg_backend == backend_t::CUDNN || cfg_backend == backend_t::CUDNNGRAPH) {
                auto context = std::make_unique<BackendContext>();
                m_backend[gnum]->m_context.emplace_back(std::move(context));
            }
        }
    }
#else
    (void) channels;
    (void) model_hash;
#endif
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
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
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
                if (cfg_backend == backend_t::TENSORRT) {
                    void *w_mem;
                    cudaHostGetDevicePointer((void**)&w_mem, *it, 0);
                    if (w_mem) {
                        cudaFreeAsync(w_mem, cudaStreamDefault);
                    }
                    cudaFreeHost(*it);
                } else {
                    cudaFreeAsync(*it, cudaStreamDefault);
                }
            }
        }
    }
    for (const auto& backend : m_backend) {
        for (const auto& context : backend->m_context) {
            if (cfg_backend == backend_t::TENSORRT) {
                if (context->m_buffers_allocated) {
#if defined(USE_TENSOR_RT)
                    for (auto ptr: context->mBuffers) {
                        cudaFreeAsync(ptr.second, cudaStreamDefault);
                    }
#endif
                }
            } else if (context->m_buffers_allocated) {
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
#endif
}

template <typename net_t>
bool GPUScheduler<net_t>::needs_autodetect()
{
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
    for (auto& backend : m_backend) {
        // If any card has no native fp16 compute, we'll have to benchmark.
        if (!backend->has_fp16_compute() && !backend->has_tensor_cores()) {
            return true;
        }
    }
#endif
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
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
    for (const auto& backend : m_backend) {
        float scale = 1.0f;
        backend->push_input_convolution(
            filter_size,
            channels,
            outputs,
            weights->m_conv_weights[weight_index],
            weights->m_batchnorm_means[weight_index],
            scale
        );
    }
#else
    (void) filter_size;
    (void) channels;
    (void) outputs;
    (void) weight_index;
    (void) weights;
#endif
}

template <typename net_t>
void GPUScheduler<net_t>::push_residual(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
    for (const auto& backend : m_backend) {
        /* Convolution alpha */
        float scale_1 = 1.0f;
        float scale_2 = 1.0f;
        /* Residual add alpha */
        float scale_3 = 1.0f;
        backend->push_residual(
            filter_size,
            channels,
            outputs,
            weights->m_conv_weights[weight_index],
            weights->m_batchnorm_means[weight_index],
            weights->m_conv_weights[weight_index + 1],
            weights->m_batchnorm_means[weight_index + 1],
            scale_1,
            scale_2,
            scale_3
        );
    }
#else
    (void) filter_size;
    (void) channels;
    (void) outputs;
    (void) weight_index;
    (void) weights;
#endif
}

template <typename net_t>
void GPUScheduler<net_t>::push_residual_se(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
    for (const auto& backend : m_backend) {
        /* Convolution alpha */
        float scale_1 = 1.0f;
        float scale_2 = 1.0f;
        /* Residual add alpha */
        float scale_3 = 1.0f;
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
            weights->m_se_biases[weight_index],
            scale_1,
            scale_2,
            scale_3
        );
    }
#else
    (void) filter_size;
    (void) channels;
    (void) outputs;
    (void) weight_index;
    (void) weights;
#endif
}

template <typename net_t>
void GPUScheduler<net_t>::push_convolve(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
    for (const auto& backend : m_backend) {
        if (outputs == Network::OUTPUTS_POLICY) {
            backend->push_convolve(
                filter_size,
                channels,
                outputs,
                weights->m_conv_pol_w,
                weights->m_conv_pol_b,
                weights->m_bn_pol_w2,
                weights->m_ip_pol_w,
                weights->m_ip_pol_b,
                weights->m_ip_pol_w, // The following are dummy arguments
                weights->m_ip_pol_b  // The following are dummy arguments
            );
        } else {
            backend->push_convolve(
                filter_size,
                channels,
                outputs,
                weights->m_conv_val_w,
                weights->m_conv_val_b,
                weights->m_bn_val_w2,
                weights->m_ip1_val_w,
                weights->m_ip1_val_b,
                weights->m_ip2_val_w,
                weights->m_ip2_val_b
            );
        }
    }
#else
    (void) filter_size;
    (void) channels;
    (void) outputs;
    (void) weights;
#endif
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
}

template <typename net_t>
void GPUScheduler<net_t>::forward(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val)
{
    auto entry =
        std::make_shared<ForwardQueueEntry>(input, output_pol, output_val);
    std::unique_lock<std::mutex> lk(entry->mutex);
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.emplace_back(entry);
        if (m_single_eval_in_progress.load()) {
            m_waittime += 2;
        }
    }
    m_cv.notify_one();
    entry->cv.wait(lk);
    if (m_ep) {
        throw NetworkHaltException();
    }
    if (m_draining) {
        throw NetworkHaltException();
    }
}

#ifndef NDEBUG
struct batch_stats_t batch_stats;
#endif

template <typename net_t>
void GPUScheduler<net_t>::batch_worker(
    const size_t gnum,
    const size_t tid)
{
    constexpr auto in_size = Network::INPUT_CHANNELS * NUM_INTERSECTIONS;
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
                if (timeout
                    && m_single_eval_in_progress.exchange(true) == false) {
                    // Waited long enough but couldn't form a batch.
                    // Check if there is any other single eval in progress,
                    // and if not, do one from this thread.
                    if (m_waittime > 1) {
                        m_waittime--;
                    }
                    if (cfg_execute_context == execute_t::SINGLE)
                        count = m_forward_queue.size();
                    else
                        count = 1;
                    break;
                }
            }
        }
        // Move 'count' evals from shared queue to local list.
        auto end = begin(m_forward_queue);
        std::advance(end, count);
        std::move(begin(m_forward_queue), end, std::back_inserter(inputs));
        m_forward_queue.erase(begin(m_forward_queue), end);
        return inputs;
    };
    auto batch_input = std::vector<float>();
    auto batch_output_pol = std::vector<float>();
    auto batch_output_val = std::vector<float>();
    while (true) {
        auto inputs = pickup_task();
        auto count = inputs.size();
        if (!m_running) {
            return;
        }
#ifndef NDEBUG
        if (count == 1) {
            batch_stats.single_evals++;
        } else {
            batch_stats.batch_evals++;
        }
#endif
        // prepare input for forward() call
        batch_input.resize(in_size * count);
        batch_output_pol.resize(m_out_pol_size * count);
        batch_output_val.resize(m_out_val_size * count);
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
        // run the NN evaluation
        try {
            if (cfg_backend == backend_t::OPENCL) {
                m_networks[gnum]->forward(
                    batch_input,
                    batch_output_pol,
                    batch_output_val,
                    context,
                    (const int)count
                );
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
            } else {
                m_backend[gnum]->forward(
                    batch_input,
                    batch_output_pol,
                    batch_output_val,
                    tid,
                    (const int)count
                );
#endif
            }
        } catch(...) {
            m_ep = std::current_exception();
            m_running = false;
        }
        // Get output and copy back
        index = 0;
        for (auto& x : inputs) {
            std::copy(
                begin(batch_output_pol) + m_out_pol_size * index,
                begin(batch_output_pol) + m_out_pol_size * (index + 1),
                begin(x->out_p)
            );
            std::copy(
                begin(batch_output_val) + m_out_val_size * index,
                begin(batch_output_val) + m_out_val_size * (index + 1),
                begin(x->out_v)
            );
            x->cv.notify_all();
            index++;
        }
        if (count < cfg_batch_size) {
            m_single_eval_in_progress = false;
        }
    }
}

template <typename net_t>
void GPUScheduler<net_t>::drain()
{
    // When signaled to drain requests, this method picks up all pending
    // requests and wakes them up.  Throws exception once the woken up request
    // sees m_draining.
    m_draining = true;
    std::list<std::shared_ptr<ForwardQueueEntry>> fq;
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        std::move(
            m_forward_queue.begin(),
            m_forward_queue.end(),
            std::back_inserter(fq)
        );
        m_forward_queue.clear();
    }
    for (auto& x : fq) {
        {
            // dummy lock/unlock to make sure thread in forward() is sleeping
            std::unique_lock<std::mutex> lk(x->mutex);
        }
        x->cv.notify_all();
    }
}

template <typename net_t>
void GPUScheduler<net_t>::resume()
{
    // UCTNode::think() should wait for all child threads to complete before resuming.
    assert(m_forward_queue.empty());
    m_draining = false;
}

template class GPUScheduler<float>;
#ifdef USE_HALF
template class GPUScheduler<half_float::half>;
#else
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
template class GPUScheduler<half_float::half>;
#endif
#endif

#endif

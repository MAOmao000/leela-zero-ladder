/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Junhee Yoo and contributors

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

#include "GTP.h"
#include "Random.h"
#include "Network.h"
#include "TRTScheduler.h"
#include "Utils.h"

template <typename net_t>
TRTScheduler<net_t>::~TRTScheduler() {
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_running = false;
    }
    m_cv.notify_all();
    for (auto& x : m_worker_threads) {
        x.join();
    }
    for (const auto& trt : m_networks) {
        for (auto iter = std::begin(trt->m_layers); iter != std::end(trt->m_layers); iter++) {
            const auto& layer = *iter;
            for (auto it = layer.weights.begin(); it != layer.weights.end(); ++it) {
                void *w_mem;
                cudaHostGetDevicePointer((void**)&w_mem, *it, 0);
                if (w_mem) cudaFreeAsync(w_mem, cudaStreamDefault);
                cudaFreeHost(*it);
            }
        }
    }
    for (const auto& trt: m_networks) {
        for (const auto& context: trt->m_context) {
            if (context->m_buffers_allocated) {
                for (auto ptr: context->mBuffers) {
                    cudaFreeAsync(ptr.second, cudaStreamDefault);
                }
            }
        }
    }
    cudaStreamSynchronize(cudaStreamDefault);
    for (auto& trt: m_networks) {
        trt.release();
    }
}

template <typename net_t>
TRTScheduler<net_t>::TRTScheduler() {
    // multi-gpu?
    auto gpus = cfg_gpus;

    // An empty GPU list from the command line represents autodetect.
    // Put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }

    auto silent{false};

    for (auto gpu : gpus) {
        auto net = std::make_unique<TRT<net_t>>(gpu, silent);
        m_networks.emplace_back(std::move(net));
        // Starting next GPU, let's not dump full list of GPUs.
        silent = true;
    }
}

template <typename net_t>
void TRTScheduler<net_t>::initialize(const int net_type, const std::string &model_hash) {
    m_net_type = net_type;

    // Launch the worker threads.  Minimum 1 worker per GPU, but use enough
    // threads so that we can at least concurrently schedule something to the GPU.
    auto num_worker_threads =
        cfg_num_threads / cfg_batch_size / (m_networks.size() + 1) + 1;

    auto gnum = 0;
    for (auto& trt : m_networks) {
        trt->initialize(net_type, num_worker_threads, model_hash);

        for (auto i = unsigned{0}; i < num_worker_threads; i++) {
            auto t =
                std::thread(&TRTScheduler<net_t>::batch_worker, this, gnum, i);
            m_worker_threads.emplace_back(std::move(t));
        }
        gnum++;
    }
}

template <typename net_t>
bool TRTScheduler<net_t>::needs_autodetect() {
    for (auto& trt : m_networks) {
        // If any card has no native fp16 compute, we'll have to benchmark.
        if (!trt->has_fp16_compute() && !trt->has_tensor_cores()) {
            return true;
        }
    }
    return false;
}

template <typename net_t>
void TRTScheduler<net_t>::push_input_convolution(unsigned int filter_size,
                                                 unsigned int channels,
                                                 unsigned int outputs,
                                                 const std::vector<float>& weights,
                                                 const std::vector<float>& means) {
    for (const auto& trt : m_networks) {
        trt->push_input_convolution(
            filter_size, channels, outputs,
            weights, means
        );
    }
}

template <typename net_t>
void TRTScheduler<net_t>::push_residual(unsigned int filter_size,
                                        unsigned int channels,
                                        unsigned int outputs,
                                        const std::vector<float>& weights_1,
                                        const std::vector<float>& means_1,
                                        const std::vector<float>& weights_2,
                                        const std::vector<float>& means_2) {

    for (const auto& trt: m_networks) {
        trt->push_residual(filter_size, channels, outputs,
                           weights_1,
                           means_1,
                           weights_2,
                           means_2);
    }
}

template <typename net_t>
void TRTScheduler<net_t>::push_residual_se(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& weights_1,
                                           const std::vector<float>& means_1,
                                           const std::vector<float>& weights_2,
                                           const std::vector<float>& means_2,
                                           const std::vector<float>& fc1_w,
                                           const std::vector<float>& fc1_b,
                                           const std::vector<float>& fc2_w,
                                           const std::vector<float>& fc2_b) {

    for (const auto& trt: m_networks) {
        trt->push_residual_se(
            filter_size,
            channels,
            outputs,
            weights_1,
            means_1,
            weights_2,
            means_2,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b);
    }
}

template <typename net_t>
void TRTScheduler<net_t>::push_convolve(unsigned int filter_size,
                                        unsigned int channels,
                                        unsigned int outputs,
                                        const std::vector<float>& weights,
                                        const std::vector<float>& biases,
                                        const std::vector<float>& ip1_w,
                                        const std::vector<float>& ip1_b,
                                        const std::vector<float>& ip2_w,
                                        const std::vector<float>& ip2_b) {

    for (const auto& trt: m_networks) {
        trt->push_convolve(filter_size, channels, outputs,
            weights, biases, ip1_w, ip1_b, ip2_w, ip2_b);
    }
}

template <typename net_t>
void TRTScheduler<net_t>::push_weights(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    std::shared_ptr<const ForwardPipeWeights> weights) {

    auto weight_index = size_t{0};

    push_input_convolution(filter_size, channels, outputs,
                           weights->m_conv_weights[weight_index],
                           weights->m_batchnorm_means[weight_index]);
    weight_index++;

    if (m_net_type == int(NetworkType::LEELA_ZERO)) {
        // residual blocks : except the first entry,
        // the second ~ last entry is all on residual topwer
        for (auto i = size_t{0}; i < weights->m_conv_weights.size() / 2; i++) {
            push_residual(filter_size, outputs, outputs,
                weights->m_conv_weights[weight_index],
                weights->m_batchnorm_means[weight_index],
                weights->m_conv_weights[weight_index + 1],
                weights->m_batchnorm_means[weight_index + 1]);
            weight_index += 2;
        }
    } else if (m_net_type == int(NetworkType::MINIGO_SE)) {
        // residual blocks : except the first entry,
        // the second ~ last entry is all on residual topwer
        for (auto i = size_t{0}; i < weights->m_conv_weights.size() / 2; i++) {
            push_residual_se(filter_size, outputs, outputs,
                weights->m_conv_weights[weight_index],
                weights->m_batchnorm_means[weight_index],
                weights->m_conv_weights[weight_index + 1],
                weights->m_batchnorm_means[weight_index + 1],
                weights->m_se_weights[weight_index - 1],
                weights->m_se_biases[weight_index - 1],
                weights->m_se_weights[weight_index],
                weights->m_se_biases[weight_index]);
            weight_index += 2;
        }
    }
    // Output head convolutions
    push_convolve(1, outputs, Network::OUTPUTS_POLICY,
                  weights->m_conv_pol_w, weights->m_conv_pol_b,
                  weights->m_ip_pol_w, weights->m_ip_pol_b,
                  weights->m_ip_pol_w, weights->m_ip_pol_b);
    push_convolve(1, outputs, Network::OUTPUTS_VALUE,
                  weights->m_conv_val_w, weights->m_conv_val_b,
                  weights->m_ip1_val_w, weights->m_ip1_val_b,
                  weights->m_ip2_val_w, weights->m_ip2_val_b);
}

template <typename net_t>
void TRTScheduler<net_t>::forward(const std::vector<float>& input,
                                  std::vector<float>& output_pol,
                                  std::vector<float>& output_val) {
    auto entry = std::make_shared<ForwardQueueEntry>(input, output_pol, output_val);
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

template <typename net_t>
void TRTScheduler<net_t>::batch_worker(size_t gnum, size_t tid) {
    constexpr auto in_size = Network::INPUT_CHANNELS * NUM_INTERSECTIONS;
    constexpr auto out_pol_size = POTENTIAL_MOVES;
    constexpr auto out_val_size = Network::OUTPUTS_VALUE;

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
                });

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

        // prepare input for forward() call
        batch_input.resize(in_size * count);
        batch_output_pol.resize(out_pol_size * count);
        batch_output_val.resize(out_val_size * count);

        auto index = size_t{0};
        for (auto& x : inputs) {
            std::unique_lock<std::mutex> lk(x->mutex);
            std::copy(begin(x->in), end(x->in),
                      begin(batch_input) + in_size * index);
            index++;
        }

        // run the NN evaluation
        try {
            m_networks[gnum]->forward(
                batch_input,
                batch_output_pol,
                batch_output_val,
                tid,
                (const int)count
            );
        } catch(...) {
            m_ep = std::current_exception();
            m_running = false;
        }
        // Get output and copy back
        index = 0;
        for (auto& x : inputs) {
            std::copy(begin(batch_output_pol) + out_pol_size * index,
                      begin(batch_output_pol) + out_pol_size * (index + 1),
                      begin(x->out_p));
            std::copy(begin(batch_output_val) + out_val_size * index,
                      begin(batch_output_val) + out_val_size * (index + 1),
                      begin(x->out_v));
            x->cv.notify_all();
            index++;
        }
        if (count < cfg_batch_size) {
            m_single_eval_in_progress = false;
        }
    }
}

template <typename net_t>
void TRTScheduler<net_t>::drain() {
    // When signaled to drain requests, this method picks up all pending
    // requests and wakes them up.  Throws exception once the woken up request
    // sees m_draining.
    m_draining = true;

    std::list<std::shared_ptr<ForwardQueueEntry>> fq;
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        std::move(m_forward_queue.begin(), m_forward_queue.end(),
                  std::back_inserter(fq));
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
void TRTScheduler<net_t>::resume() {
    // UCTNode::think() should wait for all child threads to complete before resuming.
    assert(m_forward_queue.empty());

    m_draining = false;
}
template class TRTScheduler<float>;
template class TRTScheduler<__half>;
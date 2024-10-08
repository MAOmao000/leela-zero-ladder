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

#ifndef GPUSCHEDULER_H_INCLUDED
#define GPUSCHEDULER_H_INCLUDED
#include "config.h"

#include <list>
#include <thread>
#include <vector>

#include "ForwardPipe.h"
#include "Backend.h"
#include "GTP.h"
#include "SMP.h"
#include "ThreadPool.h"
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
#include "Backend.h"
#endif

#ifndef NDEBUG
struct batch_stats_t {
    std::atomic<size_t> single_evals{0};
    std::atomic<size_t> batch_evals{0};
};
extern batch_stats_t batch_stats;
#endif

template <typename net_t>
class GPUScheduler : public ForwardPipe {
    class ForwardQueueEntry {
    public:
        std::mutex mutex;
        std::condition_variable cv;
        const std::vector<float>& in;
        std::vector<float>& out_p;
        std::vector<float>& out_v;
        ForwardQueueEntry(
            const std::vector<float>& input,
            std::vector<float>& output_pol,
            std::vector<float>& output_val)
            : in(input),
              out_p(output_pol),
              out_v(output_val) {}
    };

public:
    GPUScheduler();
    ~GPUScheduler() override;

    virtual void initialize(
        const int channels,
        const NetworkType net_type,
        const std::string &model_hash = nullptr
    ) override;
    bool needs_autodetect() override;
    void push_weights(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::shared_ptr<const ForwardPipeWeights> weights
    ) override;
    void forward(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val
    ) override;
    void batch_worker(
        const size_t gnum,
        const size_t tid = -1
    );

private:
    void drain() override;
    void resume() override;
    virtual void push_input_convolution(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const size_t weight_index,
        const std::shared_ptr<const ForwardPipeWeights> weights
    );
    virtual void push_residual(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const size_t weight_index,
        const std::shared_ptr<const ForwardPipeWeights> weights
    );
    virtual void push_residual_se(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const size_t weight_index,
        const std::shared_ptr<const ForwardPipeWeights> weights
    );
    virtual void push_convolve(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::shared_ptr<const ForwardPipeWeights> weights
    );

    std::atomic<bool> m_draining{false};
    // start with 10 milliseconds : lock protected
    int m_waittime{10};
    // set to true when single (non-batch) eval is in progress
    std::atomic<bool> m_single_eval_in_progress{false};
    std::list<std::shared_ptr<ForwardQueueEntry>> m_forward_queue;
    std::exception_ptr m_ep;
#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
    std::vector<std::unique_ptr<Backend<net_t>>> m_backend;
#endif

protected: // Member variables used by OpenCLSheduler
    bool m_running = true;
    std::vector<std::unique_ptr<OpenCL_Network<net_t>>> m_networks;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::list<std::thread> m_worker_threads;
    size_t m_out_pol_size;
    size_t m_out_val_size;
    NetworkType m_net_type{NetworkType::LEELA_ZERO};
};
#endif

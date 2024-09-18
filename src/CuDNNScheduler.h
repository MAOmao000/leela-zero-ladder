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

#ifndef CUDNNSCHEDULER_H_INCLUDED
#define CUDNNSCHEDULER_H_INCLUDED
#include "config.h"

#include <list>
#include <thread>
#include <vector>
#include <sstream>
#include <zlib.h>

#include "ForwardPipe.h"
#include "CuDNN.h"
#include "SMP.h"
#include "ThreadPool.h"

#if defined(USE_TENSOR_RT)
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include "NvInfer.h"
#ifndef NDEBUG
#define BOOST_STACKTRACE_USE_BACKTRACE
#include <boost/stacktrace.hpp>
#endif

namespace trtLog {
// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    Logger(Severity severity = Severity::kERROR)
        : mReportableSeverity(severity) {}
    void log(ILogger::Severity severity, const char* msg) noexcept override {
        // suppress information level log
        if (severity <= mReportableSeverity) {
            switch (severity) {
                case Severity::kINTERNAL_ERROR:
                    std::cerr << "[F] " << msg << std::endl;
                    break;
                case Severity::kERROR:
                    std::cerr << "[E] " << msg << std::endl;
#ifndef NDEBUG
                    std::cout << boost::stacktrace::stacktrace();
                    exit(0);
#endif
                    break;
                case Severity::kWARNING:
                    std::cerr << "[W] " << msg << std::endl;
                    break;
                case Severity::kINFO:
                    std::cerr << "[I] " << msg << std::endl;
                    break;
                case Severity::kVERBOSE:
                    std::cerr << "[V] " << msg << std::endl;
                    break;
                default:
                    std::cerr << "[?] " << msg << std::endl;
            }
        }
    }
    nvinfer1::ILogger& getTRTLogger() noexcept {
        return *this;
    }
    void setReportableSeverity(Severity severity) noexcept {
        mReportableSeverity = severity;
    }
private:
    Severity mReportableSeverity;
};
}
#endif

template <typename net_t>
class CuDNNScheduler : public ForwardPipe {
    class ForwardQueueEntry {
    public:
        std::mutex mutex;
        std::condition_variable cv;
        const std::vector<float>& in;
        std::vector<float>& out_p;
        std::vector<float>& out_v;
        ForwardQueueEntry(const std::vector<float>& input,
                          std::vector<float>& output_pol,
                          std::vector<float>& output_val)
            : in(input), out_p(output_pol), out_v(output_val) {}
    };

public:
    virtual ~CuDNNScheduler();
    CuDNNScheduler();

    virtual void initialize(int channels,
                            const int net_type,
                            const std::string &model_hash = nullptr);
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val);
    virtual bool needs_autodetect();
    virtual void push_weights(
        unsigned int filter_size, unsigned int channels, unsigned int outputs,
        std::shared_ptr<const ForwardPipeWeights> weights);

    void push_input_convolution(unsigned int filter_size,
                                unsigned int channels,
                                unsigned int outputs,
                                const std::vector<float>& weights,
                                const std::vector<float>& means);

    void push_residual(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights_1,
                       const std::vector<float>& means_1,
                       const std::vector<float>& weights_2,
                       const std::vector<float>& means_2);

    void push_residual_se(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights_1,
                       const std::vector<float>& means_1,
                       const std::vector<float>& weights_2,
                       const std::vector<float>& means_2,
                       const std::vector<float>& fc1_w,
                       const std::vector<float>& fc1_b,
                       const std::vector<float>& fc2_w,
                       const std::vector<float>& fc2_b);

    void push_convolve(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights,
                       const std::vector<float>& biases,
                       const std::vector<float>& means);

private:
    bool m_running{true};
    std::atomic<bool> m_draining{false};
    std::vector<std::unique_ptr<CuDNN_Network<net_t>>> m_networks;

    std::mutex m_mutex;
    std::condition_variable m_cv;

    // start with 10 milliseconds : lock protected
    int m_waittime{10};

    // set to true when single (non-batch) eval is in progress
    std::atomic<bool> m_single_eval_in_progress{false};

    std::list<std::shared_ptr<ForwardQueueEntry>> m_forward_queue;
    std::list<std::thread> m_worker_threads;

    void batch_worker(size_t gnum, size_t tid);

    virtual void drain();
    virtual void resume();

    int m_net_type{0};
    std::exception_ptr m_ep;
};

#endif

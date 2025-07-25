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

#ifndef BACKENDTENSORRT_H_INCLUDED
#define BACKENDTENSORRT_H_INCLUDED

#include "Backend.h"

class BackendContext;
struct conv_descriptor;
struct InferDeleter;

static std::string vformat(const char *fmt, va_list ap) {
    // Allocate a buffer on the stack that's big enough for us almost
    // all the time.  Be prepared to allocate dynamically if it doesn't fit.
    size_t size = 4096;
    char stackbuf[4096];
    std::vector<char> dynamicbuf;
    char *buf = &stackbuf[0];

    int needed;
    while (true) {
        // Try to vsnprintf into our buffer.
        needed = vsnprintf(buf, size, fmt, ap);
        // NB. C99 (which modern Linux and OS X follow) says vsnprintf
        // failure returns the length it would have needed.  But older
        // glibc and current Windows return -1 for failure, i.e., not
        // telling us how much was needed.

        if (needed <= (int)size && needed >= 0)
            break;

        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So try again using a dynamic buffer.  This
        // doesn't happen very often if we chose our initial size well.
        size = (needed > 0) ? (needed+1) : (size*2);
        dynamicbuf.resize(size+1);
        buf = &dynamicbuf[0];
    }
    return std::string(buf, (size_t)needed);
}

inline std::string strprintf(const char* fmt, ...) {
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat(fmt, ap);
    va_end (ap);
    return buf;
}

inline std::string readFileBinary(
    const std::string& filename) {
    std::ifstream ifs;
    ifs.open(filename, std::ios::binary);
    std::string str((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    return str;
}

// Error Recorder for TensorRT
class TRTErrorRecorder : public nvinfer1::IErrorRecorder {
    mutable std::mutex mutex;
    std::vector<std::pair<nvinfer1::ErrorCode,std::string>> errors;
    std::atomic<int32_t> refCount;

public:
    TRTErrorRecorder()
        :mutex(),
         errors(),
         refCount(0)
    {}
    void clear() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        errors.clear();
    }
    int32_t getNbErrors() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return (int32_t)errors.size();
    }
    nvinfer1::ErrorCode getErrorCode(int32_t errorIdx) const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (errorIdx < 0 || static_cast<size_t>(errorIdx) >= errors.size())
            return nvinfer1::ErrorCode::kINVALID_ARGUMENT;
        return errors[errorIdx].first;
    }
    IErrorRecorder::ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (errorIdx < 0 || static_cast<size_t>(errorIdx) >= errors.size())
            return "";
        return errors[errorIdx].second.c_str();
    }
    bool hasOverflowed() const noexcept override {
        return false;
    }
    bool empty() const noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        return errors.size() <= 0;
    }
    bool reportError(nvinfer1::ErrorCode val, nvinfer1::IErrorRecorder::ErrorDesc desc) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        errors.push_back(std::make_pair(val, std::string(desc)));
        std::cerr << "TensorRT error : " << std::string(desc) << std::endl;
        return false;
    }
    nvinfer1::IErrorRecorder::RefCount incRefCount() noexcept override {
        return ++refCount;
    }
    nvinfer1::IErrorRecorder::RefCount decRefCount() noexcept override {
        return --refCount;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;

template <typename net_t>
class BackendTRT : public Backend<net_t> {
public:
    BackendTRT() : Backend<net_t>() {}
    BackendTRT(
        const int gpu,
        const bool silent = false)
        : Backend<net_t>(gpu, silent) {}

    void push_input_convolution(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases
    ) override;

    void push_residual(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights_1,
        const std::vector<float>& biases_1,
        const std::vector<float>& weights_2,
        const std::vector<float>& biases_2
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
        const std::vector<float>& se_fc2_b
    ) override;

    void push_convolve(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases,
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
        const size_t batch_size = 1
    ) override;

    void push_weights(
        const size_t layer,
        const std::vector<net_t>& weights,
        const bool host_mem = false
    );

    void push_weights_col_major(
        const size_t layer,
        const std::vector<net_t>& weights,
        const int row,
        const int column,
        const int channels = 1,
        const bool host_mem = false
    );

    // Builds the network engine
    bool build(
        const int num_worker_threads,
        const int64_t batch_size
    );

    // Create full model using the TensorRT network definition API and build the engine.
    bool constructNetwork(
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string& tune_desc
    );

    nvinfer1::ITensor* initInputs(
        char const *inputName,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        const int channels,
        const int rows,
        const int cols
    );

    nvinfer1::ILayer* buildConvLayer(
        nvinfer1::ITensor* input,
        unsigned int filter_size,
        int64_t weights_size,
        void* weights,
        int64_t biases_size,
        void* biases,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string& tune_desc,
        std::string op_name,
        unsigned int outputs
    );

    nvinfer1::ILayer* buildActivationLayer(
        nvinfer1::ITensor* input,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string& tune_desc,
        std::string op_name,
        nvinfer1::ActivationType act_type
    );

    nvinfer1::ILayer* applyGPoolLayer(
        nvinfer1::ITensor* input,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network
    );

    size_t get_layer_count() const override {
        return this->m_layers.size();
    }

    std::vector<std::unique_ptr<nvinfer1::IRuntime>> mRuntime;
    std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> mEngine;
    TRTErrorRecorder trtErrorRecorder;
};
#endif

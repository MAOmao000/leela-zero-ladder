/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Henrik Forsten

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

#ifndef TRT_H_INCLUDED
#define TRT_H_INCLUDED

#include "config.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <fstream>
#include <ostream>
#include <iostream>
#include <new>
#include <numeric>
#include <type_traits>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <map>
#include <iterator>
#include <filesystem>
#include <stdarg.h>

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferRuntimeBase.h"
#include "NvInferSafeRuntime.h"
#include "NvInferConsistency.h"

#include "sha2.h"

template <typename net_t> class TRT;

#define ASSERT(condition)                                         \
    do {                                                          \
        if (!(condition)) {                                       \
            myprintf_error("Assertion failure %s(%d): %s\n",      \
                __FILE__, __LINE__, #condition);                  \
            throw std::runtime_error("TensorRT error");           \
        }                                                         \
    } while (0)

#define checkCUDNN(expression)                                    \
    {                                                             \
        cudnnStatus_t status = (expression);                      \
        if (status != CUDNN_STATUS_SUCCESS) {                     \
            myprintf_error("Error on %s(%d): %s\n",               \
                __FILE__, __LINE__, cudnnGetErrorString(status)); \
            throw std::runtime_error("cuDNN error");              \
        }                                                         \
    }

#define checkCUDA(error)                                          \
    {                                                             \
        if (error != cudaSuccess) {                               \
            myprintf_error("Error on %s(%d): %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(error));   \
            throw std::runtime_error("CUDA error");               \
        }                                                         \
    }

class TRT_Layer {
    template <typename> friend class TRT;
    template <typename> friend class TRTScheduler;
private:
    unsigned int channels{0};
    unsigned int outputs{0};
    unsigned int filter_size{0};
    bool is_input_convolution{false};
    bool is_residual_block{false};
    bool is_se_block{false};
    bool is_value{false};
    bool is_policy{false};
    std::vector<void *> weights;
    std::vector<int64_t> weights_size;
    std::string name;
};

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

struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, InferDeleter>;

class TRTContext {
    template <typename> friend class TRT;
    template <typename> friend class TRTScheduler;
private:
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext_n{nullptr};
    std::map<std::string, void*> mBuffers;
    bool m_buffers_allocated{false};
};

template <typename net_t>
class TRT {
    template <typename> friend class TRTScheduler;
public:
    TRT(
        const int gpu,
        const bool silent = false
    );

    void initialize(
        const int net_type,
        const int num_worker_threads,
        const std::string &model_hash = ""
    );

    void push_input_convolution(
        const unsigned int filter_size,
        unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases
    );

    void push_residual(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights_1,
        const std::vector<float>& biases_1,
        const std::vector<float>& weights_2,
        const std::vector<float>& biases_2
    );

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
    );

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
    );

    size_t get_layer_count() const {
        return m_layers.size();
    }

    void forward(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val,
        const int tid,
        const int batch_size = 1
    );

    void forward_activations(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val,
        TRTContext& cudnn_context,
        const int tid,
        const int batch_size = 1
    );

    bool has_fp16_compute() {
        return m_fp16_compute;
    }

    bool has_tensor_cores() {
        return m_tensorcore;
    }

    std::vector<TRT_Layer> m_layers;
    std::vector<std::unique_ptr<TRTContext>> m_context;
    std::vector<std::unique_ptr<nvinfer1::IRuntime>> mRuntime;
    std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> mEngine;

private:
    void push_weights(
        const size_t layer,
        const std::vector<float>& weights
    );

    void push_weights_col_major(
        const size_t layer,
        const std::vector<float>& weights,
        const int row,
        const int column,
        const int channels = 1
    );

    // Builds the network engine
    bool build(
        const int num_worker_threads,
        const int batch_size //,
    );

    // Create full model using the TensorRT network definition API and build the engine.
    void constructNetwork(
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        nvinfer1::IOptimizationProfile* profile,
        nvinfer1::IOptimizationProfile* profile_n,
        const int batch_size
    );

    nvinfer1::ITensor* initInputs(
        char const *inputName,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        nvinfer1::IOptimizationProfile* profile,
        nvinfer1::IOptimizationProfile* profile_n,
        const int channels,
        const int rows,
        const int cols,
        const int batch_size
    );

    nvinfer1::ILayer* buildConvLayer(
        nvinfer1::ITensor* input,
        unsigned int filter_size,
        int64_t weights_size,
        void* weights,
        int64_t biases_size,
        void* biases,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name,
        unsigned int outputs
    );

    nvinfer1::ILayer* buildActivationLayer(
        nvinfer1::ITensor* input,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name,
        nvinfer1::ActivationType act_type
    );

    nvinfer1::ILayer* applyGPoolLayer(
        nvinfer1::ITensor* input,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name
    );

    // Serves as a hash of the network architecture specific to tuning
    std::string mTuneDesc;

    std::vector<cudaStream_t> m_streams;

    int m_net_type{0};
    int m_num_worker_threads{1};
    cudaDeviceProp m_device_prop;
    bool m_fp16_compute{false};
    bool m_tensorcore{false};
    std::string m_model_hash{""};
};
#endif

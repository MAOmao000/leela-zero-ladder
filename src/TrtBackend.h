/*
    This file is part of Leela Zero.
    Copyright (C) 2024 Maomao000

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
#ifndef TRTBACKEND_H_INCLUDED
#define TRTBACKEND_H_INCLUDED

#include <stdlib.h>
#include <fstream>
#include <ostream>
#include <iostream>
#include <new>
#include <vector>
#include <string>
#include <numeric>
#include <type_traits>
#include <cassert>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <memory>
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

template <typename net_t> class CuDNN_Network;
template <typename net_t> class CuDNN;
class CuDNN_Layer;

#define ASSERT(condition)                                         \
    do {                                                          \
        if (!(condition)) {                                       \
            std::cerr << "Assertion failure " << __FILE__ << "("  \
                << __LINE__ << "): " << #condition << std::endl;  \
            throw std::runtime_error("TensorRT error");           \
        }                                                         \
    } while (0)

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

inline std::string readFileBinary(const std::string& filename) {
    std::ifstream ifs;
    ifs.open(filename, std::ios::binary);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return str;
}

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    nvinfer1::ILogger& getTRTLogger() noexcept {
        return *this;
    }

    void log(ILogger::Severity severity, const char* msg) noexcept override {
        // suppress information level log
        //if (severity == Severity::kINFO || severity == Severity::kVERBOSE) return;
        //std::cout << msg << std::endl;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << msg << std::endl;
                std::exit(EXIT_FAILURE);
            case Severity::kERROR:
                std::cerr << msg << std::endl;
                std::exit(EXIT_FAILURE);
            case Severity::kWARNING:
                break;
            case Severity::kINFO:
                break;
            case Severity::kVERBOSE:
                break;
        }
    }
};

struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;

template <typename net_t>
class TrtResNet {
public:
    TrtResNet(CuDNN_Network<net_t>& cudnn_net, CuDNN<net_t>& cudnn)
        : m_cudnn_net(cudnn_net),
          m_cudnn(cudnn) {
    }

    ~TrtResNet() {
        if (mEngine) mEngine.reset();
        if (mRuntime) mRuntime.reset();
    }

    // Builds the network engine
    bool build();

    TrtUniquePtr<nvinfer1::IRuntime> mRuntime{nullptr};
    TrtUniquePtr<nvinfer1::ICudaEngine> mEngine{nullptr};
protected:
    std::map<std::string, nvinfer1::Weights> mWeightMap;

private:
    // Create full model using the TensorRT network definition API and build the engine.
    void constructNetwork(
        TrtUniquePtr<nvinfer1::IBuilder>& builder,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        TrtUniquePtr<nvinfer1::IBuilderConfig>& config,
        nvinfer1::IOptimizationProfile* profile);

    nvinfer1::ITensor* initInputs(
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        const CuDNN_Layer layer,
        nvinfer1::IOptimizationProfile* profile);

    nvinfer1::ILayer* buildConvLayer(
        nvinfer1::ITensor* input,
        unsigned int filter_size,
        int64_t weights_size,
        void* weights,
        int64_t biases_size,
        void* biases,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name,
        unsigned int channels,
        unsigned int outputs);

    nvinfer1::IScaleLayer* buildMatBiasLayer(
        nvinfer1::ITensor* input,
        int64_t biases_size,
        void* biases,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name,
        unsigned int channels,
        unsigned int outputs);

    nvinfer1::ILayer* buildActivationLayer(
        nvinfer1::ITensor* input,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name,
        nvinfer1::ActivationType act_type);

    nvinfer1::ILayer* TrtResNet<net_t>::applyGPoolLayer(
        nvinfer1::ITensor* input,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name);

    nvinfer1::ILayer* TrtResNet<net_t>::buildMatMulLayer(
        nvinfer1::ITensor* input,
        int64_t weights_size,
        void* weights,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name,
        unsigned int channels,
        unsigned int outputs);

    cudaStream_t mStream{nullptr};
    std::string mTuneDesc; // Serves as a hash of the network architecture specific to tuning

    CuDNN_Network<net_t>& m_cudnn_net;
    CuDNN<net_t>& m_cudnn;
};
#endif

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
        const std::vector<float>& biases,
        const float scale
    ) override;

    void push_residual(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights_1,
        const std::vector<float>& biases_1,
        const std::vector<float>& weights_2,
        const std::vector<float>& biases_2,
        const float scale_1,
        const float scale_2,
        const float scale_3
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
        const std::vector<float>& se_fc2_b,
        const float scale_1,
        const float scale_2,
        const float scale_3
    ) override;

    void push_convolve(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases
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
        const size_t batch_size
    );

    // Create full model using the TensorRT network definition API and build the engine.
    void constructNetwork(
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        nvinfer1::IOptimizationProfile* profile,
        nvinfer1::IOptimizationProfile* profile_n,
        const size_t batch_size
    );

    nvinfer1::ITensor* initInputs(
        char const *inputName,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        nvinfer1::IOptimizationProfile* profile,
        nvinfer1::IOptimizationProfile* profile_n,
        const int channels,
        const int rows,
        const int cols,
        const size_t batch_size
    );

    nvinfer1::ILayer* buildConvLayer(
        nvinfer1::ITensor* input,
        unsigned int filter_size,
        int64_t weights_size,
        void* weights,
        int64_t biases_size,
        void* biases,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name,
        unsigned int outputs
    );

    nvinfer1::ILayer* buildActivationLayer(
        nvinfer1::ITensor* input,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name,
        nvinfer1::ActivationType act_type
    );

    nvinfer1::ILayer* applyGPoolLayer(
        nvinfer1::ITensor* input,
        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
        std::string op_name
    );

    size_t get_layer_count() const override {
        return this->m_layers.size();
    }

    std::vector<std::unique_ptr<nvinfer1::IRuntime>> mRuntime;
    std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> mEngine;
    std::string mTuneDesc; // Serves as a hash of the network architecture specific to tuning
    std::map<std::string, nvinfer1::Weights> mWeightMap;
};
#endif

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

#ifndef CUDNN_H_INCLUDED
#define CUDNN_H_INCLUDED

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
#if defined(USE_CUDNN_GRAPH)
#include <cudnn_frontend.h>
#endif

#if defined(USE_TENSOR_RT)
#include "TrtBackend.h"
#endif

auto constexpr CONV_DESC_INPUT = 0;
auto constexpr CONV_DESC_RESIDUAL = 1;
auto constexpr CONV_DESC_VALUE = 2;
auto constexpr CONV_DESC_POLICY = 3;
#if defined(USE_CUDNN_GRAPH)
auto constexpr CONV_DESC_NO_RELU = 4;
auto constexpr CONV_DESC_ADD_RELU = 5;
#endif
auto constexpr SINGLE_BATCH = 0;
auto constexpr MULTIPLE_BATCHES = 1;

template <typename net_t> class CuDNN;
template <typename net_t> class CuDNN_Network;

#if defined(USE_CUDNN_GRAPH)
#define checkCUDNNFE(expression)                                  \
    {                                                             \
        cudnn_frontend::error_t err = (expression);               \
        if (err.is_bad()) {                                       \
            std::cerr << "Error on " << __FILE__ << "("           \
                <<  __LINE__ << "): "                             \
                << err.get_message() << std::endl;                \
            throw std::runtime_error("cuDNN FrontEnd error");     \
        }                                                         \
    }
#endif

#define checkCUDNN(expression)                                   \
    {                                                            \
        cudnnStatus_t status = (expression);                     \
        if (status != CUDNN_STATUS_SUCCESS) {                    \
            std::cerr << "Error on " << __FILE__ << "("          \
                << __LINE__ << "): "                             \
                << cudnnGetErrorString(status) << std::endl;     \
            throw std::runtime_error("cuDNN error");             \
        }                                                        \
    }

#define checkCUDA(error)                                         \
    {                                                            \
        if (error != cudaSuccess) {                              \
            std::cerr << "Error on " << __FILE__ << "("          \
                << __LINE__ << "): "                             \
                << cudaGetErrorString(error) << std::endl;       \
            throw std::runtime_error("CUDA error");              \
        }                                                        \
    }

#define checkCUBLAS(status)                                      \
    {                                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                   \
            std::cerr << "Error on " << __FILE__ << "("          \
                << __LINE__ << "): "                             \
                << cublasGetStatusString(status) << std::endl;   \
            throw std::runtime_error("cuBlas error");            \
        }                                                        \
    }

void global_average_pooling_float(
    const float *input,
    float *output,
    const int batch_size,
    const int channels,
    const int spatial
    );

void global_average_pooling_float_NHWC(
    const float *input,
    float *output,
    const int batch_size,
    const int channels,
    const int spatial
    );

void global_average_pooling_half(
    const __half *input,
    __half *output,
    const int batch_size,
    const int channels,
    const int spatial
    );

void global_average_pooling_half_NHWC(
    const __half *input,
    __half *output,
    const int batch_size,
    const int channels,
    const int spatial
    );

void add_bias_float(
    float *buf,
    const float *biases,
    const int batch_size,
    const int channels,
    const bool relu
    );

void add_bias_half(
    __half *buf,
    const __half *biases,
    const int batch_size,
    const int channels,
    const bool relu
    );

void se_scale_float(
    float *outbuf,
    const float *buf,
    const float *biases,
    const float *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial
    );

void se_scale_float_NHWC(
    float *outbuf,
    const float *buf,
    const float *biases,
    const float *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial
    );

void se_scale_half(
    __half *outbuf,
    const __half *buf,
    const __half *biases,
    const __half *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial
    );

void se_scale_half_NHWC(
    __half *outbuf,
    const __half *buf,
    const __half *biases,
    const __half *bufferIn,
    const int batch_size,
    const int channels,
    const int spatial
    );

struct conv_descriptor {
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnTensorDescriptor_t bias_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnActivationDescriptor_t activation_descriptor;
    cudnnActivationDescriptor_t activation_identity_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    cudnnConvolutionFwdAlgo_t convolution_identity_algorithm;
    size_t workspace_size{0};
    size_t workspace_identity_size{0};

#if defined(USE_CUDNN_GRAPH)
    cudnn_frontend::graph::Graph graph;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> X;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> W;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> B;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Y;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Z;
#endif
    bool is_initialized{false};
};

class CuDNN_Layer {
    template <typename> friend class CuDNN_Network;
    template <typename> friend class CuDNNScheduler;
#if defined(USE_TENSOR_RT)
    template <typename> friend class TrtResNet;
#endif
private:
    unsigned int channels{0};
    unsigned int outputs{0};
    unsigned int filter_size{0};
    bool is_input_convolution{false};
    bool is_residual_block{false};
    bool is_se_block{false};
    bool is_convolve1{false};
    conv_descriptor conv_desc[2];
#if defined(USE_CUDNN_GRAPH)
    conv_descriptor conv_no_relu_desc[2];
    conv_descriptor conv_add_relu_desc[2];
#endif
    float scale_1{1.0f};
    float scale_2{1.0f};
    float scale_3{1.0f};
    std::vector<void *> weights;
#if defined(USE_TENSOR_RT)
    std::vector<int64_t> weights_size;
    std::string name;
#endif
};

#if defined(USE_TENSOR_RT)
template <typename net_t> class TrtResNet;
#endif

class CuDNNContext {
    template <typename> friend class CuDNN;
    template <typename> friend class CuDNN_Network;
#if defined(USE_TENSOR_RT)
    template <typename> friend class TrtResNet;
    template <typename> friend class CuDNNScheduler;
#endif
private:
    void *m_workspace{nullptr};
    void *m_InBuffer{nullptr};
    void *m_OutBuffer{nullptr};
    void *m_IdentityOutBuffer{nullptr};
    void *m_PoolBuffer{nullptr};
    void *m_TempBuffer{nullptr};
    bool m_is_initialized{false};
    bool m_buffers_allocated{false};
#if defined(USE_TENSOR_RT)
    TrtUniquePtr<nvinfer1::IExecutionContext> mContext{nullptr};
    std::map<std::string, void*> mBuffers;
#endif
};

template <typename net_t>
class CuDNN_Network {

public:
    CuDNN_Network(CuDNN<net_t>& cudnn) : m_cudnn(cudnn) {}
//    ~CuDNN_Network() {
//std::cerr << "####################################################### CuDNN_Network Destructor " << std::this_thread::get_id() << std::endl;
//#if defined(USE_TENSOR_RT)
//        m_trt.reset();
//#endif
//    }
    CuDNN<net_t>& getCuDNN() {
        return m_cudnn;
    }

    void push_input_convolution(const unsigned int filter_size,
                                unsigned int channels,
                                const unsigned int outputs,
                                const std::vector<float>& weights,
                                const std::vector<float>& biases,
                                const float scale);

    void push_residual(const unsigned int filter_size,
                       const unsigned int channels,
                       const unsigned int outputs,
                       const std::vector<float>& weights_1,
                       const std::vector<float>& biases_1,
                       const std::vector<float>& weights_2,
                       const std::vector<float>& biases_2,
                       const float scale_1,
                       const float scale_2,
                       const float scale_3);

    void push_residual_se(const unsigned int filter_size,
                          const unsigned int channels,
                          const unsigned int outputs,
                          const std::vector<float>& weights_1,
                          const std::vector<float>& biases_1,
                          const std::vector<float>& weights_2,
                          const std::vector<float>& biases_2,
                          const std::vector<float>& se_fc1_w,
                          const std::vector<float>& se_fc1_b,
                          const std::vector<float>& se_fc2_w,
                          const std::vector<float>& se_fc2_b, // );
                          const float scale_1,
                          const float scale_2,
                          const float scale_3);

    void push_convolve(const unsigned int filter_size,
                       const unsigned int channels,
                       const unsigned int outputs,
                       const std::vector<float>& weights);

    size_t get_layer_count() const {
        return m_layers.size();
    }

    void forward(const std::vector<float>& input,
                 std::vector<float>& output_pol,
                 std::vector<float>& output_val,
                 std::shared_ptr<CuDNNContext> cudnn_context,
                 const int batch_size = 1);

    void forward_activations(const std::vector<float>& input,
                             std::vector<float>& output_pol,
                             std::vector<float>& output_val,
                             std::shared_ptr<CuDNNContext> cudnn_context,
                             const int batch_size = 1);

    CuDNN<net_t>& m_cudnn;
    std::vector<CuDNN_Layer> m_layers;
private:
    void push_weights(const size_t layer, const std::vector<float>& weights);
    void push_weights_trt(const size_t layer, const std::vector<float>& weights);
    void push_weights_col_major(const size_t layer,
                                const std::vector<float>& weights,
                                const int row,
                                const int column);
    void push_weights_trt_col_major(const size_t layer,
                                    const std::vector<float>& weights,
                                    const int row,
                                    const int column);

#if defined(USE_CUDNN_GRAPH)
    conv_descriptor m_conv_desc[6][2];
#else
    conv_descriptor m_conv_desc[4][2];
#endif
//#if defined(USE_TENSOR_RT)
//    std::unique_ptr<TrtResNet<net_t> > m_trt{nullptr};
//#endif
};

#if defined(USE_CUDNN_GRAPH)
namespace fe = cudnn_frontend;
#endif
template <typename net_t>
class CuDNN {
    friend class CuDNN_Network<net_t>;
    friend class CuDNN_Layer;
    friend class CuDNNScheduler<net_t>;
public:
    CuDNN(const int gpu, const bool silent = false);
//    virtual ~CuDNN();

    void initialize(const int channels, const int batch_size, const int net_type, const std::string &model_hash = "");
    bool has_fp16_compute();
    bool has_tensor_cores();

    int m_batch_size = 1;
    cudaDeviceProp m_device_prop;
#if defined(USE_TENSOR_RT)
    std::string m_model_hash;
#endif

private:
    void convolve(const void *bufferIn,
                  void *bufferOut,
                  const void *weights,
                  void *workspace,
                  const conv_descriptor& conv_desc,
                  const float alpha);

    void convolveActivation(const void *bufferIn,
                            void *bufferOut,
                            const void *weights,
                            void *residualBuffer,
                            const void *biases,
                            void *workspace,
                            const conv_descriptor& conv_desc,
                            const float alpha,
                            const float alpha2 = 1.0f);

    void convolveIdentityActivation(const void *bufferIn,
                                    void *bufferOut,
                                    const void *weights,
                                    void *residualBuffer,
                                    const void *biases,
                                    void *workspace,
                                    const conv_descriptor& conv_desc,
                                    const float alpha,
                                    const float alpha2 = 1.0f);

    void squeeze_excitation_float(const void *bufferIn1,
                                  const void *bufferIn2,
                                  void *bufferTemp,
                                  const void *fc1_weights,
                                  const void *fc1_biases,
                                  const void *fc2_weights,
                                  const void *fc2_biases,
                                  void *bufferOut,
                                  void *bufferPool,
                                  const int batch_size,
                                  const int channels,
                                  const int spatial);

    void squeeze_excitation_half(const void *bufferIn1,
                                 const void *bufferIn2,
                                 void *bufferTemp,
                                 const void *fc1_weights,
                                 const void *fc1_biases,
                                 const void *fc2_weights,
                                 const void *fc2_biases,
                                 void *bufferOut,
                                 void *bufferPool,
                                 const int batch_size,
                                 const int channels,
                                 const int spatial);

#if defined(USE_CUDNN_GRAPH)
    void convolve_fe_init(const int channels,
                          const int outputs,
                          const int filter_size,
                          conv_descriptor& conv_desc,
                          const int batch_size = 1);

    void convolve_fe_no_relu_init(const int channels,
                                  const int outputs,
                                  const int filter_size,
                                  conv_descriptor& conv_desc,
                                  const int batch_size = 1);

    void convolve_fe_add_relu_init(const int channels,
                                   const int outputs,
                                   conv_descriptor& conv_desc,
                                   const int batch_size = 1);
#endif

    void convolve_init(const int channels,
                       const int outputs,
                       const int filter_size,
                       conv_descriptor& conv_desc,
                       const int batch_size = 1);

#if defined(USE_CUDNN_GRAPH)
    void convolve_fe_head_init(const int channels,
                               const int outputs,
                               const int filter_size,
                               conv_descriptor& conv_desc,
                               const int batch_size = 1);
#endif

    cudnnHandle_t m_handle;
    cublasHandle_t m_cublas_handles;
    bool m_fp16_compute{false};
    bool m_tensorcore{false};
    bool m_init_ok{false};
    int m_net_type{0};
#if defined(USE_TENSOR_RT)
    std::unique_ptr<TrtResNet<net_t> > m_trt{nullptr};
#endif
#if defined(USE_CUDNN_GRAPH)
    using graph_and_tensors1 = std::tuple<fe::graph::Graph,
                                          std::shared_ptr<fe::graph::Tensor_attributes>,  // X
                                          std::shared_ptr<fe::graph::Tensor_attributes>,  // W
                                          std::shared_ptr<fe::graph::Tensor_attributes>,  // B
                                          std::shared_ptr<fe::graph::Tensor_attributes>   // Y
                                          >;
    using graph_and_tensors2 = std::tuple<fe::graph::Graph,
                                          std::shared_ptr<fe::graph::Tensor_attributes>,  // X
                                          std::shared_ptr<fe::graph::Tensor_attributes>,  // W
                                          std::shared_ptr<fe::graph::Tensor_attributes>,  // B
                                          std::shared_ptr<fe::graph::Tensor_attributes>   // Y
                                          >;
    using graph_and_tensors3 = std::tuple<fe::graph::Graph,
                                          std::shared_ptr<fe::graph::Tensor_attributes>,  // X
                                          std::shared_ptr<fe::graph::Tensor_attributes>,  // Z
                                          std::shared_ptr<fe::graph::Tensor_attributes>   // Y
                                          >;
    using graph_and_tensors4 = std::tuple<fe::graph::Graph,
                                          std::shared_ptr<fe::graph::Tensor_attributes>,  // X
                                          std::shared_ptr<fe::graph::Tensor_attributes>,  // W
                                          std::shared_ptr<fe::graph::Tensor_attributes>   // Y
                                          >;
    std::unordered_map<std::size_t, graph_and_tensors1> m_maintained_cache1;
    std::unordered_map<std::size_t, graph_and_tensors2> m_maintained_cache2;
    std::unordered_map<std::size_t, graph_and_tensors3> m_maintained_cache3;
    std::unordered_map<std::size_t, graph_and_tensors4> m_maintained_cache4;
#endif
};
#endif

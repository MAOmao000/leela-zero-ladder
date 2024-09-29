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
#include "config.h"

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH) || defined(USE_TENSOR_RT)
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <filesystem>

#include "GTP.h"
#include "Utils.h"
#include "CuDNN.h"

using namespace Utils;
#if defined(USE_CUDNN_GRAPH)
namespace fe = cudnn_frontend;
#endif

// Filter layout KRSC: output, rows, columns, inputs
//   K: number of output feature maps
//   R: number of rows per filter
//   S: number of columns per filter
//   C: number of input feature maps
//  CUDNN_TENSOR_NCHW = KCRS
//  CUDNN_TENSOR_NHWC = KRSC

#if defined(USE_TENSOR_RT)
// Define TRT entrypoints
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0
#include "NvInferRuntime.h"
using namespace nvinfer1;
#endif

template <typename T>
static std::vector<float> NHWC_to_NCHW(const std::vector<T>& x,
                                       unsigned int N,
                                       unsigned int H,
                                       unsigned int W,
                                       unsigned int C) {

    std::vector<float> x_out(N * H * W * C);

    for (auto n = size_t{0}; n < N; n++) {
        for (auto h = size_t{0}; h < H; h++) {
            for (auto w = size_t{0}; w < W; w++) {
                for (auto c = size_t{0}; c < C; c++) {
                    x_out[n * H * W * C + c * H * W + h * W + w] =
                    static_cast<float>(x[n * H * W * C + h * W * C + w * C + c]);
                }
            }
        }
    }
    return x_out;
}

template <typename net_t>
static std::vector<net_t> NCHW_to_NHWC(const std::vector<float> &x,
                                       unsigned int N,
                                       unsigned int H,
                                       unsigned int W,
                                       unsigned int C) {

    std::vector<net_t> x_out(N * H * W * C);

    for (auto n = size_t{0}; n < N; n++) {
        for (auto h = size_t{0}; h < H; h++) {
            for (auto w = size_t{0}; w < W; w++) {
                for (auto c = size_t{0}; c < C; c++) {
                    x_out[n * H * W * C + h * W * C + w * C + c] =
                    static_cast<net_t>(x[n * H * W * C + c * H * W + h * W + w]);
                }
            }
        }
    }
    return x_out;
}

template <typename net_t>
CuDNN_Network<net_t>::CuDNN_Network(
    const int gpu,
    const bool silent) {

    auto best_bandwidth = 0.0;
    auto found_device = false;
    auto nDevices = 0;
    auto best_device_id = 0;
    cudaDeviceProp best_device;

    cudaGetDeviceCount(&nDevices);

    if (!silent) {
        myprintf("Detected %d CUDA devices.\n", nDevices);
    }

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        auto bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        if (!silent) {
            myprintf("Device Number: %d\n", i);
            myprintf("  Device name: %s\n", prop.name);
            myprintf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            myprintf("  Peak Memory Bandwidth (GB/s): %.1f\n\n", bandwidth);
        }

        bool preferred = (gpu == i);

        if (bandwidth > best_bandwidth || preferred) {
            best_bandwidth = bandwidth;
            best_device = prop;
            best_device_id = i;
            if (preferred) {
                best_bandwidth = std::numeric_limits<decltype(best_bandwidth)>::max();
            } else {
                best_bandwidth = bandwidth;
            }
            found_device = true;
        }
    }

    if (!found_device) {
        myprintf("No suitable CUDA device found.\n");
        exit(EXIT_FAILURE);
    }

    myprintf("Selected device: %s\n", best_device.name);
    myprintf("with compute capability %d.%d.\n", best_device.major, best_device.minor);

    if (best_device.major >= 7) {
        m_tensorcore = true;
    } else if (best_device.major >= 6) {
        m_fp16_compute = true;
    }

    cudaSetDevice(best_device_id);
    m_device_prop = best_device;
}

template <typename net_t>
void CuDNN_Network<net_t>::initialize(
    const int channels,
    const int batch_size,
    const int net_type,
    const int num_worker_threads,
    const std::string &model_hash) {

    // For compatibility with OpenCL implementation
    (void)channels;
    (void)batch_size;
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    (void)model_hash;
#endif
    const char* log_level = "CUDNN_LOGLEVEL_DBG=0";
    putenv((char *)log_level);
    const char* log_dest = "CUDNN_LOGDEST_DBG=stderr";
    putenv((char *)log_dest);
    const char* module_load = "CUDA_MODULE_LOADING=LAZY";
    putenv((char *)module_load);
#if defined(USE_CUDNN_GRAPH)
    if (cfg_backend == backend_t::CUDNNGRAPH) {
        const char* log_info = "CUDNN_FRONTEND_LOG_INFO=0";
        putenv((char *)log_info);
    }
#endif
    m_net_type = net_type;
    m_num_worker_threads = num_worker_threads;

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    if (cfg_backend == backend_t::CUDNN || cfg_backend == backend_t::CUDNNGRAPH) {
        for (auto i = 0; i < m_num_worker_threads; i++) {
            cudnnHandle_t cudnn;
            checkCUDNN(cudnnCreate(&cudnn));
            cudaStream_t stream;
            checkCUDA(cudaStreamCreate(&stream));
            checkCUDNN(cudnnSetStream(cudnn, stream));
            m_handle.emplace_back(cudnn);
            if (net_type == int(NetworkType::MINIGO_SE)) {
                cublasHandle_t cublas;
                checkCUBLAS(cublasCreate(&cublas));
                checkCUBLAS(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE));
                if (m_tensorcore) {
                    checkCUBLAS(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));
                }
                checkCUBLAS(cublasSetStream(cublas, stream));
                m_streams.emplace_back(stream);
                m_cublas_handles.emplace_back(cublas);
            }
        }
#if defined(USE_TENSOR_RT)
    } else {
        m_model_hash = model_hash;
        for (auto i = 0; i < m_num_worker_threads; i++) {
            cudaStream_t stream;
            checkCUDA(cudaStreamCreate(&stream));
            m_streams.emplace_back(stream);
        }
#endif
    }
#else
#if defined(USE_TENSOR_RT)
    m_model_hash = model_hash;
    for (auto i = 0; i < m_num_worker_threads; i++) {
        cudaStream_t stream;
        checkCUDA(cudaStreamCreate(&stream));
        m_streams.emplace_back(stream);
    }
#endif
#endif
}

#if defined(USE_CUDNN)
template <typename net_t>
void CuDNN_Network<net_t>::convolve(
    const int tid,
    const void *bufferIn,
    void *bufferOut,
    const void *weights,
    void *workspace,
    const std::shared_ptr<conv_descriptor>& conv_desc,
    const float alpha) {

    const float beta = 0.0f;
    // dstValue = alpha[0] * result + beta[0] * priorDstValue
    checkCUDNN(cudnnConvolutionForward(
        /* handle               */m_handle[tid],
        /* *alpha               */&alpha,
        /* xDesc                */conv_desc->input_descriptor,
        /* *x                   */bufferIn,
        /* wDesc                */conv_desc->filter_descriptor,
        /* *w                   */weights,
        /* convDesc             */conv_desc->convolution_descriptor,
        /* algo                 */conv_desc->convolution_algorithm,
        /* *workSpace           */workspace,
        /* workSpaceSizeInBytes */conv_desc->workspace_size,
        /* *beta                */&beta,
        /* yDesc                */conv_desc->output_descriptor,
        /* *y                   */bufferOut));
}

template <typename net_t>
void CuDNN_Network<net_t>::convolveActivation(
    const int tid,
    const void *bufferIn,
    void *bufferOut,
    const void *weights,
    void *residualBuffer,
    const void *biases,
    void *workspace,
    const std::shared_ptr<conv_descriptor>& conv_desc,
    const float alpha1,
    const float alpha2) {

    void *residual = bufferOut;
    float _alpha2 = 0.0f;
    if (residualBuffer != nullptr) {
        _alpha2 = alpha2;
        residual = residualBuffer;
    }
    // y = act (alpha1 * conv(x) + alpha2 * z + bias)
    checkCUDNN(cudnnConvolutionBiasActivationForward(
        /* handle         */m_handle[tid],
        /* *alpha1        */&alpha1,
        /* xDesc          */conv_desc->input_descriptor,
        /* *x             */bufferIn,
        /* wDesc          */conv_desc->filter_descriptor,
        /* *w             */weights,
        /* convDesc       */conv_desc->convolution_descriptor,
        /* algo           */conv_desc->convolution_algorithm,
        /* *workSpace     */workspace,
        /* workSpaceSize  */conv_desc->workspace_size,
        /* *alpha2        */&_alpha2,
        /* zDesc          */conv_desc->output_descriptor,
        /* *z             */residual,
        /* biasDesc       */conv_desc->bias_descriptor,
        /* *bias          */biases,
        /* activationDesc */conv_desc->activation_descriptor,
        /* yDesc          */conv_desc->output_descriptor,
        /* *y             */bufferOut));
}

template <typename net_t>
void CuDNN_Network<net_t>::convolveIdentityActivation(
    const int tid,
    const void *bufferIn,
    void *bufferOut,
    const void *weights,
    void *residualBuffer,
    const void *biases,
    void *workspace,
    const std::shared_ptr<conv_descriptor>& conv_desc,
    const float alpha1,
    const float alpha2) {

    void *residual = bufferOut;

    float _alpha2 = 0.0f;
    if (residualBuffer != nullptr) {
        _alpha2 = alpha2;
        residual = residualBuffer;
    }

    // y = act (alpha1 * conv(x) + alpha2 * z + bias)
    checkCUDNN(cudnnConvolutionBiasActivationForward(
        /* handle         */m_handle[tid],
        /* *alpha1        */&alpha1,
        /* xDesc          */conv_desc->input_descriptor,
        /* *x             */bufferIn,
        /* wDesc          */conv_desc->filter_descriptor,
        /* *w             */weights,
        /* convDesc       */conv_desc->convolution_descriptor,
        /* algo           */conv_desc->convolution_algorithm,
        /* *workSpace     */workspace,
        /* workSpaceSize  */conv_desc->workspace_size,
        /* *alpha2        */&_alpha2,
        /* zDesc          */conv_desc->output_descriptor,
        /* *z             */residual,
        /* biasDesc       */conv_desc->bias_descriptor,
        /* *bias          */biases,
        /* activationDesc */conv_desc->activation_identity_descriptor,
        /* yDesc          */conv_desc->output_descriptor,
        /* *y             */bufferOut));
}

template <typename net_t>
std::shared_ptr<conv_descriptor> CuDNN_Network<net_t>::convolve_init(
    cudnnHandle_t handle,
    const int channels,
    const int outputs,
    const int filter_size,
    const int batch_size) {

    cudnnDataType_t data_type;
    cudnnDataType_t compute_type;
    cudnnTensorFormat_t tensor_format;
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = CUDNN_DATA_FLOAT;
        compute_type = CUDNN_DATA_FLOAT;
        tensor_format = cfg_NCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    } else {
        data_type = CUDNN_DATA_HALF;
        compute_type = CUDNN_DATA_HALF;
        tensor_format = cfg_NCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    }

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc->input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
                      /* tensorDesc     */conv_desc->input_descriptor,
                      /* format         */tensor_format,
                      /* dataType       */data_type,
                      /* N batch_size   */batch_size,
                      /* C channels     */channels,
                      /* H image_height */BOARD_SIZE,
                      /* W image_width  */BOARD_SIZE));

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc->output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
                      /* tensorDesc     */conv_desc->output_descriptor,
                      /* format         */tensor_format,
                      /* dataType       */data_type,
                      /* N batch_size   */batch_size,
                      /* C channels     */outputs,
                      /* H image_height */BOARD_SIZE,
                      /* W image_width  */BOARD_SIZE));

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc->bias_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
                      /* tensorDesc     */conv_desc->bias_descriptor,
                      /* format         */tensor_format,
                      /* dataType       */data_type,
                      /* N number of images=*/1, // Not the batch_size
                      /* C channels         */outputs,
                      /* H image_height     */1,
                      /* W image_width      */1));

    checkCUDNN(cudnnCreateFilterDescriptor(&conv_desc->filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
                      /* filterDesc     */conv_desc->filter_descriptor,
                      /* dataType       */data_type,
                      /* format         */tensor_format,
                      /* K Number of output feature maps */outputs,
                      /* C Number of input feature maps  */channels,
                      /* R Number of rows per filter     */filter_size,
                      /* S Number of columns per filter  */filter_size));

    checkCUDNN(cudnnCreateActivationDescriptor(&conv_desc->activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(
                      /* activationDesc   */conv_desc->activation_descriptor,
                      /* mode             */CUDNN_ACTIVATION_RELU,
                      /* reluNanOpt       */CUDNN_NOT_PROPAGATE_NAN,
                      /* coef             */0.));

    auto pad_size = filter_size / 2;

    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc->convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
                 /* convDesc                 */conv_desc->convolution_descriptor,
                 /* zero-padding height      */pad_size,
                 /* zero-padding width       */pad_size,
                 /* vertical filter stride   */1,
                 /* horizontal filter stride */1,
                 /* filter height dilation   */1,
                 /* filter width dilation    */1,
                 /* mode                     */CUDNN_CROSS_CORRELATION,
                 /* computeType              */compute_type));

    checkCUDNN(cudnnSetConvolutionGroupCount(conv_desc->convolution_descriptor, 8));
    checkCUDNN(cudnnSetConvolutionMathType(
                             /* convDesc */conv_desc->convolution_descriptor,
                             /* mathType */CUDNN_TENSOR_OP_MATH));

    using perf_t = cudnnConvolutionFwdAlgoPerf_t;
    int num_algos = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &num_algos));
    int returned_algo_count = 0;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
                              /* handle             */handle,
                              /* xDesc              */conv_desc->input_descriptor,
                              /* wDesc              */conv_desc->filter_descriptor,
                              /* convDesc           */conv_desc->convolution_descriptor,
                              /* yDesc              */conv_desc->output_descriptor,
                              /* requestedAlgoCount */num_algos,
                              /* *returnedAlgoCount */&returned_algo_count,
                              /* *perfResults       */perf_results.get()));

    conv_desc->convolution_algorithm = perf_results[0].algo;

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                                     /* handle       */handle,
                                     /* xDesc        */conv_desc->input_descriptor,
                                     /* wDesc        */conv_desc->filter_descriptor,
                                     /* convDesc     */conv_desc->convolution_descriptor,
                                     /* yDesc        */conv_desc->output_descriptor,
                                     /* algo         */conv_desc->convolution_algorithm,
                                     /* *sizeInBytes */&conv_desc->workspace_size));

    if (m_net_type == int(NetworkType::MINIGO_SE)) {
        checkCUDNN(cudnnCreateActivationDescriptor(&conv_desc->activation_identity_descriptor));
        checkCUDNN(cudnnSetActivationDescriptor(
                            /* activationDesc */conv_desc->activation_identity_descriptor,
                            /* mode           */CUDNN_ACTIVATION_IDENTITY,
                            /* reluNanOpt     */CUDNN_NOT_PROPAGATE_NAN,
                            /* coef           */0.));

    }
    return conv_desc;
}
#endif

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
template <typename net_t>
void CuDNN_Network<net_t>::squeeze_excitation_float(
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const CuDNNContext& cudnn_context,
    const void *bufferIn1,   // residual input(before convolve)
    const void *bufferIn2,   // residual output
    void *TempBuffer,
    const void *fc1_weights,
    const void *fc1_biases,
    const void *fc2_weights,
    const void *fc2_biases,
    void *bufferOut,
    void *bufferPool,
    const int batch_size,
    const int channels,
    const int spatial) {

    // in: batch * channels * spatial(board size * board size)
    // out: batch * channels
    if (cfg_NCHW) {
        global_average_pooling_float(
            (const float *)bufferIn2, // input: residual output
            (float *)bufferPool,      // output: GAP output
            batch_size,
            channels,
            spatial,
            stream);
        checkCUDA(cudaGetLastError());
    } else {
        global_average_pooling_float_NHWC(
            (const float *)bufferIn2, // input: residual output
            (float *)bufferPool,      // output: GAP output
            batch_size,
            channels,
            spatial,
            stream);
        checkCUDA(cudaGetLastError());
    }

    // A[channels / 2, channels], B[channels, 1], C[channels / 2, 1]
    if (m_tensorcore) {
        checkCUBLAS(cublasGemmStridedBatchedEx(
            cublas_handle,           // handle: handle to the cuBLAS library context
            CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
            CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
            channels / 2,            // m: number of rows of matrix op(A[i]) and C[i]
            1,                       // n: number of columns of op(B[i]) and C[i]
            channels,                // k: number of columns of op(A[i]) and rows of op(B[i])
            (const float *)cudnn_context.m_alpha_32, // alpha: <type> scalar used for multiplication
            (float *)fc1_weights,    // A: <type>* pointer to the A matrix corresponding to the first instance of the batch,
                                     //    with dimensions lda x k with lda>=max(1,m)
                                     //    if transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise
            CUDA_R_32F,              // Enumerant specifying the datatype of matrix A
            channels / 2,            // lda: leading dimension of two-dimensional array used to store each matrix A[i]
            (long long int)0LL,      // strideA: Value of type long long int
                                     //          that gives the offset in number of elements between A[i] and A[i+1]
            (float *)bufferPool,     // B: <type>* pointer to the B matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldb x n with ldb>=max(1,k)
                                     //    if transb==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise
            CUDA_R_32F,              // Enumerant specifying the datatype of matrix B
            channels,                // ldb: leading dimension of two-dimensional array used to store each matrix B[i]
            (long long int)channels, // strideB: Value of type long long int
                                     //          that gives the offset in number of elements between B[i] and B[i+1]
            (const float *)cudnn_context.m_beta_32, // beta: <type> scalar used for multiplication
                                                    //       If beta == 0, C does not have to be a valid input
            (float *)bufferOut,      // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldc x n with ldc>=max(1,m)
                                     //    Matrices C[i] should not overlap; otherwise,
                                     //    undefined behavior is expected
            CUDA_R_32F,              // Enumerant specifying the datatype of matrix C
            channels / 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
            (long long int)(channels / 2), // strideC: Value of type long long int
                                           //          that gives the offset in number of elements between C[i] and C[i+1]
            batch_size,              // batchCount: number of GEMMs to perform in the batch
            CUBLAS_COMPUTE_32F_FAST_16F, // Enumerant specifying the computation type
            CUBLAS_GEMM_DEFAULT));       // Enumerant specifying the algorithm
    } else {
        checkCUBLAS(cublasSgemmStridedBatched(
            cublas_handle,           // handle: handle to the cuBLAS library context
            CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
            CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
            channels / 2,            // m: number of rows of matrix op(A[i]) and C[i]
            1,                       // n: number of columns of op(B[i]) and C[i]
            channels,                // k: number of columns of op(A[i]) and rows of op(B[i])
            (const float *)cudnn_context.m_alpha_32, // alpha: <type> scalar used for multiplication
            (float *)fc1_weights,    // A: <type>* pointer to the A matrix corresponding to the first instance of the batch,
                                     //    with dimensions lda x k with lda>=max(1,m)
                                     //    if transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise
            channels / 2,            // lda: leading dimension of two-dimensional array used to store each matrix A[i]
            0,                       // strideA: Value of type long long int
                                     //          that gives the offset in number of elements between A[i] and A[i+1]
            (float *)bufferPool,     // B: <type>* pointer to the B matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldb x n with ldb>=max(1,k)
                                     //    if transb==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise
            channels,                // ldb: leading dimension of two-dimensional array used to store each matrix B[i]
            channels,                // strideB: Value of type long long int
                                     //          that gives the offset in number of elements between B[i] and B[i+1]
            (const float *)cudnn_context.m_beta_32, // beta: <type> scalar used for multiplication
                                                    //       If beta == 0, C does not have to be a valid input
            (float *)bufferOut,      // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldc x n with ldc>=max(1,m)
                                     //    Matrices C[i] should not overlap; otherwise,
                                     //    undefined behavior is expected
            channels / 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
            channels / 2,            // strideC: Value of type long long int
                                     //          that gives the offset in number of elements between C[i] and C[i+1]
            batch_size));            // batchCount: number of GEMMs to perform in the batch
    }

    add_bias_float((float *)bufferOut,  // in & out: C[1, channels / 2]
                   (float *)fc1_biases, // input: bias[channels / 2]
                   batch_size,
                   channels / 2,
                   true,
                   stream);

    checkCUDA(cudaGetLastError());

    // A[channels * 2, channels / 2], B[channels / 2, 1], C[channels * 2, 1]
    if (m_tensorcore) {
        checkCUBLAS(cublasGemmStridedBatchedEx(
            cublas_handle,           // handle: handle to the cuBLAS library context
            CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
            CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
            channels * 2,            // m: number of rows of matrix op(A[i]) and C[i]
            1,                       // n: number of columns of op(B[i]) and C[i]
            channels / 2,            // k: number of columns of op(A[i]) and rows of op(B[i])
            (const float *)cudnn_context.m_alpha_32, // alpha: <type> scalar used for multiplication
            (float *)fc2_weights,    // A: <type>* pointer to the A matrix corresponding to the first instance of the batch,
                                     //    with dimensions lda x k with lda>=max(1,m)
                                     //    if transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise
            CUDA_R_32F,              // Enumerant specifying the datatype of matrix A
            channels * 2,            // lda: leading dimension of two-dimensional array used to store each matrix A[i]
            (long long int)0LL,      // strideA: Value of type long long int
                                     //          that gives the offset in number of elements between A[i] and A[i+1]
            (float *)bufferOut,      // B: <type>* pointer to the B matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldb x n with ldb>=max(1,k)
                                     //    if transb==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise
            CUDA_R_32F,              // Enumerant specifying the datatype of matrix B
            channels / 2,            // ldb: leading dimension of two-dimensional array used to store each matrix B[i]
            (long long int)(channels / 2), // strideB: Value of type long long int
                                           //          that gives the offset in number of elements between B[i] and B[i+1]
            (const float *)cudnn_context.m_beta_32, // beta: <type> scalar used for multiplication
                                                    //       If beta == 0, C does not have to be a valid input
            (float *)TempBuffer,     // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldc x n with ldc>=max(1,m)
                                     //    Matrices C[i] should not overlap; otherwise,
                                     //    undefined behavior is expected
            CUDA_R_32F,              // Enumerant specifying the datatype of matrix C
            channels * 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
            (long long int)channels * 2, // strideC: Value of type long long int
                                         //          that gives the offset in number of elements between C[i] and C[i+1]
            batch_size,              // batchCount: number of GEMMs to perform in the batch
            CUBLAS_COMPUTE_32F_FAST_16F, // Enumerant specifying the computation type
            CUBLAS_GEMM_DEFAULT));       // Enumerant specifying the algorithm
    } else {
        checkCUBLAS(cublasSgemmStridedBatched(
            cublas_handle,           // handle: handle to the cuBLAS library context
            CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
            CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
            channels * 2,            // m: number of rows of matrix op(A[i]) and C[i]
            1,                       // n: number of columns of op(B[i]) and C[i]
            channels / 2,            // k: number of columns of op(A[i]) and rows of op(B[i])
            (const float *)cudnn_context.m_alpha_32, // alpha: <type> scalar used for multiplication
            (float *)fc2_weights,    // A: <type>* pointer to the A matrix corresponding to the first instance of the batch,
                                     //    with dimensions lda x k with lda>=max(1,m)
                                     //    if transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise
            channels * 2,            // lda: leading dimension of two-dimensional array used to store each matrix A[i]
            0,                       // strideA: Value of type long long int
                                     //          that gives the offset in number of elements between A[i] and A[i+1]
            (float *)bufferOut,      // B: <type>* pointer to the B matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldb x n with ldb>=max(1,k)
                                     //    if transb==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise
            channels / 2,            // ldb: leading dimension of two-dimensional array used to store each matrix B[i]
            channels / 2,            // strideB: Value of type long long int
                                     //          that gives the offset in number of elements between B[i] and B[i+1]
            (const float *)cudnn_context.m_beta_32, // beta: <type> scalar used for multiplication
                                                    //       If beta == 0, C does not have to be a valid input
            (float *)TempBuffer,     // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldc x n with ldc>=max(1,m)
                                     //    Matrices C[i] should not overlap; otherwise,
                                     //    undefined behavior is expected
            channels * 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
            channels * 2,            // strideC: Value of type long long int
                                     //          that gives the offset in number of elements between C[i] and C[i+1]
            batch_size));            // batchCount: number of GEMMs to perform in the batch
    }

    add_bias_float((float *)TempBuffer, // in & out: C[1, channels * 2]
                   (float *)fc2_biases, // input: bias[channels * 2]
                   batch_size,
                   channels * 2,
                   false,
                   stream);

    checkCUDA(cudaGetLastError());

    if (cfg_NCHW) {
        se_scale_float(
            (float *)bufferOut,  // output: squeeze_excitation output
            (float *)bufferIn2,  // input: residual output
            (float *)TempBuffer, // input: fc2_weights * B + fc2_biases
            (float *)bufferIn1,  // input: residual input(before convolve)
            batch_size,
            channels,
            spatial,
            stream);
    } else {
        se_scale_float_NHWC(
            (float *)bufferOut,  // output: squeeze_excitation output
            (float *)bufferIn2,  // input: residual output
            (float *)TempBuffer, // input: fc2_weights * B + fc2_biases
            (float *)bufferIn1,  // input: residual input(before convolve)
            batch_size,
            channels,
            spatial,
            stream);
    }
    checkCUDA(cudaGetLastError());
}

template <typename net_t>
void CuDNN_Network<net_t>::squeeze_excitation_half(
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const CuDNNContext& cudnn_context,
    const void *bufferIn1,   // residual input(before convolve)
    const void *bufferIn2,   // residual output
    void *TempBuffer,
    const void *fc1_weights,
    const void *fc1_biases,
    const void *fc2_weights,
    const void *fc2_biases,
    void *bufferOut,
    void *bufferPool,
    const int batch_size,
    const int channels,
    const int spatial) {

    // in: batch * channels * spatial(board size * board size)
    // out: batch * channels
    if (cfg_NCHW) {
        global_average_pooling_half(
            (const __half *)bufferIn2, // input: residual output
            (__half *)bufferPool,      // output: GAP output
            batch_size,
            channels,
            spatial,
            stream);
        checkCUDA(cudaGetLastError());
    } else {
        global_average_pooling_half_NHWC(
            (const __half *)bufferIn2, // input: residual output
            (__half *)bufferPool,      // output: GAP output
            batch_size,
            channels,
            spatial,
            stream);
        checkCUDA(cudaGetLastError());
    }

    // A[channels / 2, channels], B[channels, 1], C[channels / 2, 1]
    if (m_tensorcore) {
        checkCUBLAS(cublasGemmStridedBatchedEx(
            cublas_handle,           // handle: handle to the cuBLAS library context
            CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
            CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
            channels / 2,            // m: number of rows of matrix op(A[i]) and C[i]
            1,                       // n: number of columns of op(B[i]) and C[i]
            channels,                // k: number of columns of op(A[i]) and rows of op(B[i])
            (const float *)cudnn_context.m_alpha_32, // alpha: <type> scalar used for multiplication
            (__half *)fc1_weights,   // A: <type>* pointer to the A matrix corresponding to the first instance of the batch,
                                     //    with dimensions lda x k with lda>=max(1,m)
                                     //    if transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise
            CUDA_R_16F,              // Enumerant specifying the datatype of matrix A
            channels / 2,            // lda: leading dimension of two-dimensional array used to store each matrix A[i]
            (long long int)0LL,      // strideA: Value of type long long int
                                     //          that gives the offset in number of elements between A[i] and A[i+1]
            (__half *)bufferPool,    // B: <type>* pointer to the B matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldb x n with ldb>=max(1,k)
                                     //    if transb==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise
            CUDA_R_16F,              // Enumerant specifying the datatype of matrix B
            channels,                // ldb: leading dimension of two-dimensional array used to store each matrix B[i]
            (long long int)channels, // strideB: Value of type long long int
                                     //          that gives the offset in number of elements between B[i] and B[i+1]
            (const float *)cudnn_context.m_beta_32, // beta: <type> scalar used for multiplication
                                                    //       If beta == 0, C does not have to be a valid input
            (__half *)bufferOut,     // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldc x n with ldc>=max(1,m)
                                     //    Matrices C[i] should not overlap; otherwise,
                                     //    undefined behavior is expected
            CUDA_R_16F,              // Enumerant specifying the datatype of matrix C
            channels / 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
            (long long int)(channels / 2), // strideC: Value of type long long int
                                           //          that gives the offset in number of elements between C[i] and C[i+1]
            batch_size,              // batchCount: number of GEMMs to perform in the batch
            CUBLAS_COMPUTE_32F_FAST_16F, // Enumerant specifying the computation type
            CUBLAS_GEMM_DEFAULT));       // Enumerant specifying the algorithm
    } else {
        checkCUBLAS(cublasHgemmStridedBatched(
            cublas_handle,           // handle: handle to the cuBLAS library context
            CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
            CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
            channels / 2,            // m: number of rows of matrix op(A[i]) and C[i]
            1,                       // n: number of columns of op(B[i]) and C[i]
            channels,                // k: number of columns of op(A[i]) and rows of op(B[i])
            (const __half *)cudnn_context.m_alpha_16, // alpha: <type> scalar used for multiplication
            (__half *)fc1_weights,   // A: <type>* pointer to the A matrix corresponding to the first instance of the batch,
                                     //    with dimensions lda x k with lda>=max(1,m)
                                     //    if transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise
            channels / 2,            // lda: leading dimension of two-dimensional array used to store each matrix A[i]
            0,                       // strideA: Value of type long long int
                                     //          that gives the offset in number of elements between A[i] and A[i+1]
            (__half *)bufferPool,    // B: <type>* pointer to the B matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldb x n with ldb>=max(1,k)
                                     //    if transb==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise
            channels,                // ldb: leading dimension of two-dimensional array used to store each matrix B[i]
            channels,                // strideB: Value of type long long int
                                     //          that gives the offset in number of elements between B[i] and B[i+1]
            (const __half *)cudnn_context.m_beta_16, // beta: <type> scalar used for multiplication
                                                     //       If beta == 0, C does not have to be a valid input
            (__half *)bufferOut,     // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldc x n with ldc>=max(1,m)
                                     //    Matrices C[i] should not overlap; otherwise,
                                     //    undefined behavior is expected
            channels / 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
            channels / 2,            // strideC: Value of type long long int
                                     //          that gives the offset in number of elements between C[i] and C[i+1]
            batch_size));            // batchCount: number of GEMMs to perform in the batch
    }

    add_bias_half((__half *)bufferOut,  // in & out: C[1, channels / 2]
                  (__half *)fc1_biases, // input: bias[channels / 2]
                  batch_size,
                  channels / 2,
                  true,
                  stream);

    checkCUDA(cudaGetLastError());

    // A[channels * 2, channels / 2], B[channels / 2, 1], C[channels * 2, 1]
    if (m_tensorcore) {
        checkCUBLAS(cublasGemmStridedBatchedEx(
            cublas_handle,           // handle: handle to the cuBLAS library context
            CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
            CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
            channels * 2,            // m: number of rows of matrix op(A[i]) and C[i]
            1,                       // n: number of columns of op(B[i]) and C[i]
            channels / 2,            // k: number of columns of op(A[i]) and rows of op(B[i])
            (const float *)cudnn_context.m_alpha_32, // alpha: <type> scalar used for multiplication
            (__half *)fc2_weights,   // A: <type>* pointer to the A matrix corresponding to the first instance of the batch,
                                     //    with dimensions lda x k with lda>=max(1,m)
                                     //    if transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise
            CUDA_R_16F,              // Enumerant specifying the datatype of matrix A
            channels * 2,            // lda: leading dimension of two-dimensional array used to store each matrix A[i]
            (long long int)0LL,      // strideA: Value of type long long int
                                     //          that gives the offset in number of elements between A[i] and A[i+1]
            (__half *)bufferOut,     // B: <type>* pointer to the B matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldb x n with ldb>=max(1,k)
                                     //    if transb==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise
            CUDA_R_16F,              // Enumerant specifying the datatype of matrix B
            channels / 2,            // ldb: leading dimension of two-dimensional array used to store each matrix B[i]
            (long long int)(channels / 2), // strideB: Value of type long long int
                                           //          that gives the offset in number of elements between B[i] and B[i+1]
            (const float *)cudnn_context.m_beta_32, // beta: <type> scalar used for multiplication
                                                    //       If beta == 0, C does not have to be a valid input
            (__half *)TempBuffer,    // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldc x n with ldc>=max(1,m)
                                     //    Matrices C[i] should not overlap; otherwise,
                                     //    undefined behavior is expected
            CUDA_R_16F,              // Enumerant specifying the datatype of matrix C
            channels * 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
            (long long int)channels * 2, // strideC: Value of type long long int
                                         //          that gives the offset in number of elements between C[i] and C[i+1]
            batch_size,              // batchCount: number of GEMMs to perform in the batch
            CUBLAS_COMPUTE_32F_FAST_16F, // Enumerant specifying the computation type
            CUBLAS_GEMM_DEFAULT));       // Enumerant specifying the algorithm
    } else {
        checkCUBLAS(cublasHgemmStridedBatched(
            cublas_handle,           // handle: handle to the cuBLAS library context
            CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
            CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
            channels * 2,            // m: number of rows of matrix op(A[i]) and C[i]
            1,                       // n: number of columns of op(B[i]) and C[i]
            channels / 2,            // k: number of columns of op(A[i]) and rows of op(B[i])
            (const __half *)cudnn_context.m_alpha_16, // alpha: <type> scalar used for multiplication
            (__half *)fc2_weights,   // A: <type>* pointer to the A matrix corresponding to the first instance of the batch,
                                     //    with dimensions lda x k with lda>=max(1,m)
                                     //    if transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise
            channels * 2,            // lda: leading dimension of two-dimensional array used to store each matrix A[i]
            0,                       // strideA: Value of type long long int
                                     //          that gives the offset in number of elements between A[i] and A[i+1]
            (__half *)bufferOut,     // B: <type>* pointer to the B matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldb x n with ldb>=max(1,k)
                                     //    if transb==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise
            channels / 2,            // ldb: leading dimension of two-dimensional array used to store each matrix B[i]
            channels / 2,            // strideB: Value of type long long int
                                     //          that gives the offset in number of elements between B[i] and B[i+1]
            (const __half *)cudnn_context.m_beta_16, // beta: <type> scalar used for multiplication
                                                     //       If beta == 0, C does not have to be a valid input
            (__half *)TempBuffer,    // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                     //    with dimensions ldc x n with ldc>=max(1,m)
                                     //    Matrices C[i] should not overlap; otherwise,
                                     //    undefined behavior is expected
            channels * 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
            channels * 2,            // strideC: Value of type long long int
                                     //          that gives the offset in number of elements between C[i] and C[i+1]
            batch_size));            // batchCount: number of GEMMs to perform in the batch
    }

    add_bias_half((__half *)TempBuffer, // in & out: C[1, channels * 2]
                  (__half *)fc2_biases, // input: bias[channels * 2]
                  batch_size,
                  channels * 2,
                  false,
                  stream);

    checkCUDA(cudaGetLastError());

    if (cfg_NCHW) {
        se_scale_half(
            (__half *)bufferOut,  // output: squeeze_excitation output
            (__half *)bufferIn2,  // input: residual output
            (__half *)TempBuffer, // input: fc2_weights * B + fc2_biases
            (__half *)bufferIn1,  // input: residual input(before convolve)
            batch_size,
            channels,
            spatial,
            stream);
    } else {
        se_scale_half_NHWC(
            (__half *)bufferOut,  // output: squeeze_excitation output
            (__half *)bufferIn2,  // input: residual output
            (__half *)TempBuffer, // input: fc2_weights * B + fc2_biases
            (__half *)bufferIn1,  // input: residual input(before convolve)
            batch_size,
            channels,
            spatial,
            stream);
    }
    checkCUDA(cudaGetLastError());
}
#endif

#if defined(USE_CUDNN_GRAPH)
// Y = ReLU(Convolve(X, W) + B)
template <typename net_t>
std::shared_ptr<conv_descriptor> CuDNN_Network<net_t>::convolve_fe_init(
    cudnnHandle_t handle,
    const int channels,
    const int outputs,
    const int filter_size,
    const int batch_size) {

    int64_t n = batch_size;
    int64_t c = channels;
    int64_t h = BOARD_SIZE;
    int64_t w = BOARD_SIZE;
    int64_t k = outputs;
    int64_t r = filter_size;
    int64_t s = filter_size;
    fe::DataType_t data_type;
    fe::DataType_t compute_type;
    fe::DataType_t conv_compute_type;
    fe::DataType_t intermediate_type;
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = fe::DataType_t::FLOAT;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::FLOAT;
        intermediate_type = fe::DataType_t::FLOAT;
    } else {
        data_type = fe::DataType_t::HALF;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::HALF;
        intermediate_type = fe::DataType_t::HALF;
    }
    auto pad_size = filter_size / 2;
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = fe::graph::Graph();
        graph.set_io_data_type(data_type)
              .set_intermediate_data_type(intermediate_type)
              .set_compute_data_type(compute_type);
        std::shared_ptr<fe::graph::Tensor_attributes> X;
        std::shared_ptr<fe::graph::Tensor_attributes> W;
        std::shared_ptr<fe::graph::Tensor_attributes> B;
        std::shared_ptr<fe::graph::Tensor_attributes> Y;

        X = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({ n, c, h, w })
            .set_stride({ c * h * w, 1, c * w, c }));
        W = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("filter")
            .set_dim({ k, c, r, s })
            .set_stride({ c * r * s, 1, c * s, c }));
        auto conv_options = fe::graph::Conv_fprop_attributes()
                            .set_compute_data_type(conv_compute_type)
                            .set_padding({ pad_size, pad_size })
                            .set_stride({ 1, 1 })
                            .set_dilation({ 1, 1 });
        auto conv_output = graph.conv_fprop(X, W, conv_options);
        B = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("bias")
            .set_dim({ 1, k, 1, 1 })
            .set_stride({ k, 1, 1, 1 }));
        auto bias_options = fe::graph::Pointwise_attributes()
                            .set_mode(fe::PointwiseMode_t::ADD);
        auto bias_output = graph.pointwise(conv_output, B, bias_options);
        auto relu_options = fe::graph::Pointwise_attributes()
                            .set_mode(fe::PointwiseMode_t::RELU_FWD);
        Y = graph.pointwise(bias_output, relu_options);
        Y->set_output(true);

        checkCUDNNFE(graph.validate());
        auto key = graph.key();
        auto it = m_maintained_cache1.find(key);
        if (it != m_maintained_cache1.end()) {
            return it->second;
        }
        checkCUDNNFE(graph.build_operation_graph(handle));
        checkCUDNNFE(graph.create_execution_plans({ fe::HeurMode_t::A }));
        checkCUDNNFE(graph.check_support(handle));
        checkCUDNNFE(graph.build_plans(handle));
        m_maintained_cache1.insert({key, std::make_tuple(graph, X, W, B, Y)});
        return std::make_tuple(graph, X, W, B, Y);
    };

    auto [graph, X, W, B, Y] = build_new_graph(handle);
    conv_desc->graph = graph;
    conv_desc->X = X;
    conv_desc->W = W;
    conv_desc->B = B;
    conv_desc->Y = Y;
    conv_desc->workspace_size = graph.get_workspace_size();
    return conv_desc;
}

// Y = Convolve(X, W) + B
template <typename net_t>
std::shared_ptr<conv_descriptor> CuDNN_Network<net_t>::convolve_fe_no_relu_init(
    cudnnHandle_t handle,
    const int channels,
    const int outputs,
    const int filter_size,
    const int batch_size) {

    int64_t n = batch_size;
    int64_t c = channels;
    int64_t h = BOARD_SIZE;
    int64_t w = BOARD_SIZE;
    int64_t k = outputs;
    int64_t r = filter_size;
    int64_t s = filter_size;
    fe::DataType_t data_type;
    fe::DataType_t compute_type;
    fe::DataType_t conv_compute_type;
    fe::DataType_t intermediate_type;
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = fe::DataType_t::FLOAT;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::FLOAT;
        intermediate_type = fe::DataType_t::FLOAT;
    } else {
        data_type = fe::DataType_t::HALF;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::HALF;
        intermediate_type = fe::DataType_t::HALF;
    }
    auto pad_size = filter_size / 2;
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = fe::graph::Graph();
        graph.set_io_data_type(data_type)
              .set_intermediate_data_type(intermediate_type)
              .set_compute_data_type(compute_type);
        std::shared_ptr<fe::graph::Tensor_attributes> X;
        std::shared_ptr<fe::graph::Tensor_attributes> W;
        std::shared_ptr<fe::graph::Tensor_attributes> B;
        std::shared_ptr<fe::graph::Tensor_attributes> Y;

        X = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({ n, c, h, w })
            .set_stride({ c * h * w, 1, c * w, c }));
        W = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("filter")
            .set_dim({ k, c, r, s })
            .set_stride({ c * r * s, 1, c * s, c }));
        auto conv_options = fe::graph::Conv_fprop_attributes()
                            .set_compute_data_type(conv_compute_type)
                            .set_padding({ pad_size, pad_size })
                            .set_stride({ 1, 1 })
                            .set_dilation({ 1, 1 });
        auto conv_output = graph.conv_fprop(X, W, conv_options);
        B = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("bias")
            .set_dim({ 1, k, 1, 1 })
            .set_stride({ k, 1, 1, 1 }));
        auto bias_options = fe::graph::Pointwise_attributes()
                            .set_mode(fe::PointwiseMode_t::ADD);
        Y = graph.pointwise(conv_output, B, bias_options);
        Y->set_output(true);

        checkCUDNNFE(graph.validate());
        auto key = graph.key();
        auto it = m_maintained_cache2.find(key);
        if (it != m_maintained_cache2.end()) {
            return it->second;
        }
        checkCUDNNFE(graph.build_operation_graph(handle));
        checkCUDNNFE(graph.create_execution_plans({ fe::HeurMode_t::A }));
        checkCUDNNFE(graph.check_support(handle));
        checkCUDNNFE(graph.build_plans(handle));
        m_maintained_cache2.insert({key, std::make_tuple(graph, X, W, B, Y)});
        return std::make_tuple(graph, X, W, B, Y);
    };

    auto [graph, X, W, B, Y] = build_new_graph(handle);
    conv_desc->graph = graph;
    conv_desc->X = X;
    conv_desc->W = W;
    conv_desc->B = B;
    conv_desc->Y = Y;
    conv_desc->workspace_size = graph.get_workspace_size();
    return conv_desc;
}

// Y = ReLU(X + Z)
template <typename net_t>
std::shared_ptr<conv_descriptor> CuDNN_Network<net_t>::convolve_fe_add_relu_init(
    cudnnHandle_t handle,
    const int channels,
    const int outputs,
    const int batch_size) {

    int64_t n = batch_size;
    int64_t c = channels;
    int64_t h = BOARD_SIZE;
    int64_t w = BOARD_SIZE;
    int64_t k = outputs;
    fe::DataType_t data_type;
    fe::DataType_t compute_type;
    fe::DataType_t intermediate_type;
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = fe::DataType_t::FLOAT;
        compute_type = fe::DataType_t::FLOAT;
        intermediate_type = fe::DataType_t::FLOAT;
    } else {
        data_type = fe::DataType_t::HALF;
        compute_type = fe::DataType_t::FLOAT;
        intermediate_type = fe::DataType_t::HALF;
    }
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = fe::graph::Graph();
        graph.set_io_data_type(data_type)
              .set_intermediate_data_type(intermediate_type)
              .set_compute_data_type(compute_type);
        std::shared_ptr<fe::graph::Tensor_attributes> X;
        std::shared_ptr<fe::graph::Tensor_attributes> Z;
        std::shared_ptr<fe::graph::Tensor_attributes> Y;

        X = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({ n, c, h, w })
            .set_stride({ c * h * w, 1, c * w, c }));
        Z = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("feature")
            .set_dim({ n, c, h, w })
            .set_stride({ c * h * w, 1, c * w, c }));  // Should be p,q
        auto add_options = fe::graph::Pointwise_attributes()
                           .set_mode(fe::PointwiseMode_t::ADD);
        auto add_output = graph.pointwise(X, Z, add_options);
        auto relu_options = fe::graph::Pointwise_attributes()
                            .set_mode(fe::PointwiseMode_t::RELU_FWD);
        Y = graph.pointwise(add_output, relu_options);
        Y->set_output(true).set_stride({ k * h * w, 1, k * w, k });

        checkCUDNNFE(graph.validate());
        auto key = graph.key();
        auto it = m_maintained_cache3.find(key);
        if (it != m_maintained_cache3.end()) {
            return it->second;
        }
        checkCUDNNFE(graph.build_operation_graph(handle));
        checkCUDNNFE(graph.create_execution_plans({ fe::HeurMode_t::A }));
        checkCUDNNFE(graph.check_support(handle));
        checkCUDNNFE(graph.build_plans(handle));
        m_maintained_cache3.insert({key, std::make_tuple(graph, X, Z, Y)});
        return std::make_tuple(graph, X, Z, Y);
    };

    auto [graph, X, Z, Y] = build_new_graph(handle);
    conv_desc->graph = graph;
    conv_desc->X = X;
    conv_desc->Z = Z;
    conv_desc->Y = Y;
    conv_desc->workspace_size = graph.get_workspace_size();
    return conv_desc;
}

// Y = Convolve(X, W)
template <typename net_t>
std::shared_ptr<conv_descriptor> CuDNN_Network<net_t>::convolve_fe_head_init(
    cudnnHandle_t handle,
    const int channels,
    const int outputs,
    const int filter_size,
    const int batch_size) {

    int64_t n = batch_size;
    int64_t c = channels;
    int64_t h = BOARD_SIZE;
    int64_t w = BOARD_SIZE;
    int64_t k = outputs;
    int64_t r = filter_size;
    int64_t s = filter_size;
    fe::DataType_t data_type;
    fe::DataType_t compute_type;
    fe::DataType_t conv_compute_type;
    fe::DataType_t intermediate_type;
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = fe::DataType_t::FLOAT;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::FLOAT;
        intermediate_type = fe::DataType_t::FLOAT;
    } else {
        data_type = fe::DataType_t::HALF;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::HALF;
        intermediate_type = fe::DataType_t::HALF;
    }
    auto pad_size = filter_size / 2;
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = fe::graph::Graph();
        graph.set_io_data_type(data_type)
              .set_intermediate_data_type(intermediate_type)
              .set_compute_data_type(compute_type);
        std::shared_ptr<fe::graph::Tensor_attributes> X;
        std::shared_ptr<fe::graph::Tensor_attributes> W;
        std::shared_ptr<fe::graph::Tensor_attributes> Y;

        X = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({ n, c, h, w })
            .set_stride({ c * h * w, 1, c * w, c }));
        W = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("filter")
            .set_dim({ k, c, r, s })
            .set_stride({ c * r * s, 1, c * s, c }));
        auto conv_options = fe::graph::Conv_fprop_attributes()
                            .set_compute_data_type(conv_compute_type)
                            .set_padding({ pad_size, pad_size })
                            .set_stride({ 1, 1 })
                            .set_dilation({ 1, 1 });
        Y = graph.conv_fprop(X, W, conv_options);
        Y->set_output(true);

        checkCUDNNFE(graph.validate());
        auto key = graph.key();
        auto it = m_maintained_cache4.find(key);
        if (it != m_maintained_cache4.end()) {
            return it->second;
        }
        checkCUDNNFE(graph.build_operation_graph(handle));
        checkCUDNNFE(graph.create_execution_plans({ fe::HeurMode_t::A }));
        checkCUDNNFE(graph.check_support(handle));
        checkCUDNNFE(graph.build_plans(handle));
        m_maintained_cache4.insert({key, std::make_tuple(graph, X, W, Y)});
        return std::make_tuple(graph, X, W, Y);
    };

    auto [graph, X, W, Y] = build_new_graph(handle);
    //std::cout << graph << std::endl;
    conv_desc->graph = graph;
    conv_desc->X = X;
    conv_desc->W = W;
    conv_desc->Y = Y;
    conv_desc->workspace_size = graph.get_workspace_size();
    return conv_desc;
}
#endif

template <typename net_t>
void CuDNN_Network<net_t>::push_weights(
    const size_t layer,
    const std::vector<float>& weights) {

    if (layer >= m_layers.size()) {
        m_layers.emplace_back(CuDNN_Layer());
    }
    auto weights_net_t = std::vector<net_t>(weights.size());
    std::copy(weights.begin(), weights.end(), weights_net_t.begin());
    void *device_mem;
    checkCUDA(cudaMalloc((void**)&device_mem, weights.size() * sizeof(net_t)));
    checkCUDA(cudaMemcpy(device_mem,
                         (net_t *)&weights_net_t[0],
                         weights.size() * sizeof(net_t),
                         cudaMemcpyHostToDevice));
    m_layers.back().weights.emplace_back(device_mem);
}

template <typename net_t>
void CuDNN_Network<net_t>::push_weights_col_major(
    const size_t layer,
    const std::vector<float>& weights,
    const int row,
    const int column) {

    if (layer >= m_layers.size()) {
        m_layers.emplace_back(CuDNN_Layer());
    }

    // Transpose from model's CK to cublas's KC
    auto weightSize = weights.size() * sizeof(net_t);
    auto transposed_weights = std::vector<net_t>(weights.size());
    for (int i = 0; i < column; i++) {
        for (int j = 0; j < row; j++) {
            transposed_weights[i * row + j] = (net_t)weights[i + j * column];
        }
    }
    void *device_mem;
    checkCUDA(cudaMalloc((void**)&device_mem, weightSize));
    checkCUDA(cudaMemcpy(device_mem,
                         (net_t *)&transposed_weights[0],
                         weightSize,
                         cudaMemcpyHostToDevice));
    m_layers.back().weights.emplace_back(device_mem);
}

#if defined(USE_TENSOR_RT)
template <typename net_t>
void CuDNN_Network<net_t>::push_weights_trt(
    const size_t layer,
    const std::vector<float>& weights) {

    if (layer >= m_layers.size()) {
        m_layers.emplace_back(CuDNN_Layer());
    }
    // When TensorRT chooses a precision for a layer,
    // it automatically converts weights as necessary to run the layer
    void *host_mem;
    checkCUDA(cudaHostAlloc((void **)&host_mem,
                            weights.size() * sizeof(float),
                            cudaHostAllocMapped));
    memcpy(host_mem, (float *)&weights[0], weights.size() * sizeof(float));
    m_layers.back().weights.emplace_back(host_mem);
    m_layers.back().weights_size.emplace_back((int64_t)weights.size());
}

template <typename net_t>
void CuDNN_Network<net_t>::push_weights_trt_col_major(
    const size_t layer,
    const std::vector<float>& weights,
    const int row,
    const int column,
    const int channels) {

    if (layer >= m_layers.size()) {
        m_layers.emplace_back(CuDNN_Layer());
    }
    // When TensorRT chooses a precision for a layer,
    // it automatically converts weights as necessary to run the layer
    // Transpose from model's CK to TensorRT's KC
    auto weightSize = weights.size() * sizeof(float);
    auto transposed_weights = std::vector<float>(weights.size());
    for (int ch = 0; ch < channels; ch++) {
        for (int i = 0; i < column; i++) {
            for (int j = 0; j < row; j++) {
                transposed_weights[ch * column * row + j * column + i] =
                    (float)weights[ch * column * row + i * row + j];
            }
        }
    }
    void *host_mem;
    checkCUDA(cudaHostAlloc((void **)&host_mem,
                            weightSize,
                            cudaHostAllocMapped));
    memcpy(host_mem, (float*)&transposed_weights[0], weightSize);
    m_layers.back().weights.emplace_back(host_mem);
    m_layers.back().weights_size.emplace_back((int64_t)weights.size());
}
#endif

template <typename net_t>
void CuDNN_Network<net_t>::push_input_convolution(
    const unsigned int filter_size,
    unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    const float scale) {

#if !defined(USE_CUDNN) && !defined(USE_CUDNN_GRAPH)
    (void)scale;
#endif

    size_t layer = get_layer_count();
#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        push_weights_trt(layer, weights);
        push_weights_trt(layer, biases);
    } else
#endif
    if (cfg_NCHW) {
        push_weights(layer, weights); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases);  // Here it is still float(Convert precision with push_weights)
    } else {
        auto weights_convert = NCHW_to_NHWC<float>(weights, outputs, filter_size, filter_size, channels);
        push_weights(layer, weights_convert); // Convert precision with push_weights
        push_weights(layer, biases);          // Convert precision with push_weights
    }
    m_layers[layer].is_input_convolution = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;
#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        m_layers[layer].name = "in." + std::to_string(layer);
        return;
    }
#endif

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    m_layers[layer].scale_1 = 1.0f / scale;
    m_layers[layer].scale_2 = 1.0f / scale;
    m_layers[layer].scale_3 = 1.0f;

#if defined(USE_CUDNN_GRAPH)
    if (cfg_backend == backend_t::CUDNNGRAPH) {
        for (auto i = 0; i < m_num_worker_threads; i++) {
            auto conv_desc_single
                = convolve_fe_init(m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
            auto conv_desc_multi
                = convolve_fe_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
        }
#if defined(USE_CUDNN)
    } else {
#endif
#endif
#if defined(USE_CUDNN)
        for (auto i = 0; i < m_num_worker_threads; i++) {
            auto conv_desc_single
                = convolve_init(m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
            auto conv_desc_multi
                = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
        }
#endif
#if defined(USE_CUDNN_GRAPH)
    }
#endif
#endif
}

template <typename net_t>
void CuDNN_Network<net_t>::push_residual(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights_1,
    const std::vector<float>& biases_1,
    const std::vector<float>& weights_2,
    const std::vector<float>& biases_2,
    const float scale_1,
    const float scale_2,
    const float scale_3) {

    size_t layer = get_layer_count();
#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        push_weights_trt(layer, weights_1);
        push_weights_trt(layer, biases_1);
        push_weights_trt(layer, weights_2);
        push_weights_trt(layer, biases_2);
    } else
#endif
    if (cfg_NCHW) {
        push_weights(layer, weights_1); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
        push_weights(layer, weights_2); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
    } else {
        auto weights_convert_1 = NCHW_to_NHWC<float>(
            weights_1, outputs, filter_size, filter_size, channels);
        auto weights_convert_2 = NCHW_to_NHWC<float>(
            weights_2, outputs, filter_size, filter_size, channels);
        push_weights(layer, weights_convert_1); // Convert precision with push_weights
        push_weights(layer, biases_1);          // Convert precision with push_weights
        push_weights(layer, weights_convert_2); // Convert precision with push_weights
        push_weights(layer, biases_2);          // Convert precision with push_weights
    }
    m_layers[layer].is_residual_block = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;

#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        m_layers[layer].name = "res." + std::to_string(layer);
        return;
    }
#endif

    m_layers[layer].scale_1 = 1.0f / scale_1;
    m_layers[layer].scale_2 = 1.0f / scale_2;
    m_layers[layer].scale_3 = 1.0f / scale_3;

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    if (layer == 1) {
#if defined(USE_CUDNN_GRAPH)
        if (cfg_backend == backend_t::CUDNNGRAPH) {
            for (auto i = 0; i < m_num_worker_threads; i++) {
                auto conv_desc_single
                    = convolve_fe_init(m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = convolve_fe_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
                auto conv_desc_no_relu_single
                    = convolve_fe_no_relu_init(m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_no_relu_desc_single.emplace_back(conv_desc_no_relu_single);
                auto conv_desc_no_relu_multi
                    = convolve_fe_no_relu_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_no_relu_desc_multi.emplace_back(conv_desc_no_relu_multi);
                auto conv_desc_add_relu_single
                    = convolve_fe_add_relu_init(m_handle[i], channels, outputs);
                m_layers[layer].conv_add_relu_desc_single.emplace_back(conv_desc_add_relu_single);
                auto conv_desc_add_relu_multi
                    = convolve_fe_add_relu_init(m_handle[i], channels, outputs, cfg_batch_size);
                m_layers[layer].conv_add_relu_desc_multi.emplace_back(conv_desc_add_relu_multi);
            }
        } else {
#endif
#if defined(USE_CUDNN)
            for (auto i = 0; i < m_num_worker_threads; i++) {
                auto conv_desc_single
                    = convolve_init(m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desk_multi
                    = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desk_multi);
            }
#endif
#if defined(USE_CUDNN_GRAPH)
        }
#endif
    }
#endif
}

template <typename net_t>
void CuDNN_Network<net_t>::push_residual_se(
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
    const float scale_3) {

#if !defined(USE_CUDNN) && !defined(USE_CUDNN_GRAPH)
    (void)scale_1;
    (void)scale_2;
    (void)scale_3;
#endif

    size_t layer = get_layer_count();
#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        push_weights_trt(layer, weights_1);
        push_weights_trt(layer, biases_1);
        push_weights_trt(layer, weights_2);
        push_weights_trt(layer, biases_2);
        push_weights_trt(layer, se_fc1_w);
        push_weights_trt(layer, se_fc1_b);
        push_weights_trt(layer, se_fc2_w);
        push_weights_trt(layer, se_fc2_b);
    } else
#endif
    if (cfg_NCHW) {
        push_weights(layer, weights_1); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
        push_weights(layer, weights_2); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
        push_weights_col_major(layer, se_fc1_w, channels / 2, channels);
        push_weights(layer, se_fc1_b);
        push_weights_col_major(layer, se_fc2_w, channels * 2, channels / 2);
        push_weights(layer, se_fc2_b);
    } else {
        auto weights_convert_1 = NCHW_to_NHWC<float>(
            weights_1, outputs, filter_size, filter_size, channels);
        auto weights_convert_2 = NCHW_to_NHWC<float>(
            weights_2, outputs, filter_size, filter_size, channels);
        push_weights(layer, weights_convert_1); // Convert precision with push_weights
        push_weights(layer, biases_1);          // Convert precision with push_weights
        push_weights(layer, weights_convert_2); // Convert precision with push_weights
        push_weights(layer, biases_2);          // Convert precision with push_weights
        push_weights_col_major(layer, se_fc1_w, channels / 2, channels);
        push_weights(layer, se_fc1_b);
        push_weights_col_major(layer, se_fc2_w, channels * 2, channels / 2);
        push_weights(layer, se_fc2_b);
    }
    m_layers[layer].is_residual_block = true;
    m_layers[layer].is_se_block = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;

#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        m_layers[layer].name = "res." + std::to_string(layer);
        return;
    }
#endif

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    m_layers[layer].scale_1 = 1.0f / scale_1;
    m_layers[layer].scale_2 = 1.0f / scale_2;
    m_layers[layer].scale_3 = 1.0f / scale_3;

    if (layer == 1) {
#if defined(USE_CUDNN_GRAPH)
        if (cfg_backend == backend_t::CUDNNGRAPH) {
            for (auto i = 0; i < m_num_worker_threads; i++) {
                auto conv_desc_single
                    = convolve_fe_init(m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = convolve_fe_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
                auto conv_desc_no_relu_single
                    = convolve_fe_no_relu_init(m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_no_relu_desc_single.emplace_back(conv_desc_no_relu_single);
                auto conv_desc_no_relu_multi
                    = convolve_fe_no_relu_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_no_relu_desc_multi.emplace_back(conv_desc_no_relu_multi);
            }
        } else {
#endif
#if defined(USE_CUDNN)
            for (auto i = 0; i < m_num_worker_threads; i++) {
                auto conv_desc_single
                    = convolve_init(m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
            }
#endif
#if defined(USE_CUDNN_GRAPH)
        }
#endif
    }
#endif
}

template <typename net_t>
void CuDNN_Network<net_t>::push_convolve(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    const std::vector<float>& stddevs,
    const std::vector<float>& ip1_w,
    const std::vector<float>& ip1_b,
    const std::vector<float>& ip2_w,
    const std::vector<float>& ip2_b) {

    size_t layer = get_layer_count();
#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        push_weights_trt(layer, weights);
        if (cfg_head_bn == head_bn_t::GPU_A) {
            push_weights_trt(layer, biases);
            if (outputs == Network::OUTPUTS_VALUE) {
                push_weights_trt_col_major(layer, ip1_w, NUM_INTERSECTIONS, channels);
            } else {
                push_weights_trt(layer, ip1_w);
            }
            push_weights_trt(layer, ip1_b);
            if (outputs == Network::OUTPUTS_VALUE) {
                push_weights_trt(layer, ip2_w);
                push_weights_trt(layer, ip2_b);
            }
        } else if (cfg_head_bn == head_bn_t::GPU_B) {
            push_weights_trt(layer, biases);
            push_weights_trt(layer, stddevs);
        }
        m_layers[layer].outputs = outputs;
        m_layers[layer].channels = channels;
        m_layers[layer].filter_size = filter_size;
        if (outputs == Network::OUTPUTS_POLICY) {
            m_layers[layer].is_policy = true;
            m_layers[layer].name = "pol." + std::to_string(layer);
            return;
        }
        m_layers[layer].is_value = true;
        m_layers[layer].name = "val." + std::to_string(layer);

        if (build(m_num_worker_threads, cfg_batch_size)) {
            return;
        }
        exit(EXIT_FAILURE);
    }
#else
    (void) biases;
    (void) stddevs;
    (void) ip1_w;
    (void) ip1_b;
    (void) ip2_w;
    (void) ip2_b;
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    if (cfg_NCHW) {
        push_weights(layer, weights); // Here it is still float(Convert precision with push_weights)
    } else {
        auto weights_convert = NCHW_to_NHWC<float>(
            weights, outputs, filter_size, filter_size, channels);
        push_weights(layer, weights_convert); // Convert precision with push_weights
    }
    m_layers[layer].outputs = outputs;
    m_layers[layer].channels = channels;
    m_layers[layer].filter_size = filter_size;

    if (outputs == Network::OUTPUTS_VALUE) {
        m_layers[layer].is_value = true;
#if defined(USE_CUDNN_GRAPH)
        if (cfg_backend == backend_t::CUDNNGRAPH) {
            for (auto i = 0; i < m_num_worker_threads; i++) {
                auto conv_desc_single
                    = convolve_fe_head_init(m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = convolve_fe_head_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
            }
#if defined(USE_CUDNN)
        } else {
#endif
#endif
#if defined(USE_CUDNN)
            for (auto i = 0; i < m_num_worker_threads; i++) {
                auto conv_desc_single
                    = convolve_init(m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
            }
#endif
#if defined(USE_CUDNN_GRAPH)
        }
#endif
    } else {
        m_layers[layer].is_policy = true;
#if defined(USE_CUDNN_GRAPH)
        if (cfg_backend == backend_t::CUDNNGRAPH) {
            for (auto i = 0; i < m_num_worker_threads; i++) {
                auto conv_desc_single
                    = convolve_fe_head_init(m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = convolve_fe_head_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
            }
#if defined(USE_CUDNN)
        } else {
#endif
#endif
#if defined(USE_CUDNN)
            for (auto i = 0; i < m_num_worker_threads; i++) {
                std::shared_ptr<conv_descriptor> conv_desc_single
                    = convolve_init(m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                std::shared_ptr<conv_descriptor> conv_desc_multi
                    = convolve_init(m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
            }
#endif
#if defined(USE_CUDNN_GRAPH)
        }
#endif
    }
#endif
#endif
}

template <typename net_t>
void CuDNN_Network<net_t>::forward_activations(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val,
    CuDNNContext& cudnn_context,
    const int tid,
    const int batch_size) {

    const auto inSize = batch_size * sizeof(net_t) * m_layers[0].channels * NUM_INTERSECTIONS;
    const auto pol_elements
        = batch_size * m_layers[m_layers.size() - 2].outputs * NUM_INTERSECTIONS;
    const auto val_elements
        = batch_size * m_layers.back().outputs * NUM_INTERSECTIONS;

#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        size_t pol_elements_trt;
        size_t val_elements_trt;
        if (cfg_head_bn == head_bn_t::GPU_A) {
            pol_elements_trt = batch_size * POTENTIAL_MOVES;
            val_elements_trt = batch_size;
        } else {
            pol_elements_trt = pol_elements;
            val_elements_trt = val_elements;
        }
        std::vector<net_t> pol_net_trt_t = std::vector<net_t>(pol_elements_trt);
        std::vector<net_t> val_net_trt_t = std::vector<net_t>(val_elements_trt);
        auto search = cudnn_context.mBuffers.find("InputFeature");
        assert(search != cudnn_context.mBuffers.end());
        if (typeid(net_t) == typeid(float)) {
            checkCUDA(cudaMemcpyAsync(
                search->second,
                (net_t*)&input[0],
                inSize,
                cudaMemcpyHostToDevice,
                m_streams[tid]));
        } else {
            auto input_net_t = std::vector<net_t>(batch_size * m_layers[0].channels * NUM_INTERSECTIONS);
            std::copy(input.begin(), input.end(), input_net_t.begin());
            cudaMemcpyAsync(
                search->second,
                (net_t*)&input_net_t[0],
                inSize,
                cudaMemcpyHostToDevice,
                m_streams[tid]);
        }
        if (cfg_execute_context == execute_t::SINGLE || batch_size == 1) {
            cudnn_context.mContext->setInputShape("InputFeature",
                Dims4(batch_size, m_layers[0].channels, BOARD_SIZE, BOARD_SIZE));
        } else {
            cudnn_context.mContext_n->setInputShape("InputFeature",
                Dims4(batch_size, m_layers[0].channels, BOARD_SIZE, BOARD_SIZE));
        }
        if (m_net_type == int(NetworkType::MINIGO_SE)) {
            if (cfg_execute_context == execute_t::SINGLE || batch_size == 1) {
                cudnn_context.mContext->setInputShape("BatchSize",
                    Dims({4, {(unsigned int)batch_size, m_layers[1].channels, 1, 1}}));
            } else {
                cudnn_context.mContext_n->setInputShape("BatchSize",
                    Dims({4, {(unsigned int)batch_size, m_layers[1].channels, 1, 1}}));
            }
        }
        if (cfg_execute_context == execute_t::SINGLE || batch_size == 1) {
            ASSERT(cudnn_context.mContext->enqueueV3(m_streams[tid]));
        } else {
            ASSERT(cudnn_context.mContext_n->enqueueV3(m_streams[tid]));
        }
        search = cudnn_context.mBuffers.find("OutputPolicy");
        assert(search != cudnn_context.mBuffers.end());
        checkCUDA(cudaMemcpyAsync(&output_pol[0],
                                  search->second,
                                  pol_elements_trt * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  m_streams[tid]));
        search = cudnn_context.mBuffers.find("OutputValue");
        assert(search != cudnn_context.mBuffers.end());
        checkCUDA(cudaMemcpyAsync(&output_val[0],
                                  search->second,
                                  val_elements_trt * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  m_streams[tid]));
        // Asynchronously enqueue the inference work
        cudaStreamSynchronize(m_streams[tid]);
        return;
    }
#endif

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    // input: input(float) 18 chanels * (BOARD_SIZE * BOARD_SIZE)

    auto pol_net_t = std::vector<net_t>(pol_elements);
    auto val_net_t = std::vector<net_t>(val_elements);
    // Always allocates enough space for floats
    constexpr auto one_plane = NUM_INTERSECTIONS * sizeof(float);

    if (!cudnn_context.m_buffers_allocated) {
        auto max_wsize = size_t{0};
        auto max_channels = unsigned{0};
        int layer_i = 0;
        for (const auto& layer : m_layers) {
            for (auto i = 0; i < m_num_worker_threads; i++) {
                if (!layer.is_residual_block || layer_i == 1) {
                    max_wsize = std::max(max_wsize,
                                         layer.conv_desc_single[i]->workspace_size);
                    max_wsize = std::max(max_wsize,
                                         layer.conv_desc_multi[i]->workspace_size);
                    if (cfg_backend == backend_t::CUDNNGRAPH) {
                        if (layer.conv_no_relu_desc_single.size() > 0) {
                            max_wsize = std::max(max_wsize,
                                                 layer.conv_no_relu_desc_single[i]->workspace_size);
                            max_wsize = std::max(max_wsize,
                                                 layer.conv_no_relu_desc_multi[i]->workspace_size);
                        }
                        if (layer.conv_add_relu_desc_single.size() > 0) {
                            max_wsize = std::max(max_wsize,
                                                 layer.conv_add_relu_desc_single[i]->workspace_size);
                            max_wsize = std::max(max_wsize,
                                                 layer.conv_add_relu_desc_multi[i]->workspace_size);
                        }
                    }
                    max_channels = std::max(max_channels,
                                            std::max(layer.channels, layer.outputs));
                }
            }
            layer_i++;
        }
        auto alloc_insize = cfg_batch_size * max_channels * one_plane;

        void *d_workspace;
        checkCUDA(cudaMalloc((void**)&d_workspace, max_wsize));

        void *d_InBuffer;
        checkCUDA(cudaMalloc((void**)&d_InBuffer, alloc_insize));

        void *d_OutBuffer;
        checkCUDA(cudaMalloc((void**)&d_OutBuffer, alloc_insize));

        void *d_TempBuffer;
        checkCUDA(cudaMalloc((void**)&d_TempBuffer, alloc_insize));

        cudnn_context.m_workspace = d_workspace;
        cudnn_context.m_InBuffer = d_InBuffer;
        cudnn_context.m_OutBuffer = d_OutBuffer;
        cudnn_context.m_TempBuffer = d_TempBuffer;
        cudnn_context.m_buffers_allocated = true;

        if (m_net_type == int(NetworkType::MINIGO_SE)) {
            void *d_IdentityOutBuffer;
            checkCUDA(cudaMalloc((void**)&d_IdentityOutBuffer, alloc_insize));

            void *d_PoolBuffer;
            checkCUDA(cudaMalloc((void**)&d_PoolBuffer,
                                 cfg_batch_size * max_channels * sizeof(net_t)));

            cudnn_context.m_IdentityOutBuffer = d_IdentityOutBuffer;
            cudnn_context.m_PoolBuffer = d_PoolBuffer;

            const __half alpha_16 = __float2half(1.0f);
            const float alpha_32 = 1.0f;
            const __half beta_16 = __float2half(0.0f);
            const float beta_32 = 0.0f;
            void *d_m_alpha_16;
            checkCUDA(cudaMalloc((void**)&d_m_alpha_16, sizeof(alpha_16)));
            checkCUDA(cudaMemcpyAsync(d_m_alpha_16, (__half*)&alpha_16, sizeof(alpha_16),
                      cudaMemcpyHostToDevice, m_streams[tid]));
            void *d_m_alpha_32;
            checkCUDA(cudaMalloc((void**)&d_m_alpha_32, sizeof(alpha_32)));
            checkCUDA(cudaMemcpyAsync(d_m_alpha_32, (float*)&alpha_32, sizeof(alpha_32),
                      cudaMemcpyHostToDevice, m_streams[tid]));
            void *d_m_beta_16;
            checkCUDA(cudaMalloc((void**)&d_m_beta_16, sizeof(beta_16)));
            checkCUDA(cudaMemcpyAsync(d_m_beta_16, (__half*)&beta_16, sizeof(beta_16),
                      cudaMemcpyHostToDevice, m_streams[tid]));
            void *d_m_beta_32;
            checkCUDA(cudaMalloc((void**)&d_m_beta_32, sizeof(beta_32)));
            checkCUDA(cudaMemcpyAsync(d_m_beta_32, (float*)&beta_32, sizeof(beta_32),
                      cudaMemcpyHostToDevice, m_streams[tid]));
            cudaStreamSynchronize(m_streams[tid]);
            cudnn_context.m_alpha_32 = d_m_alpha_32;
            cudnn_context.m_alpha_16 = d_m_alpha_16;
            cudnn_context.m_beta_32 = d_m_beta_32;
            cudnn_context.m_beta_16 = d_m_beta_16;
        }
    }

    auto workspace = cudnn_context.m_workspace;
    auto InBuffer = cudnn_context.m_InBuffer;
    auto OutBuffer = cudnn_context.m_OutBuffer;
    auto IdentityOutBuffer = cudnn_context.m_IdentityOutBuffer;
    auto PoolBuffer = cudnn_context.m_PoolBuffer;
    auto TempBuffer = cudnn_context.m_TempBuffer;

    if (typeid(net_t) == typeid(float) && cfg_NCHW) {
        checkCUDA(cudaMemcpy(InBuffer, (net_t*)&input[0], inSize, cudaMemcpyHostToDevice));
    } else if (typeid(net_t) == typeid(__half) && cfg_NCHW) {
        auto input_net_t = std::vector<net_t>(batch_size * m_layers[0].channels * NUM_INTERSECTIONS);
        std::copy(input.begin(), input.end(), input_net_t.begin());
        checkCUDA(cudaMemcpy(InBuffer, (net_t*)&input_net_t[0], inSize, cudaMemcpyHostToDevice));
    } else {
        auto input_net_t = std::vector<net_t>(batch_size * m_layers[0].channels * NUM_INTERSECTIONS);
        input_net_t = NCHW_to_NHWC<net_t>(
            input, batch_size, BOARD_SIZE, BOARD_SIZE, m_layers[0].channels);
        checkCUDA(cudaMemcpy(InBuffer, (net_t*)&input_net_t[0], inSize, cudaMemcpyHostToDevice));
    }

    for (auto iter = std::begin(m_layers); iter != std::end(m_layers); iter++) {
        const auto& layer = *iter;
        const auto niter = std::next(iter);

        if (layer.is_input_convolution) {
            // input: InBuffer
            assert(niter != std::end(m_layers));
            auto conv_weights = begin(layer.weights);
            auto conv_biases = begin(layer.weights) + 1;

#if defined(USE_CUDNN_GRAPH)
            if (cfg_backend == backend_t::CUDNNGRAPH) {
                if (batch_size == 1) {
                    // Y = ReLU(Convolve(X, W) + B)
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                        {layer.conv_desc_single[tid]->X, InBuffer},
                        {layer.conv_desc_single[tid]->W, conv_weights[0]},
                        {layer.conv_desc_single[tid]->B, conv_biases[0]},
                        {layer.conv_desc_single[tid]->Y, OutBuffer} };
                    checkCUDNNFE(layer.conv_desc_single[tid]->graph.execute(m_handle[tid],
                                                                            variant_pack, workspace));
                } else {
                    // Y = ReLU(Convolve(X, W) + B)
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                        {layer.conv_desc_multi[tid]->X, InBuffer},
                        {layer.conv_desc_multi[tid]->W, conv_weights[0]},
                        {layer.conv_desc_multi[tid]->B, conv_biases[0]},
                        {layer.conv_desc_multi[tid]->Y, OutBuffer} };
                    checkCUDNNFE(layer.conv_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                           variant_pack, workspace));
                }
#if defined(USE_CUDNN)
            } else {
#endif
#endif
#if defined(USE_CUDNN)
                if (batch_size == 1) {
                    convolveActivation(tid,
                                       InBuffer,
                                       OutBuffer,
                                       conv_weights[0],
                                       nullptr,
                                       conv_biases[0],
                                       workspace,
                                       layer.conv_desc_single[tid],
                                       layer.scale_1,
                                       1.0f);
                } else {
                    convolveActivation(tid,
                                       InBuffer,
                                       OutBuffer,
                                       conv_weights[0],
                                       nullptr,
                                       conv_biases[0],
                                       workspace,
                                       layer.conv_desc_multi[tid],
                                       layer.scale_1,
                                       1.0f);
                }
#endif
#if defined(USE_CUDNN_GRAPH)
            }
#endif
            // output: OutBuffer
        } else if (layer.is_residual_block && !layer.is_se_block) {
            // input: OutBuffer
            assert(layer.channels == layer.outputs);
            assert(niter != std::end(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases  = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases  = begin(layer.weights) + 3;

#if defined(USE_CUDNN_GRAPH)
            if (cfg_backend == backend_t::CUDNNGRAPH) {
                if (batch_size == 1) {
                    // Y = ReLU(Convolve(X, W) + B)
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                        {m_layers[1].conv_desc_single[tid]->X, OutBuffer},
                        {m_layers[1].conv_desc_single[tid]->W, conv1_weights[0]},
                        {m_layers[1].conv_desc_single[tid]->B, conv1_biases[0]},
                        {m_layers[1].conv_desc_single[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_desc_single[tid]->graph.execute(m_handle[tid],
                                                                                  variant_pack1, workspace));
                    // Y = Convolve(X, W) + B
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                        {m_layers[1].conv_no_relu_desc_single[tid]->X, InBuffer},
                        {m_layers[1].conv_no_relu_desc_single[tid]->W, conv2_weights[0]},
                        {m_layers[1].conv_no_relu_desc_single[tid]->B, conv2_biases[0]},
                        {m_layers[1].conv_no_relu_desc_single[tid]->Y, TempBuffer} };
                    checkCUDNNFE(m_layers[1].conv_no_relu_desc_single[tid]->graph.execute(m_handle[tid],
                                                                                          variant_pack2, workspace));
                    // Y = ReLU(X + Z)
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack3 = {
                        {m_layers[1].conv_add_relu_desc_single[tid]->X, TempBuffer},
                        {m_layers[1].conv_add_relu_desc_single[tid]->Z, OutBuffer},
                        {m_layers[1].conv_add_relu_desc_single[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_add_relu_desc_single[tid]->graph.execute(m_handle[tid],
                                                                                           variant_pack3, workspace));
                } else {
                    // Y = ReLU(Convolve(X, W) + B)
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                        {m_layers[1].conv_desc_multi[tid]->X, OutBuffer},
                        {m_layers[1].conv_desc_multi[tid]->W, conv1_weights[0]},
                        {m_layers[1].conv_desc_multi[tid]->B, conv1_biases[0]},
                        {m_layers[1].conv_desc_multi[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                                 variant_pack1, workspace));
                    // Y = Convolve(X, W) + B
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                        {m_layers[1].conv_no_relu_desc_multi[tid]->X, InBuffer},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->W, conv2_weights[0]},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->B, conv2_biases[0]},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->Y, TempBuffer} };
                    checkCUDNNFE(m_layers[1].conv_no_relu_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                                         variant_pack2, workspace));
                    // Y = ReLU(X + Z)
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack3 = {
                        {m_layers[1].conv_add_relu_desc_multi[tid]->X, TempBuffer},
                        {m_layers[1].conv_add_relu_desc_multi[tid]->Z, OutBuffer},
                        {m_layers[1].conv_add_relu_desc_multi[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_add_relu_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                                          variant_pack3, workspace));
                }
                std::swap(InBuffer, OutBuffer);
                // output: OutBuffer
#if defined(USE_CUDNN)
            } else {
#endif
#endif
#if defined(USE_CUDNN)
                if (batch_size == 1) {
                    convolveActivation(tid,
                                       OutBuffer,
                                       InBuffer,
                                       conv1_weights[0],
                                       nullptr,
                                       conv1_biases[0],
                                       workspace,
                                       m_layers[1].conv_desc_single[tid],
                                       layer.scale_1,
                                       1.0f);

                    convolveActivation(tid,
                                       InBuffer,
                                       OutBuffer,
                                       conv2_weights[0],
                                       OutBuffer,          // *residualBuffer: first input
                                       conv2_biases[0],
                                       workspace,
                                       m_layers[1].conv_desc_single[tid],
                                       layer.scale_2,
                                       layer.scale_3);
                } else {
                    convolveActivation(tid,
                                       OutBuffer,
                                       InBuffer,
                                       conv1_weights[0],
                                       nullptr,
                                       conv1_biases[0],
                                       workspace,
                                       m_layers[1].conv_desc_multi[tid],
                                       layer.scale_1,
                                       1.0f);

                    convolveActivation(tid,
                                       InBuffer,
                                       OutBuffer,
                                       conv2_weights[0],
                                       OutBuffer,          // *residualBuffer: first input
                                       conv2_biases[0],
                                       workspace,
                                       m_layers[1].conv_desc_multi[tid],
                                       layer.scale_2,
                                       layer.scale_3);
                }
                // output: OutBuffer
#endif
#if defined(USE_CUDNN_GRAPH)
            }
#endif
        } else if (layer.is_residual_block && layer.is_se_block) {
            // input: OutBuffer
            assert(layer.channels == layer.outputs);
            assert(niter != std::end(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases  = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases  = begin(layer.weights) + 3;
            auto fc1_weights   = begin(layer.weights) + 4;
            auto fc1_biases    = begin(layer.weights) + 5;
            auto fc2_weights   = begin(layer.weights) + 6;
            auto fc2_biases    = begin(layer.weights) + 7;

#if defined(USE_CUDNN_GRAPH)
            if (cfg_backend == backend_t::CUDNNGRAPH) {
                if (batch_size == 1) {
                    // Y = ReLU(Convolve(X, W) + B)
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                        {m_layers[1].conv_desc_single[tid]->X, OutBuffer},
                        {m_layers[1].conv_desc_single[tid]->W, conv1_weights[0]},
                        {m_layers[1].conv_desc_single[tid]->B, conv1_biases[0]},
                        {m_layers[1].conv_desc_single[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_desc_single[tid]->graph.execute(m_handle[tid],
                                                                                  variant_pack1, workspace));
                    // Y = Convolve(X, W) + B
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                        {m_layers[1].conv_no_relu_desc_single[tid]->X, InBuffer},
                        {m_layers[1].conv_no_relu_desc_single[tid]->W, conv2_weights[0]},
                        {m_layers[1].conv_no_relu_desc_single[tid]->B, conv2_biases[0]},
                        {m_layers[1].conv_no_relu_desc_single[tid]->Y, TempBuffer} };
                    checkCUDNNFE(m_layers[1].conv_no_relu_desc_single[tid]->graph.execute(m_handle[tid],
                                                                                          variant_pack2, workspace));
                } else {
                    // Y = ReLU(Convolve(X, W) + B)
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                        {m_layers[1].conv_desc_multi[tid]->X, OutBuffer},
                        {m_layers[1].conv_desc_multi[tid]->W, conv1_weights[0]},
                        {m_layers[1].conv_desc_multi[tid]->B, conv1_biases[0]},
                        {m_layers[1].conv_desc_multi[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                                 variant_pack1, workspace));
                    // Y = Convolve(X, W) + B
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                        {m_layers[1].conv_no_relu_desc_multi[tid]->X, InBuffer},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->W, conv2_weights[0]},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->B, conv2_biases[0]},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->Y, TempBuffer} };
                    checkCUDNNFE(m_layers[1].conv_no_relu_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                                         variant_pack2, workspace));
                }

                std::swap(TempBuffer, IdentityOutBuffer);
#if defined(USE_CUDNN)
            } else {
#endif
#endif
#if defined(USE_CUDNN)
                if (batch_size == 1) {
                    convolveActivation(tid,
                                       OutBuffer,        // *bufferIn
                                       InBuffer,         // *bufferOut
                                       conv1_weights[0],
                                       nullptr,
                                       conv1_biases[0],
                                       workspace,
                                       m_layers[1].conv_desc_single[tid],
                                       layer.scale_1,
                                       1.0f);

                    convolveIdentityActivation(tid,
                                               InBuffer,          // *bufferIn
                                               IdentityOutBuffer, // *bufferOut
                                               conv2_weights[0],
                                               nullptr,
                                               conv2_biases[0],
                                               workspace,
                                               m_layers[1].conv_desc_single[tid],
                                               layer.scale_2,
                                               layer.scale_3);
                } else {
                    convolveActivation(tid,
                                       OutBuffer,        // *bufferIn
                                       InBuffer,         // *bufferOut
                                       conv1_weights[0],
                                       nullptr,
                                       conv1_biases[0],
                                       workspace,
                                       m_layers[1].conv_desc_multi[tid],
                                       layer.scale_1,
                                       1.0f);

                    convolveIdentityActivation(tid,
                                               InBuffer,          // *bufferIn
                                               IdentityOutBuffer, // *bufferOut
                                               conv2_weights[0],
                                               nullptr,
                                               conv2_biases[0],
                                               workspace,
                                               m_layers[1].conv_desc_multi[tid],
                                               layer.scale_2,
                                               layer.scale_3);
                }
#endif
#if defined(USE_CUDNN_GRAPH)
            }
#endif
            if (typeid(net_t) == typeid(float)) {
                squeeze_excitation_float(m_cublas_handles[tid],
                                         m_streams[tid],
                                         cudnn_context,
                                         OutBuffer,         // *bufferIn1: first input
                                         IdentityOutBuffer, // *bufferIn2: second output
                                         TempBuffer,
                                         fc1_weights[0],
                                         fc1_biases[0],
                                         fc2_weights[0],
                                         fc2_biases[0],
                                         InBuffer,          // *bufferOut
                                         PoolBuffer,
                                         batch_size,
                                         layer.outputs,
                                         NUM_INTERSECTIONS);
            } else {
                squeeze_excitation_half(m_cublas_handles[tid],
                                        m_streams[tid],
                                        cudnn_context,
                                        OutBuffer,         // *bufferIn1: first input
                                        IdentityOutBuffer, // *bufferIn2: second output
                                        TempBuffer,
                                        fc1_weights[0],
                                        fc1_biases[0],
                                        fc2_weights[0],
                                        fc2_biases[0],
                                        InBuffer,          // *bufferOut
                                        PoolBuffer,
                                        batch_size,
                                        layer.outputs,
                                        NUM_INTERSECTIONS);
            }
            std::swap(InBuffer, OutBuffer);
            // output: OutBuffer
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
        } else {
            auto conv_weights = begin(layer.weights);
#endif
            // input: OutBuffer(net_t is float or __half)
#if defined(USE_CUDNN_GRAPH)
            if (cfg_backend == backend_t::CUDNNGRAPH) {
                if (batch_size == 1) {
                    // Y = Convolve(X, W)
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                        {layer.conv_desc_single[tid]->X, OutBuffer},
                        {layer.conv_desc_single[tid]->W, conv_weights[0]},
                        {layer.conv_desc_single[tid]->Y, InBuffer} };
                    checkCUDNNFE(layer.conv_desc_single[tid]->graph.execute(m_handle[tid],
                                                                            variant_pack, workspace));
                } else {
                    // Y = Convolve(X, W)
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                        {layer.conv_desc_multi[tid]->X, OutBuffer},
                        {layer.conv_desc_multi[tid]->W, conv_weights[0]},
                        {layer.conv_desc_multi[tid]->Y, InBuffer} };
                    checkCUDNNFE(layer.conv_desc_multi[tid]->graph.execute(m_handle[tid],
                                                                           variant_pack, workspace));
                }
#if defined(USE_CUDNN)
            } else {
#endif
#endif
#if defined(USE_CUDNN)
                if (batch_size == 1) {
                    convolve(tid,
                             OutBuffer,
                             InBuffer,
                             conv_weights[0],
                             workspace,
                             layer.conv_desc_single[tid],
                             layer.scale_1);
                } else {
                    convolve(tid,
                             OutBuffer,
                             InBuffer,
                             conv_weights[0],
                             workspace,
                             layer.conv_desc_multi[tid],
                             layer.scale_1);
                }
#endif
#if defined(USE_CUDNN_GRAPH)
            }
#endif
            if (niter == std::end(m_layers)) {
                // Value input: InBuffer
                checkCUDA(cudaMemcpy(&val_net_t[0], InBuffer,
                                     val_elements * sizeof(net_t),
                                     cudaMemcpyDeviceToHost));
                // output: val_net_t
            } else {
                // Policy input: InBuffer
                checkCUDA(cudaMemcpy(&pol_net_t[0], InBuffer,
                                     pol_elements * sizeof(net_t),
                                     cudaMemcpyDeviceToHost));
                // output: pol_net_t
            }
        }
    }

    // input: val_net_t(net_t), pol_net_t(net_t)
    if (cfg_NCHW) {
        std::copy(val_net_t.begin(), val_net_t.end(), output_val.begin()); 
        std::copy(pol_net_t.begin(), pol_net_t.end(), output_pol.begin());
    } else {
        output_val = NHWC_to_NCHW<net_t>(
            val_net_t, batch_size, BOARD_SIZE, BOARD_SIZE, Network::OUTPUTS_VALUE);
        output_pol = NHWC_to_NCHW<net_t>(
            pol_net_t, batch_size, BOARD_SIZE, BOARD_SIZE, Network::OUTPUTS_POLICY);
    }
    // output: output_val(float) 1 chanels * (BOARD_SIZE * BOARD_SIZE)
    // output: output_pol(float) 2 chanels * (BOARD_SIZE * BOARD_SIZE)
#endif
}

template <typename net_t>
void CuDNN_Network<net_t>::forward(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val,
    const int tid,
    const int batch_size) {

    forward_activations(input, output_pol, output_val, *m_context[tid], tid, batch_size);
}

#if defined(USE_TENSOR_RT)
template <typename net_t>
bool CuDNN_Network<net_t>::build(
    const int num_worker_threads,
    const int batch_size) {

    // Bump this when between program versions we want to forcibly drop old timing caches and plan caches.
    mTuneDesc = strprintf(
        R"|("salt"(%s%s)"model %s"(%s,%d,%d,%d))|",
        PROGRAM_VERSION_MAJOR,
        PROGRAM_VERSION_MINOR,
        typeid(net_t) == typeid(float) ? "single" : "half",
        "1.0",                    // modelVersion,
        Network::INPUT_CHANNELS,  // numInputChannels,
        cfg_execute_context,
        batch_size
    );
    auto builder
        = TrtUniquePtr<IBuilder>(createInferBuilder(cfg_logger.getTRTLogger()));
    if (!builder) {
        std::cerr << "TensorRT backend: failed to create builder" << std::endl;
        return false;
    }
    auto config = TrtUniquePtr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "TensorRT backend: failed to create builder config" << std::endl;
        return false;
    }
    bool usingFP16 = false;
    if (builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
        usingFP16 = true;
    }

    const auto explicitBatchFlag =
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TrtUniquePtr<INetworkDefinition>(builder->createNetworkV2(explicitBatchFlag));
    if (!network) {
        std::cerr << "TensorRT backend: failed to create network definition" << std::endl;
        return false;
    }
    std::filesystem::path path = cfg_weightsfile;
    std::string filename = path.filename().string();
    auto ext_i = filename.find_last_of(".");
    std::string weightsfile = filename.substr(0, ext_i);
    network->setName(weightsfile.c_str());

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "TensorRT backend: failed to create optimization profile" << std::endl;
        return false;
    }
    if (cfg_execute_context == execute_t::SINGLE) {
        constructNetwork(network, profile, nullptr, batch_size);
        config->addOptimizationProfile(profile);
    } else {
        auto profile_n = builder->createOptimizationProfile();
        if (!profile_n) {
            std::cerr << "TensorRT backend: failed to create optimization profile" << std::endl;
            return false;
        }
        constructNetwork(network, profile, profile_n, batch_size);
        config->addOptimizationProfile(profile);
        config->addOptimizationProfile(profile_n);
    }

    if (m_device_prop.major >= 8) {
        // This is to avoid tactics that have shape switching overhead
        config->setTacticSources(1U << static_cast<uint32_t>(TacticSource::kJIT_CONVOLUTIONS));
        config->setBuilderOptimizationLevel(2);
    }
    // So that there are no concurrent kernel executions probably from other parts of code while profiling
    // See CUDA Runtime API document for more details related to NULL stream and synchronization behaviors
    config->setProfileStream(cudaStreamLegacy);
    // Typical runtime allocation is much less than the 1 GiB specified below
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);

    std::string plan;
    {
        static std::mutex tuneMutex;
        tuneMutex.lock();
        std::string cacheDir = Utils::leelaz_file("trtcache");
        std::filesystem::create_directory(cacheDir);
        assert(std::filesystem::exists(cacheDir));
        assert(std::filesystem::is_directory(cacheDir));

        uint8_t deviceHash[32];
        SHA2::get256(m_device_prop.name, deviceHash);

        // Truncated to 4 bytes
        char deviceIdent[4 * 2 + 1];
        for(int i = 0; i < 4; i++) {
            sprintf(deviceIdent + i * 2, "%02x", static_cast<unsigned char>(deviceHash[i]));
        }
        deviceIdent[sizeof(deviceIdent) - 1] = 0;

        std::string precision = typeid(net_t) == typeid(float) ? "single" : "half";
        std::string sep_char{std::filesystem::path::preferred_separator};

        uint8_t tuneHash[32];
        SHA2::get256(mTuneDesc.c_str(), tuneHash);
        // Truncated to 6 bytes
        char tuneIdent[6 * 2 + 1];
        for(int i = 0; i < 6; i++) {
            sprintf(tuneIdent + i * 2, "%02x", static_cast<unsigned char>(tuneHash[i]));
        }
        tuneIdent[sizeof(tuneIdent) - 1] = 0;

        if (cfg_cache_plan) {
            auto planCacheFile = strprintf(
                "%s%strt-%d_gpu-%s_tune-%s_net-%s_%s%s_%dx%d_%d_%d_batch%d_fp%d_%s",
                cacheDir.c_str(),
                sep_char.c_str(),
                getInferLibVersion(),
                deviceIdent,
                tuneIdent,
                network->getName(),
                PROGRAM_VERSION_MAJOR,
                PROGRAM_VERSION_MINOR,
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_execute_context,
                cfg_head_bn,
                batch_size,
                usingFP16 ? 16 : 32,
                precision.c_str()
            );
            std::string paramStr = strprintf(
                "_%d_%s_%s%s_%d_%d_%d_%d_%d_%s",
                getInferLibVersion(),
                deviceIdent,
                PROGRAM_VERSION_MAJOR,
                PROGRAM_VERSION_MINOR,
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_execute_context,
                batch_size,
                usingFP16 ? 16 : 32,
                precision.c_str()
            );
            try {
                plan = readFileBinary(planCacheFile);
            } catch (std::exception const& e) {
                (void) e;
            };

            if (plan.size() > 0) {
                if (plan.size() < 64 + paramStr.size()) {
                    std::cout << "Could not parse plan, unexpected size in " + planCacheFile << std::endl;
                    plan.clear();
                } else {
                    std::string cachedParamStr = plan.substr(plan.size() - paramStr.size());
                    std::string modelHash = plan.substr(plan.size() - 64 - paramStr.size(), 64);
                    if (modelHash != m_model_hash) {
                        std::cout << "Plan cache is corrupted or is for the wrong model in " + planCacheFile << std::endl;
                        plan.clear();
                    } else if (cachedParamStr != paramStr) {
                        std::cout << "Plan cache is corrupted or is for the wrong parameters in " + planCacheFile << std::endl;
                        plan.clear();
                    } else {
                        plan.erase(plan.size() - 64 - paramStr.size());
                    }
                }
            }

            if (plan.size() <= 0) {
                std::cout << "Creating new plan cache" << std::endl;
                auto planBuffer = std::unique_ptr<IHostMemory>(
                    builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    std::cerr << "TensorRT backend: failed to create plan" << std::endl;
                    return false;
                }
                plan.insert(
                    plan.end(),
                    static_cast<char*>(planBuffer->data()),
                    static_cast<char*>(planBuffer->data()) + planBuffer->size()
                );
                if (m_model_hash.size() != 64) {
                    std::cerr << "Unexpected model hash size" << std::endl;
                    return false;
                }
                plan.insert(
                    plan.end(),
                    m_model_hash.begin(),
                    m_model_hash.end()
                );
                plan.insert(
                    plan.end(),
                    paramStr.begin(),
                    paramStr.end()
                );
                std::ofstream ofs;
                ofs.open(planCacheFile, std::ios_base::out | std::ios_base::binary);
                ofs.write(plan.data(), plan.size());
                ofs.close();
                std::cout << "Saved new plan cache to " + planCacheFile << std::endl;
                plan.erase(plan.size() - 64 - paramStr.size());
                tuneMutex.unlock();
            } else {
                tuneMutex.unlock();
                std::cout << "Using existing plan cache at " + planCacheFile << std::endl;
            }
        } else {
            auto timingCacheFile = strprintf(
                "%s%strt-%d_gpu-%s_tune-%s_%dx%d_%d_%d_batch%d_fp%d_%s",
                cacheDir.c_str(),
                sep_char.c_str(),
                getInferLibVersion(),
                deviceIdent,
                tuneIdent,
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_execute_context,
                cfg_head_bn,
                batch_size,
                usingFP16 ? 16 : 32,
                precision.c_str()
            );

            std::string timingCacheBlob;
            try {
                timingCacheBlob = readFileBinary(timingCacheFile);
            } catch (std::exception const& e) {
                (void) e;
            };
            if (timingCacheBlob.size() > 0)
                std::cout << "Using existing timing cache at " << timingCacheFile << std::endl;
            else
                std::cout << "Creating new timing cache" << std::endl;

            auto timingCache =
                std::unique_ptr<ITimingCache>(
                    config->createTimingCache(timingCacheBlob.data(), timingCacheBlob.size()));
            auto invalidTimingCache = !config->setTimingCache(*timingCache, false);
            if (invalidTimingCache) {
                std::cout << "Invalid timing cache, using new one instead" << std::endl;
                timingCache.reset(config->createTimingCache(nullptr, 0));
                config->setTimingCache(*timingCache, false);
            }

            std::unique_ptr<IHostMemory> planBuffer;
            if (invalidTimingCache || !timingCacheBlob.size()) {
                planBuffer.reset(builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    std::cerr << "TensorRT backend: failed to create plan" << std::endl;
                    return false;
                }
                auto serializedTimingCache = std::unique_ptr<IHostMemory>(
                    config->getTimingCache()->serialize());
                std::ofstream ofs;
                ofs.open(timingCacheFile, std::ios_base::out | std::ios_base::binary);
                ofs.write(static_cast<char*>(serializedTimingCache->data()), serializedTimingCache->size());
                ofs.close();
                std::cout << "Saved new timing cache to " << timingCacheFile << std::endl;
                tuneMutex.unlock();
            } else {
                tuneMutex.unlock();
                planBuffer.reset(builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    std::cerr << "TensorRT backend: failed to create plan" << std::endl;
                    return false;
                }
            }
            plan.insert(
                plan.end(),
                static_cast<char*>(planBuffer->data()),
                static_cast<char*>(planBuffer->data()) + planBuffer->size());
        }
    }

    for (auto i = 0; i < num_worker_threads; i++) {
        std::unique_ptr<IRuntime> runtime
            = std::unique_ptr<IRuntime>(createInferRuntime(cfg_logger.getTRTLogger()));
        if (!runtime) {
            std::cerr << "createInferRuntime error: " << std::endl;
            return false;
        }
        std::unique_ptr<ICudaEngine> engine
            = std::unique_ptr<ICudaEngine>(
                runtime->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine) {
            std::cerr << "deserializeCudaEngine error: " << std::endl;
            return false;
        }
        std::unique_ptr<CuDNNContext> context = std::make_unique<CuDNNContext>();
        context->mContext.reset(engine->createExecutionContext());
        if (cfg_execute_context == execute_t::DOUBLE) {
            context->mContext_n.reset(engine->createExecutionContext());
        }
        for (auto i = 0; i < engine->getNbIOTensors(); i++) {
            void* buffer = nullptr;
            auto name = engine->getIOTensorName(i);
            auto dims = engine->getTensorShape(name);
            std::string_view name_str{name};
            size_t size_byte;
            if (name_str == "BatchSize") {
                size_byte = sizeof(int32_t);
            } else if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
                size_byte = sizeof(float);
            } else {
                size_byte = sizeof(net_t);
            }
            size_t bytes = std::accumulate(dims.d + 1,
                                           dims.d + dims.nbDims,
                                           batch_size * size_byte,
                                           std::multiplies<size_t>());
            checkCUDA(cudaMalloc(&buffer, bytes));
            if (name_str == "BatchSize") {
                auto input_batch = std::vector<int32_t>(batch_size * m_layers[1].channels, 0);
                checkCUDA(cudaMemcpy(
                    buffer,
                    (int32_t*)&input_batch[0],
                    bytes,
                    cudaMemcpyHostToDevice));
            }
            context->mBuffers.emplace(std::make_pair(name, buffer));
            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                context->mContext->setInputTensorAddress(name, buffer);
                if (cfg_execute_context == execute_t::DOUBLE) {
                    context->mContext_n->setInputTensorAddress(name, buffer);
                }
            } else {
                context->mContext->setOutputTensorAddress(name, buffer);
                if (cfg_execute_context == execute_t::DOUBLE) {
                    context->mContext_n->setOutputTensorAddress(name, buffer);
                }
            }
        }
        context->mContext->setOptimizationProfileAsync(0, cudaStreamPerThread);
        if (cfg_execute_context == execute_t::DOUBLE) {
            context->mContext_n->setOptimizationProfileAsync(1, cudaStreamPerThread);
        }
        cudaStreamSynchronize(cudaStreamPerThread);
        context->m_buffers_allocated = true;
        mRuntime.emplace_back(std::move(runtime));
        mEngine.emplace_back(std::move(engine));
        m_context.emplace_back(std::move(context));
    }
    return true;
}

template <typename net_t>
void CuDNN_Network<net_t>::constructNetwork(
    TrtUniquePtr<INetworkDefinition>& network,
    IOptimizationProfile* profile,
    IOptimizationProfile* profile_n,
    const int batch_size) {

    ITensor* inputFeature = nullptr;
    ITensor* outputConv = nullptr;
    ILayer* outPolicyLayer = nullptr;
    ILayer* outValueLayer = nullptr;
    ILayer* shapeLayer = nullptr;
    IShapeLayer* inShapeLayer = nullptr;
    ICastLayer* castLayer = nullptr;

    if (m_net_type == int(NetworkType::MINIGO_SE)) {
        auto batchSizeTensor = initInputs("BatchSize",
                                          network,
                                          profile,
                                          profile_n,
                                          m_layers[1].channels,
                                          1,
                                          1,
                                          batch_size);

        // See. https://github.com/NVIDIA/TensorRT/issues/2282
        inShapeLayer = network->addShape(*batchSizeTensor);
        castLayer = network->addCast(*inShapeLayer->getOutput(0), DataType::kINT32);

        shapeLayer = network->addUnary(
            *castLayer->getOutput(0),
            UnaryOperation::kABS);
    }

    for (auto iter = std::begin(m_layers);
         iter != std::end(m_layers); iter++) {

        const auto& layer = *iter;
        if (layer.is_input_convolution) {
            inputFeature = initInputs("InputFeature",
                                      network,
                                      profile,
                                      profile_n,
                                      layer.channels,
                                      BOARD_SIZE,
                                      BOARD_SIZE,
                                      batch_size);
            auto conv_weights = begin(layer.weights);
            auto conv_biases = begin(layer.weights) + 1;
            auto initialConvLayer = buildConvLayer(
                inputFeature,
                layer.filter_size,
                layer.weights_size[0],
                conv_weights[0],
                layer.weights_size[1],
                conv_biases[0],
                network,
                layer.name + ".conv",
                layer.outputs);
            auto outputConvLayer = buildActivationLayer(
                initialConvLayer->getOutput(0),
                network,
                layer.name + ".activation",
                ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else if (layer.is_residual_block && !layer.is_se_block) {
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases  = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases  = begin(layer.weights) + 3;
            auto firstConvLayer = buildConvLayer(
                outputConv,
                layer.filter_size,
                layer.weights_size[0],
                conv1_weights[0],
                layer.weights_size[1],
                conv1_biases[0],
                network,
                layer.name + ".conv.first",
                layer.outputs);
            auto firstActivationConvLayer = buildActivationLayer(
                firstConvLayer->getOutput(0),
                network,
                layer.name + ".activation.first",
                ActivationType::kRELU);
            auto secondConvLayer = buildConvLayer(
                firstActivationConvLayer->getOutput(0),
                layer.filter_size,
                layer.weights_size[2],
                conv2_weights[0],
                layer.weights_size[3],
                conv2_biases[0],
                network,
                layer.name + ".conv.second",
                layer.outputs);
            auto mergeLayer = network->addElementWise(
                *outputConv, *secondConvLayer->getOutput(0), ElementWiseOperation::kSUM);
            mergeLayer->setName((layer.name + ".merge").c_str());
            auto outputConvLayer = buildActivationLayer(
                mergeLayer->getOutput(0),
                network,
                layer.name + ".activation.final",
                ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else if (layer.is_residual_block && layer.is_se_block) {
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases  = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases  = begin(layer.weights) + 3;
            auto fc1_weights   = begin(layer.weights) + 4;
            auto fc1_biases    = begin(layer.weights) + 5;
            auto fc2_weights   = begin(layer.weights) + 6;
            auto fc2_biases    = begin(layer.weights) + 7;
            auto firstConvLayer = buildConvLayer(
                outputConv,
                layer.filter_size,
                layer.weights_size[0],
                conv1_weights[0],
                layer.weights_size[1],
                conv1_biases[0],
                network,
                layer.name + ".conv.first",
                layer.outputs);
            auto firstActivationConvLayer = buildActivationLayer(
                firstConvLayer->getOutput(0),
                network,
                layer.name + ".activation.first",
                ActivationType::kRELU);
            auto secondConvLayer = buildConvLayer(
                firstActivationConvLayer->getOutput(0),
                layer.filter_size,
                layer.weights_size[2],
                conv2_weights[0],
                layer.weights_size[3],
                conv2_biases[0],
                network,
                layer.name + ".conv.second",
                layer.outputs);
            // pool = tf.layers.average_pooling2d(residual, pool_size=go.N, strides=1, padding='valid')
            auto gpoolLayer = applyGPoolLayer(
                secondConvLayer->getOutput(0),
                network,
                layer.name + ".gpool");
            // fc1 = tf.layers.dense(pool, units=channels // 2)
            auto thirdMatMulLayer = buildConvLayer(
                gpoolLayer->getOutput(0),
                1,
                layer.weights_size[4],
                fc1_weights[0],
                layer.weights_size[5],
                fc1_biases[0],
                network,
                layer.name + ".conv.third",
                layer.outputs / 2);
            // squeeze = tf.nn.relu(fc1)
            auto thirdActivationMatLayer = buildActivationLayer(
                thirdMatMulLayer->getOutput(0),
                network,
                layer.name + ".activation.third",
                ActivationType::kRELU);
            // fc2 = tf.layers.dense(squeeze, units=2*channels)
            auto fourthMatMulLayer = buildConvLayer(
                thirdActivationMatLayer->getOutput(0),
                1,
                layer.weights_size[6],
                fc2_weights[0],
                layer.weights_size[7],
                fc2_biases[0],
                network,
                layer.name + ".conv.fourth",
                layer.outputs * 2);
            // gamma, bias = tf.split(fc2, 2, axis=3)
            auto gammaLayer = network->addSlice(
                *fourthMatMulLayer->getOutput(0),
                {4 ,{0, 0, 0, 0}},
                {4 ,{0, layer.channels, 1, 1}},
                {4 ,{1, 1, 1, 1}}
            );
            gammaLayer->setInput(2, *shapeLayer->getOutput(0));
            gammaLayer->setName((layer.name + ".gamma").c_str());
            // gamma, bias = tf.split(fc2, 2, axis=3)
            auto biasLayer = network->addSlice(
                *fourthMatMulLayer->getOutput(0),
                {4 ,{0, layer.channels, 0, 0}},
                {4 ,{0, layer.channels, 1, 1}},
                {4 ,{1, 1, 1, 1}}
            );
            biasLayer->setInput(2, *shapeLayer->getOutput(0));
            biasLayer->setName((layer.name + ".bias").c_str());
            // sig = tf.nn.sigmoid(gamma)
            auto sigLayer = buildActivationLayer(
                gammaLayer->getOutput(0),
                network,
                layer.name + ".activation.sig",
                ActivationType::kSIGMOID);
            sigLayer->setName((layer.name + ".sig").c_str());
            // scale = tf.reshape(sig, [-1, 1, 1, channels])
            // excitation = tf.multiply(scale, residual) + bias
            auto scaleLayer = network->addElementWise(
                *sigLayer->getOutput(0),
                *secondConvLayer->getOutput(0),
                ElementWiseOperation::kPROD
            );
            scaleLayer->setName((layer.name + ".scale").c_str());
            // excitation = tf.multiply(scale, residual) + bias
            auto excitationLayer = network->addElementWise(
                *scaleLayer->getOutput(0),
                *biasLayer->getOutput(0),
                ElementWiseOperation::kSUM
            );
            excitationLayer->setName((layer.name + ".excitation").c_str());
            // (inputs + excitation)
            auto mergeLayer = network->addElementWise(
                *outputConv,
                *excitationLayer->getOutput(0),
                ElementWiseOperation::kSUM);
            mergeLayer->setName((layer.name + ".merge").c_str());
            // shared_output = tf.nn.relu(inputs + excitation)
            auto outputConvLayer = buildActivationLayer(
                mergeLayer->getOutput(0),
                network,
                layer.name + ".activation.final",
                ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else {
            const auto niter = std::next(iter);
            auto weights = begin(layer.weights);
            if (niter == std::end(m_layers)) {
                if (cfg_head_bn == head_bn_t::GPU_A) {
                    auto conv_val_bias = begin(layer.weights)  + 1;
                    auto ip1_val_weight = begin(layer.weights) + 2;
                    auto ip1_val_bias = begin(layer.weights)   + 3;
                    auto ip2_val_weight = begin(layer.weights) + 4;
                    auto ip2_val_bias = begin(layer.weights)   + 5;
                    // value_conv = tf.layers.conv2d(shared_output, filters=1, kernel_size=1, padding='same', use_bias=False)
                    // value_conv = tf.layers.batch_normalization(value_conv, axis=1, momentum=.95, epsilon=1e-5, center=False, scale=False, fused=True, training=False)
                    auto valueConvLayer = buildConvLayer(
                        outputConv,
                        layer.filter_size,
                        layer.weights_size[0],
                        weights[0],
                        layer.weights_size[1],
                        conv_val_bias[0],
                        network,
                        layer.name + ".conv",
                        layer.outputs);
                    // value_conv = tf.nn.relu(value_conv)
                    auto actValueLayer = buildActivationLayer(
                        valueConvLayer->getOutput(0),
                        network,
                        layer.name + ".act",
                        ActivationType::kRELU);
                    // value_conv = tf.reshape(value_conv, [-1, 1 * go.N * go.N])
                    int32_t const batch = actValueLayer->getOutput(0)->getDimensions().d[0];
                    int32_t const mmInputs = actValueLayer->getOutput(0)->getDimensions().d[1]
                        * actValueLayer->getOutput(0)->getDimensions().d[2]
                        * actValueLayer->getOutput(0)->getDimensions().d[3]; 
                    auto inputReshape = network->addShuffle(*actValueLayer->getOutput(0));
                    inputReshape->setReshapeDimensions(Dims{2, {batch, mmInputs}});
                    inputReshape->setName((layer.name + ".shuffle1").c_str());
                    auto filter1Const =
                        network->addConstant(
                            Dims{2, {NUM_INTERSECTIONS, layer.channels}},
                            {DataType::kFLOAT, ip1_val_weight[0], layer.weights_size[2]}
                        );
                    // value_fc_hidden = tf.layers.dense(value_conv, units=256)
                    auto val1MatMulLayer = network->addMatrixMultiply(
                        *inputReshape->getOutput(0),
                        MatrixOperation::kNONE,
                        *filter1Const->getOutput(0),
                        MatrixOperation::kNONE);
                    val1MatMulLayer->setName((layer.name + ".matmul1").c_str());
                    // value_fc_hidden = tf.layers.dense(value_conv, units=256)
                    auto bias1Const =
                        network->addConstant(
                            Dims{2, {1, layer.channels}},
                            {DataType::kFLOAT, ip1_val_bias[0], layer.weights_size[3]}
                        );
                    auto val1BiasLayer = network->addElementWise(
                        *val1MatMulLayer->getOutput(0),
                        *bias1Const->getOutput(0),
                        ElementWiseOperation::kSUM);
                    val1BiasLayer->setName((layer.name + ".bias1").c_str());
                    // value_fc_hidden = tf.nn.relu(value_fc_hidden)
                    auto ip1ActValueLayer = buildActivationLayer(
                        val1BiasLayer->getOutput(0),
                        network,
                        layer.name + ".ip1act",
                        ActivationType::kRELU);
                    // value_fc_hidden = tf.layers.dense(value_conv, units=1)
                    auto filter2Const =
                        network->addConstant(
                            Dims{2, {layer.channels, 1}},
                            {DataType::kFLOAT, ip2_val_weight[0], layer.weights_size[4]}
                        );
                    auto val2MatMulLayer = network->addMatrixMultiply(
                        *ip1ActValueLayer->getOutput(0),
                        MatrixOperation::kNONE,
                        *filter2Const->getOutput(0),
                        MatrixOperation::kNONE);
                    val2MatMulLayer->setName((layer.name + ".matmul2").c_str());
                    // value_fc_hidden = tf.layers.dense(value_conv, units=1)
                    auto bias2Const =
                        network->addConstant(
                            Dims{2, {1, 1}},
                            {DataType::kFLOAT, ip2_val_bias[0], layer.weights_size[5]}
                        );
                    auto val2BiasLayer = network->addElementWise(
                        *val2MatMulLayer->getOutput(0),
                        *bias2Const->getOutput(0),
                        ElementWiseOperation::kSUM);
                    val2BiasLayer->setName((layer.name + ".bias2").c_str());
                    // value_fc_hidden = tf.reshape(value_fc_hidden, [-1])
                    // value_output = tf.nn.tanh(value_fc_hidden)
                    outValueLayer = buildActivationLayer(
                        val2BiasLayer->getOutput(0),
                        network,
                        layer.name + ".tanh",
                        ActivationType::kTANH);
                } else if (cfg_head_bn == head_bn_t::GPU_B) {
                    auto conv_val_bias = begin(layer.weights) + 1;
                    auto bn_val_stddevs = begin(layer.weights) + 2;
                    auto valueConvLayer = buildConvLayer(
                        outputConv,
                        layer.filter_size,
                        layer.weights_size[0],
                        weights[0],
                        0,
                        nullptr,
                        network,
                        layer.name + ".conv",
                        layer.outputs);
                    auto valueBatchNormLayer = network->addScale(
                        *valueConvLayer->getOutput(0),
                        ScaleMode::kCHANNEL,
                        {DataType::kFLOAT, conv_val_bias[0], static_cast<int64_t>(layer.weights_size[1])},
                        {DataType::kFLOAT, bn_val_stddevs[0], static_cast<int64_t>(layer.weights_size[2])},
                        {DataType::kFLOAT, nullptr, 0});
                    valueBatchNormLayer->setName((layer.name + ".bn").c_str());
                    outValueLayer = buildActivationLayer(
                        valueBatchNormLayer->getOutput(0),
                        network,
                        layer.name + ".act",
                        ActivationType::kRELU);
                } else {
                    outValueLayer = buildConvLayer(
                        outputConv,
                        layer.filter_size,
                        layer.weights_size[0],
                        weights[0],
                        0,
                        nullptr,
                        network,
                        layer.name + ".conv",
                        layer.outputs);
                }
            } else {
                if (cfg_head_bn == head_bn_t::GPU_A) {
                    auto conv_pol_bias = begin(layer.weights) + 1;
                    auto ip_pol_weight = begin(layer.weights) + 2;
                    auto ip_pol_bias = begin(layer.weights)   + 3;
                    // policy_conv = tf.layers.conv2d(shared_output, filters=2, kernel_size=1, padding='same', use_bias=False)
                    // policy_conv = tf.layers.batch_normalization(policy_conv, axis=1, momentum=.95, epsilon=1e-5, center=False, scale=False, fused=True, training=False)
                    auto policyConvLayer = buildConvLayer(
                        outputConv,
                        layer.filter_size,
                        layer.weights_size[0],
                        weights[0],
                        layer.weights_size[1],
                        conv_pol_bias[0],
                        network,
                        layer.name + ".conv",
                        layer.outputs);
                    // policy_conv = tf.nn.relu(policy_conv)
                    auto actPolicyLayer = buildActivationLayer(
                        policyConvLayer->getOutput(0),
                        network,
                        layer.name + ".act",
                        ActivationType::kRELU);
                    // policy_conv = tf.reshape(policy_conv, [-1, 2 * go.N * go.N])
                    int32_t const batch = actPolicyLayer->getOutput(0)->getDimensions().d[0];
                    int32_t const mmInputs = actPolicyLayer->getOutput(0)->getDimensions().d[1]
                        * actPolicyLayer->getOutput(0)->getDimensions().d[2]
                        * actPolicyLayer->getOutput(0)->getDimensions().d[3]; 
                    auto inputReshape = network->addShuffle(*actPolicyLayer->getOutput(0));
                    inputReshape->setReshapeDimensions(Dims{2, {batch, mmInputs}});
                    inputReshape->setName((layer.name + ".shuffle1").c_str());
                    // logits = tf.layers.dense(policy_conv, units=go.N * go.N + 1)
                    auto filterConst =
                        network->addConstant(
                            Dims{2, {POTENTIAL_MOVES, layer.outputs * NUM_INTERSECTIONS}},
                            {DataType::kFLOAT, ip_pol_weight[0], layer.weights_size[2]}
                        );
                    auto polMatMulLayer = network->addMatrixMultiply(
                        *inputReshape->getOutput(0),
                        MatrixOperation::kNONE,
                        *filterConst->getOutput(0),
                        MatrixOperation::kTRANSPOSE
                        );
                    polMatMulLayer->setName((layer.name + ".matmul").c_str());
                    // logits = tf.layers.dense(policy_conv, units=go.N * go.N + 1)
                    auto biasConst =
                        network->addConstant(
                            Dims{2, {1, POTENTIAL_MOVES}},
                            {DataType::kFLOAT, ip_pol_bias[0], layer.weights_size[3]}
                        );
                    auto polBiasLayer = network->addElementWise(
                        *polMatMulLayer->getOutput(0),
                        *biasConst->getOutput(0),
                        ElementWiseOperation::kSUM);
                    polBiasLayer->setName((layer.name + ".bias").c_str());
                    // policy_output = tf.nn.softmax(logits)
                    outPolicyLayer = network->addSoftMax(*polBiasLayer->getOutput(0));	
                    static_cast<ISoftMaxLayer*>(outPolicyLayer)->setAxes(1U << 1);	
                    outPolicyLayer->setName((layer.name + ".softmax").c_str());
                } else if (cfg_head_bn == head_bn_t::GPU_B) {
                    auto conv_pol_bias = begin(layer.weights) + 1;
                    auto bn_pol_stddevs = begin(layer.weights) + 2;
                    auto policyConvLayer = buildConvLayer(
                        outputConv,
                        layer.filter_size,
                        layer.weights_size[0],
                        weights[0],
                        0,
                        nullptr,
                        network,
                        layer.name + ".conv",
                        layer.outputs);
                    auto policyBatchNormLayer = network->addScale(
                        *policyConvLayer->getOutput(0),
                        ScaleMode::kCHANNEL,
                        {DataType::kFLOAT, conv_pol_bias[0], static_cast<int64_t>(layer.weights_size[1])},
                        {DataType::kFLOAT, bn_pol_stddevs[0], static_cast<int64_t>(layer.weights_size[2])},
                        {DataType::kFLOAT, nullptr, 0});
                    policyBatchNormLayer->setName((layer.name + ".bn").c_str());
                    outPolicyLayer = buildActivationLayer(
                        policyBatchNormLayer->getOutput(0),
                        network,
                        layer.name + ".act",
                        ActivationType::kRELU);
                } else {
                    outPolicyLayer = buildConvLayer(
                        outputConv,
                        layer.filter_size,
                        layer.weights_size[0],
                        weights[0],
                        0,
                        nullptr,
                        network,
                        layer.name + ".conv",
                        layer.outputs);
                }
            }
        }
    }
    // Mark the outputs for the network
    auto outputPolicy = outPolicyLayer->getOutput(0);
    network->markOutput(*outputPolicy);
    outputPolicy->setName("OutputPolicy");
    outputPolicy->setType(DataType::kFLOAT);
    outputPolicy->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    auto outputValue = outValueLayer->getOutput(0);
    network->markOutput(*outputValue);
    outputValue->setName("OutputValue");
    outputValue->setType(DataType::kFLOAT);
    outputValue->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    std::cout << "Done constructing network..." << std::endl;
}

template <typename net_t>
ITensor* CuDNN_Network<net_t>::initInputs(
    char const *inputName,
    TrtUniquePtr<INetworkDefinition>& network,
    IOptimizationProfile* profile,
    IOptimizationProfile* profile_n,
    const int channels,
    const int rows,
    const int cols,
    const int batch_size) {

    ITensor* inputFeature;

    std::string_view name_str{inputName};
    if (name_str == "BatchSize") {
        inputFeature
            = network->addInput(inputName,
                                DataType::kINT32,
                                {4, {-1, channels, rows, cols}});
    } else if (typeid(net_t) == typeid(float)) {
        inputFeature
            = network->addInput(inputName,
                                DataType::kFLOAT,
                                {4, {-1, channels, rows, cols}});
    } else {
        inputFeature
            = network->addInput(inputName,
                                DataType::kHALF,
                                {4, {-1, channels, rows, cols}});
    }
    assert(inputFeature != nullptr);
    inputFeature->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    if (cfg_execute_context == execute_t::SINGLE) {
        profile->setDimensions(inputName,
                               OptProfileSelector::kMIN,
                               Dims4(1, channels, rows, cols));
        profile->setDimensions(inputName,
                               OptProfileSelector::kOPT,
                               Dims4(batch_size, channels, rows, cols));
        profile->setDimensions(inputName,
                               OptProfileSelector::kMAX,
                               Dims4(batch_size, channels, rows, cols));
    } else {
        profile->setDimensions(inputName,
                               OptProfileSelector::kMIN,
                               Dims4(1, channels, rows, cols));
        profile->setDimensions(inputName,
                               OptProfileSelector::kOPT,
                               Dims4(1, channels, rows, cols));
        profile->setDimensions(inputName,
                               OptProfileSelector::kMAX,
                               Dims4(1, channels, rows, cols));
        profile_n->setDimensions(inputName,
                                 OptProfileSelector::kMIN,
                                 Dims4(batch_size, channels, rows, cols));
        profile_n->setDimensions(inputName,
                                 OptProfileSelector::kOPT,
                                 Dims4(batch_size, channels, rows, cols));
        profile_n->setDimensions(inputName,
                                 OptProfileSelector::kMAX,
                                 Dims4(batch_size, channels, rows, cols));
    }
    return inputFeature;
}

template <typename net_t>
ILayer* CuDNN_Network<net_t>::buildConvLayer(
    ITensor* input,
    unsigned int filter_size,
    int64_t weights_size,
    void* weights,
    int64_t biases_size,
    void* biases,
    TrtUniquePtr<INetworkDefinition>& network,
    std::string op_name,
    unsigned int outputs) {

    mTuneDesc += strprintf(
        R"|("%s"(%d,%d,%d))|",
        op_name.c_str(),
        filter_size,
        filter_size,
        outputs);

    // For convenience, both I/O tensors have 3 dimentions (in addition to batch), so that
    // matmul is mathmatically equivalent to a 2D convolution of 1x1 features and 1x1 kernels.
    IConvolutionLayer *convLayer;
    convLayer = network->addConvolutionNd(
        *input,
        outputs,
        {2, {filter_size, filter_size}},
        {
            DataType::kFLOAT,
            weights,
            weights_size
        },
        {
            DataType::kFLOAT,
            biases,
            biases_size
        }
    );
    convLayer->setName(op_name.c_str());
    if (filter_size == 1) {
        return convLayer;
    }
    convLayer->setDilationNd({2, {1, 1}});
    convLayer->setPaddingMode(PaddingMode::kSAME_UPPER);
    return convLayer;
}

template <typename net_t>
ILayer* CuDNN_Network<net_t>::buildActivationLayer(
    ITensor* input,
    TrtUniquePtr<INetworkDefinition>& network,
    std::string op_name,
    ActivationType act_type) {

    mTuneDesc += strprintf(
        R"|("%s"(%d))|",
        op_name.c_str(),
        (int)act_type);

    auto activationLayer = network->addActivation(*input, act_type);
    activationLayer->setName(op_name.c_str());
    return activationLayer;
}

template <typename net_t>
ILayer* CuDNN_Network<net_t>::applyGPoolLayer(
    ITensor* input,
    TrtUniquePtr<INetworkDefinition>& network,
    std::string op_name) {

    IPoolingLayer* gpoolMeanLayer
        = network->addPoolingNd(
            *input,
            PoolingType::kAVERAGE,
            DimsHW{BOARD_SIZE, BOARD_SIZE});
    auto gpoolMeanLayerName = op_name + "/gpmean";
    gpoolMeanLayer->setName(gpoolMeanLayerName.c_str());
    return gpoolMeanLayer;
}
#endif

template class CuDNN_Network<float>;
#ifdef USE_HALF
template class CuDNN_Network<__half>;
#endif

#endif

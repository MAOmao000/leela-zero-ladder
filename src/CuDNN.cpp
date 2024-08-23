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
CuDNN<net_t>::CuDNN(const int gpu, const bool silent) {
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
//void CuDNN<net_t>::initialize(const int channels, const int batch_size, const int net_type, const std::string &model_hash) {
void CuDNN<net_t>::initialize(const int channels,
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
            m_handle.emplace_back(cudnn);
            if (net_type == int(NetworkType::MINIGO_SE)) {
                cublasHandle_t cublas;
                checkCUBLAS(cublasCreate(&cublas));
                m_cublas_handles.emplace_back(cublas);
            }
        }
#if defined(USE_TENSOR_RT)
    } else {
        m_model_hash = model_hash;
#endif
    }
#else
#if defined(USE_TENSOR_RT)
    m_model_hash = model_hash;
#endif
#endif
}

#if defined(USE_CUDNN)
template <typename net_t>
void CuDNN<net_t>::convolve(
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
void CuDNN<net_t>::convolveActivation(
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
void CuDNN<net_t>::convolveIdentityActivation(
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
        /* algo           */conv_desc->convolution_identity_algorithm,
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
#endif

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
template <typename net_t>
void CuDNN<net_t>::squeeze_excitation_float(
    cublasHandle_t cublas_handle,
    const void *bufferIn1,   // residual input(before convolve)
    const void *bufferIn2,   // residual output
    void *TempBuffer,
    const void *fc1_weights, // [256, 128]
    const void *fc1_biases,  // [128]
    const void *fc2_weights, // [128, 512]
    const void *fc2_biases,  // [512]
    void *bufferOut,
    void *bufferPool,        // [N, 256, 1, 1]
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
            spatial);
        checkCUDA(cudaGetLastError());
    } else {
        global_average_pooling_float_NHWC(
            (const float *)bufferIn2, // input: residual output
            (float *)bufferPool,      // output: GAP output
            batch_size,
            channels,
            spatial);
        checkCUDA(cudaGetLastError());
    }

    const float alpha = 1.0f;
    const float beta_first = 0.0f;
    const float beta_second = 0.0f;

    // A[channels / 2, channels], B[channels, 1], C[channels / 2, 1]
    checkCUBLAS(cublasSgemmStridedBatched(
        cublas_handle,           // handle: handle to the cuBLAS library context
        CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
        CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
        channels / 2,            // m: number of rows of matrix op(A[i]) and C[i]
        1,                       // n: number of columns of op(B[i]) and C[i]
        channels,                // k: number of columns of op(A[i]) and rows of op(B[i])
        &alpha,                  // alpha: <type> scalar used for multiplication
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
        &beta_first,             // beta: <type> scalar used for multiplication
                                 //       If beta == 0, C does not have to be a valid input
        (float *)bufferOut,      // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                 //    with dimensions ldc x n with ldc>=max(1,m)
                                 //    Matrices C[i] should not overlap; otherwise,
                                 //    undefined behavior is expected
        channels / 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
        channels / 2,            // strideC: Value of type long long int
                                 //          that gives the offset in number of elements between C[i] and C[i+1]
        batch_size));            // batchCount: number of GEMMs to perform in the batch

    add_bias_float((float *)bufferOut,  // in & out: C[1, channels / 2]
                   (float *)fc1_biases, // input: bias[channels / 2]
                   batch_size,
                   channels / 2,
                   true);

    checkCUDA(cudaGetLastError());

    // A[channels * 2, channels / 2], B[channels / 2, 1], C[channels * 2, 1]
    checkCUBLAS(cublasSgemmStridedBatched(
        cublas_handle,           // handle: handle to the cuBLAS library context
        CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
        CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
        channels * 2,            // m: number of rows of matrix op(A[i]) and C[i]
        1,                       // n: number of columns of op(B[i]) and C[i]
        channels / 2,            // k: number of columns of op(A[i]) and rows of op(B[i])
        &alpha,                  // alpha: <type> scalar used for multiplication
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
        &beta_second,            // beta: <type> scalar used for multiplication
                                 //       If beta == 0, C does not have to be a valid input
        (float *)TempBuffer,     // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                 //    with dimensions ldc x n with ldc>=max(1,m)
                                 //    Matrices C[i] should not overlap; otherwise,
                                 //    undefined behavior is expected
        channels * 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
        channels * 2,            // strideC: Value of type long long int
                                 //          that gives the offset in number of elements between C[i] and C[i+1]
        batch_size));            // batchCount: number of GEMMs to perform in the batch

    add_bias_float((float *)TempBuffer, // in & out: C[1, channels * 2]
                   (float *)fc2_biases, // input: bias[channels * 2]
                   batch_size,
                   channels * 2,
                   false);

    checkCUDA(cudaGetLastError());

    if (cfg_NCHW) {
        se_scale_float(
            (float *)bufferOut,  // output: squeeze_excitation output
            (float *)bufferIn2,  // input: residual output
            (float *)TempBuffer, // input: fc2_weights * B + fc2_biases
            (float *)bufferIn1,  // input: residual input(before convolve)
            batch_size,
            channels,
            spatial);
    } else {
        se_scale_float_NHWC(
            (float *)bufferOut,  // output: squeeze_excitation output
            (float *)bufferIn2,  // input: residual output
            (float *)TempBuffer, // input: fc2_weights * B + fc2_biases
            (float *)bufferIn1,  // input: residual input(before convolve)
            batch_size,
            channels,
            spatial);
    }
    checkCUDA(cudaGetLastError());
}

template <typename net_t>
void CuDNN<net_t>::squeeze_excitation_half(
    cublasHandle_t cublas_handle,
    const void *bufferIn1,   // residual input(before convolve)
    const void *bufferIn2,   // residual output
    void *TempBuffer,
    const void *fc1_weights, // [256, 128]
    const void *fc1_biases,  // [128]
    const void *fc2_weights, // [128, 512]
    const void *fc2_biases,  // [512]
    void *bufferOut,
    void *bufferPool,        // [N, 256, 1, 1]
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
            spatial);
        checkCUDA(cudaGetLastError());
    } else {
        global_average_pooling_half_NHWC(
            (const __half *)bufferIn2, // input: residual output
            (__half *)bufferPool,      // output: GAP output
            batch_size,
            channels,
            spatial);
        checkCUDA(cudaGetLastError());
    }

    const __half alpha = __float2half(1.0f);
    const __half beta_first = __float2half(0.0f);
    const __half beta_second = __float2half(0.0f);

    // A[channels / 2, channels], B[channels, 1], C[channels / 2, 1]
    checkCUBLAS(cublasHgemmStridedBatched(
        cublas_handle,           // handle: handle to the cuBLAS library context
        CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
        CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
        channels / 2,            // m: number of rows of matrix op(A[i]) and C[i]
        1,                       // n: number of columns of op(B[i]) and C[i]
        channels,                // k: number of columns of op(A[i]) and rows of op(B[i])
        &alpha,                  // alpha: <type> scalar used for multiplication
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
        &beta_first,             // beta: <type> scalar used for multiplication
                                 //       If beta == 0, C does not have to be a valid input
        (__half *)bufferOut,     // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                 //    with dimensions ldc x n with ldc>=max(1,m)
                                 //    Matrices C[i] should not overlap; otherwise,
                                 //    undefined behavior is expected
        channels / 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
        channels / 2,            // strideC: Value of type long long int
                                 //          that gives the offset in number of elements between C[i] and C[i+1]
        batch_size));            // batchCount: number of GEMMs to perform in the batch

    add_bias_half((__half *)bufferOut,  // in & out: C[1, channels / 2]
                  (__half *)fc1_biases, // input: bias[channels / 2]
                  batch_size,
                  channels / 2,
                  true);

    checkCUDA(cudaGetLastError());

    // A[channels * 2, channels / 2], B[channels / 2, 1], C[channels * 2, 1]
    checkCUBLAS(cublasHgemmStridedBatched(
        cublas_handle,           // handle: handle to the cuBLAS library context
        CUBLAS_OP_N,             // transa: operation op(A[i]) that is non- or (conj.) transpose
        CUBLAS_OP_N,             // transb: operation op(B[i]) that is non- or (conj.) transpose
        channels * 2,            // m: number of rows of matrix op(A[i]) and C[i]
        1,                       // n: number of columns of op(B[i]) and C[i]
        channels / 2,            // k: number of columns of op(A[i]) and rows of op(B[i])
        &alpha,                  // alpha: <type> scalar used for multiplication
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
        &beta_second,            // beta: <type> scalar used for multiplication
                                 //       If beta == 0, C does not have to be a valid input
        (__half *)TempBuffer,    // C: <type>* pointer to the C matrix corresponding to the first instance of the batch,
                                 //    with dimensions ldc x n with ldc>=max(1,m)
                                 //    Matrices C[i] should not overlap; otherwise,
                                 //    undefined behavior is expected
        channels * 2,            // ldc: leading dimension of two-dimensional array used to store each matrix C[i]
        channels * 2,            // strideC: Value of type long long int
                                 //          that gives the offset in number of elements between C[i] and C[i+1]
        batch_size));            // batchCount: number of GEMMs to perform in the batch

    add_bias_half((__half *)TempBuffer, // in & out: C[1, channels * 2]
                  (__half *)fc2_biases, // input: bias[channels * 2]
                  batch_size,
                  channels * 2,
                  false);

    checkCUDA(cudaGetLastError());

    if (cfg_NCHW) {
        se_scale_half(
            (__half *)bufferOut,  // output: squeeze_excitation output
            (__half *)bufferIn2,  // input: residual output
            (__half *)TempBuffer, // input: fc2_weights * B + fc2_biases
            (__half *)bufferIn1,  // input: residual input(before convolve)
            batch_size,
            channels,
            spatial);
    } else {
        se_scale_half_NHWC(
            (__half *)bufferOut,  // output: squeeze_excitation output
            (__half *)bufferIn2,  // input: residual output
            (__half *)TempBuffer, // input: fc2_weights * B + fc2_biases
            (__half *)bufferIn1,  // input: residual input(before convolve)
            batch_size,
            channels,
            spatial);
    }
    checkCUDA(cudaGetLastError());
}
#endif

#if defined(USE_CUDNN_GRAPH)
template <typename net_t>
std::shared_ptr<conv_descriptor> CuDNN<net_t>::convolve_fe_init(
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
    } else { // typeid: __half
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
        Y->set_output(true); // is_virtual = false

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

template <typename net_t>
std::shared_ptr<conv_descriptor> CuDNN<net_t>::convolve_fe_no_relu_init(
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
    } else { // typeid: __half
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

template <typename net_t>
std::shared_ptr<conv_descriptor> CuDNN<net_t>::convolve_fe_add_relu_init(
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
    } else { // typeid: __half
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

template <typename net_t>
std::shared_ptr<conv_descriptor> CuDNN<net_t>::convolve_fe_head_init(
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
    std::shared_ptr<conv_descriptor> conv_desc = std::make_shared<conv_descriptor>();

    if (typeid(net_t) == typeid(float)) {
        data_type = fe::DataType_t::FLOAT;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::FLOAT;
    } else { // typeid: __half
        data_type = fe::DataType_t::HALF;
        compute_type = fe::DataType_t::FLOAT;
        conv_compute_type = fe::DataType_t::HALF;
    }
    auto pad_size = filter_size / 2;
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = fe::graph::Graph();
        graph.set_io_data_type(data_type)
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
        Y->set_output(true); // is_virtual = false

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

#if defined(USE_CUDNN)
template <typename net_t>
std::shared_ptr<conv_descriptor> CuDNN<net_t>::convolve_init(
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
    } else { // typeid: __half
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

        // Only the CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM algo is enabled with CUDNN_ACTIVATION_IDENTITY.
        conv_desc->convolution_identity_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                                         /* handle       */handle,
                                         /* xDesc        */conv_desc->input_descriptor,
                                         /* wDesc        */conv_desc->filter_descriptor,
                                         /* convDesc     */conv_desc->convolution_descriptor,
                                         /* yDesc        */conv_desc->output_descriptor,
                                         /* algo         */conv_desc->convolution_identity_algorithm,
                                         /* *sizeInBytes */&conv_desc->workspace_identity_size));
    }
    return conv_desc;
}
#endif

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
template <typename net_t>
void CuDNN_Network<net_t>::push_weights(
    const size_t layer,
    const std::vector<float>& weights) {

    if (layer >= m_layers.size()) {
        m_layers.emplace_back(CuDNN_Layer());
    }
    if (typeid(net_t) == typeid(float)) {
        auto weightSize = weights.size() * sizeof(float);

        void *device_mem;
        checkCUDA(cudaMalloc((void**)&device_mem, weightSize));
        checkCUDA(cudaMemcpy(device_mem,
                             (net_t*)&weights[0],
                             weightSize,
                             cudaMemcpyHostToDevice));
        m_layers.back().weights.emplace_back(device_mem);
    } else {
        auto converted_weights = std::vector<net_t>();
        for(auto i = size_t{0}; i < weights.size(); i++) {
            converted_weights.emplace_back((net_t)weights[i]);
        }

        auto weightSize = weights.size() * sizeof(net_t);

        void *device_mem;
        checkCUDA(cudaMalloc((void**)&device_mem, weightSize));
        checkCUDA(cudaMemcpy(device_mem,
                             (net_t *)&converted_weights[0],
                             weightSize,
                             cudaMemcpyHostToDevice));
        m_layers.back().weights.emplace_back(device_mem);
    }
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
#endif

#if defined(USE_TENSOR_RT)
template <typename net_t>
void CuDNN_Network<net_t>::push_weights_trt(
    const size_t layer,
    const std::vector<float>& weights) {

    if (layer >= m_layers.size()) {
        m_layers.emplace_back(CuDNN_Layer());
    }
    if (typeid(net_t) == typeid(float)) {
        auto weightSize = weights.size() * sizeof(float);
        void *host_mem;
        cudaHostAlloc((void **)&host_mem, weightSize, cudaHostAllocMapped);
        memcpy(host_mem, (net_t*)&weights[0], weightSize);
        m_layers.back().weights.emplace_back(host_mem);
        m_layers.back().weights_size.emplace_back((int64_t)weights.size());
    } else {
        auto converted_weights = std::vector<net_t>();
        for(auto i = size_t{0}; i < weights.size(); i++) {
            converted_weights.emplace_back((net_t)weights[i]);
        }
        auto weightSize = weights.size() * sizeof(net_t);
        void *host_mem;
        cudaHostAlloc((void **)&host_mem, weightSize, cudaHostAllocMapped);
        memcpy(host_mem, (net_t *)&converted_weights[0], weightSize);
        m_layers.back().weights.emplace_back(host_mem);
        m_layers.back().weights_size.emplace_back((int64_t)weights.size());
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::push_weights_trt_col_major(
    const size_t layer,
    const std::vector<float>& weights,
    const int row,
    const int column) {

    if (layer >= m_layers.size()) {
        m_layers.emplace_back(CuDNN_Layer());
    }
    // Transpose from model's CK to TensorRT's KC
    auto weightSize = weights.size() * sizeof(net_t);
    auto transposed_weights = std::vector<net_t>(weights.size());
    for (int i = 0; i < column; i++) {
        for (int j = 0; j < row; j++) {
            transposed_weights[j * column + i] = (net_t)weights[i * row + j];
        }
    }
    void *host_mem;
    cudaHostAlloc((void **)&host_mem, weightSize, cudaHostAllocMapped);
    memcpy(host_mem, (net_t*)&transposed_weights[0], weightSize);
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
    if (cfg_NCHW) {
#if defined(USE_TENSOR_RT)
        if (cfg_backend == backend_t::TENSORRT) {
            push_weights_trt(layer, weights); // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, biases);  // Here it is still float(Convert precision with push_weights)
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
        } else {
            push_weights(layer, weights); // Here it is still float(Convert precision with push_weights)
            push_weights(layer, biases);  // Here it is still float(Convert precision with push_weights)
#endif
        }
#else
        push_weights(layer, weights); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases);  // Here it is still float(Convert precision with push_weights)
#endif
    } else {
        auto weights_convert = NCHW_to_NHWC<float>(weights, outputs, filter_size, filter_size, channels);
#if defined(USE_TENSOR_RT)
        if (cfg_backend == backend_t::TENSORRT) {
            push_weights_trt(layer, weights_convert); // Convert precision with push_weights
            push_weights_trt(layer, biases);          // Convert precision with push_weights
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
        } else {
            push_weights(layer, weights_convert); // Convert precision with push_weights
            push_weights(layer, biases);          // Convert precision with push_weights
#endif
        }
#else
        push_weights(layer, weights_convert); // Convert precision with push_weights
        push_weights(layer, biases);          // Convert precision with push_weights
#endif
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
        for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
            auto conv_desc_single
                = m_cudnn.convolve_fe_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
            auto conv_desc_multi
                = m_cudnn.convolve_fe_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
            m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
        }
#if defined(USE_CUDNN)
    } else {
#endif
#endif
#if defined(USE_CUDNN)
        for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
            auto conv_desc_single
                = m_cudnn.convolve_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
            m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
            auto conv_desc_multi
                = m_cudnn.convolve_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
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
    if (cfg_NCHW) {
#if defined(USE_TENSOR_RT)
        if (cfg_backend == backend_t::TENSORRT) {
            push_weights_trt(layer, weights_1); // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, weights_2); // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
        } else {
            push_weights(layer, weights_1); // Here it is still float(Convert precision with push_weights)
            push_weights(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
            push_weights(layer, weights_2); // Here it is still float(Convert precision with push_weights)
            push_weights(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
#endif
        }
#else
        push_weights(layer, weights_1); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
        push_weights(layer, weights_2); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
#endif
    } else {
        auto weights_convert_1 = NCHW_to_NHWC<float>(
            weights_1, outputs, filter_size, filter_size, channels);
        auto weights_convert_2 = NCHW_to_NHWC<float>(
            weights_2, outputs, filter_size, filter_size, channels);
#if defined(USE_TENSOR_RT)
        if (cfg_backend == backend_t::TENSORRT) {
            push_weights_trt(layer, weights_convert_1); // Convert precision with push_weights
            push_weights_trt(layer, biases_1);          // Convert precision with push_weights
            push_weights_trt(layer, weights_convert_2); // Convert precision with push_weights
            push_weights_trt(layer, biases_2);          // Convert precision with push_weights
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
        } else {
            push_weights(layer, weights_convert_1); // Convert precision with push_weights
            push_weights(layer, biases_1);          // Convert precision with push_weights
            push_weights(layer, weights_convert_2); // Convert precision with push_weights
            push_weights(layer, biases_2);          // Convert precision with push_weights
#endif
        }
#else
        push_weights(layer, weights_convert_1); // Convert precision with push_weights
        push_weights(layer, biases_1);          // Convert precision with push_weights
        push_weights(layer, weights_convert_2); // Convert precision with push_weights
        push_weights(layer, biases_2);          // Convert precision with push_weights
#endif
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
            for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
                auto conv_desc_single
                    = m_cudnn.convolve_fe_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = m_cudnn.convolve_fe_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
                auto conv_desc_no_relu_single
                    = m_cudnn.convolve_fe_no_relu_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_no_relu_desc_single.emplace_back(conv_desc_no_relu_single);
                auto conv_desc_no_relu_multi
                    = m_cudnn.convolve_fe_no_relu_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_no_relu_desc_multi.emplace_back(conv_desc_no_relu_multi);
                auto conv_desc_add_relu_single
                    = m_cudnn.convolve_fe_add_relu_init(m_cudnn.m_handle[i], channels, outputs);
                m_layers[layer].conv_add_relu_desc_single.emplace_back(conv_desc_add_relu_single);
                auto conv_desc_add_relu_multi
                    = m_cudnn.convolve_fe_add_relu_init(m_cudnn.m_handle[i], channels, outputs, cfg_batch_size);
                m_layers[layer].conv_add_relu_desc_multi.emplace_back(conv_desc_add_relu_multi);
            }
        } else {
#endif
#if defined(USE_CUDNN)
            for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
                auto conv_desc_single
                    = m_cudnn.convolve_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desk_multi
                    = m_cudnn.convolve_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
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
    if (cfg_NCHW) {
#if defined(USE_TENSOR_RT)
        if (cfg_backend == backend_t::TENSORRT) {
            push_weights_trt(layer, weights_1); // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, weights_2); // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, se_fc1_w);
            push_weights_trt(layer, se_fc1_b);
            push_weights_trt(layer, se_fc2_w);
            push_weights_trt(layer, se_fc2_b);
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
        } else {
            push_weights(layer, weights_1); // Here it is still float(Convert precision with push_weights)
            push_weights(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
            push_weights(layer, weights_2); // Here it is still float(Convert precision with push_weights)
            push_weights(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
            push_weights_col_major(layer, se_fc1_w, channels / 2, channels);
            push_weights(layer, se_fc1_b);
            push_weights_col_major(layer, se_fc2_w, channels * 2, channels / 2);
            push_weights(layer, se_fc2_b);
#endif
        }
#else
        push_weights(layer, weights_1); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
        push_weights(layer, weights_2); // Here it is still float(Convert precision with push_weights)
        push_weights(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
        push_weights_col_major(layer, se_fc1_w, channels / 2, channels);
        push_weights(layer, se_fc1_b);
        push_weights_col_major(layer, se_fc2_w, channels * 2, channels / 2);
        push_weights(layer, se_fc2_b);
#endif
    } else {
        auto weights_convert_1 = NCHW_to_NHWC<float>(
            weights_1, outputs, filter_size, filter_size, channels);
        auto weights_convert_2 = NCHW_to_NHWC<float>(
            weights_2, outputs, filter_size, filter_size, channels);
#if defined(USE_TENSOR_RT)
        if (cfg_backend == backend_t::TENSORRT) {
            push_weights_trt(layer, weights_convert_1); // Convert precision with push_weights
            push_weights_trt(layer, biases_1);          // Convert precision with push_weights
            push_weights_trt(layer, weights_convert_2); // Convert precision with push_weights
            push_weights_trt(layer, biases_2);          // Convert precision with push_weights
            push_weights_trt_col_major(layer, se_fc1_w, channels / 2, channels);
            push_weights_trt(layer, se_fc1_b);
            push_weights_trt_col_major(layer, se_fc2_w, channels * 2, channels / 2);
            push_weights_trt(layer, se_fc2_b);
#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
        } else {
            push_weights(layer, weights_convert_1); // Convert precision with push_weights
            push_weights(layer, biases_1);          // Convert precision with push_weights
            push_weights(layer, weights_convert_2); // Convert precision with push_weights
            push_weights(layer, biases_2);          // Convert precision with push_weights
            push_weights_col_major(layer, se_fc1_w, channels / 2, channels);
            push_weights(layer, se_fc1_b);
            push_weights_col_major(layer, se_fc2_w, channels * 2, channels / 2);
            push_weights(layer, se_fc2_b);
#endif
        }
#else
        push_weights(layer, weights_convert_1); // Convert precision with push_weights
        push_weights(layer, biases_1);          // Convert precision with push_weights
        push_weights(layer, weights_convert_2); // Convert precision with push_weights
        push_weights(layer, biases_2);          // Convert precision with push_weights
        push_weights_col_major(layer, se_fc1_w, channels / 2, channels);
        push_weights(layer, se_fc1_b);
        push_weights_col_major(layer, se_fc2_w, channels * 2, channels / 2);
        push_weights(layer, se_fc2_b);
#endif
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
            for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
                auto conv_desc_single
                    = m_cudnn.convolve_fe_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = m_cudnn.convolve_fe_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
                auto conv_desc_no_relu_single
                    = m_cudnn.convolve_fe_no_relu_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_no_relu_desc_single.emplace_back(conv_desc_no_relu_single);
                auto conv_desc_no_relu_multi
                    = m_cudnn.convolve_fe_no_relu_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_no_relu_desc_multi.emplace_back(conv_desc_no_relu_multi);
            }
        } else {
#endif
#if defined(USE_CUDNN)
            for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
                auto conv_desc_single
                    = m_cudnn.convolve_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = m_cudnn.convolve_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
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
    const std::vector<float>& weights) {

    size_t layer = get_layer_count();
#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        if (cfg_NCHW) {
            push_weights_trt(layer, weights); // Here it is still float(Convert precision with push_weights)
        } else {
            auto weights_convert = NCHW_to_NHWC<float>(
                weights, outputs, filter_size, filter_size, channels);
            push_weights_trt(layer, weights_convert); // Convert precision with push_weights
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

        if (build(m_cudnn.m_num_worker_threads, cfg_batch_size)) {
            return;
        }
        exit(EXIT_FAILURE);
    }
#else
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
            for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
                auto conv_desc_single
                    = m_cudnn.convolve_fe_head_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = m_cudnn.convolve_fe_head_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
            }
#if defined(USE_CUDNN)
        } else {
#endif
#endif
#if defined(USE_CUDNN)
            for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
                auto conv_desc_single
                    = m_cudnn.convolve_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = m_cudnn.convolve_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
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
            for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
                auto conv_desc_single
                    = m_cudnn.convolve_fe_head_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                auto conv_desc_multi
                    = m_cudnn.convolve_fe_head_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
                m_layers[layer].conv_desc_multi.emplace_back(conv_desc_multi);
            }
#if defined(USE_CUDNN)
        } else {
#endif
#endif
#if defined(USE_CUDNN)
            for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
                std::shared_ptr<conv_descriptor> conv_desc_single
                    = m_cudnn.convolve_init(m_cudnn.m_handle[i], channels, outputs, filter_size);
                m_layers[layer].conv_desc_single.emplace_back(conv_desc_single);
                std::shared_ptr<conv_descriptor> conv_desc_multi
                    = m_cudnn.convolve_init(m_cudnn.m_handle[i], channels, outputs, filter_size, cfg_batch_size);
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
    std::shared_ptr<CuDNNContext> cudnn_context,
    const int tid,
    const int batch_size) {

#if defined(USE_TENSOR_RT)
    (void)tid;
#endif
    const auto inSize = batch_size * sizeof(net_t) * m_layers[0].channels * NUM_INTERSECTIONS;
    const auto pol_elements
        = batch_size * m_layers[m_layers.size() - 2].outputs * NUM_INTERSECTIONS;
    const auto val_elements
        = batch_size * m_layers.back().outputs * NUM_INTERSECTIONS;
    auto pol_net_t = std::vector<net_t>(pol_elements);
    auto val_net_t = std::vector<net_t>(val_elements);

#if defined(USE_TENSOR_RT)
    if (cfg_backend == backend_t::TENSORRT) {
        auto search = cudnn_context->mBuffers.find("InputFeature");
        assert(search != cudnn_context->mBuffers.end());
        if (typeid(net_t) == typeid(float) && cfg_NCHW) {
            checkCUDA(cudaMemcpyAsync(
                search->second,
                (net_t*)&input[0],
                inSize,
                cudaMemcpyHostToDevice));
        } else if (typeid(net_t) == typeid(half_float::half) && cfg_NCHW) {
            auto input_net_t = std::vector<net_t>(batch_size * m_layers[0].channels * NUM_INTERSECTIONS);
            std::copy(input.begin(), input.end(), input_net_t.begin());
            cudaMemcpyAsync(
                search->second,
                (net_t*)&input_net_t[0],
                inSize,
                cudaMemcpyHostToDevice);
        } else {
            auto input_net_t = std::vector<net_t>(batch_size * m_layers[0].channels * NUM_INTERSECTIONS);
            input_net_t = NCHW_to_NHWC<net_t>(
                input, batch_size, BOARD_SIZE, BOARD_SIZE, m_layers[0].channels);
            cudaMemcpyAsync(
                search->second,
                (net_t*)&input_net_t[0],
                inSize,
                cudaMemcpyHostToDevice);
        }
        if (cfg_execute_context == execute_t::SINGLE || batch_size == 1) {
            cudnn_context->mContext->setInputShape("InputFeature",
                nvinfer1::Dims4(batch_size, m_layers[0].channels, BOARD_SIZE, BOARD_SIZE));
        } else {
            cudnn_context->mContext_n->setInputShape("InputFeature",
                nvinfer1::Dims4(batch_size, m_layers[0].channels, BOARD_SIZE, BOARD_SIZE));
        }
        if (m_cudnn.m_net_type == int(NetworkType::MINIGO_SE)) {
            if (cfg_execute_context == execute_t::SINGLE || batch_size == 1) {
                cudnn_context->mContext->setInputShape("BatchSize",
                    nvinfer1::Dims({4, {(unsigned int)batch_size, m_layers[1].channels, 1, 1}}));
            } else {
                cudnn_context->mContext_n->setInputShape("BatchSize",
                    nvinfer1::Dims({4, {(unsigned int)batch_size, m_layers[1].channels, 1, 1}}));
            }
        }
        // Asynchronously enqueue the inference work
        if (cfg_execute_context == execute_t::SINGLE || batch_size == 1) {
            ASSERT(cudnn_context->mContext->enqueueV3(cudaStreamPerThread));
        } else {
            ASSERT(cudnn_context->mContext_n->enqueueV3(cudaStreamPerThread));
        }
        search = cudnn_context->mBuffers.find("OutputPolicy");
        assert(search != cudnn_context->mBuffers.end());
        checkCUDA(cudaMemcpy(
            &pol_net_t[0],
            search->second,
            pol_elements * sizeof(net_t),
            cudaMemcpyDeviceToHost));
        search = cudnn_context->mBuffers.find("OutputValue");
        assert(search != cudnn_context->mBuffers.end());
        checkCUDA(cudaMemcpy(
            &val_net_t[0],
            search->second,
            val_elements * sizeof(net_t),
            cudaMemcpyDeviceToHost));

        if (cfg_NCHW) {
            std::copy(val_net_t.begin(), val_net_t.end(), output_val.begin()); 
            std::copy(pol_net_t.begin(), pol_net_t.end(), output_pol.begin());
        } else {
            output_val = NHWC_to_NCHW<net_t>(
                val_net_t, batch_size, BOARD_SIZE, BOARD_SIZE, Network::OUTPUTS_VALUE);
            output_pol = NHWC_to_NCHW<net_t>(
                pol_net_t, batch_size, BOARD_SIZE, BOARD_SIZE, Network::OUTPUTS_POLICY);
        }
        return;
    }
#endif

#if defined(USE_CUDNN) || defined(USE_CUDNN_GRAPH)
    // input: input(float) 18 chanels * (BOARD_SIZE * BOARD_SIZE)

    // Always allocates enough space for floats
    constexpr auto one_plane = NUM_INTERSECTIONS * sizeof(float);

    if (!cudnn_context->m_buffers_allocated) {
        auto max_wsize = size_t{0};
        auto max_channels = unsigned{0};
        int layer_i = 0;
        for (const auto& layer : m_layers) {
            for (auto i = 0; i < m_cudnn.m_num_worker_threads; i++) {
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
                    if (m_cudnn.m_net_type == int(NetworkType::MINIGO_SE)) {
                        max_wsize = std::max(max_wsize,
                                             layer.conv_desc_single[i]->workspace_identity_size);
                        max_wsize = std::max(max_wsize,
                                             layer.conv_desc_multi[i]->workspace_identity_size);
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

        cudnn_context->m_workspace = d_workspace;
        cudnn_context->m_InBuffer = d_InBuffer;
        cudnn_context->m_OutBuffer = d_OutBuffer;
        cudnn_context->m_TempBuffer = d_TempBuffer;
        cudnn_context->m_buffers_allocated = true;

        if (m_cudnn.m_net_type == int(NetworkType::MINIGO_SE)) {
            void *d_IdentityOutBuffer;
            checkCUDA(cudaMalloc((void**)&d_IdentityOutBuffer, alloc_insize));

            void *d_PoolBuffer;
            checkCUDA(cudaMalloc((void**)&d_PoolBuffer,
                                 cfg_batch_size * max_channels * sizeof(net_t)));

            cudnn_context->m_IdentityOutBuffer = d_IdentityOutBuffer;
            cudnn_context->m_PoolBuffer = d_PoolBuffer;
        }
    }

    auto workspace = cudnn_context->m_workspace;
    auto InBuffer = cudnn_context->m_InBuffer;
    auto OutBuffer = cudnn_context->m_OutBuffer;
    auto IdentityOutBuffer = cudnn_context->m_IdentityOutBuffer;
    auto PoolBuffer = cudnn_context->m_PoolBuffer;
    auto TempBuffer = cudnn_context->m_TempBuffer;

    if (typeid(net_t) == typeid(float) && cfg_NCHW) {
        checkCUDA(cudaMemcpy(InBuffer, (net_t*)&input[0], inSize, cudaMemcpyHostToDevice));
    } else if (typeid(net_t) == typeid(half_float::half) && cfg_NCHW) {
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
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                        {layer.conv_desc_single[tid]->X, InBuffer},
                        {layer.conv_desc_single[tid]->W, conv_weights[0]},
                        {layer.conv_desc_single[tid]->B, conv_biases[0]},
                        {layer.conv_desc_single[tid]->Y, OutBuffer} };
                    checkCUDNNFE(layer.conv_desc_single[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                            variant_pack, workspace));
                } else {
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                        {layer.conv_desc_multi[tid]->X, InBuffer},
                        {layer.conv_desc_multi[tid]->W, conv_weights[0]},
                        {layer.conv_desc_multi[tid]->B, conv_biases[0]},
                        {layer.conv_desc_multi[tid]->Y, OutBuffer} };
                    checkCUDNNFE(layer.conv_desc_multi[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                           variant_pack, workspace));
                }
#if defined(USE_CUDNN)
            } else {
#endif
#endif
#if defined(USE_CUDNN)
                if (batch_size == 1) {
                    m_cudnn.convolveActivation(tid,
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
                    m_cudnn.convolveActivation(tid,
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
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                        {m_layers[1].conv_desc_single[tid]->X, OutBuffer},
                        {m_layers[1].conv_desc_single[tid]->W, conv1_weights[0]},
                        {m_layers[1].conv_desc_single[tid]->B, conv1_biases[0]},
                        {m_layers[1].conv_desc_single[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_desc_single[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                                  variant_pack1, workspace));
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                        {m_layers[1].conv_no_relu_desc_single[tid]->X, InBuffer},
                        {m_layers[1].conv_no_relu_desc_single[tid]->W, conv2_weights[0]},
                        {m_layers[1].conv_no_relu_desc_single[tid]->B, conv2_biases[0]},
                        {m_layers[1].conv_no_relu_desc_single[tid]->Y, TempBuffer} };
                    checkCUDNNFE(m_layers[1].conv_no_relu_desc_single[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                                          variant_pack2, workspace));
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack3 = {
                        {m_layers[1].conv_add_relu_desc_single[tid]->X, TempBuffer},
                        {m_layers[1].conv_add_relu_desc_single[tid]->Z, OutBuffer},
                        {m_layers[1].conv_add_relu_desc_single[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_add_relu_desc_single[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                                           variant_pack3, workspace));
                } else {
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                        {m_layers[1].conv_desc_multi[tid]->X, OutBuffer},
                        {m_layers[1].conv_desc_multi[tid]->W, conv1_weights[0]},
                        {m_layers[1].conv_desc_multi[tid]->B, conv1_biases[0]},
                        {m_layers[1].conv_desc_multi[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_desc_multi[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                                 variant_pack1, workspace));
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                        {m_layers[1].conv_no_relu_desc_multi[tid]->X, InBuffer},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->W, conv2_weights[0]},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->B, conv2_biases[0]},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->Y, TempBuffer} };
                    checkCUDNNFE(m_layers[1].conv_no_relu_desc_multi[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                                         variant_pack2, workspace));
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack3 = {
                        {m_layers[1].conv_add_relu_desc_multi[tid]->X, TempBuffer},
                        {m_layers[1].conv_add_relu_desc_multi[tid]->Z, OutBuffer},
                        {m_layers[1].conv_add_relu_desc_multi[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_add_relu_desc_multi[tid]->graph.execute(getCuDNN().m_handle[tid],
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
                    m_cudnn.convolveActivation(tid,
                                               OutBuffer,
                                               InBuffer,
                                               conv1_weights[0],
                                               nullptr,
                                               conv1_biases[0],
                                               workspace,
                                               m_layers[1].conv_desc_single[tid],
                                               layer.scale_1,
                                               1.0f);

                    m_cudnn.convolveActivation(tid,
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
                    m_cudnn.convolveActivation(tid,
                                               OutBuffer,
                                               InBuffer,
                                               conv1_weights[0],
                                               nullptr,
                                               conv1_biases[0],
                                               workspace,
                                               m_layers[1].conv_desc_multi[tid],
                                               layer.scale_1,
                                               1.0f);

                    m_cudnn.convolveActivation(tid,
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
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                        {m_layers[1].conv_desc_single[tid]->X, OutBuffer},
                        {m_layers[1].conv_desc_single[tid]->W, conv1_weights[0]},
                        {m_layers[1].conv_desc_single[tid]->B, conv1_biases[0]},
                        {m_layers[1].conv_desc_single[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_desc_single[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                                  variant_pack1, workspace));
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                        {m_layers[1].conv_no_relu_desc_single[tid]->X, InBuffer},
                        {m_layers[1].conv_no_relu_desc_single[tid]->W, conv2_weights[0]},
                        {m_layers[1].conv_no_relu_desc_single[tid]->B, conv2_biases[0]},
                        {m_layers[1].conv_no_relu_desc_single[tid]->Y, TempBuffer} };
                    checkCUDNNFE(m_layers[1].conv_no_relu_desc_single[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                                          variant_pack2, workspace));
                } else {
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                        {m_layers[1].conv_desc_multi[tid]->X, OutBuffer},
                        {m_layers[1].conv_desc_multi[tid]->W, conv1_weights[0]},
                        {m_layers[1].conv_desc_multi[tid]->B, conv1_biases[0]},
                        {m_layers[1].conv_desc_multi[tid]->Y, InBuffer} };
                    checkCUDNNFE(m_layers[1].conv_desc_multi[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                                 variant_pack1, workspace));
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                        {m_layers[1].conv_no_relu_desc_multi[tid]->X, InBuffer},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->W, conv2_weights[0]},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->B, conv2_biases[0]},
                        {m_layers[1].conv_no_relu_desc_multi[tid]->Y, TempBuffer} };
                    checkCUDNNFE(m_layers[1].conv_no_relu_desc_multi[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                                         variant_pack2, workspace));
                }

                std::swap(TempBuffer, IdentityOutBuffer);
#if defined(USE_CUDNN)
            } else {
#endif
#endif
#if defined(USE_CUDNN)
                if (batch_size == 1) {
                    m_cudnn.convolveActivation(tid,
                                               OutBuffer,        // *bufferIn
                                               InBuffer,         // *bufferOut
                                               conv1_weights[0],
                                               nullptr,
                                               conv1_biases[0],
                                               workspace,
                                               m_layers[1].conv_desc_single[tid],
                                               layer.scale_1,
                                               1.0f);

                    m_cudnn.convolveIdentityActivation(tid,
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
                    m_cudnn.convolveActivation(tid,
                                               OutBuffer,        // *bufferIn
                                               InBuffer,         // *bufferOut
                                               conv1_weights[0],
                                               nullptr,
                                               conv1_biases[0],
                                               workspace,
                                               m_layers[1].conv_desc_multi[tid],
                                               layer.scale_1,
                                               1.0f);

                    m_cudnn.convolveIdentityActivation(tid,
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
                m_cudnn.squeeze_excitation_float(getCuDNN().m_cublas_handles[tid],
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
                m_cudnn.squeeze_excitation_half(getCuDNN().m_cublas_handles[tid],
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
#endif
            // input: OutBuffer(net_t is float or __half)
#if defined(USE_CUDNN_GRAPH)
            if (cfg_backend == backend_t::CUDNNGRAPH) {
                if (batch_size == 1) {
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                        {layer.conv_desc_single[tid]->X, OutBuffer},
                        {layer.conv_desc_single[tid]->W, layer.weights[0]},
                        {layer.conv_desc_single[tid]->Y, InBuffer} };
                    checkCUDNNFE(layer.conv_desc_single[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                            variant_pack, workspace));
                } else {
                    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                        {layer.conv_desc_multi[tid]->X, OutBuffer},
                        {layer.conv_desc_multi[tid]->W, layer.weights[0]},
                        {layer.conv_desc_multi[tid]->Y, InBuffer} };
                    checkCUDNNFE(layer.conv_desc_multi[tid]->graph.execute(getCuDNN().m_handle[tid],
                                                                           variant_pack, workspace));
                }
#if defined(USE_CUDNN)
            } else {
#endif
#endif
#if defined(USE_CUDNN)
                if (batch_size == 1) {
                    m_cudnn.convolve(tid,
                                     OutBuffer,
                                     InBuffer,
                                     layer.weights[0],
                                     workspace,
                                     layer.conv_desc_single[tid],
                                     layer.scale_1);
                } else {
                    m_cudnn.convolve(tid,
                                     OutBuffer,
                                     InBuffer,
                                     layer.weights[0],
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
                                     val_elements * sizeof(net_t), cudaMemcpyDeviceToHost));
                // output: val_net_t
            } else {
                // Policy input: InBuffer
                checkCUDA(cudaMemcpy(&pol_net_t[0], InBuffer,
                                     pol_elements * sizeof(net_t), cudaMemcpyDeviceToHost));
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

    forward_activations(input, output_pol, output_val, m_context[tid], tid, batch_size);
}

template <typename net_t>
bool CuDNN<net_t>::has_fp16_compute() {
    return m_fp16_compute;
}

template <typename net_t>
bool CuDNN<net_t>::has_tensor_cores() {
    return m_tensorcore;
}

#if defined(USE_TENSOR_RT)
template <typename net_t>
bool CuDNN_Network<net_t>::build(
    const int num_worker_threads,
    const int batch_size) {

    // Bump this when between program versions we want to forcibly drop old timing caches and plan caches.
    if (typeid(net_t) == typeid(float)) {
        mTuneDesc = strprintf(
            R"|("salt"(%s%s)"model float"(%s,%d,%d,%d))|",
            PROGRAM_VERSION_MAJOR,
            PROGRAM_VERSION_MINOR,
            "1.0",                    // modelVersion,
            Network::INPUT_CHANNELS,  // numInputChannels,
            cfg_execute_context,
            cfg_execute_context == execute_t::MULTI ? num_worker_threads : batch_size);
    } else {
        mTuneDesc = strprintf(
            R"|("salt"(%s%s)"model half"(%s,%d,%d,%d))|",
            PROGRAM_VERSION_MAJOR,
            PROGRAM_VERSION_MINOR,
            "1.0",                    // modelVersion,
            Network::INPUT_CHANNELS,  // numInputChannels,
            cfg_execute_context,
            cfg_execute_context == execute_t::MULTI ? num_worker_threads : batch_size);
    }
    //Logger logger;
    auto builder
        = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*m_logger));
    if (!builder) {
        std::cerr << "TensorRT backend: failed to create builder" << std::endl;
        return false;
    }
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "TensorRT backend: failed to create builder config" << std::endl;
        return false;
    }
    bool usingFP16 = false;
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        usingFP16 = true;
    }
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);

    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<int>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
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

    if (getCuDNN().m_device_prop.major >= 8) {
        // This is to avoid tactics that have shape switching overhead
        config->setTacticSources(1U << static_cast<uint32_t>(nvinfer1::TacticSource::kJIT_CONVOLUTIONS));
        config->setBuilderOptimizationLevel(2);
    }
    // So that there are no concurrent kernel executions probably from other parts of code while profiling
    // See CUDA Runtime API document for more details related to NULL stream and synchronization behaviors
    config->setProfileStream(cudaStreamLegacy);
    // Typical runtime allocation is much less than the 1 GiB specified below
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);

    std::string plan;
    {
        static std::mutex tuneMutex;
        tuneMutex.lock();
        std::string cacheDir = Utils::leelaz_file("trtcache");
        std::filesystem::create_directory(cacheDir);
        assert(std::filesystem::exists(cacheDir));
        assert(std::filesystem::is_directory(cacheDir));

        uint8_t deviceHash[32];
        SHA2::get256(getCuDNN().m_device_prop.name, deviceHash);

        // Truncated to 4 bytes
        char deviceIdent[4 * 2 + 1];
        for(int i = 0; i < 4; i++) {
            sprintf(deviceIdent + i * 2, "%02x", static_cast<unsigned char>(deviceHash[i]));
        }
        deviceIdent[sizeof(deviceIdent) - 1] = 0;

        std::string precision = typeid(net_t) == typeid(float) ? "single" : "half";
        std::string sep_char{std::filesystem::path::preferred_separator};
        if (cfg_cache_plan) {
            auto planCacheFile = strprintf(
                "%s%strt-%d_gpu-%s_net-%s_%s%s_%dx%d_%d_batch%d_fp%d_%s",
                cacheDir.c_str(),
                sep_char.c_str(),
                getInferLibVersion(),
                deviceIdent,
                network->getName(),
                PROGRAM_VERSION_MAJOR,
                PROGRAM_VERSION_MINOR,
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_execute_context,
                cfg_execute_context == execute_t::MULTI ? num_worker_threads : batch_size,
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
                cfg_execute_context == execute_t::MULTI ? num_worker_threads : batch_size,
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
                    if (modelHash != getCuDNN().m_model_hash) {
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
                auto planBuffer = std::unique_ptr<nvinfer1::IHostMemory>(
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
                if (getCuDNN().m_model_hash.size() != 64) {
                    std::cerr << "Unexpected model hash size" << std::endl;
                    return false;
                }
                plan.insert(
                    plan.end(),
                    getCuDNN().m_model_hash.begin(),
                    getCuDNN().m_model_hash.end()
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
            uint8_t tuneHash[32];
            SHA2::get256(mTuneDesc.c_str(), tuneHash);
            // Truncated to 6 bytes
            char tuneIdent[6 * 2 + 1];
            for(int i = 0; i < 6; i++) {
                sprintf(tuneIdent + i * 2, "%02x", static_cast<unsigned char>(tuneHash[i]));
            }
            tuneIdent[sizeof(tuneIdent) - 1] = 0;

            auto timingCacheFile = strprintf(
                "%s%strt-%d_gpu-%s_tune-%s_%dx%d_%d_batch%d_fp%d_%s",
                cacheDir.c_str(),
                sep_char.c_str(),
                getInferLibVersion(),
                deviceIdent,
                tuneIdent,
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_execute_context,
                cfg_execute_context == execute_t::MULTI ? num_worker_threads : batch_size,
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
                std::unique_ptr<nvinfer1::ITimingCache>(
                    config->createTimingCache(timingCacheBlob.data(), timingCacheBlob.size()));
            auto invalidTimingCache = !config->setTimingCache(*timingCache, false);
            if (invalidTimingCache) {
                std::cout << "Invalid timing cache, using new one instead" << std::endl;
                timingCache.reset(config->createTimingCache(nullptr, 0));
                config->setTimingCache(*timingCache, false);
            }

            std::unique_ptr<nvinfer1::IHostMemory> planBuffer;
            if (invalidTimingCache || !timingCacheBlob.size()) {
                planBuffer.reset(builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    std::cerr << "TensorRT backend: failed to create plan" << std::endl;
                    return false;
                }
                auto serializedTimingCache = std::unique_ptr<nvinfer1::IHostMemory>(
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
        std::shared_ptr<nvinfer1::IRuntime> runtime
            = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_logger));
        if (!runtime) {
            std::cerr << "createInferRuntime error: " << std::endl;
            return false;
        }
        std::shared_ptr<nvinfer1::ICudaEngine> engine
            = std::shared_ptr<nvinfer1::ICudaEngine>(
                runtime->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine) {
            std::cerr << "deserializeCudaEngine error: " << std::endl;
            return false;
        }
        std::shared_ptr<CuDNNContext> context = std::make_shared<CuDNNContext>();
        context->mContext.reset(engine->createExecutionContext());
        if (cfg_execute_context == execute_t::DOUBLE) {
            context->mContext_n.reset(engine->createExecutionContext());
        }
        for (auto i = 0; i < engine->getNbIOTensors(); i++) {
            void* buffer = nullptr;
            auto name = engine->getIOTensorName(i);
            auto dims = engine->getTensorShape(name);
            std::string_view name_str{name};
            size_t size_byte = (name_str == "BatchSize") ? sizeof(int32_t) : sizeof(net_t);
            size_t bytes = std::accumulate(dims.d + 1,
                                           dims.d + dims.nbDims,
                                           batch_size * size_byte,
                                           std::multiplies<size_t>());
            checkCUDA(cudaMalloc(&buffer, bytes));
            if (name_str == "BatchSize") {
                auto input_batch = std::vector<int32_t>(batch_size * m_layers[1].channels, 0);
                checkCUDA(cudaMemcpyAsync(
                    buffer,
                    (int32_t*)&input_batch[0],
                    bytes,
                    cudaMemcpyHostToDevice));
            }
            context->mBuffers.emplace(std::make_pair(name, buffer));
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
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
        mRuntime.emplace_back(runtime);
        mEngine.emplace_back(engine);
        m_context.emplace_back(context);
    }
    return true;
}

template <typename net_t>
void CuDNN_Network<net_t>::constructNetwork(
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    nvinfer1::IOptimizationProfile* profile,
    nvinfer1::IOptimizationProfile* profile_n,
    const int batch_size) {

    nvinfer1::ITensor* inputFeature = nullptr;
    nvinfer1::ILayer* initialConvLayer = nullptr;
    nvinfer1::ITensor* outputConv = nullptr;
    nvinfer1::ILayer* policyConvLayer = nullptr;
    nvinfer1::ILayer* valueConvLayer = nullptr;
    nvinfer1::ILayer* shapeLayer = nullptr;
    nvinfer1::IShapeLayer* inShapeLayer = nullptr;
    nvinfer1::ICastLayer* castLayer = nullptr;
    nvinfer1::ISliceLayer* gammaLayer = nullptr;
    nvinfer1::ISliceLayer* biasLayer = nullptr;
    nvinfer1::ITensor* batchSizeLayer = nullptr;

    if (m_cudnn.m_net_type == int(NetworkType::MINIGO_SE)) {
        batchSizeLayer
            = network->addInput(
                "BatchSize",
                nvinfer1::DataType::kINT32,
                {nvinfer1::Dims{4, {-1, m_layers[1].channels, 1, 1}}});
        batchSizeLayer->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
        if (cfg_execute_context == execute_t::SINGLE) {
            profile->setDimensions("BatchSize",
                                   nvinfer1::OptProfileSelector::kMIN,
                                   nvinfer1::Dims({4,
                                       {1, m_layers[1].channels, 1, 1}}));
            profile->setDimensions("BatchSize",
                                   nvinfer1::OptProfileSelector::kOPT,
                                   nvinfer1::Dims({4,
                                       {(unsigned int)batch_size, m_layers[1].channels, 1, 1}}));
            profile->setDimensions("BatchSize",
                                   nvinfer1::OptProfileSelector::kMAX,
                                   nvinfer1::Dims({4,
                                       {(unsigned int)batch_size, m_layers[1].channels, 1, 1}}));
        } else {
            profile->setDimensions("BatchSize",
                                   nvinfer1::OptProfileSelector::kMIN,
                                   nvinfer1::Dims({4,
                                       {1, m_layers[1].channels, 1, 1}}));
            profile->setDimensions("BatchSize",
                                   nvinfer1::OptProfileSelector::kOPT,
                                   nvinfer1::Dims({4,
                                       {1, m_layers[1].channels, 1, 1}}));
            profile->setDimensions("BatchSize",
                                   nvinfer1::OptProfileSelector::kMAX,
                                   nvinfer1::Dims({4,
                                       {1, m_layers[1].channels, 1, 1}}));
            profile_n->setDimensions("BatchSize",
                                     nvinfer1::OptProfileSelector::kMIN,
                                     nvinfer1::Dims({4,
                                         {(unsigned int)batch_size, m_layers[1].channels, 1, 1}}));
            profile_n->setDimensions("BatchSize",
                                     nvinfer1::OptProfileSelector::kOPT,
                                     nvinfer1::Dims({4,
                                         {(unsigned int)batch_size, m_layers[1].channels, 1, 1}}));
            profile_n->setDimensions("BatchSize",
                                     nvinfer1::OptProfileSelector::kMAX,
                                     nvinfer1::Dims({4,
                                         {(unsigned int)batch_size, m_layers[1].channels, 1, 1}}));
        }

        // See. https://github.com/NVIDIA/TensorRT/issues/2282
        inShapeLayer = network->addShape(*batchSizeLayer);
        castLayer = network->addCast(*inShapeLayer->getOutput(0), nvinfer1::DataType::kINT32);

        shapeLayer = network->addUnary(
            *castLayer->getOutput(0),
            nvinfer1::UnaryOperation::kABS);
    }

    for (auto iter = std::begin(m_layers);
         iter != std::end(m_layers); iter++) {

        const auto& layer = *iter;
        if (layer.is_input_convolution) {
            inputFeature = initInputs(network, layer, profile, profile_n, batch_size);
            auto conv_weights = begin(layer.weights);
            auto conv_biases = begin(layer.weights) + 1;
            initialConvLayer = buildConvLayer(
                inputFeature,
                layer.filter_size,
                layer.weights_size[0],
                conv_weights[0],
                layer.weights_size[1],
                conv_biases[0],
                network,
                layer.name + ".conv",
                layer.channels,
                layer.outputs);
            auto outputConvLayer = buildActivationLayer(
                initialConvLayer->getOutput(0),
                network,
                layer.name + ".activation",
                nvinfer1::ActivationType::kRELU);
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
                layer.channels,
                layer.outputs);
            auto firstActivationConvLayer = buildActivationLayer(
                firstConvLayer->getOutput(0),
                network,
                layer.name + ".activation.first",
                nvinfer1::ActivationType::kRELU);
            auto secondConvLayer = buildConvLayer(
                firstActivationConvLayer->getOutput(0),
                layer.filter_size,
                layer.weights_size[2],
                conv2_weights[0],
                layer.weights_size[3],
                conv2_biases[0],
                network,
                layer.name + ".conv.second",
                layer.channels,
                layer.outputs);
            auto mergeLayer = network->addElementWise(
                *outputConv, *secondConvLayer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
            mergeLayer->setName((layer.name + ".merge").c_str());
            auto outputConvLayer = buildActivationLayer(
                mergeLayer->getOutput(0),
                network,
                layer.name + ".activation.final",
                nvinfer1::ActivationType::kRELU);
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
                layer.channels,
                layer.outputs);
            auto firstActivationConvLayer = buildActivationLayer(
                firstConvLayer->getOutput(0),
                network,
                layer.name + ".activation.first",
                nvinfer1::ActivationType::kRELU);
            auto secondConvLayer = buildConvLayer(
                firstActivationConvLayer->getOutput(0),
                layer.filter_size,
                layer.weights_size[2],
                conv2_weights[0],
                layer.weights_size[3],
                conv2_biases[0],
                network,
                layer.name + ".conv.second",
                layer.channels,
                layer.outputs);
            auto gpoolLayer = applyGPoolLayer(
                secondConvLayer->getOutput(0),
                network,
                layer.name + ".gpool");
            auto thirdMatMulLayer = buildMatMulLayer(
                gpoolLayer->getOutput(0),
                layer.weights_size[4],
                fc1_weights[0],
                layer.weights_size[5],
                fc1_biases[0],
                network,
                layer.name + ".conv.third",
                layer.channels,
                layer.outputs / 2);
            auto thirdActivationMatLayer = buildActivationLayer(
                thirdMatMulLayer->getOutput(0),
                network,
                layer.name + ".activation.third",
                nvinfer1::ActivationType::kRELU);
            auto fourthMatMulLayer = buildMatMulLayer(
                thirdActivationMatLayer->getOutput(0),
                layer.weights_size[6],
                fc2_weights[0],
                layer.weights_size[7],
                fc2_biases[0],
                network,
                layer.name + ".conv.fourth",
                layer.channels / 2,
                layer.outputs * 2);

                gammaLayer = network->addSlice(
                    *fourthMatMulLayer->getOutput(0),
                    {4 ,{0, 0, 0, 0}},
                    {4 ,{0, layer.channels, 1, 1}},
                    {4 ,{1, 1, 1, 1}}
                );
                gammaLayer->setInput(2, *shapeLayer->getOutput(0));
            gammaLayer->setName((layer.name + ".gamma").c_str());

                biasLayer = network->addSlice(
                    *fourthMatMulLayer->getOutput(0),
                    {4 ,{0, layer.channels, 0, 0}},
                    {4 ,{0, layer.channels, 1, 1}},
                    {4 ,{1, 1, 1, 1}}
                );
                biasLayer->setInput(2, *shapeLayer->getOutput(0));
            biasLayer->setName((layer.name + ".bias").c_str());

            auto sigLayer = buildActivationLayer(
                gammaLayer->getOutput(0),
                network,
                layer.name + ".activation.sig",
                nvinfer1::ActivationType::kSIGMOID);
            sigLayer->setName((layer.name + ".sig").c_str());

            auto scaleLayer = network->addElementWise(
                *sigLayer->getOutput(0),
                *secondConvLayer->getOutput(0),
                nvinfer1::ElementWiseOperation::kPROD
            );
            scaleLayer->setName((layer.name + ".scale").c_str());
            auto excitationLayer = network->addElementWise(
                *scaleLayer->getOutput(0),
                *biasLayer->getOutput(0),
                nvinfer1::ElementWiseOperation::kSUM
            );
            excitationLayer->setName((layer.name + ".excitation").c_str());
            auto mergeLayer = network->addElementWise(
                *outputConv,
                *excitationLayer->getOutput(0),
                nvinfer1::ElementWiseOperation::kSUM);
            mergeLayer->setName((layer.name + ".merge").c_str());
            auto outputConvLayer = buildActivationLayer(
                mergeLayer->getOutput(0),
                network,
                layer.name + ".activation.final",
                nvinfer1::ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else {
            const auto niter = std::next(iter);
            if (niter == std::end(m_layers)) {
                valueConvLayer = buildConvLayer(
                    outputConv,
                    layer.filter_size,
                    layer.weights_size[0],
                    layer.weights[0],
                    0,
                    nullptr,
                    network,
                    layer.name + ".value",
                    layer.channels,
                    layer.outputs);
            } else {
                policyConvLayer = buildConvLayer(
                    outputConv,
                    layer.filter_size,
                    layer.weights_size[0],
                    layer.weights[0],
                    0,
                    nullptr,
                    network,
                    layer.name + ".policy",
                    layer.channels,
                    layer.outputs);
            }
        }
    }
    // Mark the outputs for the network
    auto outputPolicy = policyConvLayer->getOutput(0);
    network->markOutput(*outputPolicy);
    outputPolicy->setName("OutputPolicy");
    if (typeid(net_t) == typeid(float)) {
        outputPolicy->setType(nvinfer1::DataType::kFLOAT);
    } else {
        outputPolicy->setType(nvinfer1::DataType::kHALF);
    }
    outputPolicy->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    auto outputValue = valueConvLayer->getOutput(0);
    network->markOutput(*outputValue);
    outputValue->setName("OutputValue");
    if (typeid(net_t) == typeid(float)) {
        outputValue->setType(nvinfer1::DataType::kFLOAT);
    } else {
        outputValue->setType(nvinfer1::DataType::kHALF);
    }
    outputValue->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    std::cout << "Done constructing network..." << std::endl;
}

template <typename net_t>
nvinfer1::ITensor* CuDNN_Network<net_t>::initInputs(
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    const CuDNN_Layer layer,
    nvinfer1::IOptimizationProfile* profile,
    nvinfer1::IOptimizationProfile* profile_n,
    const int batch_size) {

    auto numInChannels = layer.channels;
    auto nnYLen = BOARD_SIZE;
    auto nnXLen = BOARD_SIZE;
    nvinfer1::ITensor* inputFeature;

    if (typeid(net_t) == typeid(float)) {
        inputFeature
            = network->addInput("InputFeature",
                                nvinfer1::DataType::kFLOAT,
                                {4, {-1, numInChannels, nnYLen, nnXLen}});
    } else {
        inputFeature
            = network->addInput("InputFeature",
                                nvinfer1::DataType::kHALF,
                                {4, {-1, numInChannels, nnYLen, nnXLen}});
    }
    assert(inputFeature != nullptr);
    inputFeature->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));

    if (cfg_execute_context == execute_t::SINGLE) {
        profile->setDimensions("InputFeature",
                               nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims4(1, numInChannels, nnYLen, nnXLen));
        profile->setDimensions("InputFeature",
                               nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims4(batch_size, numInChannels, nnYLen, nnXLen));
        profile->setDimensions("InputFeature",
                               nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims4(batch_size, numInChannels, nnYLen, nnXLen));
    } else {
        profile->setDimensions("InputFeature",
                               nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims4(1, numInChannels, nnYLen, nnXLen));
        profile->setDimensions("InputFeature",
                               nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims4(1, numInChannels, nnYLen, nnXLen));
        profile->setDimensions("InputFeature",
                               nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims4(1, numInChannels, nnYLen, nnXLen));
        profile_n->setDimensions("InputFeature",
                                 nvinfer1::OptProfileSelector::kMIN,
                                 nvinfer1::Dims4(batch_size, numInChannels, nnYLen, nnXLen));
        profile_n->setDimensions("InputFeature",
                                 nvinfer1::OptProfileSelector::kOPT,
                                 nvinfer1::Dims4(batch_size, numInChannels, nnYLen, nnXLen));
        profile_n->setDimensions("InputFeature",
                                 nvinfer1::OptProfileSelector::kMAX,
                                 nvinfer1::Dims4(batch_size, numInChannels, nnYLen, nnXLen));
    }
    return inputFeature;
}

template <typename net_t>
nvinfer1::ILayer* CuDNN_Network<net_t>::buildConvLayer(
    nvinfer1::ITensor* input,
    unsigned int filter_size,
    int64_t weights_size,
    void* weights,
    int64_t biases_size,
    void* biases,
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    std::string op_name,
    unsigned int channels,
    unsigned int outputs) {

    auto dilationX = 1;
    auto dilationY = 1;

    mTuneDesc += strprintf(
        R"|("%s"(%d,%d,%d,%d,%d,%d))|",
        op_name.c_str(),
        filter_size,
        filter_size,
        channels,
        outputs,
        dilationX,
        dilationY);

    nvinfer1::IConvolutionLayer *convLayer;
    if (biases_size > 0) {
        if (typeid(net_t) == typeid(float)) {
            convLayer = network->addConvolutionNd(
                *input,
                outputs,
                {2, {filter_size, filter_size}},
                {
                    nvinfer1::DataType::kFLOAT,
                    weights,
                    weights_size
                },
                {
                    nvinfer1::DataType::kFLOAT,
                    biases,
                    biases_size
                }
            );
        } else {
            convLayer = network->addConvolutionNd(
                *input,
                outputs,
                {2, {filter_size, filter_size}},
                {
                    nvinfer1::DataType::kHALF,
                    weights,
                    weights_size
                },
                {
                    nvinfer1::DataType::kHALF,
                    biases,
                    biases_size
                }
            );
        }
    } else {
        if (typeid(net_t) == typeid(float)) {
            convLayer = network->addConvolutionNd(
                *input,
                outputs,
                {2, {filter_size, filter_size}},
                {
                    nvinfer1::DataType::kFLOAT,
                    weights,
                    weights_size
                },
                {nvinfer1::DataType::kFLOAT, nullptr, 0}
            );
        } else {
            convLayer = network->addConvolutionNd(
                *input,
                outputs,
                {2, {filter_size, filter_size}},
                {
                    nvinfer1::DataType::kHALF,
                    weights,
                    weights_size
                },
                {nvinfer1::DataType::kHALF, nullptr, 0}
            );
        }
    }
    convLayer->setDilationNd({2, {dilationY, dilationX}});
    convLayer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    convLayer->setName(op_name.c_str());
    return convLayer;
}

template <typename net_t>
nvinfer1::ILayer* CuDNN_Network<net_t>::buildActivationLayer(
    nvinfer1::ITensor* input,
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    std::string op_name,
    nvinfer1::ActivationType act_type) {

    mTuneDesc += strprintf(
        R"|("%s"(%d))|",
        op_name.c_str(),
        (int)act_type);

    auto activationLayer = network->addActivation(*input, act_type);
    activationLayer->setName(op_name.c_str());
    return activationLayer;
}

template <typename net_t>
nvinfer1::ILayer* CuDNN_Network<net_t>::applyGPoolLayer(
    nvinfer1::ITensor* input,
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    std::string op_name) {

    nvinfer1::IPoolingLayer* gpoolMeanLayer
        = network->addPoolingNd(
            *input,
            nvinfer1::PoolingType::kAVERAGE,
            nvinfer1::DimsHW{BOARD_SIZE, BOARD_SIZE});
    auto gpoolMeanLayerName = op_name + "/gpmean";
    gpoolMeanLayer->setName(gpoolMeanLayerName.c_str());
    return gpoolMeanLayer;
}

template <typename net_t>
nvinfer1::ILayer* CuDNN_Network<net_t>::buildMatMulLayer(
    nvinfer1::ITensor* input,
    int64_t weights_size,
    void* weights,
    int64_t biases_size,
    void* biases,
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    std::string op_name,
    unsigned int channels,
    unsigned int outputs) {

    mTuneDesc += strprintf(
        R"|("%s"(%d,%d))|",
        op_name.c_str(),
        channels,
        outputs);

    // For convenience, both I/O tensors have 3 dimentions (in addition to batch), so that
    // matmul is mathmatically equivalent to a 2D convolution of 1x1 features and 1x1 kernels.
    nvinfer1::IConvolutionLayer *matMulLayer;
    if (typeid(net_t) == typeid(float)) {
        matMulLayer = network->addConvolutionNd(
            *input,
            outputs,
            {2, {1, 1}},
            {
                nvinfer1::DataType::kFLOAT,
                weights,
                weights_size
            },
            {
                nvinfer1::DataType::kFLOAT,
                biases,
                biases_size
            }
        );
    } else {
        matMulLayer = network->addConvolutionNd(
            *input,
            outputs,
            {2, {1, 1}},
            {
                nvinfer1::DataType::kHALF,
                weights,
                weights_size
            },
            {
                nvinfer1::DataType::kHALF,
                biases,
                biases_size
            }
        );
    }
    matMulLayer->setName(op_name.c_str());
    return matMulLayer;
}
#endif

template class CuDNN<float>;
template class CuDNN_Network<float>;
#ifdef USE_HALF
template class CuDNN<half_float::half>;
template class CuDNN_Network<half_float::half>;
#endif

#endif

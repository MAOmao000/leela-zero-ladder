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
//#define CACHE_TENSORRT_PLAN
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
        throw std::runtime_error("No suitable CUDA device found.");
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
CuDNN<net_t>::~CuDNN() {
    cublasDestroy(m_cublas_handles);
    cudnnDestroy(m_handle);
}

template <typename net_t>
void CuDNN<net_t>::initialize(const int channels, const int batch_size, const int net_type, const std::string &model_hash) {
    // For compatibility with OpenCL implementation
    (void)channels;

    putenv("CUDNN_LOGGLEVEL_DBG=0");
    if (cfg_backend == backend_t::CUDNNGRAPH) {
        putenv("CUDNN_FRONTEND_LOG_INFO=0");
    }
    m_net_type = net_type;
    m_batch_size = batch_size;

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cublasHandle_t cublas;
    checkCUBLAS(cublasCreate(&cublas));

    m_handle = cudnn;
    m_cublas_handles = cublas;
    m_model_hash = model_hash;
    m_init_ok = true;
}

template <typename net_t>
void CuDNN<net_t>::convolve(const void *bufferIn,
                            void *bufferOut,
                            const void *weights,
                            void *workspace,
                            const conv_descriptor& conv_desc,
                            const float alpha) {

    const float beta = 0.0f;

    // dstValue = alpha[0]*result + beta[0]*priorDstValue
    checkCUDNN(cudnnConvolutionForward(
        /* handle               */m_handle,
        /* *alpha               */&alpha,
        /* xDesc                */conv_desc.input_descriptor,
        /* *x                   */bufferIn,
        /* wDesc                */conv_desc.filter_descriptor,
        /* *w                   */weights,
        /* convDesc             */conv_desc.convolution_descriptor,
        /* algo                 */conv_desc.convolution_algorithm,
        /* *workSpace           */workspace,
        /* workSpaceSizeInBytes */conv_desc.workspace_size,
        /* *beta                */&beta,
        /* yDesc                */conv_desc.output_descriptor,
        /* *y                   */bufferOut));
}

template <typename net_t>
void CuDNN<net_t>::convolveActivation(const void *bufferIn,
                                      void *bufferOut,
                                      const void *weights,
                                      void *residualBuffer,
                                      const void *biases,
                                      void *workspace,
                                      const conv_descriptor& conv_desc,
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
        /* handle         */m_handle,
        /* *alpha1        */&alpha1,
        /* xDesc          */conv_desc.input_descriptor,
        /* *x             */bufferIn,
        /* wDesc          */conv_desc.filter_descriptor,
        /* *w             */weights,
        /* convDesc       */conv_desc.convolution_descriptor,
        /* algo           */conv_desc.convolution_algorithm,
        /* *workSpace     */workspace,
        /* workSpaceSize  */conv_desc.workspace_size,
        /* *alpha2        */&_alpha2,
        /* zDesc          */conv_desc.output_descriptor,
        /* *z             */residual,
        /* biasDesc       */conv_desc.bias_descriptor,
        /* *bias          */biases,
        /* activationDesc */conv_desc.activation_descriptor,
        /* yDesc          */conv_desc.output_descriptor,
        /* *y             */bufferOut));
}

template <typename net_t>
void CuDNN<net_t>::convolveIdentityActivation(const void *bufferIn,
                                              void *bufferOut,
                                              const void *weights,
                                              void *residualBuffer,
                                              const void *biases,
                                              void *workspace,
                                              const conv_descriptor& conv_desc,
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
        /* handle         */m_handle,
        /* *alpha1        */&alpha1,
        /* xDesc          */conv_desc.input_descriptor,
        /* *x             */bufferIn,
        /* wDesc          */conv_desc.filter_descriptor,
        /* *w             */weights,
        /* convDesc       */conv_desc.convolution_descriptor,
        /* algo           */conv_desc.convolution_identity_algorithm,
        /* *workSpace     */workspace,
        /* workSpaceSize  */conv_desc.workspace_size,
        /* *alpha2        */&_alpha2,
        /* zDesc          */conv_desc.output_descriptor,
        /* *z             */residual,
        /* biasDesc       */conv_desc.bias_descriptor,
        /* *bias          */biases,
        /* activationDesc */conv_desc.activation_identity_descriptor,
        /* yDesc          */conv_desc.output_descriptor,
        /* *y             */bufferOut));
}

template <typename net_t>
void CuDNN<net_t>::squeeze_excitation_float(const void *bufferIn1,   // residual input(before convolve)
                                            const void *bufferIn2,   // residual output
                                            void *TempBuffer,
                                            const void *fc1_weights, // [256, 128]
                                            const void *fc1_biases,  // [128]
                                            const void *fc2_weights, // [128, 512]
                                            const void *fc2_biases,  // [512]
                                            void *bufferOut,
                                            void *bufferPool,        // [N, 256, 19, 19]
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
        m_cublas_handles,        // handle: handle to the cuBLAS library context
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
        m_cublas_handles,        // handle: handle to the cuBLAS library context
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
void CuDNN<net_t>::squeeze_excitation_half(const void *bufferIn1,   // residual input(before convolve)
                                           const void *bufferIn2,   // residual output
                                           void *TempBuffer,
                                           const void *fc1_weights, // [256, 128]
                                           const void *fc1_biases,  // [128]
                                           const void *fc2_weights, // [128, 512]
                                           const void *fc2_biases,  // [512]
                                           void *bufferOut,
                                           void *bufferPool,        // [N, 256, 19, 19]
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
        m_cublas_handles,        // handle: handle to the cuBLAS library context
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
        m_cublas_handles,        // handle: handle to the cuBLAS library context
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

#if defined(USE_CUDNN_GRAPH)
template <typename net_t>
void CuDNN<net_t>::convolve_fe_init(const int channels,
                                    const int outputs,
                                    const int filter_size,
                                    conv_descriptor& conv_desc,
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

    auto [graph, X, W, B, Y] = build_new_graph(m_handle);
    conv_desc.graph = graph;
    conv_desc.X = X;
    conv_desc.W = W;
    conv_desc.B = B;
    conv_desc.Y = Y;
    conv_desc.workspace_size = graph.get_workspace_size();
}

template <typename net_t>
void CuDNN<net_t>::convolve_fe_no_relu_init(const int channels,
                                            const int outputs,
                                            const int filter_size,
                                            conv_descriptor& conv_desc,
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

    auto [graph, X, W, B, Y] = build_new_graph(m_handle);
    conv_desc.graph = graph;
    conv_desc.X = X;
    conv_desc.W = W;
    conv_desc.B = B;
    conv_desc.Y = Y;
    conv_desc.workspace_size = graph.get_workspace_size();
}

template <typename net_t>
void CuDNN<net_t>::convolve_fe_add_relu_init(const int channels,
                                             const int outputs,
                                             conv_descriptor& conv_desc,
                                             const int batch_size) {

    int64_t n = batch_size;
    int64_t c = channels;
    int64_t h = BOARD_SIZE;
    int64_t w = BOARD_SIZE;
    int64_t k = outputs;

    fe::DataType_t data_type;
    fe::DataType_t compute_type;
    fe::DataType_t intermediate_type;
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

    auto [graph, X, Z, Y] = build_new_graph(m_handle);
    conv_desc.graph = graph;
    conv_desc.X = X;
    conv_desc.Z = Z;
    conv_desc.Y = Y;
    conv_desc.workspace_size = graph.get_workspace_size();
}

template <typename net_t>
void CuDNN<net_t>::convolve_fe_head_init(const int channels,
                                         const int outputs,
                                         const int filter_size,
                                         conv_descriptor& conv_desc,
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

    auto [graph, X, W, Y] = build_new_graph(m_handle);
    conv_desc.graph = graph;
    conv_desc.X = X;
    conv_desc.W = W;
    conv_desc.Y = Y;
    conv_desc.workspace_size = graph.get_workspace_size();
}
#endif

template <typename net_t>
void CuDNN<net_t>::convolve_init(const int channels,
                                 const int outputs,
                                 const int filter_size,
                                 conv_descriptor& conv_desc,
                                 const int batch_size) {

    cudnnDataType_t data_type;
    cudnnDataType_t compute_type;
    cudnnTensorFormat_t tensor_format;

    if (typeid(net_t) == typeid(float)) {
        data_type = CUDNN_DATA_FLOAT;
        compute_type = CUDNN_DATA_FLOAT;
        tensor_format = cfg_NCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    } else { // typeid: __half
        data_type = CUDNN_DATA_HALF;
        compute_type = CUDNN_DATA_HALF;
        tensor_format = cfg_NCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    }

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
                      /* tensorDesc     */conv_desc.input_descriptor,
                      /* format         */tensor_format,
                      /* dataType       */data_type,
                      /* N batch_size   */batch_size,
                      /* C channels     */channels,
                      /* H image_height */BOARD_SIZE,
                      /* W image_width  */BOARD_SIZE));

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
                      /* tensorDesc     */conv_desc.output_descriptor,
                      /* format         */tensor_format,
                      /* dataType       */data_type,
                      /* N batch_size   */batch_size,
                      /* C channels     */outputs,
                      /* H image_height */BOARD_SIZE,
                      /* W image_width  */BOARD_SIZE));

    checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.bias_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
                      /* tensorDesc     */conv_desc.bias_descriptor,
                      /* format         */tensor_format,
                      /* dataType       */data_type,
                      /* N number of images=*/1, // Not the batch_size
                      /* C channels         */outputs,
                      /* H image_height     */1,
                      /* W image_width      */1));

    checkCUDNN(cudnnCreateFilterDescriptor(&conv_desc.filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
                      /* filterDesc     */conv_desc.filter_descriptor,
                      /* dataType       */data_type,
                      /* format         */tensor_format,
                      /* K Number of output feature maps */outputs,
                      /* C Number of input feature maps  */channels,
                      /* R Number of rows per filter     */filter_size,
                      /* S Number of columns per filter  */filter_size));

    checkCUDNN(cudnnCreateActivationDescriptor(&conv_desc.activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(
                      /* activationDesc   */conv_desc.activation_descriptor,
                      /* mode             */CUDNN_ACTIVATION_RELU,
                      /* reluNanOpt       */CUDNN_NOT_PROPAGATE_NAN,
                      /* coef             */0.));

    auto pad_size = filter_size / 2;

    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc.convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
                 /* convDesc                 */conv_desc.convolution_descriptor,
                 /* zero-padding height      */pad_size,
                 /* zero-padding width       */pad_size,
                 /* vertical filter stride   */1,
                 /* horizontal filter stride */1,
                 /* filter height dilation   */1,
                 /* filter width dilation    */1,
                 /* mode                     */CUDNN_CROSS_CORRELATION,
                 /* computeType              */compute_type));
    checkCUDNN(cudnnSetConvolutionGroupCount(conv_desc.convolution_descriptor, 8));
    checkCUDNN(cudnnSetConvolutionMathType(
                             /* convDesc */conv_desc.convolution_descriptor,
                             /* mathType */CUDNN_TENSOR_OP_MATH));

    using perf_t = cudnnConvolutionFwdAlgoPerf_t;
    int num_algos = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(m_handle, &num_algos));
    int returned_algo_count = 0;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
                              /* handle             */m_handle,
                              /* xDesc              */conv_desc.input_descriptor,
                              /* wDesc              */conv_desc.filter_descriptor,
                              /* convDesc           */conv_desc.convolution_descriptor,
                              /* yDesc              */conv_desc.output_descriptor,
                              /* requestedAlgoCount */num_algos,
                              /* *returnedAlgoCount */&returned_algo_count,
                              /* *perfResults       */perf_results.get()));

    conv_desc.convolution_algorithm = perf_results[0].algo;

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                                     /* handle       */m_handle,
                                     /* xDesc        */conv_desc.input_descriptor,
                                     /* wDesc        */conv_desc.filter_descriptor,
                                     /* convDesc     */conv_desc.convolution_descriptor,
                                     /* yDesc        */conv_desc.output_descriptor,
                                     /* algo         */conv_desc.convolution_algorithm,
                                     /* *sizeInBytes */&conv_desc.workspace_size));

    if (m_net_type == int(NetworkType::MINIGO_SE)) {
        checkCUDNN(cudnnCreateActivationDescriptor(&conv_desc.activation_identity_descriptor));
        checkCUDNN(cudnnSetActivationDescriptor(
                            /* activationDesc */conv_desc.activation_identity_descriptor,
                            /* mode           */CUDNN_ACTIVATION_IDENTITY,
                            /* reluNanOpt     */CUDNN_NOT_PROPAGATE_NAN,
                            /* coef           */0.));

        // Only the CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM algo is enabled with CUDNN_ACTIVATION_IDENTITY.
        conv_desc.convolution_identity_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                                         /* handle       */m_handle,
                                         /* xDesc        */conv_desc.input_descriptor,
                                         /* wDesc        */conv_desc.filter_descriptor,
                                         /* convDesc     */conv_desc.convolution_descriptor,
                                         /* yDesc        */conv_desc.output_descriptor,
                                         /* algo         */conv_desc.convolution_identity_algorithm,
                                         /* *sizeInBytes */&conv_desc.workspace_identity_size));
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::push_weights(const size_t layer,
                                        const std::vector<float>& weights) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(CuDNN_Layer());
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
void CuDNN_Network<net_t>::push_weights_trt(const size_t layer,
                                            const std::vector<float>& weights) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(CuDNN_Layer());
    }
    if (typeid(net_t) == typeid(float)) {
        auto weightSize = weights.size() * sizeof(float);

        void *host_mem;
        host_mem = malloc(weightSize);
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
        host_mem = malloc(weightSize);
        memcpy(host_mem, (net_t *)&converted_weights[0], weightSize);
        m_layers.back().weights.emplace_back(host_mem);
        m_layers.back().weights_size.emplace_back((int64_t)weights.size());
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::push_weights_col_major(const size_t layer,
                                                  const std::vector<float>& weights,
                                                  const int row,
                                                  const int column) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(CuDNN_Layer());
    }

    auto weightSize = weights.size() * sizeof(net_t);
    auto converted_weights = std::vector<net_t>(weights.size());
    for (int i = 0; i < column; i++) {
        for (int j = 0; j < row; j++) {
            converted_weights[i * row + j] = (net_t)weights[i + j * column];
        }
    }
    void *device_mem;
    checkCUDA(cudaMalloc((void**)&device_mem, weightSize));
    checkCUDA(cudaMemcpy(device_mem,
                         (net_t *)&converted_weights[0],
                         weightSize,
                         cudaMemcpyHostToDevice));
    m_layers.back().weights.emplace_back(device_mem);
}

template <typename net_t>
void CuDNN_Network<net_t>::push_weights_trt_col_major(const size_t layer,
                                                      const std::vector<float>& weights,
                                                      const int row,
                                                      const int column) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(CuDNN_Layer());
    }

    auto weightSize = weights.size() * sizeof(net_t);
    auto converted_weights = std::vector<net_t>(weights.size());
    for (int i = 0; i < column; i++) {
        for (int j = 0; j < row; j++) {
            converted_weights[i * row + j] = (net_t)weights[i + j * column];
        }
    }
    void *host_mem;
    host_mem = malloc(weightSize);
    memcpy(host_mem, (net_t*)&converted_weights[0], weightSize);
    m_layers.back().weights.emplace_back(host_mem);
    m_layers.back().weights_size.emplace_back((int64_t)weights.size());
}

template <typename net_t>
void CuDNN_Network<net_t>::push_input_convolution(const unsigned int filter_size,
                                                  unsigned int channels,
                                                  const unsigned int outputs,
                                                  const std::vector<float>& weights,
                                                  const std::vector<float>& biases,
                                                  const float scale) {
    size_t layer = get_layer_count();
    if (cfg_backend == backend_t::TENSORRT) {
        if (cfg_NCHW) {
            push_weights_trt(layer, weights); // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, biases);  // Here it is still float(Convert precision with push_weights)
        } else {
            auto weights_convert = NCHW_to_NHWC<float>(weights, outputs, filter_size, filter_size, channels);
            push_weights_trt(layer, weights_convert); // Convert precision with push_weights
            push_weights_trt(layer, biases);          // Convert precision with push_weights
        }
        m_layers[layer].is_input_convolution = true;
        m_layers[layer].outputs = outputs;
        m_layers[layer].filter_size = filter_size;
        m_layers[layer].channels = channels;
        m_layers[layer].name = "in." + std::to_string(layer);
        return;
    }
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
    m_layers[layer].scale_1 = 1.0f / scale;
    m_layers[layer].scale_2 = 1.0f / scale;
    m_layers[layer].scale_3 = 1.0f;

    if (!m_conv_desc[CONV_DESC_INPUT][SINGLE_BATCH].is_initialized) {
        if (cfg_backend == backend_t::CUDNNGRAPH) {
            m_cudnn.convolve_fe_init(channels, outputs, filter_size,
                                     m_conv_desc[CONV_DESC_INPUT][SINGLE_BATCH]);
            m_cudnn.convolve_fe_init(channels, outputs, filter_size,
                                     m_conv_desc[CONV_DESC_INPUT][MULTIPLE_BATCHES],
                                     cfg_batch_size);
        } else {
            m_cudnn.convolve_init(channels, outputs, filter_size,
                                  m_conv_desc[CONV_DESC_INPUT][SINGLE_BATCH]);
            m_cudnn.convolve_init(channels, outputs, filter_size,
                                  m_conv_desc[CONV_DESC_INPUT][MULTIPLE_BATCHES],
                                  cfg_batch_size);
        }
        m_conv_desc[CONV_DESC_INPUT][SINGLE_BATCH].is_initialized = true;
        m_conv_desc[CONV_DESC_INPUT][MULTIPLE_BATCHES].is_initialized = true;
    }
    m_layers[layer].conv_desc[SINGLE_BATCH]
        = m_conv_desc[CONV_DESC_INPUT][SINGLE_BATCH];
    m_layers[layer].conv_desc[MULTIPLE_BATCHES]
        = m_conv_desc[CONV_DESC_INPUT][MULTIPLE_BATCHES];
}

template <typename net_t>
void CuDNN_Network<net_t>::push_residual(const unsigned int filter_size,
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
    if (cfg_backend == backend_t::TENSORRT) {
        if (cfg_NCHW) {
            push_weights_trt(layer, weights_1); // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, weights_2); // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
        } else {
            auto weights_convert_1 = NCHW_to_NHWC<float>(
                weights_1, outputs, filter_size, filter_size, channels);
            auto weights_convert_2 = NCHW_to_NHWC<float>(
                weights_2, outputs, filter_size, filter_size, channels);
            push_weights_trt(layer, weights_convert_1); // Convert precision with push_weights
            push_weights_trt(layer, biases_1);          // Convert precision with push_weights
            push_weights_trt(layer, weights_convert_2); // Convert precision with push_weights
            push_weights_trt(layer, biases_2);          // Convert precision with push_weights
        }
        m_layers[layer].is_residual_block = true;
        m_layers[layer].outputs = outputs;
        m_layers[layer].filter_size = filter_size;
        m_layers[layer].channels = channels;
        m_layers[layer].name = "res." + std::to_string(layer);
        return;
    }
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
    m_layers[layer].scale_1 = 1.0f / scale_1;
    m_layers[layer].scale_2 = 1.0f / scale_2;
    m_layers[layer].scale_3 = 1.0f / scale_3;

    if (!m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH].is_initialized) {
        if (cfg_backend == backend_t::CUDNNGRAPH) {
            m_cudnn.convolve_fe_init(channels, outputs, filter_size,
                                     m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH]);
            m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH].is_initialized = true;
            m_cudnn.convolve_fe_init(channels, outputs, filter_size,
                                     m_conv_desc[CONV_DESC_RESIDUAL][MULTIPLE_BATCHES],
                                     cfg_batch_size);
            m_conv_desc[CONV_DESC_RESIDUAL][MULTIPLE_BATCHES].is_initialized = true;

            m_cudnn.convolve_fe_no_relu_init(channels, outputs, filter_size,
                                             m_conv_desc[CONV_DESC_NO_RELU][SINGLE_BATCH]);
            m_conv_desc[CONV_DESC_NO_RELU][SINGLE_BATCH].is_initialized = true;
            m_cudnn.convolve_fe_no_relu_init(channels, outputs, filter_size,
                                             m_conv_desc[CONV_DESC_NO_RELU][MULTIPLE_BATCHES],
                                             cfg_batch_size);
            m_conv_desc[CONV_DESC_NO_RELU][MULTIPLE_BATCHES].is_initialized = true;

            m_cudnn.convolve_fe_add_relu_init(channels, outputs,
                                              m_conv_desc[CONV_DESC_ADD_RELU][SINGLE_BATCH]);
            m_conv_desc[CONV_DESC_ADD_RELU][SINGLE_BATCH].is_initialized = true;
            m_cudnn.convolve_fe_add_relu_init(channels, outputs,
                                              m_conv_desc[CONV_DESC_ADD_RELU][MULTIPLE_BATCHES],
                                              cfg_batch_size);
            m_conv_desc[CONV_DESC_ADD_RELU][MULTIPLE_BATCHES].is_initialized = true;
        } else {
            m_cudnn.convolve_init(channels, outputs, filter_size,
                                  m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH]);
            m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH].is_initialized = true;
            m_cudnn.convolve_init(channels, outputs, filter_size,
                                  m_conv_desc[CONV_DESC_RESIDUAL][MULTIPLE_BATCHES],
                                  cfg_batch_size);
            m_conv_desc[CONV_DESC_RESIDUAL][MULTIPLE_BATCHES].is_initialized = true;
        }
    }

    m_layers[layer].conv_desc[SINGLE_BATCH]
        = m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH];
    m_layers[layer].conv_desc[MULTIPLE_BATCHES]
        = m_conv_desc[CONV_DESC_RESIDUAL][MULTIPLE_BATCHES];
    if (cfg_backend == backend_t::CUDNNGRAPH) {
        m_layers[layer].conv_no_relu_desc[SINGLE_BATCH]
            = m_conv_desc[CONV_DESC_NO_RELU][SINGLE_BATCH];
        m_layers[layer].conv_no_relu_desc[MULTIPLE_BATCHES]
            = m_conv_desc[CONV_DESC_NO_RELU][MULTIPLE_BATCHES];
        m_layers[layer].conv_add_relu_desc[SINGLE_BATCH]
            = m_conv_desc[CONV_DESC_ADD_RELU][SINGLE_BATCH];
        m_layers[layer].conv_add_relu_desc[MULTIPLE_BATCHES]
            = m_conv_desc[CONV_DESC_ADD_RELU][MULTIPLE_BATCHES];
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::push_residual_se(const unsigned int filter_size,
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
    size_t layer = get_layer_count();
    if (cfg_backend == backend_t::TENSORRT) {
        if (cfg_NCHW) {
            push_weights_trt(layer, weights_1); // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, biases_1);  // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, weights_2); // Here it is still float(Convert precision with push_weights)
            push_weights_trt(layer, biases_2);  // Here it is still float(Convert precision with push_weights)
            push_weights_trt_col_major(layer, se_fc1_w, channels / 2, channels);
            push_weights_trt(layer, se_fc1_b);
            push_weights_trt_col_major(layer, se_fc2_w, channels * 2, channels / 2);
            push_weights_trt(layer, se_fc2_b);
        } else {
            auto weights_convert_1 = NCHW_to_NHWC<float>(
                weights_1, outputs, filter_size, filter_size, channels);
            auto weights_convert_2 = NCHW_to_NHWC<float>(
                weights_2, outputs, filter_size, filter_size, channels);
            push_weights_trt(layer, weights_convert_1); // Convert precision with push_weights
            push_weights_trt(layer, biases_1);          // Convert precision with push_weights
            push_weights_trt(layer, weights_convert_2); // Convert precision with push_weights
            push_weights_trt(layer, biases_2);          // Convert precision with push_weights
            push_weights_trt_col_major(layer, se_fc1_w, channels / 2, channels);
            push_weights_trt(layer, se_fc1_b);
            push_weights_trt_col_major(layer, se_fc2_w, channels * 2, channels / 2);
            push_weights_trt(layer, se_fc2_b);
        }
        m_layers[layer].is_residual_block = true;
        m_layers[layer].is_se_block = true;
        m_layers[layer].outputs = outputs;
        m_layers[layer].filter_size = filter_size;
        m_layers[layer].channels = channels;
        m_layers[layer].name = "res." + std::to_string(layer);
        return;
    }
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
    m_layers[layer].scale_1 = 1.0f / scale_1;
    m_layers[layer].scale_2 = 1.0f / scale_2;
    m_layers[layer].scale_3 = 1.0f / scale_3;

    if (!m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH].is_initialized) {
        if (cfg_backend == backend_t::CUDNNGRAPH) {
            m_cudnn.convolve_fe_init(channels, outputs, filter_size,
                                     m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH]);
            m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH].is_initialized = true;
            m_cudnn.convolve_fe_init(channels, outputs, filter_size,
                                     m_conv_desc[CONV_DESC_RESIDUAL][MULTIPLE_BATCHES],
                                     cfg_batch_size);
            m_conv_desc[CONV_DESC_RESIDUAL][MULTIPLE_BATCHES].is_initialized = true;

            m_cudnn.convolve_fe_no_relu_init(channels, outputs, filter_size,
                                             m_conv_desc[CONV_DESC_NO_RELU][SINGLE_BATCH]);
            m_conv_desc[CONV_DESC_NO_RELU][SINGLE_BATCH].is_initialized = true;
            m_cudnn.convolve_fe_no_relu_init(channels, outputs, filter_size,
                                             m_conv_desc[CONV_DESC_NO_RELU][MULTIPLE_BATCHES],
                                             cfg_batch_size);
            m_conv_desc[CONV_DESC_NO_RELU][MULTIPLE_BATCHES].is_initialized = true;
        } else {
            m_cudnn.convolve_init(channels, outputs, filter_size,
                                  m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH]);
            m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH].is_initialized = true;
            m_cudnn.convolve_init(channels, outputs, filter_size,
                                  m_conv_desc[CONV_DESC_RESIDUAL][MULTIPLE_BATCHES],
                                  cfg_batch_size);
            m_conv_desc[CONV_DESC_RESIDUAL][MULTIPLE_BATCHES].is_initialized = true;
        }
    }

    m_layers[layer].conv_desc[SINGLE_BATCH]
        = m_conv_desc[CONV_DESC_RESIDUAL][SINGLE_BATCH];
    m_layers[layer].conv_desc[MULTIPLE_BATCHES]
        = m_conv_desc[CONV_DESC_RESIDUAL][MULTIPLE_BATCHES];
    if (cfg_backend == backend_t::CUDNNGRAPH) {
        m_layers[layer].conv_no_relu_desc[SINGLE_BATCH]
            = m_conv_desc[CONV_DESC_NO_RELU][SINGLE_BATCH];
        m_layers[layer].conv_no_relu_desc[MULTIPLE_BATCHES]
            = m_conv_desc[CONV_DESC_NO_RELU][MULTIPLE_BATCHES];
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::push_convolve(const unsigned int filter_size,
                                         const unsigned int channels,
                                         const unsigned int outputs,
                                         const std::vector<float>& weights) {
    size_t layer = get_layer_count();
    if (cfg_backend == backend_t::TENSORRT) {
        if (cfg_NCHW) {
            push_weights_trt(layer, weights); // Here it is still float(Convert precision with push_weights)
        } else {
            auto weights_convert = NCHW_to_NHWC<float>(
                weights, outputs, filter_size, filter_size, channels);
            push_weights_trt(layer, weights_convert); // Convert precision with push_weights
        }
        m_layers[layer].is_convolve1 = true;
        m_layers[layer].outputs = outputs;
        m_layers[layer].filter_size = filter_size;
        m_layers[layer].channels = channels;
        if (outputs != Network::OUTPUTS_VALUE) {
            m_layers[layer].name = "pol." + std::to_string(layer);
            return;
        }
        m_layers[layer].name = "val." + std::to_string(layer);
        m_trt.reset(new TrtResNet<net_t>(*this, m_cudnn));
        if (!m_trt->build()) {
            return;
        }
        return;
    }
    if (cfg_NCHW) {
        push_weights(layer, weights); // Here it is still float(Convert precision with push_weights)
    } else {
        auto weights_convert = NCHW_to_NHWC<float>(
            weights, outputs, filter_size, filter_size, channels);
        push_weights(layer, weights_convert); // Convert precision with push_weights
    }
    m_layers[layer].is_convolve1 = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].channels = channels;
    m_layers[layer].filter_size = filter_size;

    if (outputs == Network::OUTPUTS_VALUE) {
        if (!m_conv_desc[CONV_DESC_VALUE][SINGLE_BATCH].is_initialized) {
            if (cfg_backend == backend_t::CUDNNGRAPH) {
                m_cudnn.convolve_fe_head_init(channels, outputs, filter_size,
                                              m_conv_desc[CONV_DESC_VALUE][SINGLE_BATCH]);
                m_cudnn.convolve_fe_head_init(channels, outputs, filter_size,
                                              m_conv_desc[CONV_DESC_VALUE][MULTIPLE_BATCHES],
                                              cfg_batch_size);
            } else {
                m_cudnn.convolve_init(channels, outputs, filter_size,
                                      m_conv_desc[CONV_DESC_VALUE][SINGLE_BATCH]);
                m_cudnn.convolve_init(channels, outputs, filter_size,
                                      m_conv_desc[CONV_DESC_VALUE][MULTIPLE_BATCHES],
                                      cfg_batch_size);
            }
            m_conv_desc[CONV_DESC_VALUE][SINGLE_BATCH].is_initialized = true;
            m_conv_desc[CONV_DESC_VALUE][MULTIPLE_BATCHES].is_initialized = true;
        }
        m_layers[layer].conv_desc[SINGLE_BATCH]
            = m_conv_desc[CONV_DESC_VALUE][SINGLE_BATCH];
        m_layers[layer].conv_desc[MULTIPLE_BATCHES]
            = m_conv_desc[CONV_DESC_VALUE][MULTIPLE_BATCHES];
    } else {
        if (!m_conv_desc[CONV_DESC_POLICY][SINGLE_BATCH].is_initialized) {
            if (cfg_backend == backend_t::CUDNNGRAPH) {
                m_cudnn.convolve_fe_head_init(channels, outputs, filter_size,
                                              m_conv_desc[CONV_DESC_POLICY][SINGLE_BATCH]);
                m_cudnn.convolve_fe_head_init(channels, outputs, filter_size,
                                              m_conv_desc[CONV_DESC_POLICY][MULTIPLE_BATCHES],
                                              cfg_batch_size);
            } else {
                m_cudnn.convolve_init(channels, outputs, filter_size,
                                      m_conv_desc[CONV_DESC_POLICY][SINGLE_BATCH]);
                m_cudnn.convolve_init(channels, outputs, filter_size,
                                      m_conv_desc[CONV_DESC_POLICY][MULTIPLE_BATCHES],
                                      cfg_batch_size);
            }
            m_conv_desc[CONV_DESC_POLICY][SINGLE_BATCH].is_initialized = true;
            m_conv_desc[CONV_DESC_POLICY][MULTIPLE_BATCHES].is_initialized = true;
        }
        m_layers[layer].conv_desc[SINGLE_BATCH]
            = m_conv_desc[CONV_DESC_POLICY][SINGLE_BATCH];
        m_layers[layer].conv_desc[MULTIPLE_BATCHES]
            = m_conv_desc[CONV_DESC_POLICY][MULTIPLE_BATCHES];
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::forward_activations(const std::vector<float>& input,
                                               std::vector<float>& output_pol,
                                               std::vector<float>& output_val,
                                               CuDNNContext& cudnn_context,
                                               const int batch_size) {

    const auto inSize = batch_size * sizeof(net_t) * m_layers[0].channels * NUM_INTERSECTIONS;
    const auto pol_elements
        = batch_size * m_layers[m_layers.size() - 2].outputs * NUM_INTERSECTIONS;
    const auto val_elements
        = batch_size * m_layers.back().outputs * NUM_INTERSECTIONS;
    auto pol_net_t = std::vector<net_t>(pol_elements);
    auto val_net_t = std::vector<net_t>(val_elements);

    if (cfg_backend == backend_t::TENSORRT) {
        if (!cudnn_context.m_buffers_allocated) {
            cudnn_context.mContext.reset(m_trt->mEngine->createExecutionContext());
            assert(cudnn_context.mContext);
            for (int i = 0; i < m_trt->mEngine->getNbIOTensors(); i++) {
                void* buffer = nullptr;
                auto name = m_trt->mEngine->getIOTensorName(i);
                auto dims = m_trt->mEngine->getTensorShape(name);
                size_t bytes = std::accumulate(dims.d + 1,
                                               dims.d + dims.nbDims,
                                               batch_size * sizeof(net_t),
                                               std::multiplies<size_t>());
                checkCUDA(cudaMalloc(&buffer, bytes));
                cudnn_context.mBuffers.emplace(std::make_pair(name, buffer));
                cudnn_context.mContext->setTensorAddress(name, buffer);
            }
            cudnn_context.mContext->setOptimizationProfileAsync(0, cudaStreamPerThread);
            cudaStreamSynchronize(cudaStreamPerThread);
            cudnn_context.m_buffers_allocated = true;
        }
        auto search = cudnn_context.mBuffers.find("InputFeature");
        assert(search != cudnn_context.mBuffers.end());
        if (typeid(net_t) == typeid(float) && cfg_NCHW) {
            checkCUDA(cudaMemcpyAsync(
                search->second,
                (net_t*)&input[0],
                inSize,
                cudaMemcpyHostToDevice));
        } else if (typeid(net_t) == typeid(half_float::half) && cfg_NCHW) {
            auto input_net_t = std::vector<net_t>(batch_size * m_layers[0].channels * NUM_INTERSECTIONS);
            std::copy(input.begin(), input.end(), input_net_t.begin());
            checkCUDA(cudaMemcpyAsync(
                search->second,
                (net_t*)&input_net_t[0],
                inSize,
                cudaMemcpyHostToDevice));
        } else {
            auto input_net_t = std::vector<net_t>(batch_size * m_layers[0].channels * NUM_INTERSECTIONS);
            input_net_t = NCHW_to_NHWC<net_t>(
                input, batch_size, BOARD_SIZE, BOARD_SIZE, m_layers[0].channels);
            checkCUDA(cudaMemcpyAsync(
                search->second,
                (net_t*)&input_net_t[0],
                inSize,
                cudaMemcpyHostToDevice));
        }
        auto dims = m_trt->mEngine->getTensorShape("InputFeature");
        assert(dims.nbDims != -1);
        dims.d[0] = batch_size;
        cudnn_context.mContext->setInputShape("InputFeature", dims);
        // Asynchronously enqueue the inference work
        ASSERT(cudnn_context.mContext->enqueueV3(cudaStreamPerThread));
        search = cudnn_context.mBuffers.find("OutputPolicy");
        assert(search != cudnn_context.mBuffers.end());
        checkCUDA(cudaMemcpy(
            &pol_net_t[0],
            search->second,
            pol_elements * sizeof(net_t),
            cudaMemcpyDeviceToHost));
        search = cudnn_context.mBuffers.find("OutputValue");
        assert(search != cudnn_context.mBuffers.end());
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
    // input: input(float) 18 chanels * (BOARD_SIZE * BOARD_SIZE)
    int conv_desc_idx = SINGLE_BATCH;
    if (batch_size > 1) {
        conv_desc_idx = MULTIPLE_BATCHES;
    }
    // Always allocates enough space for floats
    constexpr auto one_plane = NUM_INTERSECTIONS * sizeof(float);

    if (!cudnn_context.m_buffers_allocated) {
        auto max_wsize = size_t{0};
        auto max_channels = unsigned{0};
        for (const auto& layer : m_layers) {
            max_wsize = std::max(max_wsize,
                                 layer.conv_desc[conv_desc_idx].workspace_size);
            if (cfg_backend == backend_t::CUDNNGRAPH) {
                max_wsize = std::max(max_wsize,
                                     layer.conv_no_relu_desc[conv_desc_idx].workspace_size);
                max_wsize = std::max(max_wsize,
                                     layer.conv_add_relu_desc[conv_desc_idx].workspace_size);
            }
            if (m_cudnn.m_net_type == int(NetworkType::MINIGO_SE))
                max_wsize = std::max(max_wsize,
                                     layer.conv_desc[conv_desc_idx].workspace_identity_size);
            max_channels = std::max(max_channels,
                                    std::max(layer.channels, layer.outputs));
        }
        auto alloc_insize = batch_size * max_channels * one_plane;

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

        if (m_cudnn.m_net_type == int(NetworkType::MINIGO_SE)) {
            void *d_IdentityOutBuffer;
            checkCUDA(cudaMalloc((void**)&d_IdentityOutBuffer, alloc_insize));

            void *d_PoolBuffer;
            checkCUDA(cudaMalloc((void**)&d_PoolBuffer,
                                 batch_size * max_channels * sizeof(net_t)));

            cudnn_context.m_IdentityOutBuffer = d_IdentityOutBuffer;
            cudnn_context.m_PoolBuffer = d_PoolBuffer;
        }
    }

    auto workspace = cudnn_context.m_workspace;
    auto InBuffer = cudnn_context.m_InBuffer;
    auto OutBuffer = cudnn_context.m_OutBuffer;
    auto IdentityOutBuffer = cudnn_context.m_IdentityOutBuffer;
    auto PoolBuffer = cudnn_context.m_PoolBuffer;
    auto TempBuffer = cudnn_context.m_TempBuffer;

//    const auto inSize = batch_size * sizeof(net_t) * m_layers[0].channels * NUM_INTERSECTIONS;
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

            if (cfg_backend == backend_t::CUDNNGRAPH) {
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                    {layer.conv_desc[conv_desc_idx].X, InBuffer},
                    {layer.conv_desc[conv_desc_idx].W, conv_weights[0]},
                    {layer.conv_desc[conv_desc_idx].B, conv_biases[0]},
                    {layer.conv_desc[conv_desc_idx].Y, OutBuffer} };
                checkCUDNNFE(layer.conv_desc[conv_desc_idx].graph.execute(getCuDNN().m_handle,
                                                                          variant_pack, workspace));
            } else {
                m_cudnn.convolveActivation(InBuffer,
                                           OutBuffer,
                                           conv_weights[0],
                                           nullptr,
                                           conv_biases[0],
                                           workspace,
                                           layer.conv_desc[conv_desc_idx],
                                           layer.scale_1,
                                           1.0f);
            }
            // output: OutBuffer

        } else if (layer.is_residual_block && !layer.is_se_block) {
            // input: OutBuffer
            assert(layer.channels == layer.outputs);
            assert(niter != std::end(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases = begin(layer.weights) + 3;

            if (cfg_backend == backend_t::CUDNNGRAPH) {
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                    {layer.conv_desc[conv_desc_idx].X, OutBuffer},
                    {layer.conv_desc[conv_desc_idx].W, conv1_weights[0]},
                    {layer.conv_desc[conv_desc_idx].B, conv1_biases[0]},
                    {layer.conv_desc[conv_desc_idx].Y, InBuffer} };
                checkCUDNNFE(layer.conv_desc[conv_desc_idx].graph.execute(getCuDNN().m_handle,
                                                                          variant_pack1, workspace));

                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                    {layer.conv_no_relu_desc[conv_desc_idx].X, InBuffer},
                    {layer.conv_no_relu_desc[conv_desc_idx].W, conv2_weights[0]},
                    {layer.conv_no_relu_desc[conv_desc_idx].B, conv2_biases[0]},
                    {layer.conv_no_relu_desc[conv_desc_idx].Y, TempBuffer} };
                checkCUDNNFE(layer.conv_no_relu_desc[conv_desc_idx].graph.execute(getCuDNN().m_handle,
                                                                                  variant_pack2, workspace));

                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack3 = {
                    {layer.conv_add_relu_desc[conv_desc_idx].X, TempBuffer},
                    {layer.conv_add_relu_desc[conv_desc_idx].Z, OutBuffer},
                    {layer.conv_add_relu_desc[conv_desc_idx].Y, InBuffer} };
                checkCUDNNFE(layer.conv_add_relu_desc[conv_desc_idx].graph.execute(getCuDNN().m_handle,
                                                                                   variant_pack3, workspace));
                std::swap(InBuffer, OutBuffer);
            } else {
                m_cudnn.convolveActivation(OutBuffer,
                                           InBuffer,
                                           conv1_weights[0],
                                           nullptr,
                                           conv1_biases[0],
                                           workspace,
                                           layer.conv_desc[conv_desc_idx],
                                           layer.scale_1,
                                           1.0f);

                m_cudnn.convolveActivation(InBuffer,
                                           OutBuffer,
                                           conv2_weights[0],
                                           OutBuffer,          // *residualBuffer: first input
                                           conv2_biases[0],
                                           workspace,
                                           layer.conv_desc[conv_desc_idx],
                                           layer.scale_2,
                                           layer.scale_3);
            }
            // output: OutBuffer

        } else if (layer.is_residual_block && layer.is_se_block) {
            // input: OutBuffer
            assert(layer.channels == layer.outputs);
            assert(niter != std::end(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases = begin(layer.weights) + 3;
            auto fc1_weights = begin(layer.weights) + 4;
            auto fc1_biases = begin(layer.weights) + 5;
            auto fc2_weights = begin(layer.weights) + 6;
            auto fc2_biases = begin(layer.weights) + 7;

            if (cfg_backend == backend_t::CUDNNGRAPH) {
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack1 = {
                    {layer.conv_desc[conv_desc_idx].X, OutBuffer},
                    {layer.conv_desc[conv_desc_idx].W, conv1_weights[0]},
                    {layer.conv_desc[conv_desc_idx].B, conv1_biases[0]},
                    {layer.conv_desc[conv_desc_idx].Y, InBuffer} };
                checkCUDNNFE(layer.conv_desc[conv_desc_idx].graph.execute(getCuDNN().m_handle,
                                                                          variant_pack1, workspace));

                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack2 = {
                    {layer.conv_no_relu_desc[conv_desc_idx].X, InBuffer},
                    {layer.conv_no_relu_desc[conv_desc_idx].W, conv2_weights[0]},
                    {layer.conv_no_relu_desc[conv_desc_idx].B, conv2_biases[0]},
                    {layer.conv_no_relu_desc[conv_desc_idx].Y, TempBuffer} };
                checkCUDNNFE(layer.conv_no_relu_desc[conv_desc_idx].graph.execute(getCuDNN().m_handle,
                                                                                  variant_pack2, workspace));

                std::swap(TempBuffer, IdentityOutBuffer);
            } else {
                m_cudnn.convolveActivation(OutBuffer,        // *bufferIn
                                           InBuffer,         // *bufferOut
                                           conv1_weights[0],
                                           nullptr,
                                           conv1_biases[0],
                                           workspace,
                                           layer.conv_desc[conv_desc_idx],
                                           layer.scale_1,
                                           1.0f);

                m_cudnn.convolveIdentityActivation(InBuffer,          // *bufferIn
                                                   IdentityOutBuffer, // *bufferOut
                                                   conv2_weights[0],
                                                   nullptr,
                                                   conv2_biases[0],
                                                   workspace,
                                                   layer.conv_desc[conv_desc_idx],
                                                   layer.scale_2,
                                                   layer.scale_3);
            }

            if (typeid(net_t) == typeid(float)) {
                m_cudnn.squeeze_excitation_float(OutBuffer,         // *bufferIn1: first input
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
                m_cudnn.squeeze_excitation_half(OutBuffer,         // *bufferIn1: first input
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

        } else {
            // input: OutBuffer(net_t is float or __half)
            assert(layer.is_convolve1);
            // input: OutBuffer
            if (cfg_backend == backend_t::CUDNNGRAPH) {
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                    {layer.conv_desc[conv_desc_idx].X, OutBuffer},
                    {layer.conv_desc[conv_desc_idx].W, layer.weights[0]},
                    {layer.conv_desc[conv_desc_idx].Y, InBuffer} };
                checkCUDNNFE(layer.conv_desc[conv_desc_idx].graph.execute(getCuDNN().m_handle,
                                                                          variant_pack, workspace));
            } else {
                m_cudnn.convolve(OutBuffer,
                                 InBuffer,
                                 layer.weights[0],
                                 workspace,
                                 layer.conv_desc[conv_desc_idx],
                                 layer.scale_1);
            }
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
    //         output_pol(float) 2 chanels * (BOARD_SIZE * BOARD_SIZE)
}

template <typename net_t>
void CuDNN_Network<net_t>::forward(const std::vector<float>& input,
                                   std::vector<float>& output_pol,
                                   std::vector<float>& output_val,
                                   CuDNNContext& cudnn_context,
                                   const int batch_size) {

    forward_activations(input, output_pol, output_val, cudnn_context, batch_size);
}

template <typename net_t>
bool CuDNN<net_t>::has_fp16_compute() {
    return m_fp16_compute;
}

template <typename net_t>
bool CuDNN<net_t>::has_tensor_cores() {
    return m_tensorcore;
}

template class CuDNN<float>;
template class CuDNN_Network<float>;
#ifdef USE_HALF
template class CuDNN<half_float::half>;
template class CuDNN_Network<half_float::half>;
#endif

#endif

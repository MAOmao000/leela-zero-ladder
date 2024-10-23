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

#if defined(USE_CUDNN) || defined(USE_TENSOR_RT)
#include "Backend.h"
#include "GTP.h"

using namespace Utils;

void BE::squeeze_excitation_float(
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const BackendContext& cudnn_context,
    const void *bufferIn1,   // residual input(before convolve)
    const void *bufferIn2,   // residual output
    void *TempBuffer,
    const void *fc1_weights,
    const void *fc1_biases,
    const void *fc2_weights,
    const void *fc2_biases,
    void *bufferOut,
    void *bufferPool,
    const size_t batch_size,
    const int channels,
    const int spatial,
    const bool isNCHW,
    const bool isTensorCore) {

    // in: batch * channels * spatial(board size * board size)
    // out: batch * channels
    if (isNCHW) {
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
    if (isTensorCore) {
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
    if (isTensorCore) {
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

    if (isNCHW) {
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

void BE::squeeze_excitation_half(
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const BackendContext& cudnn_context,
    const void *bufferIn1,   // residual input(before convolve)
    const void *bufferIn2,   // residual output
    void *TempBuffer,
    const void *fc1_weights,
    const void *fc1_biases,
    const void *fc2_weights,
    const void *fc2_biases,
    void *bufferOut,
    void *bufferPool,
    const size_t batch_size,
    const int channels,
    const int spatial,
    const bool isNCHW,
    const bool isTensorCore) {

    // in: batch * channels * spatial(board size * board size)
    // out: batch * channels
    if (isNCHW) {
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
    if (isTensorCore) {
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
    if (isTensorCore) {
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

    if (isNCHW) {
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

template <typename net_t>
Backend<net_t>::Backend(
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
void Backend<net_t>::initialize(
    const int channels,
    const size_t batch_size,
    const NetworkType net_type,
    const size_t num_worker_threads,
    const std::string &model_hash) {

    // For compatibility with OpenCL implementation
    (void)channels;
    (void)batch_size;

    const char* log_level = "CUDNN_LOGLEVEL_DBG=0";
    putenv((char *)log_level);
    const char* log_dest = "CUDNN_LOGDEST_DBG=stderr";
    putenv((char *)log_dest);
    const char* module_load = "CUDA_MODULE_LOADING=LAZY";
    putenv((char *)module_load);
    if (cfg_backend == backend_t::CUDNNGRAPH) {
        const char* log_info = "CUDNN_FRONTEND_LOG_INFO=0";
        putenv((char *)log_info);
    }

    m_net_type = net_type;
    m_num_worker_threads = num_worker_threads;
    m_model_hash = model_hash;

    for (auto i = 0; i < m_num_worker_threads; i++) {
        if (cfg_backend == backend_t::TENSORRT) {
            continue;
        }
        cudnnHandle_t cudnn;
        checkCUDNN(cudnnCreate(&cudnn));
        checkCUDNN(cudnnSetStream(cudnn, cudaStreamPerThread));
        m_handle.emplace_back(cudnn);
        if (net_type == NetworkType::MINIGO_SE) {
            cublasHandle_t cublas;
            checkCUBLAS(cublasCreate(&cublas));
            checkCUBLAS(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE));
            if (m_tensorcore) {
                checkCUBLAS(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));
            }
            checkCUBLAS(cublasSetStream(cublas, cudaStreamPerThread));
            m_cublas_handles.emplace_back(cublas);
        }
    }
}

template <typename net_t>
void Backend<net_t>::forward(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val,
    const int tid,
    const size_t batch_size) {

    forward_activations(input, output_pol, output_val, *m_context[tid], tid, batch_size);
}

template class Backend<float>;
template class Backend<half_float::half>;

#endif

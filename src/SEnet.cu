#include "config.h"
#include "Utils.h"

#ifdef USE_CUDNN
#include <stdint.h>
#include <stdexcept>
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_fp16.h>

// TODO maybe tune this number, it varies by GPU.
static const int targetNumThreads = 128;

inline int DivUp(int a, int b) {
    return (a + b - 1) / b;
}

void splitThreadsAcrossDim01(
    int dim0Size,
    int dim1Size,
    int& threads0,
    int& blocks0,
    int& threads1,
    int& blocks1) {

    if (dim0Size > targetNumThreads) {
        threads0 = targetNumThreads / 2;
        blocks0 = (dim0Size + threads0 - 1) / threads0;
        threads1 = 1;
        blocks1 = dim1Size;
    }
    else if (dim0Size > targetNumThreads / 2) {
        threads0 = dim0Size;
        blocks0 = 1;
        threads1 = 1;
        blocks1 = dim1Size;
    }
    else {
        threads0 = dim0Size;
        blocks0 = 1;
        threads1 = targetNumThreads / dim0Size;
        blocks1 = (dim1Size + threads1 - 1) / threads1;
    }
}

__global__ void global_average_pooling_kernel_float(
    const float* input,
    float* output,
    const int N,
    const int C,
    const int spatial) {

    extern __shared__ float pool_shared_float[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int shared_size = blockDim.x;
    int s = threadIdx.x;

    int nc_index = index / shared_size;
    int n = nc_index / C;
    int c = nc_index % C;

    // the pools
    float* sum_pool_shared = pool_shared_float;

    // assign the pool
    if (s < spatial && n < N) {
        float val = input[(n * C + c) * spatial + s];
        sum_pool_shared[s] = val;
    }
    else {
        // out of the board
        sum_pool_shared[s] = 0.0f;
    }
    __syncthreads();

    if (s < spatial && n < N) {
        for (int shift = shared_size >> 1; shift > 0; shift >>= 1) {
            if (s < shift) {
                sum_pool_shared[s] += sum_pool_shared[s + shift];
            }
            __syncthreads();
        }

        if (s == 0) {
            float vsum = sum_pool_shared[s];
            float vmean = vsum / (float)spatial;
            output[n * C + c] = vmean;
        }
    }
}

void global_average_pooling_float(
    const float* input,
    float* output,
    const int N,
    const int C,
    const int spatial) {

    int shared_size_base = 1;
    while (shared_size_base < spatial) {
        shared_size_base *= 2;
    }
    const int total_elements = N * C * shared_size_base;
    const int block_size = shared_size_base;
    const int blocks = DivUp(total_elements, block_size);
    const int shared_size = sizeof(float) * shared_size_base * 2;

    global_average_pooling_kernel_float<<<blocks, block_size, shared_size>>>(
        input, output, N, C, spatial);
}

__global__ void global_average_pooling_kernel_float_NHWC(
    const float* input,
    float* output,
    const int inputSize,
    const int outputSize) {

    const int elementsPerThread = 361;  // 19x19 board.
    int blockStart = blockIdx.x * blockDim.x;
    float S = 0;

#pragma unroll
    for (int i = 0; i < elementsPerThread; i++) {
        int localIndex = i * blockDim.x + threadIdx.x;
        int inputIndex = blockStart * elementsPerThread + localIndex;
        if (inputIndex < inputSize) S += input[inputIndex];
    }

    float avg = S / elementsPerThread;

    int opIndex = blockStart + threadIdx.x;
    if (opIndex < outputSize) output[opIndex] = avg;
}

void global_average_pooling_float_NHWC(
    const float* input,
    float* output,
    const int N,
    const int C,
    const int spatial) {

    // For NHWC fp32, simply launch N blocks, each with C threads.
    global_average_pooling_kernel_float_NHWC<<<N, C>>>(
        input, output, N * C * spatial, N * C);
}

__global__ void global_average_pooling_kernel_half(
    const __half* input,
    __half* output,
    const int N,
    const int C,
    const int spatial) {

    extern __shared__ float pool_shared_half[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int shared_size = blockDim.x;
    int s = threadIdx.x;

    int nc_index = index / shared_size;
    int n = nc_index / C;
    int c = nc_index % C;

    // the pools
    float* sum_pool_shared = pool_shared_half;

    // assign the pool
    if (s < spatial && n < N) {
        sum_pool_shared[s] = __half2float(input[(n * C + c) * spatial + s]);
    }
    else {
        // out of the board
        sum_pool_shared[s] = 0.0f;
    }
    __syncthreads();

    if (s < spatial && n < N) {
        for (int shift = shared_size >> 1; shift > 0; shift >>= 1) {
            if (s < shift) {
                sum_pool_shared[s] += sum_pool_shared[s + shift];
            }
            __syncthreads();
        }

        if (s == 0) {
            float vsum = sum_pool_shared[s];
            float vmean = vsum / (float)spatial;
            output[n * C + c] = __float2half(vmean);
        }
    }
}

void global_average_pooling_half(
    const __half* input,
    __half* output,
    const int N,
    const int C,
    const int spatial) {
    int shared_size_base = 1;
    while (shared_size_base < spatial) {
        shared_size_base *= 2;
    }
    const int total_elements = N * C * shared_size_base;
    const int block_size = shared_size_base;
    const int blocks = DivUp(total_elements, block_size);
    const int shared_size = sizeof(float) * shared_size_base * 2;

    global_average_pooling_kernel_half << <blocks, block_size, shared_size >> > (
        input, output, N, C, spatial);
}

__global__ void global_average_pooling_kernel_half_NHWC(
    const __half* input,
    __half* output,
    const int inputSize,
    const int outputSize) {

    const int elementsPerThread = 361;  // 19x19 board.
    int blockStart = blockIdx.x * blockDim.x;
    float S = 0;

#pragma unroll
    for (int i = 0; i < elementsPerThread; i++) {
        int localIndex = i * blockDim.x + threadIdx.x;
        int inputIndex = blockStart * elementsPerThread + localIndex;
        if (inputIndex < inputSize) S += __half2float(input[inputIndex]);
    }

    float avg = S / elementsPerThread;

    int opIndex = blockStart + threadIdx.x;
    if (opIndex < outputSize) output[opIndex] = __float2half(avg);
}

void global_average_pooling_half_NHWC(
    const __half* input,
    __half* output,
    const int N,
    const int C,
    const int spatial) {

    // For NHWC fp16, simply launch N blocks, each with C threads.
    global_average_pooling_kernel_half_NHWC<<<N, C>>>(
        (__half*)input, (__half*)output, N * C * spatial, N * C);
}

__global__ void add_bias_kernel_float(
    float* buf,
    const float* biases,
    const int N,
    const int C,
    const bool relu) {

    int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (cIdx < C && nIdx < N) {
        int idx = nIdx * C + cIdx;
        if (relu)
            buf[idx] = fmaxf(buf[idx] + biases[cIdx], 0.0f);
        else
            buf[idx] = buf[idx] + biases[cIdx];
    }
}

void add_bias_float(
    float* buf,
    const float* biases,
    const int N,
    const int C,
    const bool is_relu) {

    int cThreads;
    int cBlocks;
    int nThreads;
    int nBlocks;
    splitThreadsAcrossDim01(C, N, cThreads, cBlocks, nThreads, nBlocks);
    if (nBlocks > 65536)
        throw std::runtime_error("add_bias_float: batch size too large given channel size");
    dim3 grid(cBlocks, nBlocks, 1);
    dim3 threads(cThreads, nThreads, 1);
    add_bias_kernel_float<<<grid, threads>>>(buf, biases, N, C, is_relu);
}

__global__ void add_bias_kernel_half(
    __half* buf,
    const __half* biases,
    const int N,
    const int C,
    const bool relu) {

    int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
    const __half halfzero = __float2half(0.0f);
    if (cIdx < C && nIdx < N) {
        int idx = nIdx * C + cIdx;
        if (relu) {
            __half a = __hadd(buf[idx], biases[cIdx]);
            buf[idx] = __hmax(a, halfzero);
        }
        else {
            buf[idx] = __hadd(buf[idx], biases[cIdx]);
        }
    }
}

void add_bias_half(
    __half* buf,
    const __half* biases,
    const int N,
    const int C,
    const bool is_relu) {

    int cThreads;
    int cBlocks;
    int nThreads;
    int nBlocks;
    splitThreadsAcrossDim01(C, N, cThreads, cBlocks, nThreads, nBlocks);
    if (nBlocks > 65536)
        throw std::runtime_error("add_bias_half: batch size too large given channel size");
    dim3 grid(cBlocks, nBlocks, 1);
    dim3 threads(cThreads, nThreads, 1);
    add_bias_kernel_half<<<grid, threads>>>(buf, biases, N, C, is_relu);
}

__global__ void se_scale_kernel_float(
    float* output,
    const float* input,
    const float* biases,
    const float* residual,
    const int N,
    const int C,
    const int spatial) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int total_elements = N * C * spatial;
    if (index < total_elements) {
        int c = (index / spatial) % C;
        int n = (index / spatial) / C;
        int start_idx = n * 2 * C;

        float val = input[index];
        float gamma = biases[start_idx + c];
        gamma = 1.0f / (1.0f + expf(-gamma));

        float beta = biases[start_idx + c + C];

        float res = residual[index];

        val = gamma * val + beta + res;

        if (val < 0.0f) {
            val = 0.0f;
        }
        output[index] = val;
    }
}

void se_scale_float(
    float* out_buf,
    const float* buf,
    const float* biases,
    const float* bufferIn,
    const int N,
    const int C,
    const int spatial) {

    const int total_elements = C * spatial * N;
    const int block_size = 256;
    const int blocks = DivUp(total_elements, block_size);
    se_scale_kernel_float<<<blocks, block_size>>>(
        out_buf, buf, biases, bufferIn, N, C, spatial);
}

__global__ void se_scale_kernel_float_NHWC(
    float* output,
    const float* input,
    const float* biases,
    const float* residual,
    const int N,
    const int C,
    const int spatial) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int total_elements = N * C * spatial;
    if (index < total_elements) {
        int c = index % C;
        int n = index / (spatial * C);
        int start_idx = n * 2 * C;

        float val = input[index];
        float gamma = biases[start_idx + c];    // first biases
        gamma = 1.0f / (1.0f + expf(-gamma));   // sigmoid

        float beta = biases[start_idx + c + C]; // second biases

        float res = residual[index];

        val = gamma * val + beta + res;

        if (val < 0.0f) {
            val = 0.0f;
        }
        output[index] = val;
    }
}

void se_scale_float_NHWC(
    float* out_buf,
    const float* buf,
    const float* biases,
    const float* bufferIn,
    const int N,
    const int C,
    const int spatial) {

    const int total_elements = C * spatial * N;
    const int block_size = 256;
    const int blocks = DivUp(total_elements, block_size);

    se_scale_kernel_float_NHWC<<<blocks, block_size>>>(
        out_buf, buf, biases, bufferIn, N, C, spatial);
}

__global__ void se_scale_kernel_half(
    __half* output,
    const __half* input,
    const __half* biases,
    const __half* residual,
    const int N,
    const int C,
    const int spatial) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int total_elements = N * C * spatial;
    if (index < total_elements) {
        int c = (index / spatial) % C;
        int n = (index / spatial) / C;
        int start_idx = n * 2 * C;

        float val = __half2float(input[index]);
        float gamma = __half2float(biases[start_idx + c]);    // first biases
        gamma = 1.0f / (1.0f + expf(-gamma));   // sigmoid

        float beta = __half2float(biases[start_idx + c + C]); // second biases

        float res = __half2float(residual[index]);

        val = gamma * val + beta + res;

        if (val < 0.0f) {
            val = 0.0f;
        }
        output[index] = __float2half(val);
    }
}

void se_scale_half(
    __half* out_buf,
    const __half* buf,
    const __half* biases,
    const __half* bufferIn,
    const int N,
    const int C,
    const int spatial) {

    const int total_elements = C * spatial * N;
    const int block_size = 256;
    const int blocks = DivUp(total_elements, block_size);
    se_scale_kernel_half<<<blocks, block_size>>>(
        out_buf, buf, biases, bufferIn, N, C, spatial);
}

__global__ void se_scale_kernel_half_NHWC(
    __half* output,
    const __half* input,
    const __half* biases,
    const __half* residual,
    const int N,
    const int C,
    const int spatial) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int total_elements = N * C * spatial;
    if (index < total_elements) {
        int c = index % C;
        int n = index / (spatial * C);
        int start_idx = n * 2 * C;

        float val = __half2float(input[index]);
        float gamma = __half2float(biases[start_idx + c]);    // first biases
        gamma = 1.0f / (1.0f + expf(-gamma));                 // sigmoid

        float beta = __half2float(biases[start_idx + c + C]); // second biases

        float res = __half2float(residual[index]);

        val = gamma * val + beta + res;

        if (val < 0.0f) {
            val = 0.0f;
        }
        output[index] = __float2half(val);
    }
}

void se_scale_half_NHWC(
    __half* out_buf,
    const __half* buf,
    const __half* biases,
    const __half* bufferIn,
    const int N,
    const int C,
    const int spatial) {

    const int total_elements = C * spatial * N;
    const int block_size = 256;
    const int blocks = DivUp(total_elements, block_size);
    se_scale_kernel_half_NHWC<<<blocks, block_size>>>(
        out_buf, buf, biases, bufferIn, N, C, spatial);
}

void se_scale_float_stream(
    float* out_buf,
    const float* buf,
    const float* biases,
    const float* bufferIn,
    const int N,
    const int C,
    const int spatial,
    cudaStream_t stream) {

    const int total_elements = C * spatial * N;
    const int block_size = 256;
    const int blocks = DivUp(total_elements, block_size);
    se_scale_kernel_float<<<blocks, block_size , 0, stream>>> (
        out_buf, buf, biases, bufferIn, N, C, spatial);
}

void se_scale_half_stream(
    __half* out_buf,
    const __half* buf,
    const __half* biases,
    const __half* bufferIn,
    const int N,
    const int C,
    const int spatial,
    cudaStream_t stream) {

    const int total_elements = C * spatial * N;
    const int block_size = 256;
    const int blocks = DivUp(total_elements, block_size);
    se_scale_kernel_half<<<blocks, block_size, 0, stream>>>(
        out_buf, buf, biases, bufferIn, N, C, spatial);
}
#endif

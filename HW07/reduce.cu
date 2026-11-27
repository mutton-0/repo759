#include <cuda_runtime.h>
#include <utility>
#include "reduce.cuh"

// Reduction #4: first add during global load. Each block reads 2*blockDim.x
// elements, adds the pair while loading, then does a sequential-addressing
// tree reduction in shared memory. Block b writes its partial sum to g_odata[b].
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    const unsigned int tid = threadIdx.x;
    const size_t i = (size_t)blockIdx.x * (blockDim.x * 2) + tid;

    float v = 0.0f;
    if (i < n) v = g_idata[i];
    if (i + blockDim.x < n) v += g_idata[i + blockDim.x];
    sdata[tid] = v;
    __syncthreads();

    // smallest power of two >= blockDim.x, so blockDim.x need not be a power of 2
    unsigned int p = 1;
    while (p < blockDim.x) p <<= 1;
    for (unsigned int s = p / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    unsigned int n = N;
    const size_t shmem = threads_per_block * sizeof(float);
    while (true) {
        unsigned int blocks = (n + 2 * threads_per_block - 1) / (2 * threads_per_block);
        reduce_kernel<<<blocks, threads_per_block, shmem>>>(*input, *output, n);
        // the partial sums become the input of the next call; swapping the two
        // buffers also leaves the final sum in (*input)[0]
        std::swap(*input, *output);
        n = blocks;
        if (n == 1) break;
    }
    cudaDeviceSynchronize();
}

#include "stencil.cuh"

// Dynamic shared memory layout (floats):
//   [ mask: 2R+1 | image tile: blockDim.x + 2R | output tile: blockDim.x ]
__global__ void stencil_kernel(const float *image, const float *mask, float *output, unsigned int n,
                               unsigned int R) {
    extern __shared__ float shared[];
    float *s_mask = shared;
    float *s_image = s_mask + (2 * R + 1);
    float *s_out = s_image + (blockDim.x + 2 * R);

    unsigned int tid = threadIdx.x;
    long long block_start = (long long)blockIdx.x * blockDim.x;

    if (tid < 2 * R + 1) {
        s_mask[tid] = mask[tid];
    }

    // s_image[s] holds image[block_start - R + s]; outside [0, n) the value is 1
    for (unsigned int s = tid; s < blockDim.x + 2 * R; s += blockDim.x) {
        long long g = block_start - (long long)R + s;
        s_image[s] = (g >= 0 && g < (long long)n) ? image[g] : 1.0f;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int j = -(int)R; j <= (int)R; j++) {
        sum += s_image[tid + R + j] * s_mask[j + R];
    }
    s_out[tid] = sum;
    __syncthreads();

    unsigned long long i = (unsigned long long)block_start + tid;
    if (i < n) {
        output[i] = s_out[tid];
    }
}

__host__ void stencil(const float *image, const float *mask, float *output, unsigned int n,
                      unsigned int R, unsigned int threads_per_block) {
    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_bytes = ((2 * R + 1) + (threads_per_block + 2 * R) + threads_per_block) * sizeof(float);
    stencil_kernel<<<blocks, threads_per_block, shared_bytes>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}

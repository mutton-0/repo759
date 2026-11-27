#include <cuda_runtime.h>
#include "matmul.cuh"

// Tiled matrix multiplication. Each block computes one block_dim x block_dim
// tile of C. In every phase the block loads one tile of A and one tile of B
// into shared memory, then each thread accumulates its partial dot product.
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
    extern __shared__ __align__(sizeof(double)) unsigned char smem[];
    const unsigned int bd = blockDim.x;
    T *As = reinterpret_cast<T *>(smem);
    T *Bs = As + bd * bd;

    const unsigned int tx = threadIdx.x, ty = threadIdx.y;
    const unsigned int row = blockIdx.y * bd + ty;
    const unsigned int col = blockIdx.x * bd + tx;

    T sum = 0;
    const unsigned int phases = (n + bd - 1) / bd;
    for (unsigned int p = 0; p < phases; p++) {
        const unsigned int a_col = p * bd + tx;
        const unsigned int b_row = p * bd + ty;
        // out-of-range entries are padded with 0 so they do not change the sum
        As[ty * bd + tx] = (row < n && a_col < n) ? A[(size_t)row * n + a_col] : T(0);
        Bs[ty * bd + tx] = (b_row < n && col < n) ? B[(size_t)b_row * n + col] : T(0);
        __syncthreads();
        for (unsigned int k = 0; k < bd; k++) {
            sum += As[ty * bd + k] * Bs[k * bd + tx];
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        C[(size_t)row * n + col] = sum;
    }
}

template <typename T>
static void matmul_launch(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim) {
    dim3 block(block_dim, block_dim);
    unsigned int g = (n + block_dim - 1) / block_dim;
    dim3 grid(g, g);
    size_t shmem = 2 * (size_t)block_dim * block_dim * sizeof(T);
    matmul_kernel<T><<<grid, block, shmem>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    matmul_launch<int>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    matmul_launch<float>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n,
                       unsigned int block_dim) {
    matmul_launch<double>(A, B, C, n, block_dim);
}

#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "matmul.cuh"

// Fills A and B on the host, copies them to device memory,
// runs one matmul version and prints C[0], C[n*n-1] and the time in ms.
template <typename T, typename Fill, typename Mul>
void run(size_t n, unsigned int block_dim, Fill fill, Mul mul, const char *fmt) {
    size_t bytes = n * n * sizeof(T);
    T *hA = new T[n * n];
    T *hB = new T[n * n];
    T *hC = new T[n * n];
    for (size_t i = 0; i < n * n; i++) {
        hA[i] = fill();
        hB[i] = fill();
    }

    T *A, *B, *C;
    cudaMalloc((void **)&A, bytes);
    cudaMalloc((void **)&B, bytes);
    cudaMalloc((void **)&C, bytes);
    cudaMemcpy(A, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    mul(A, B, C, (unsigned int)n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(hC, C, bytes, cudaMemcpyDeviceToHost);

    std::printf(fmt, hC[0]);
    std::printf(fmt, hC[n * n - 1]);
    std::printf("%f\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    delete[] hA;
    delete[] hB;
    delete[] hC;
}

int main(int argc, char *argv[]) {
    size_t n = std::atol(argv[1]);
    unsigned int block_dim = std::atoi(argv[2]);

    std::random_device entropy_source;
    std::mt19937 gen(entropy_source());
    std::uniform_int_distribution<int> idist(-10, 10);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    std::uniform_real_distribution<double> ddist(-1.0, 1.0);

    run<int>(n, block_dim, [&] { return idist(gen); }, matmul_1, "%d\n");
    run<float>(n, block_dim, [&] { return fdist(gen); }, matmul_2, "%f\n");
    run<double>(n, block_dim, [&] { return ddist(gen); }, matmul_3, "%f\n");
    return 0;
}

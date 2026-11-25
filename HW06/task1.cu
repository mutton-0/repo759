#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "matmul.cuh"

int main(int argc, char *argv[]) {
    size_t n = std::atol(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t bytes = n * n * sizeof(float);
    float *hA = new float[n * n];
    float *hB = new float[n * n];
    float *hC = new float[n * n];
    for (size_t i = 0; i < n * n; i++) {
        hA[i] = dist(generator);
        hB[i] = dist(generator);
    }

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, bytes);
    cudaMalloc((void **)&dB, bytes);
    cudaMalloc((void **)&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul(dA, dB, dC, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    std::printf("%f\n", hC[n * n - 1]);
    std::printf("%f\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;
    return 0;
}

#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "vscale.cuh"

int main(int argc, char *argv[]) {
    unsigned int n = std::atoi(argv[1]);
    unsigned int threads_per_block = (argc > 2) ? std::atoi(argv[2]) : 512;

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dist_a(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_b(0.0f, 1.0f);

    float *hA = new float[n];
    float *hB = new float[n];
    for (unsigned int i = 0; i < n; i++) {
        hA[i] = dist_a(generator);
        hB[i] = dist_b(generator);
    }

    float *dA, *dB;
    cudaMalloc((void **)&dA, n * sizeof(float));
    cudaMalloc((void **)&dB, n * sizeof(float));
    cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vscale<<<blocks, threads_per_block>>>(dA, dB, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hB, dB, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::printf("%f\n", ms);
    std::printf("%f\n", hB[0]);
    std::printf("%f\n", hB[n - 1]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    delete[] hA;
    delete[] hB;
    return 0;
}

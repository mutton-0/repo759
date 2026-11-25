#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "stencil.cuh"

int main(int argc, char *argv[]) {
    unsigned int n = std::atol(argv[1]);
    unsigned int R = std::atoi(argv[2]);
    unsigned int threads_per_block = std::atoi(argv[3]);

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float *hImage = new float[n];
    float *hOutput = new float[n];
    float *hMask = new float[2 * R + 1];
    for (unsigned int i = 0; i < n; i++) {
        hImage[i] = dist(generator);
    }
    for (unsigned int i = 0; i < 2 * R + 1; i++) {
        hMask[i] = dist(generator);
    }

    float *dImage, *dMask, *dOutput;
    cudaMalloc((void **)&dImage, n * sizeof(float));
    cudaMalloc((void **)&dMask, (2 * R + 1) * sizeof(float));
    cudaMalloc((void **)&dOutput, n * sizeof(float));
    cudaMemcpy(dImage, hImage, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, hMask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stencil(dImage, dMask, dOutput, n, R, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(hOutput, dOutput, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::printf("%f\n", hOutput[n - 1]);
    std::printf("%f\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dImage);
    cudaFree(dMask);
    cudaFree(dOutput);
    delete[] hImage;
    delete[] hOutput;
    delete[] hMask;
    return 0;
}

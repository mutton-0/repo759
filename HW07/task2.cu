#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "reduce.cuh"

int main(int argc, char *argv[]) {
    unsigned int N = std::atol(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    std::random_device entropy_source;
    std::mt19937 gen(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float *h = new float[N];
    for (unsigned int i = 0; i < N; i++) h[i] = dist(gen);

    unsigned int blocks = (N + 2 * threads_per_block - 1) / (2 * threads_per_block);
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, (size_t)N * sizeof(float));
    cudaMalloc((void **)&d_out, (size_t)blocks * sizeof(float));
    cudaMemcpy(d_in, h, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce(&d_in, &d_out, N, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float sum;
    cudaMemcpy(&sum, d_in, sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("%f\n", sum);
    std::printf("%f\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h;
    return 0;
}

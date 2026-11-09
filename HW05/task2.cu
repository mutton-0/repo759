#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

__global__ void axpy(int a, int *dA) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    dA[blockIdx.x * blockDim.x + threadIdx.x] = a * x + y;
}

int main() {
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_int_distribution<int> dist(1, 10);
    int a = dist(generator);

    int hA[16];
    int *dA;
    cudaMalloc((void **)&dA, 16 * sizeof(int));

    axpy<<<2, 8>>>(a, dA);
    cudaMemcpy(hA, dA, 16 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++) {
        std::printf("%d%c", hA[i], i == 15 ? '\n' : ' ');
    }

    cudaFree(dA);
    return 0;
}

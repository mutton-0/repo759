#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "matmul.h"

int main(int argc, char *argv[]) {
    std::size_t n = std::atol(argv[1]);
    int t = std::atoi(argv[2]);

    float *A = new float[n * n];
    float *B = new float[n * n];
    float *C = new float[n * n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n * n; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    omp_set_num_threads(t);

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    mmul(A, B, C, n);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << C[0] << std::endl;
    std::cout << C[n * n - 1] << std::endl;
    std::cout << elapsed.count() << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

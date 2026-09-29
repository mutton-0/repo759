#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "matmul.h"

int main() {
    const unsigned int n = 1024;

    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C = new double[n * n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (unsigned int i = 0; i < n * n; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    std::vector<double> Av(A, A + n * n);
    std::vector<double> Bv(B, B + n * n);

    std::cout << n << std::endl;

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double, std::milli> elapsed;

    start = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << elapsed.count() << std::endl;
    std::cout << C[n * n - 1] << std::endl;

    start = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << elapsed.count() << std::endl;
    std::cout << C[n * n - 1] << std::endl;

    start = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << elapsed.count() << std::endl;
    std::cout << C[n * n - 1] << std::endl;

    start = std::chrono::high_resolution_clock::now();
    mmul4(Av, Bv, C, n);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << elapsed.count() << std::endl;
    std::cout << C[n * n - 1] << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

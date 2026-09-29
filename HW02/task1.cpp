#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "scan.h"

int main(int argc, char *argv[]) {
    std::size_t n = std::atol(argv[1]);

    float *arr = new float[n];
    float *out = new float[n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; i++) {
        arr[i] = dist(gen);
    }

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    scan(arr, out, n);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << elapsed.count() << std::endl;
    std::cout << out[0] << std::endl;
    std::cout << out[n - 1] << std::endl;

    delete[] arr;
    delete[] out;

    return 0;
}

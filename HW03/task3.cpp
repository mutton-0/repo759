#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "msort.h"

int main(int argc, char *argv[]) {
    std::size_t n = std::atol(argv[1]);
    int t = std::atoi(argv[2]);
    std::size_t ts = std::atol(argv[3]);

    int *arr = new int[n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (std::size_t i = 0; i < n; i++) {
        arr[i] = dist(gen);
    }

    omp_set_num_threads(t);

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    msort(arr, n, ts);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << arr[0] << std::endl;
    std::cout << arr[n - 1] << std::endl;
    std::cout << elapsed.count() << std::endl;

    delete[] arr;

    return 0;
}

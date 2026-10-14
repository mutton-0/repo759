#include "msort.h"

#include <algorithm>
#include <vector>

static void merge_sort(int *arr, int *tmp, std::size_t n, std::size_t threshold) {
    if (n <= threshold || n < 2) {
        std::sort(arr, arr + n);
        return;
    }

    std::size_t mid = n / 2;

#pragma omp task
    merge_sort(arr, tmp, mid, threshold);
#pragma omp task
    merge_sort(arr + mid, tmp + mid, n - mid, threshold);
#pragma omp taskwait

    std::merge(arr, arr + mid, arr + mid, arr + n, tmp);
    std::copy(tmp, tmp + n, arr);
}

void msort(int *arr, const std::size_t n, const std::size_t threshold) {
    std::vector<int> tmp(n);
#pragma omp parallel
#pragma omp single
    merge_sort(arr, tmp.data(), n, threshold);
}

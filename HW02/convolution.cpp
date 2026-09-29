#include "convolution.h"

// Returns f[r][c] with the assignment's boundary convention:
//   - both r,c in range [0,n)      -> actual image value
//   - exactly one of r,c out of range (an "edge")   -> 1.0
//   - both r,c out of range (a "corner")             -> 0.0
static float padded_value(const float *image, long r, long c, std::size_t n) {
    bool row_ok = (r >= 0) && (static_cast<std::size_t>(r) < n);
    bool col_ok = (c >= 0) && (static_cast<std::size_t>(c) < n);

    if (row_ok && col_ok) {
        return image[static_cast<std::size_t>(r) * n + static_cast<std::size_t>(c)];
    }
    if (row_ok != col_ok) {
        return 1.0f;
    }
    return 0.0f;
}

void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) {
    long offset = (static_cast<long>(m) - 1) / 2;

    for (std::size_t x = 0; x < n; x++) {
        for (std::size_t y = 0; y < n; y++) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < m; j++) {
                    long r = static_cast<long>(x) + static_cast<long>(i) - offset;
                    long c = static_cast<long>(y) + static_cast<long>(j) - offset;
                    sum += mask[i * m + j] * padded_value(image, r, c, n);
                }
            }
            output[x * n + y] = sum;
        }
    }
}

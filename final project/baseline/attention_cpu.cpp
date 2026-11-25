#include "attention_cpu.h"

#include <algorithm>
#include <cmath>
#include <vector>

// One query row: scores, softmax, weighted sum of V. s has Nk floats.
static void attention_row(const float* q, const float* K, const float* V,
                          float* o, float* s, int Nk, int d) {
  const float scale = 1.0f / std::sqrt((float)d);

  float m = -INFINITY;
  for (int j = 0; j < Nk; ++j) {
    const float* k = K + (size_t)j * d;
    float dot = 0.0f;
    for (int c = 0; c < d; ++c) dot += q[c] * k[c];
    s[j] = dot * scale;
    m = std::max(m, s[j]);
  }

  float sum = 0.0f;
  for (int j = 0; j < Nk; ++j) {
    s[j] = std::exp(s[j] - m);
    sum += s[j];
  }

  for (int c = 0; c < d; ++c) o[c] = 0.0f;
  for (int j = 0; j < Nk; ++j) {
    const float* v = V + (size_t)j * d;
    const float p = s[j];
    for (int c = 0; c < d; ++c) o[c] += p * v[c];
  }
  const float inv = 1.0f / sum;
  for (int c = 0; c < d; ++c) o[c] *= inv;
}

void attention_serial(const float* Q, const float* K, const float* V,
                      float* O, int Nq, int Nk, int d) {
  std::vector<float> s(Nk);
  for (int i = 0; i < Nq; ++i)
    attention_row(Q + (size_t)i * d, K, V, O + (size_t)i * d, s.data(), Nk, d);
}

void attention_omp(const float* Q, const float* K, const float* V, float* O,
                   int Nq, int Nk, int d) {
#pragma omp parallel
  {
    std::vector<float> s(Nk);  // one score row per thread
#pragma omp for schedule(static)
    for (int i = 0; i < Nq; ++i)
      attention_row(Q + (size_t)i * d, K, V, O + (size_t)i * d, s.data(), Nk,
                    d);
  }
}

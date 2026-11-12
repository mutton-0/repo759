// v0: naive attention, three kernels, one thread per output element.
// The thread index threadIdx.x walks the ROW dimension (i), which is how one
// would write it straight from the formula. Neighboring threads in a warp
// then touch addresses that are a whole row apart (stride d or Nk), so the
// global loads/stores of S and O are not coalesced. v1 fixes this.
#include <cmath>

#include "common.cuh"
#include "memopt.cuh"

// S[i][j] = dot(Q[i], K[j]) * scale, x -> i, y -> j
__global__ void qk_v0(const float* Q, const float* K, float* S, int Nq, int Nk,
                      int d, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= Nq || j >= Nk) return;
  float sum = 0.0f;
  for (int k = 0; k < d; ++k) sum += Q[i * d + k] * K[j * d + k];
  S[(size_t)i * Nk + j] = sum * scale;
}

// one thread per row: max, then exp and sum, then divide
__global__ void softmax_v0(float* S, int Nq, int Nk) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Nq) return;
  float* row = S + (size_t)i * Nk;
  float m = -INFINITY;
  for (int j = 0; j < Nk; ++j) m = fmaxf(m, row[j]);
  float sum = 0.0f;
  for (int j = 0; j < Nk; ++j) {
    row[j] = expf(row[j] - m);
    sum += row[j];
  }
  for (int j = 0; j < Nk; ++j) row[j] /= sum;
}

// O[i][c] = sum_j S[i][j] * V[j][c], x -> i, y -> c
__global__ void pv_v0(const float* S, const float* V, float* O, int Nq, int Nk,
                      int d) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= Nq || c >= d) return;
  float sum = 0.0f;
  for (int j = 0; j < Nk; ++j) sum += S[(size_t)i * Nk + j] * V[j * d + c];
  O[i * d + c] = sum;
}

static void qk(const float* Q, const float* K, float* S, int Nq, int Nk,
               int d) {
  const dim3 block(32, 8);
  const dim3 grid((Nq + 31) / 32, (Nk + 7) / 8);
  qk_v0<<<grid, block>>>(Q, K, S, Nq, Nk, d, 1.0f / std::sqrt((float)d));
}

static void softmax(float* S, int Nq, int Nk) {
  softmax_v0<<<(Nq + 127) / 128, 128>>>(S, Nq, Nk);
}

static void pv(const float* S, const float* V, float* O, int Nq, int Nk,
               int d) {
  const dim3 block(32, 8);
  const dim3 grid((Nq + 31) / 32, (d + 7) / 8);
  pv_v0<<<grid, block>>>(S, V, O, Nq, Nk, d);
}

extern const AttentionStages kV0 = {qk, softmax, pv};

void attention_memopt_v0(const float* Q, const float* K, const float* V,
                         float* O, int N, int d) {
  run_attention(kV0, Q, K, V, O, N, N, d);
}

void attention_memopt_v0_decode(const float* Q, const float* K,
                                const float* V, float* O, int Nq, int Nk,
                                int d) {
  run_attention(kV0, Q, K, V, O, Nq, Nk, d);
}

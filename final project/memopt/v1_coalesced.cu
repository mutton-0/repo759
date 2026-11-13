// v1: same work as v0, but threadIdx.x now walks the dimension that is
// contiguous in memory, so a warp reads/writes 32 neighboring floats.
//   qk:      x -> column j of S. K is transposed first (Kt is d x Nk), so
//            Kt[k*Nk + j] is contiguous in j.
//   softmax: one block per row, the threads stride over the row together,
//            max and sum are reduced in shared memory.
//   pv:      x -> column c of O, V[j*d + c] is contiguous in c.
#include <cmath>

#include "common.cuh"
#include "memopt.cuh"

static const int kSoftmaxThreads = 256;

// Kt[k][j] = K[j][k], x -> j so the writes are coalesced. The reads are
// strided, but K is only Nk x d, small next to the Nq x Nk work of qk.
__global__ void transpose_v1(const float* K, float* Kt, int Nk, int d) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (j >= Nk || k >= d) return;
  Kt[(size_t)k * Nk + j] = K[(size_t)j * d + k];
}

// S[i][j] = dot(Q[i], K[j]) * scale, x -> j, y -> i
__global__ void qk_v1(const float* Q, const float* Kt, float* S, int Nq,
                      int Nk, int d, float scale) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= Nq || j >= Nk) return;
  float sum = 0.0f;
  for (int k = 0; k < d; ++k) sum += Q[i * d + k] * Kt[(size_t)k * Nk + j];
  S[(size_t)i * Nk + j] = sum * scale;
}

// tree reduction over the block, op = max or sum; all threads get the result
template <bool kIsMax>
__device__ float block_reduce(float v, float* buf) {
  const int t = threadIdx.x;
  buf[t] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (t < s) {
      buf[t] = kIsMax ? fmaxf(buf[t], buf[t + s]) : buf[t] + buf[t + s];
    }
    __syncthreads();
  }
  const float r = buf[0];
  __syncthreads();  // buf is reused by the next reduction
  return r;
}

// one block per row, thread t handles j = t, t + blockDim.x, ...
__global__ void softmax_v1(float* S, int Nk) {
  __shared__ float buf[kSoftmaxThreads];
  float* row = S + (size_t)blockIdx.x * Nk;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < Nk; j += blockDim.x) m = fmaxf(m, row[j]);
  m = block_reduce<true>(m, buf);
  float sum = 0.0f;
  for (int j = threadIdx.x; j < Nk; j += blockDim.x) {
    const float e = expf(row[j] - m);
    row[j] = e;
    sum += e;
  }
  sum = block_reduce<false>(sum, buf);
  const float inv = 1.0f / sum;
  for (int j = threadIdx.x; j < Nk; j += blockDim.x) row[j] *= inv;
}

// O[i][c] = sum_j S[i][j] * V[j][c], x -> c, y -> i
__global__ void pv_v1(const float* S, const float* V, float* O, int Nq, int Nk,
                      int d) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= Nq || c >= d) return;
  float sum = 0.0f;
  for (int j = 0; j < Nk; ++j) sum += S[(size_t)i * Nk + j] * V[j * d + c];
  O[i * d + c] = sum;
}

static void qk(const float* Q, const float* K, float* S, int Nq, int Nk,
               int d) {
  float* Kt = get_workspace(1, (size_t)Nk * d);
  const dim3 block(32, 8);
  transpose_v1<<<dim3((Nk + 31) / 32, (d + 7) / 8), block>>>(K, Kt, Nk, d);
  const dim3 grid((Nk + 31) / 32, (Nq + 7) / 8);
  qk_v1<<<grid, block>>>(Q, Kt, S, Nq, Nk, d, 1.0f / std::sqrt((float)d));
}

static void softmax(float* S, int Nq, int Nk) {
  softmax_v1<<<Nq, kSoftmaxThreads>>>(S, Nk);
}

static void pv(const float* S, const float* V, float* O, int Nq, int Nk,
               int d) {
  const dim3 block(32, 8);
  const dim3 grid((d + 31) / 32, (Nq + 7) / 8);
  pv_v1<<<grid, block>>>(S, V, O, Nq, Nk, d);
}

extern const AttentionStages kV1 = {qk, softmax, pv};

void attention_memopt_v1(const float* Q, const float* K, const float* V,
                         float* O, int N, int d) {
  run_attention(kV1, Q, K, V, O, N, N, d);
}

void attention_memopt_v1_decode(const float* Q, const float* K,
                                const float* V, float* O, int Nq, int Nk,
                                int d) {
  run_attention(kV1, Q, K, V, O, Nq, Nk, d);
}

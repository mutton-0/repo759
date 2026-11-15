// v2: qk and pv load 32 x 32 tiles into shared memory; softmax is v1's.
// Every value loaded from global memory is now reused by 32 threads from
// shared memory instead of being loaded again.
//
// The template parameter PAD is the padding of each shared row:
//   v2a (PAD = 0): in qk, the 32 threads of a warp read Ks[tx][kk], i.e.
//                  addresses 32 floats apart, which all fall in the same
//                  bank -> 32-way bank conflict.
//   v2b (PAD = 1): rows are 33 floats long, so Ks[tx][kk] is in bank
//                  (tx + kk) % 32 and the 32 reads hit 32 different banks.
// In pv the warp reads Vs[jj][tx] (consecutive) and Ps[ty][jj] (broadcast),
// which have no conflict, so pv is not padded (v2a and v2b share it).
// Padding pv would only cost: a 33-float row is not 16-byte aligned, so the
// compiler can no longer read Ps with 128-bit LDS.128 instructions.
#include <cmath>

#include "common.cuh"
#include "memopt.cuh"

extern const AttentionStages kV1;

static const int kTile = 32;

// S tile (i0.., j0..) = Q[i0.., :] K[j0.., :]^T * scale, x -> j, y -> i
template <int PAD>
__global__ void qk_v2(const float* Q, const float* K, float* S, int Nq, int Nk,
                      int d, float scale) {
  __shared__ float Qs[kTile][kTile + PAD];
  __shared__ float Ks[kTile][kTile + PAD];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int i0 = blockIdx.y * kTile, j0 = blockIdx.x * kTile;
  float sum = 0.0f;
  for (int k0 = 0; k0 < d; k0 += kTile) {
    // both loads walk k with tx, so they are coalesced; out of range -> 0
    const int k = k0 + tx;
    Qs[ty][tx] = (i0 + ty < Nq && k < d) ? Q[(i0 + ty) * d + k] : 0.0f;
    Ks[ty][tx] = (j0 + ty < Nk && k < d) ? K[(j0 + ty) * d + k] : 0.0f;
    __syncthreads();
    for (int kk = 0; kk < kTile; ++kk) sum += Qs[ty][kk] * Ks[tx][kk];
    __syncthreads();
  }
  if (i0 + ty < Nq && j0 + tx < Nk)
    S[(size_t)(i0 + ty) * Nk + j0 + tx] = sum * scale;
}

// O tile (i0.., c0..) = S[i0.., :] V[:, c0..], x -> c, y -> i
__global__ void pv_v2(const float* S, const float* V, float* O, int Nq, int Nk,
                      int d) {
  __shared__ float Ps[kTile][kTile];
  __shared__ float Vs[kTile][kTile];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int i0 = blockIdx.y * kTile, c0 = blockIdx.x * kTile;
  float sum = 0.0f;
  for (int j0 = 0; j0 < Nk; j0 += kTile) {
    Ps[ty][tx] = (i0 + ty < Nq && j0 + tx < Nk)
                     ? S[(size_t)(i0 + ty) * Nk + j0 + tx]
                     : 0.0f;
    Vs[ty][tx] =
        (j0 + ty < Nk && c0 + tx < d) ? V[(j0 + ty) * d + c0 + tx] : 0.0f;
    __syncthreads();
    for (int jj = 0; jj < kTile; ++jj) sum += Ps[ty][jj] * Vs[jj][tx];
    __syncthreads();
  }
  if (i0 + ty < Nq && c0 + tx < d) O[(i0 + ty) * d + c0 + tx] = sum;
}

template <int PAD>
static void qk(const float* Q, const float* K, float* S, int Nq, int Nk,
               int d) {
  const dim3 block(kTile, kTile);
  const dim3 grid((Nk + kTile - 1) / kTile, (Nq + kTile - 1) / kTile);
  qk_v2<PAD><<<grid, block>>>(Q, K, S, Nq, Nk, d, 1.0f / std::sqrt((float)d));
}

static void softmax(float* S, int Nq, int Nk) { kV1.softmax(S, Nq, Nk); }

static void pv(const float* S, const float* V, float* O, int Nq, int Nk,
               int d) {
  const dim3 block(kTile, kTile);
  const dim3 grid((d + kTile - 1) / kTile, (Nq + kTile - 1) / kTile);
  pv_v2<<<grid, block>>>(S, V, O, Nq, Nk, d);
}

extern const AttentionStages kV2a = {qk<0>, softmax, pv};
extern const AttentionStages kV2b = {qk<1>, softmax, pv};

void attention_memopt_v2a(const float* Q, const float* K, const float* V,
                          float* O, int N, int d) {
  run_attention(kV2a, Q, K, V, O, N, N, d);
}

void attention_memopt_v2a_decode(const float* Q, const float* K,
                                 const float* V, float* O, int Nq, int Nk,
                                 int d) {
  run_attention(kV2a, Q, K, V, O, Nq, Nk, d);
}

void attention_memopt_v2b(const float* Q, const float* K, const float* V,
                          float* O, int N, int d) {
  run_attention(kV2b, Q, K, V, O, N, N, d);
}

void attention_memopt_v2b_decode(const float* Q, const float* K,
                                 const float* V, float* O, int Nq, int Nk,
                                 int d) {
  run_attention(kV2b, Q, K, V, O, Nq, Nk, d);
}

// v3: v2b with float4 (128-bit) loads and stores.
// v2 showed that the inner loops are limited by the NUMBER of load
// instructions (two per multiply-add), so v3 moves 4 floats per instruction
// wherever a thread needs 4 consecutive floats:
//   qk:      the inner loop reads Qs[ty][kk..kk+3] and Ks[tx][kk..kk+3] as
//            float4. The rows are padded to 36 floats: 36 * 4 bytes is a
//            multiple of 16 (float4 alignment), and a 128-bit load is served
//            8 threads at a time, whose rows start in banks 0, 4, ..., 28,
//            so the 8 x 4 banks are all different (no conflict).
//   pv:      reads Ps[ty][jj..jj+3] as float4 (v2's compiler already did
//            this on its own, so the inner loop of pv does not change).
//   both:    global -> shared tile loads are float4 (256 float4 per tile,
//            done by the first 8 warps of the block).
//   softmax: reads and writes the row as float4.
// float4 needs 16-byte aligned addresses: d % 4 == 0 for Q, K, V rows and
// Nk % 4 == 0 for S rows. Other sizes fall back to the v2b / v1 stages.
#include <cmath>

#include "common.cuh"
#include "memopt.cuh"

extern const AttentionStages kV1;
extern const AttentionStages kV2b;

static const int kTile = 32;
static const int kPad = 4;
static const int kSoftmaxThreads = 256;

__global__ void qk_v3(const float* Q, const float* K, float* S, int Nq, int Nk,
                      int d, float scale) {
  __shared__ __align__(16) float Qs[kTile][kTile + kPad];
  __shared__ __align__(16) float Ks[kTile][kTile + kPad];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * kTile + tx;
  const int i0 = blockIdx.y * kTile, j0 = blockIdx.x * kTile;
  float sum = 0.0f;
  for (int k0 = 0; k0 < d; k0 += kTile) {
    if (tid < kTile * kTile / 4) {
      const int r = tid / 8, c = (tid % 8) * 4;
      float4 q = make_float4(0.0f, 0.0f, 0.0f, 0.0f), k = q;
      if (i0 + r < Nq && k0 + c < d)
        q = *reinterpret_cast<const float4*>(&Q[(i0 + r) * d + k0 + c]);
      if (j0 + r < Nk && k0 + c < d)
        k = *reinterpret_cast<const float4*>(&K[(j0 + r) * d + k0 + c]);
      *reinterpret_cast<float4*>(&Qs[r][c]) = q;
      *reinterpret_cast<float4*>(&Ks[r][c]) = k;
    }
    __syncthreads();
    for (int kk = 0; kk < kTile; kk += 4) {
      const float4 a = *reinterpret_cast<const float4*>(&Qs[ty][kk]);
      const float4 b = *reinterpret_cast<const float4*>(&Ks[tx][kk]);
      sum += a.x * b.x;
      sum += a.y * b.y;
      sum += a.z * b.z;
      sum += a.w * b.w;
    }
    __syncthreads();
  }
  if (i0 + ty < Nq && j0 + tx < Nk)
    S[(size_t)(i0 + ty) * Nk + j0 + tx] = sum * scale;
}

__global__ void softmax_v3(float* S, int Nk) {
  __shared__ float buf[kSoftmaxThreads];
  float4* row = reinterpret_cast<float4*>(S + (size_t)blockIdx.x * Nk);
  const int n4 = Nk / 4;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < n4; j += blockDim.x) {
    const float4 v = row[j];
    m = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
  }
  m = block_reduce<true>(m, buf);
  float sum = 0.0f;
  for (int j = threadIdx.x; j < n4; j += blockDim.x) {
    float4 v = row[j];
    v.x = expf(v.x - m);
    v.y = expf(v.y - m);
    v.z = expf(v.z - m);
    v.w = expf(v.w - m);
    row[j] = v;
    sum += v.x + v.y + v.z + v.w;
  }
  sum = block_reduce<false>(sum, buf);
  const float inv = 1.0f / sum;
  for (int j = threadIdx.x; j < n4; j += blockDim.x) {
    float4 v = row[j];
    v.x *= inv;
    v.y *= inv;
    v.z *= inv;
    v.w *= inv;
    row[j] = v;
  }
}

__global__ void pv_v3(const float* S, const float* V, float* O, int Nq, int Nk,
                      int d) {
  __shared__ __align__(16) float Ps[kTile][kTile];
  __shared__ __align__(16) float Vs[kTile][kTile];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * kTile + tx;
  const int i0 = blockIdx.y * kTile, c0 = blockIdx.x * kTile;
  float sum = 0.0f;
  for (int j0 = 0; j0 < Nk; j0 += kTile) {
    if (tid < kTile * kTile / 4) {
      const int r = tid / 8, c = (tid % 8) * 4;
      float4 p = make_float4(0.0f, 0.0f, 0.0f, 0.0f), v = p;
      if (i0 + r < Nq && j0 + c < Nk)
        p = *reinterpret_cast<const float4*>(
            &S[(size_t)(i0 + r) * Nk + j0 + c]);
      if (j0 + r < Nk && c0 + c < d)
        v = *reinterpret_cast<const float4*>(&V[(j0 + r) * d + c0 + c]);
      *reinterpret_cast<float4*>(&Ps[r][c]) = p;
      *reinterpret_cast<float4*>(&Vs[r][c]) = v;
    }
    __syncthreads();
    for (int jj = 0; jj < kTile; jj += 4) {
      const float4 p = *reinterpret_cast<const float4*>(&Ps[ty][jj]);
      sum += p.x * Vs[jj][tx];
      sum += p.y * Vs[jj + 1][tx];
      sum += p.z * Vs[jj + 2][tx];
      sum += p.w * Vs[jj + 3][tx];
    }
    __syncthreads();
  }
  if (i0 + ty < Nq && c0 + tx < d) O[(i0 + ty) * d + c0 + tx] = sum;
}

static void qk(const float* Q, const float* K, float* S, int Nq, int Nk,
               int d) {
  if (d % 4 != 0) {
    kV2b.qk(Q, K, S, Nq, Nk, d);
    return;
  }
  const dim3 block(kTile, kTile);
  const dim3 grid((Nk + kTile - 1) / kTile, (Nq + kTile - 1) / kTile);
  qk_v3<<<grid, block>>>(Q, K, S, Nq, Nk, d, 1.0f / std::sqrt((float)d));
}

static void softmax(float* S, int Nq, int Nk) {
  if (Nk % 4 != 0) {
    kV1.softmax(S, Nq, Nk);
    return;
  }
  softmax_v3<<<Nq, kSoftmaxThreads>>>(S, Nk);
}

static void pv(const float* S, const float* V, float* O, int Nq, int Nk,
               int d) {
  if (d % 4 != 0 || Nk % 4 != 0) {
    kV2b.pv(S, V, O, Nq, Nk, d);
    return;
  }
  const dim3 block(kTile, kTile);
  const dim3 grid((d + kTile - 1) / kTile, (Nq + kTile - 1) / kTile);
  pv_v3<<<grid, block>>>(S, V, O, Nq, Nk, d);
}

extern const AttentionStages kV3 = {qk, softmax, pv};

void attention_memopt_v3(const float* Q, const float* K, const float* V,
                         float* O, int N, int d) {
  run_attention(kV3, Q, K, V, O, N, N, d);
}

void attention_memopt_v3_decode(const float* Q, const float* K,
                                const float* V, float* O, int Nq, int Nk,
                                int d) {
  run_attention(kV3, Q, K, V, O, Nq, Nk, d);
}

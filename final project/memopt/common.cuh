#ifndef MEMOPT_COMMON_CUH
#define MEMOPT_COMMON_CUH

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t err_ = (call);                                        \
    if (err_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                   cudaGetErrorString(err_), __FILE__, __LINE__);     \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

// One version of attention split into its three stages, so that main.cu can
// time every stage on its own. Shapes: Q, O: Nq x d; K, V: Nk x d; S: Nq x Nk.
//   qk:      S = Q K^T / sqrt(d)
//   softmax: every row of S is replaced by softmax(row), in place
//   pv:      O = S V
struct AttentionStages {
  void (*qk)(const float* Q, const float* K, float* S, int Nq, int Nk, int d);
  void (*softmax)(float* S, int Nq, int Nk);
  void (*pv)(const float* S, const float* V, float* O, int Nq, int Nk, int d);
};

// Device scratch buffer with at least count floats. It only grows, so after
// the first call with the largest size there are no more cudaMalloc calls.
// slot 0 is the score matrix S; other slots are free for a version to use.
inline float* get_workspace(int slot, size_t count) {
  static float* buf[4] = {nullptr, nullptr, nullptr, nullptr};
  static size_t cap[4] = {0, 0, 0, 0};
  if (count > cap[slot]) {
    if (buf[slot] != nullptr) CUDA_CHECK(cudaFree(buf[slot]));
    CUDA_CHECK(cudaMalloc(&buf[slot], count * sizeof(float)));
    cap[slot] = count;
  }
  return buf[slot];
}

inline void run_attention(const AttentionStages& s, const float* Q,
                          const float* K, const float* V, float* O, int Nq,
                          int Nk, int d) {
  float* S = get_workspace(0, (size_t)Nq * Nk);
  s.qk(Q, K, S, Nq, Nk, d);
  s.softmax(S, Nq, Nk);
  s.pv(S, V, O, Nq, Nk, d);
}

#endif

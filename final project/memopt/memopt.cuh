#ifndef MEMOPT_CUH
#define MEMOPT_CUH

// Attention O = softmax(Q K^T / sqrt(d)) V, one head, float32, row-major.
// All pointers are DEVICE pointers. The functions only launch kernels
// (no host<->device copies); the N x N score matrix lives in a scratch
// buffer that is allocated once and reused.
//
//   attention_memopt_vX(Q, K, V, O, N, d)              Q, K, V, O: N x d
//   attention_memopt_vX_decode(Q, K, V, O, Nq, Nk, d)
//                                  Q, O: Nq x d; K, V: Nk x d
//
// The _decode entry is the same kernels with a different number of query
// rows, e.g. Nq = 1 for one new token against a KV cache of Nk tokens.
//
// Versions (each one adds one memory optimization on top of the previous);
// a line is added here together with each new version:
//   v0  naive, three kernels, threadIdx.x walks the row dimension
//   v1  coalesced: threadIdx.x walks the contiguous dimension

void attention_memopt_v0(const float* Q, const float* K, const float* V,
                         float* O, int N, int d);
void attention_memopt_v0_decode(const float* Q, const float* K,
                                const float* V, float* O, int Nq, int Nk,
                                int d);

void attention_memopt_v1(const float* Q, const float* K, const float* V,
                         float* O, int N, int d);
void attention_memopt_v1_decode(const float* Q, const float* K,
                                const float* V, float* O, int Nq, int Nk,
                                int d);

#endif

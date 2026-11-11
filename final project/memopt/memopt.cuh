#ifndef MEMOPT_CUH
#define MEMOPT_CUH

// Attention O = softmax(Q K^T / sqrt(d)) V, one head, float32, row-major.
// All pointers are DEVICE pointers. The functions only launch kernels
// (no host<->device copies); the N x N score matrix lives in a scratch
// buffer that is allocated once and reused.
//
//   attention_memopt_vX(Q, K, V, O, N, d)              Q, K, V, O: N x d
//   attention_memopt_vX_decode(Q, K, V, O, Nq, Nk, d)  Q, O: Nq x d; K, V: Nk x d
//
// The _decode entry is the same kernels with a different number of query
// rows, e.g. Nq = 1 for one new token against a KV cache of Nk tokens.
//
// Versions (each one adds one memory optimization on top of the previous);
// a line is added here together with each new version:

#endif

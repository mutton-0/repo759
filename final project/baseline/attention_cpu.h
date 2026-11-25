#ifndef ATTENTION_CPU_H
#define ATTENTION_CPU_H

// Attention O = softmax(Q K^T / sqrt(d)) V, one head, float32, row-major.
// Q, O: Nq x d; K, V: Nk x d. Prefill is Nq == Nk, decoding is a small Nq
// (e.g. 1 new token) against a KV cache of Nk tokens.
// Same shapes and layout as the CUDA versions in ../memopt, but these work on
// host pointers and never build the Nq x Nk score matrix: every query row
// uses one temporary row of Nk scores.

// Serial baseline.
void attention_serial(const float* Q, const float* K, const float* V,
                      float* O, int Nq, int Nk, int d);

// Query rows are split over the OpenMP threads (OMP_NUM_THREADS).
void attention_omp(const float* Q, const float* K, const float* V, float* O,
                   int Nq, int Nk, int d);

#endif

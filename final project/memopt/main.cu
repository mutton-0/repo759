// Driver for the memopt versions.
//   ./memopt list               print the names of all versions
//   ./memopt <version> N d [Nq] run one version, Nq query rows (default N)
//                               against N key/value rows
// Prints the time of each stage, GFLOP/s and the max error against a CPU
// reference computed in double precision.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "common.cuh"
#include "memopt.cuh"

struct Version {
  const char* name;
  const AttentionStages* stages;
};

// One line per version, added together with its .cu file.
static const std::vector<Version> kVersions = {};

static const int kWarmup = 3;
static const int kRepeat = 10;
static const double kTolerance = 1e-4;
static const int kMaxCheckRows = 256;

// O[i, :] for a single query row i, in double precision.
static void reference_row(const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V, int i, int Nk, int d,
                          std::vector<double>& out) {
  std::vector<double> s(Nk);
  const double scale = 1.0 / std::sqrt((double)d);
  double m = -INFINITY;
  for (int j = 0; j < Nk; ++j) {
    double dot = 0.0;
    for (int k = 0; k < d; ++k) dot += (double)Q[i * d + k] * K[j * d + k];
    s[j] = dot * scale;
    m = std::max(m, s[j]);
  }
  double sum = 0.0;
  for (int j = 0; j < Nk; ++j) {
    s[j] = std::exp(s[j] - m);
    sum += s[j];
  }
  out.assign(d, 0.0);
  for (int j = 0; j < Nk; ++j)
    for (int c = 0; c < d; ++c) out[c] += s[j] * V[j * d + c];
  for (int c = 0; c < d; ++c) out[c] /= sum;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "list") == 0) {
    for (const Version& v : kVersions) std::printf("%s\n", v.name);
    return 0;
  }
  if (argc < 4) {
    std::printf("usage: %s list | <version> N d [Nq]\n", argv[0]);
    return 1;
  }
  const Version* ver = nullptr;
  for (const Version& v : kVersions)
    if (std::strcmp(v.name, argv[1]) == 0) ver = &v;
  if (ver == nullptr) {
    std::printf("unknown version %s\n", argv[1]);
    return 1;
  }
  const int Nk = std::atoi(argv[2]);
  const int d = std::atoi(argv[3]);
  const int Nq = argc > 4 ? std::atoi(argv[4]) : Nk;

  // random inputs in [-1, 1], fixed seed so every version sees the same data
  std::vector<float> Q((size_t)Nq * d), K((size_t)Nk * d), V((size_t)Nk * d);
  std::mt19937 gen(759);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& x : Q) x = dist(gen);
  for (float& x : K) x = dist(gen);
  for (float& x : V) x = dist(gen);

  float *dQ, *dK, *dV, *dO;
  CUDA_CHECK(cudaMalloc(&dQ, Q.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK, K.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV, V.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dO, Q.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, K.data(), K.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, V.data(), V.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  float* dS = get_workspace(0, (size_t)Nq * Nk);

  // events around each stage; for every stage and for the total we keep the
  // minimum over kRepeat runs, after kWarmup untimed runs
  cudaEvent_t ev[4];
  for (cudaEvent_t& e : ev) CUDA_CHECK(cudaEventCreate(&e));
  float best[4] = {1e30f, 1e30f, 1e30f, 1e30f};  // qk, softmax, pv, total
  const AttentionStages& s = *ver->stages;
  for (int r = 0; r < kWarmup + kRepeat; ++r) {
    CUDA_CHECK(cudaEventRecord(ev[0]));
    s.qk(dQ, dK, dS, Nq, Nk, d);
    CUDA_CHECK(cudaEventRecord(ev[1]));
    s.softmax(dS, Nq, Nk);
    CUDA_CHECK(cudaEventRecord(ev[2]));
    s.pv(dS, dV, dO, Nq, Nk, d);
    CUDA_CHECK(cudaEventRecord(ev[3]));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(ev[3]));
    if (r < kWarmup) continue;
    float ms[4];
    for (int t = 0; t < 3; ++t)
      CUDA_CHECK(cudaEventElapsedTime(&ms[t], ev[t], ev[t + 1]));
    CUDA_CHECK(cudaEventElapsedTime(&ms[3], ev[0], ev[3]));
    for (int t = 0; t < 4; ++t) best[t] = std::min(best[t], ms[t]);
  }

  std::vector<float> O(Q.size());
  CUDA_CHECK(cudaMemcpy(O.data(), dO, O.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // check up to kMaxCheckRows rows spread evenly over [0, Nq), last row
  // included; a full CPU reference for N = 8192 would take too long
  const int rows = std::min(Nq, kMaxCheckRows);
  double max_err = 0.0;
  std::vector<double> ref;
  for (int r = 0; r < rows; ++r) {
    const int i = rows == 1 ? 0 : (int)((long long)r * (Nq - 1) / (rows - 1));
    reference_row(Q, K, V, i, Nk, d, ref);
    for (int c = 0; c < d; ++c) {
      const double err = std::fabs(ref[c] - (double)O[(size_t)i * d + c]);
      if (!(err <= max_err)) max_err = err;  // also catches NaN
    }
  }
  const bool pass = max_err < kTolerance;

  // 2*Nq*Nk*d flops for QK^T and 2*Nq*Nk*d for PV; softmax is not counted.
  // For softmax we report the bandwidth of the minimum traffic: read S once
  // and write it once.
  const double flops = 4.0 * Nq * Nk * d;
  const double gflops = flops / (best[3] * 1e6);
  const double softmax_gbs = 8.0 * Nq * Nk / (best[1] * 1e6);

  std::printf("version %s  Nq %d  Nk %d  d %d\n", ver->name, Nq, Nk, d);
  std::printf("time_ms  qk %.4f  softmax %.4f  pv %.4f  total %.4f\n",
              best[0], best[1], best[2], best[3]);
  std::printf("gflops %.1f  softmax_gbs %.1f\n", gflops, softmax_gbs);
  std::printf("max_err %.3e (%d rows checked)  %s\n", max_err, rows,
              pass ? "PASS" : "FAIL");
  // one machine-readable line for the scripts
  std::printf("csv,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%.3e,%s\n", ver->name,
              Nq, Nk, d, best[0], best[1], best[2], best[3], gflops, max_err,
              pass ? "PASS" : "FAIL");

  for (cudaEvent_t& e : ev) CUDA_CHECK(cudaEventDestroy(e));
  CUDA_CHECK(cudaFree(dQ));
  CUDA_CHECK(cudaFree(dK));
  CUDA_CHECK(cudaFree(dV));
  CUDA_CHECK(cudaFree(dO));
  return pass ? 0 : 2;
}

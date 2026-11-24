// CPU baseline and benchmark driver for attention.
//   ./bench list
//   ./bench <version> Nk d [Nq] [--reps R] [--dump DIR]
// Nq defaults to Nk (prefill); Nq = 1 is the decoding case. Inputs are random
// in [-1, 1] with the same seed and order as ../memopt/main.cu, so the CPU and
// GPU versions see exactly the same data. OpenMP threads: OMP_NUM_THREADS.
// --dump writes Q, K, V and the output as raw float32 files (for torch_ref.py).

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "attention_cpu.h"

struct Version {
  const char* name;
  void (*fn)(const float*, const float*, const float*, float*, int, int, int);
};
static const Version kVersions[] = {
    {"cpu_serial", attention_serial},
    {"cpu_omp", attention_omp},
};

static const double kTolerance = 1e-4;
static const int kMaxCheckRows = 16;

// reference for one output row in double precision
static void reference_row(const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V, int i, int Nk, int d,
                          std::vector<double>& out) {
  std::vector<double> s(Nk);
  double m = -1e300;
  for (int j = 0; j < Nk; ++j) {
    double dot = 0.0;
    for (int c = 0; c < d; ++c)
      dot += (double)Q[(size_t)i * d + c] * (double)K[(size_t)j * d + c];
    s[j] = dot / std::sqrt((double)d);
    m = std::max(m, s[j]);
  }
  double sum = 0.0;
  for (int j = 0; j < Nk; ++j) {
    s[j] = std::exp(s[j] - m);
    sum += s[j];
  }
  out.assign(d, 0.0);
  for (int j = 0; j < Nk; ++j)
    for (int c = 0; c < d; ++c) out[c] += s[j] * (double)V[(size_t)j * d + c];
  for (int c = 0; c < d; ++c) out[c] /= sum;
}

static void dump(const std::string& dir, const char* name,
                 const std::vector<float>& v) {
  const std::string path = dir + "/" + name;
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(1);
  }
  std::fwrite(v.data(), sizeof(float), v.size(), f);
  std::fclose(f);
}

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "list") == 0) {
    for (const Version& v : kVersions) std::printf("%s\n", v.name);
    return 0;
  }
  if (argc < 4) {
    std::printf("usage: %s list | <version> Nk d [Nq] [--reps R] [--dump DIR]\n",
                argv[0]);
    return 1;
  }
  const Version* ver = nullptr;
  for (const Version& v : kVersions)
    if (std::strcmp(v.name, argv[1]) == 0) ver = &v;
  if (!ver) {
    std::printf("unknown version %s\n", argv[1]);
    return 1;
  }
  const int Nk = std::atoi(argv[2]);
  const int d = std::atoi(argv[3]);
  int Nq = Nk;
  int reps = 5;
  std::string dump_dir;
  for (int a = 4; a < argc; ++a) {
    if (std::strcmp(argv[a], "--reps") == 0 && a + 1 < argc)
      reps = std::atoi(argv[++a]);
    else if (std::strcmp(argv[a], "--dump") == 0 && a + 1 < argc)
      dump_dir = argv[++a];
    else
      Nq = std::atoi(argv[a]);
  }

  std::vector<float> Q((size_t)Nq * d), K((size_t)Nk * d), V((size_t)Nk * d);
  std::mt19937 gen(759);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& x : Q) x = dist(gen);
  for (float& x : K) x = dist(gen);
  for (float& x : V) x = dist(gen);
  std::vector<float> O(Q.size());

  // one untimed run, then the minimum over reps runs
  ver->fn(Q.data(), K.data(), V.data(), O.data(), Nq, Nk, d);
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    ver->fn(Q.data(), K.data(), V.data(), O.data(), Nq, Nk, d);
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  // check rows spread evenly over [0, Nq), last row included
  const int rows = std::min(Nq, kMaxCheckRows);
  double max_err = 0.0;
  std::vector<double> ref;
  for (int r = 0; r < rows; ++r) {
    const int i = rows == 1 ? Nq - 1 : (int)((long long)r * (Nq - 1) / (rows - 1));
    reference_row(Q, K, V, i, Nk, d, ref);
    for (int c = 0; c < d; ++c)
      max_err = std::max(max_err, std::fabs(ref[c] - (double)O[(size_t)i * d + c]));
  }
  const bool pass = max_err < kTolerance;

  if (!dump_dir.empty()) {
    dump(dump_dir, "Q.bin", Q);
    dump(dump_dir, "K.bin", K);
    dump(dump_dir, "V.bin", V);
    dump(dump_dir, "O.bin", O);
  }

  const int threads = std::strcmp(ver->name, "cpu_omp") == 0 ? omp_get_max_threads() : 1;
  const double gflops = 4.0 * Nq * Nk * d / (best * 1e6);
  std::printf("version %s  Nq %d  Nk %d  d %d  threads %d\n", ver->name, Nq, Nk, d, threads);
  std::printf("time_ms total %.4f\n", best);
  std::printf("gflops %.1f\n", gflops);
  std::printf("max_err %.3e (%d rows checked)  %s\n", max_err, rows, pass ? "PASS" : "FAIL");
  // same columns as the csv line of ../memopt/main.cu (no per-stage times on the CPU)
  std::printf("csv,%s,%d,%d,%d,-,-,-,%.4f,%.1f,%.3e,%s\n", ver->name, Nq, Nk, d,
              best, gflops, max_err, pass ? "PASS" : "FAIL");
  return pass ? 0 : 2;
}

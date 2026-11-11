# memopt: memory-access optimizations for attention (part C)

Attention for one head, float32, row-major:

    O = softmax(Q K^T / sqrt(d)) V,   Q, K, V, O: N x d,   S = Q K^T: N x N

This folder starts from a naive three-kernel CUDA version (QK^T, softmax, PV)
and adds one memory-access optimization per version, so every version can be
compared with the one before it. Each version is its own `.cu` file.

## Plan

| Version | File | Change (on top of the previous version) | Status |
|---|---|---|---|
| v0 | `v0_naive.cu` | naive: one thread per output, thread index walks the row dimension | todo |
| v1 | `v1_coalesced.cu` | coalescing: thread index walks the contiguous dimension | todo |
| v2a / v2b | `v2_smem_pad.cu` | shared-memory tiles without / with padding (bank conflicts) | todo |
| v3 | `v3_float4.cu` | `float4` vectorized loads | todo |
| v4 | `v4_regblock.cu` | register blocking (each thread computes a small tile) | todo |
| v5 | `v5_unroll.cu` | loop unrolling with compile-time sizes | todo |
| v6 | `v6_double_buffer.cu` | double buffering of the shared-memory tiles | todo |

Shared-memory tiling itself is part B's topic (`cuda_basic/tiled.cu`). It
appears here only because padding needs a tiled kernel; the gain claimed by
part C for v2 is v2a -> v2b.

## Interface (for the benchmark and the other parts)

`memopt.cuh` declares, for every version `vX`:

```cpp
// Q, K, V, O: N x d
void attention_memopt_vX(const float* Q, const float* K, const float* V,
                         float* O, int N, int d);
// decode shape: Q, O: Nq x d (e.g. Nq = 1); K, V: Nk x d (the KV cache)
void attention_memopt_vX_decode(const float* Q, const float* K,
                                const float* V, float* O, int Nq, int Nk,
                                int d);
```

- All pointers are **device** pointers; the functions only launch kernels,
  there are no host/device copies inside.
- The N x N score matrix is kept in a scratch buffer that is allocated on
  the first call and reused (`get_workspace` in `common.cuh`), so a timed
  call does not include `cudaMalloc`.
- Inside, every version is three stages (`AttentionStages` in `common.cuh`):
  `qk` (S = Q K^T / sqrt(d)), `softmax` (row-wise, in place) and `pv`
  (O = S V). The driver times each stage separately.

## Build and run

```bash
make                 # Euler, RTX 4000 Ada (sm_89)
make ARCH=sm_120     # another GPU: set its compute capability
./memopt list                 # names of all versions
./memopt <version> N d [Nq]   # Nq query rows (default N), N key/value rows
```

On Euler (compute nodes only), from this folder: `sbatch run_euler.sh`.

Output of one run:

```
version v1  Nq 4096  Nk 4096  d 64
time_ms  qk ...  softmax ...  pv ...  total ...
gflops ...  softmax_gbs ...
max_err ... (256 rows checked)  PASS
csv,v1,4096,4096,64,...
```

## Method

- **Inputs:** uniform random in [-1, 1] with a fixed seed, so every version
  gets the same data.
- **Correctness:** each output row is compared with a CPU reference in double
  precision. Up to 256 rows spread evenly over the output are checked (all
  rows if Nq <= 256, the first and last row always). Pass if the max absolute
  error is below 1e-4, the team's criterion against PyTorch fp32.
- **Timing:** CUDA events around each stage; 3 warm-up runs, then the minimum
  over 10 runs, for each stage and for the total.
- **GFLOP/s:** `4 N^2 d / time` (2 N^2 d for QK^T and 2 N^2 d for PV;
  softmax not counted).
- **softmax_gbs:** `8 N^2 / softmax time`, i.e. the bandwidth if S were read
  once and written once (the minimum traffic of the softmax stage).
- **Sizes:** main table N = 4096, d = 64; sweeps N = 256 ... 8192 for d = 64
  and d = 128; decode shape Nq = 1.

## Results (Euler, RTX 4000 Ada)

N = 4096, d = 64. Each row is "before -> after" for one optimization.

| Version | qk (ms) | softmax (ms) | pv (ms) | total (ms) | GFLOP/s | speedup vs previous | max err |
|---|---|---|---|---|---|---|---|
| v0 | | | | | | - | |

## Notes per optimization

One section per version: what was slow, what changed, why it is faster, the
measured before/after, and the profiler evidence. Added with each version.

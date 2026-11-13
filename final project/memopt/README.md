# memopt: memory-access optimizations for attention (part C)

Attention for one head, float32, row-major:

    O = softmax(Q K^T / sqrt(d)) V,   Q, K, V, O: N x d,   S = Q K^T: N x N

This folder starts from a naive three-kernel CUDA version (QK^T, softmax, PV)
and adds one memory-access optimization per version, so every version can be
compared with the one before it. Each version is its own `.cu` file.

## Plan

| Version | File | Change (on top of the previous version) | Status |
|---|---|---|---|
| v0 | `v0_naive.cu` | naive: one thread per output, thread index walks the row dimension | done |
| v1 | `v1_coalesced.cu` | coalescing: thread index walks the contiguous dimension | done |
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
Profiler counters for some versions: `sbatch ncu_euler.sh v0 v1`.

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

## Results (Euler, RTX 4000 SFF Ada, cc 8.9)

N = 4096, d = 64. Each row is "before -> after" for one optimization.

| Version | qk (ms) | softmax (ms) | pv (ms) | total (ms) | GFLOP/s | speedup vs previous | max err |
|---|---|---|---|---|---|---|---|
| v0 | 10.549 | 3.064 | 10.553 | 24.166 | 177.7 | - | 1.2e-07 |
| v1 | | | | | | | |

## Notes per optimization

One section per version: what was slow, what changed, why it is faster, the
measured before/after, and the profiler evidence. Added with each version.

### v0: naive baseline (`v0_naive.cu`)

Three kernels, written straight from the formula, one thread per output
element:

| Kernel | Thread mapping | Launch |
|---|---|---|
| `qk_v0` | x -> row i of S, y -> column j | 32 x 8 threads per block |
| `softmax_v0` | one thread per row (max, exp + sum, divide) | 128 threads per block |
| `pv_v0` | x -> row i of O, y -> column c | 32 x 8 threads per block |

The problem is the mapping x -> row. A warp is 32 threads with consecutive
`threadIdx.x`, and the hardware serves one warp-wide load in 32-byte
sectors. If the 32 threads read 32 consecutive floats (128 bytes) that is
4 sectors; here they read one float from each of 32 different rows:

| Access (per warp, per loop step) | Addresses of the 32 threads | Sectors | Useful bytes |
|---|---|---|---|
| `Q[i*d + k]` in qk | 32 rows, stride d | 32 | 12.5 % |
| `K[j*d + k]` in qk | same address | 1 (broadcast) | - |
| `S[i*Nk + j]` store in qk | 32 rows, stride Nk | 32 | 12.5 % |
| `row[j]` in softmax | 32 rows, stride Nk | 32 | 12.5 % |
| `S[i*Nk + j]` in pv | 32 rows, stride Nk | 32 | 12.5 % |
| `V[j*d + c]` in pv | same address | 1 (broadcast) | - |
| `O[i*d + c]` store in pv | 32 rows, stride d | 32 | 12.5 % |

So most memory instructions move 8x more sectors than needed. Part of this is
hidden by the L1/L2 caches (a thread reads `Q[i*d + k]` for k = 0, 1, 2, ...,
so the rest of a sector is often still in cache for the next step), which
is why the measured gain of v1 can be smaller than 8x.

The softmax kernel has a second problem: only Nq threads in total. For
N = 4096 that is 128 warps for the whole GPU, and in the decode shape
(Nq = 1) a single thread walks the whole row of length Nk.

Measured on Euler (N = 4096, d = 64): 24.17 ms in total, of which QK^T
10.55 ms (44 %), softmax 3.06 ms (13 %) and PV 10.55 ms (44 %). The two
matrix products take almost 90 % of the time, so they are where the later
optimizations (tiling, float4, register blocking) matter most.

### v1: coalesced access (`v1_coalesced.cu`)

Same arithmetic as v0; only the mapping of threads to data changes, so that
`threadIdx.x` walks the dimension that is contiguous in memory:

| Kernel | v0 mapping | v1 mapping |
|---|---|---|
| qk | x -> row i | x -> column j; K is first transposed to Kt (d x Nk) |
| softmax | one thread per row | one block (256 threads) per row, threads stride over j |
| pv | x -> row i | x -> column c |

Accesses of one warp after the change:

| Access | Addresses of the 32 threads | Sectors |
|---|---|---|
| `Q[i*d + k]` in qk | same address | 1 (broadcast) |
| `Kt[k*Nk + j]` in qk | 32 consecutive floats | 4 |
| `S[i*Nk + j]` store in qk | 32 consecutive floats | 4 |
| `row[j]` in softmax | 32 consecutive floats | 4 |
| `S[i*Nk + j]` in pv | same address | 1 (broadcast) |
| `V[j*d + c]` in pv | 32 consecutive floats | 4 |
| `O[i*d + c]` store in pv | 32 consecutive floats | 4 |

Why K is transposed: with x -> j, reading `K[j*d + k]` directly would put
neighboring threads d floats apart, the same problem as before. The
transpose kernel costs one pass over K (Nk x d floats, 1 MB for N = 4096,
d = 64) and is included in the qk time, while the qk kernel goes over all
of Kt once for every query row, so the extra pass is small (its own time is
in the profiler output).

The softmax change also fixes the parallelism problem of v0: there are now
Nq blocks of 256 threads instead of Nq threads, and the decode shape
(Nq = 1) uses a whole block instead of one thread. The max and the sum of a
row are combined with a shared-memory tree reduction (`block_reduce`); a
warp-shuffle reduction is part B's topic.

# memopt: memory-access optimizations for attention (part C)

Attention for one head, float32, row-major:

    O = softmax(Q K^T / sqrt(d)) V,   Q, K, V, O: N x d,   S = Q K^T: N x N

This folder starts from a naive three-kernel CUDA version (QK^T, softmax, PV)
and adds one memory-access optimization per version, so every version can be
compared with the one before it. Each version is its own `.cu` file.

## Summary (updated with every version)

Euler, RTX 4000 SFF Ada, N = 4096, d = 64, all versions pass (max error
about 1e-7, limit 1e-4). Details and profiler evidence are in the notes
per version below.

| Version | Optimization | Total (ms) | vs previous | vs v0 |
|---|---|---|---|---|
| v0 | naive baseline | 24.376 | - | 1.0x |
| v1 | coalesced access | 3.338 | 7.30x | 7.3x |
| v2a | shared-memory tiles (no padding) | 4.606 | 0.72x | 5.3x |
| v2b | + padding of the qk tiles | 3.270 | 1.41x | 7.5x |
| v3 | float4 (128-bit) loads | pending | | |

Main findings so far:

1. **Coalescing (v0 -> v1, 7.3x).** Mapping `threadIdx.x` to the contiguous
   dimension cuts global sectors per request from 32 to 4; the measured
   gain (7.3x) is close to the predicted 8x.
2. **Bank conflicts (v2a -> v2b, 1.41x).** Reading a K tile by column
   gives a 32-way conflict (17 wavefronts per shared load instead of 1).
   One float of padding per row removes all conflicts; qk gets 1.89x
   faster.
3. **Padding is not free.** Padding a tile that has no conflict (pv) breaks
   16-byte alignment, the compiler loses `LDS.128`, and pv gets slower.
4. **Tiling alone did not beat v1 for qk.** L1 and shared memory are the
   same hardware on Ada; v1 and v2b both issue 16.78 M load instructions
   in the inner loop. The real cost is load instructions per
   multiply-add, which v3 (float4) and v4 (register blocking) reduce.
5. **softmax is memory-bound after v1** (259 GB/s of the 280 GB/s peak at
   N = 4096); whether S fits in L2 (16.8 MB at N = 2048, 64 MB at
   N = 4096) decides which limit it hits.

## Plan

| Version | File | Change (on top of the previous version) | Status |
|---|---|---|---|
| v0 | `v0_naive.cu` | naive: one thread per output, thread index walks the row dimension | done |
| v1 | `v1_coalesced.cu` | coalescing: thread index walks the contiguous dimension | done |
| v2a / v2b | `v2_smem_pad.cu` | shared-memory tiles without / with padding (bank conflicts) | done |
| v3 | `v3_float4.cu` | `float4` vectorized loads | done |
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
Profiler counters for some versions, e.g. `sbatch ncu_euler.sh v2b v3`.

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

N = 4096, d = 64. Each row is "before -> after" for one optimization; the
speedup is against the row above. All rows come from the same Slurm job
(job 121247), so they can be compared directly.

| Version | qk (ms) | softmax (ms) | pv (ms) | total (ms) | GFLOP/s | speedup vs previous | max err |
|---|---|---|---|---|---|---|---|
| v0 | 10.685 | 3.104 | 10.587 | 24.376 | 176.2 | - | 1.2e-07 |
| v1 | 1.364 | 0.520 | 1.446 | 3.338 | 1286.6 | 7.30x | 9.0e-08 |
| v2a | 2.842 | 0.523 | 1.238 | 4.606 | 932.5 | 0.72x | 9.0e-08 |
| v2b | 1.507 | 0.520 | 1.240 | 3.270 | 1313.6 | 1.41x | 9.0e-08 |
| v3 | | | | | | | |

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

Measured on Euler (N = 4096, d = 64): 24.27 ms in total, of which QK^T
10.62 ms (44 %), softmax 3.07 ms (13 %) and PV 10.58 ms (44 %). The two
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

Measured on Euler (N = 4096, d = 64, job 118707), v0 -> v1:

| Stage | v0 (ms) | v1 (ms) | Speedup |
|---|---|---|---|
| qk (incl. transpose) | 10.621 | 1.352 | 7.9x |
| softmax | 3.071 | 0.518 | 5.9x |
| pv | 10.577 | 1.428 | 7.4x |
| total | 24.275 | 3.306 | 7.3x |

The gain is close to the 8x predicted by the sector count. The softmax now
moves S at 259 GB/s counting one read and one write, close to the 280 GB/s
peak bandwidth on the card's spec sheet (the kernel passes over a row three
times, but only the first pass comes from DRAM, see below). So after v1 the
softmax stage is memory-bound and the remaining time is in the two matrix
products (84 % of the total).

Profiler evidence (`sbatch ncu_euler.sh v0 v1`, N = 2048, d = 64, first
call of each kernel). Sectors per request = sectors / requests of the
global loads (ld) and stores (st):

| Kernel | ld sectors/request | st sectors/request | ld bytes used per sector | ld sectors in total |
|---|---|---|---|---|
| qk | 16.5 -> 2.5 | 32 -> 4 | 12.5 % -> 82.5 % | 276.8 M -> 41.9 M |
| softmax | 32 -> 4 | 32 -> 4 | 12.5 % -> 100 % | 12.6 M -> 1.6 M |
| pv | 16.5 -> 2.5 | 32 -> 4 | 12.5 % -> 82.5 % | 276.8 M -> 41.9 M |

These match the tables above exactly. In qk and pv every loop step has two
loads, one per operand: in v0 that is 32 + 1 (broadcast) sectors, so
(32 + 1) / 2 = 16.5 per request; in v1 it is 1 (broadcast) + 4, so 2.5. The
broadcast load uses only 4 of the 32 bytes of its sector, which is why v1
reaches 82.5 % and not 100 % ((4 + 4 x 32) / (5 x 32) = 82.5 %).

What the DRAM counters add:

- qk reads only 1.05 MB from DRAM in both v0 and v1, i.e. Q and K once.
  At this size the data stays in the L2 cache, so the 8x waste of v0 was
  traffic between L1 and L2, not DRAM traffic. Coalescing fixes it anyway
  because every sector still costs a transaction.
- softmax reads 16.78 MB from DRAM, which is exactly S (2048 x 2048 x 4 B)
  once, although it passes over each row three times: the second and
  third pass hit in cache.
- The transpose in v1 takes 8.9 us next to 1.00 ms for the qk kernel
  (under 1 %); its loads are 12.5 % efficient, as expected, but it does
  not matter.
- softmax_v1 has a few shared-memory bank conflicts (4,046 in total, about
  2 per block), which is negligible.

Note: ncu fixes the clocks and flushes the caches before each kernel, so its
`gpu__time_duration` is longer than the event timings; it is only used to
compare kernels with each other, not as the reported time.

The same run also gives the event timings at N = 2048, where S is only
16.8 MB: there the v1 softmax reaches 683 GB/s, above the DRAM peak, because
S written by qk is still in L2. At N = 4096 (S = 64 MB) it falls to
259 GB/s, close to DRAM peak. The size of the problem relative to the L2
cache therefore changes which limit a kernel hits.

### v2: shared-memory tiles and padding (`v2_smem_pad.cu`)

**Tiling (v1 -> v2a, part B's topic, listed for completeness).** qk and pv
are matrix products. A block of 32 x 32 threads computes a 32 x 32 tile of
the output; in steps of 32 along the inner dimension it loads one 32 x 32
tile of each operand into shared memory (coalesced, one float per thread)
and then every thread reads 32 values of each tile from shared memory. Each
float loaded from global memory is now used by 32 threads. qk loads both Q
and K tiles along k, so the transpose kernel of v1 is no longer needed.
softmax is unchanged (v1's kernel).

**Bank conflicts (v2a).** Shared memory has 32 banks, 4 bytes wide; the
float at index `a` sits in bank `a % 32`. A warp's shared load takes one
pass if the 32 threads hit 32 different banks (or the same address, a
broadcast); threads that hit different addresses in the same bank are
served one after the other. In qk a warp has fixed `ty` and `tx` = 0..31:

| Read in qk | Index in v2a (row = 32 floats) | Bank | Passes |
|---|---|---|---|
| `Qs[ty][kk]` | `ty*32 + kk` | same address for all | 1 (broadcast) |
| `Ks[tx][kk]` | `tx*32 + kk` | `kk` for all 32 threads | **32** |

So the inner loop of qk in v2a is serialized 32 ways. The tile of K has to
be read along a column because S = Q K^T needs row j of K for column j of S.

**Padding (v2a -> v2b, part C's gain).** Declaring the qk tiles as
`[32][33]` moves every row one bank further:

| Read in qk | Index in v2b (row = 33 floats) | Bank | Passes |
|---|---|---|---|
| `Ks[tx][kk]` | `tx*33 + kk` | `(tx + kk) % 32`, all different | 1 |

The cost is 32 unused floats per tile (4 % more shared memory).

**Why pv is not padded.** In pv a warp reads `Ps[ty][jj]` (broadcast) and
`Vs[jj][tx]` (32 consecutive floats, 32 banks), which is already conflict
free. We first padded both kernels and pv got slower. The SASS
(`cuobjdump -sass`, sm_89) shows why: with 32-float rows, `Ps[ty][jj..jj+3]`
is 16-byte aligned and the compiler reads it with one 128-bit `LDS.128`;
with 33-float rows the alignment is lost and every read is a scalar `LDS`.
Shared-load instructions per 32-wide tile:

| Kernel | rows of 32 floats | rows of 33 floats |
|---|---|---|
| qk | 16 `LDS.128` (Ks with 32-way conflicts) | 64 `LDS` (no conflicts) |
| pv | 8 `LDS.128` + 32 `LDS` = 40 | 64 `LDS` |

For qk, four times more load instructions is a good trade for removing the
32-way conflict; for pv it is pure cost. So v2b pads only the qk tiles, and
pv is the same in v2a and v2b. Padding is a fix for a specific access
pattern, not a free improvement.

Measured on Euler (N = 4096, d = 64, job 121247):

| Stage | v1 (ms) | v2a (ms) | v2b (ms) | v2a -> v2b |
|---|---|---|---|---|
| qk | 1.364 | 2.842 | 1.507 | 1.89x |
| softmax | 0.520 | 0.523 | 0.520 | - |
| pv | 1.446 | 1.238 | 1.240 | same kernel |
| total | 3.338 | 4.606 | 3.270 | 1.41x |

- Padding (part C's gain) makes qk 1.89x faster and the whole attention
  1.41x faster.
- Without padding, tiling makes qk 2.1x **slower** than v1: the 32-way
  conflicts cost more than the reuse gains.
- Even with padding, the tiled qk (1.507 ms) is not faster than the simple
  coalesced qk of v1 (1.364 ms); the tiled pv is 1.17x faster than v1's.

Profiler evidence (`sbatch ncu_euler.sh v1 v2a v2b`, job 121248, N = 2048,
d = 64). A *wavefront* is one pass of the shared-memory pipeline; a
conflict-free 32-bit warp load needs exactly one.

| qk kernel | shared-load instructions | wavefronts | wavefronts / instruction | bank conflicts | ncu time |
|---|---|---|---|---|---|
| v2a (rows of 32) | 4.19 M (`LDS.128`) | 71.3 M | 17.0 | 58.7 M | 2.31 ms |
| v2b (rows of 33) | 16.78 M (`LDS`) | 16.78 M | 1.0 | 0 | 1.21 ms |

The counts match the code exactly. v2a: 64 x 64 blocks x 32 warps x 2
k-tiles x 16 `LDS.128` = 4.19 M instructions; half of them read `Ks` with a
32-way conflict (32 wavefronts each) and half read `Qs` as a broadcast (2
wavefronts each), (32 + 2) / 2 = 17. v2b issues 4x more instructions
(scalar instead of 128-bit), but each takes one wavefront, so the total
work of the shared-memory pipeline drops 4.25x (71.3 M -> 16.8 M
wavefronts) and qk gets 1.9x faster.

**Why the tiled qk is not faster than v1.** Tiling did what it should for
global memory: global load requests of qk drop 32x (16.78 M in v1 ->
0.52 M in v2). But the inner loop still needs two loads per multiply-add,
now from shared memory instead of L1:

| Kernel | load instructions in the inner loop | where they are served |
|---|---|---|
| qk v1 | 16.78 M global loads | mostly L1 hits |
| qk v2b | 16.78 M shared loads | shared memory |
| pv v1 | 16.78 M global loads | mostly L1 hits |
| pv v2 | 10.49 M shared loads (8 `LDS.128` + 32 `LDS` per tile) | shared memory |

On Ada, L1 and shared memory are the same on-chip memory, so the same
number of load instructions takes about the same time, and v2 adds a
`__syncthreads()` pair per tile. qk v2b therefore ends up level with v1
(slightly slower), while pv v2 needs 1.6x fewer load instructions (thanks
to `LDS.128`) and is 1.17x faster. A padded pv would have needed 16.78 M
scalar loads, the same as v1, which is why padding it did not pay off.

The quantity to reduce from here on is **load instructions per
multiply-add**: wider loads (`float4`, v3) and reusing each loaded value
for several outputs held in registers (register blocking, v4).

### v3: float4 loads (`v3_float4.cu`)

v2 showed that the inner loops are limited by the number of load
instructions, two per multiply-add. A `float4` load moves 16 bytes in one
instruction, so v3 uses it wherever a thread needs 4 consecutive floats.

| Where | v2b | v3 |
|---|---|---|
| qk inner loop | `Qs[ty][kk]`, `Ks[tx][kk]`: 64 scalar `LDS` per tile | `Qs[ty][kk..kk+3]`, `Ks[tx][kk..kk+3]`: 16 `LDS.128` per tile |
| qk / pv tile loads (global -> shared) | 1 float per thread, 1024 threads | 1 float4 per thread, first 256 threads (8 warps) |
| pv inner loop | 8 `LDS.128` + 32 `LDS` (compiler already vectorized `Ps`) | same |
| softmax | scalar loads/stores | float4 loads/stores |

**Padding for float4.** A float4 must be 16-byte aligned, so a 33-float
row (v2b) cannot be used. Rows of 36 floats are aligned (144 bytes), and
they are also conflict-free for 128-bit loads: the hardware serves a
128-bit warp load 8 threads at a time; thread `tx` reads banks
`(4*tx + kk) % 32 ... +3`, so the 8 threads of one pass cover banks 0-31
exactly once.

| Rows of | `Ks[tx][kk]` read | Instructions per tile | Conflict |
|---|---|---|---|
| 32 floats (v2a) | `LDS.128` | 16 | 32-way |
| 33 floats (v2b) | scalar `LDS` | 64 | none |
| 36 floats (v3) | `LDS.128` | 16 | none |

So v3 combines the instruction count of v2a with the conflict-free access
of v2b. pv's inner loop is unchanged (each thread needs a different column
of `Vs`, which is not 4 consecutive floats for one thread; that needs
register blocking, v4), so only its tile loads change.

**Limits.** float4 needs 16-byte aligned addresses: `d % 4 == 0` (rows of
Q, K, V) and `Nk % 4 == 0` (rows of S). For other sizes the affected stages
fall back to the v2b / v1 kernels; all sizes used in this project satisfy
both.

# baseline: CPU versions, PyTorch reference and benchmark harness (parts A + E)

Attention for one head, float32, row-major (same as `../memopt`):

    O = softmax(Q K^T / sqrt(d)) V,   Q, O: Nq x d,   K, V: Nk x d

Prefill is `Nq == Nk`; the decoding case is `Nq = 1` against a KV cache of `Nk` tokens.

## Files

| File | What it is |
|---|---|
| `attention_cpu.h/.cpp` | `attention_serial` and `attention_omp` (query rows split over OpenMP threads). No N x N score matrix: each row uses one temporary row of Nk scores. |
| `bench.cpp` | Driver: `./bench <cpu_serial\|cpu_omp> Nk d [Nq] [--reps R] [--dump DIR]`. Min of R runs (default 5) after one warmup, checks 16 rows against a double-precision reference (limit 1e-4). Inputs use the same seed (759), order and range as `../memopt/main.cu`, so CPU and GPU see the same data. |
| `torch_ref.py` | Reads the `--dump` files, checks `torch` (fp32) against the C++ output and times `torch_naive` (matmul + softmax) and `torch_sdpa` (`scaled_dot_product_attention`) on CPU. |
| `plot.py` | Reads the `csv,...` lines of `bench`, `torch_ref.py` and `../memopt/memopt` and draws time vs N (prefill) and time vs Nk (decoding). Takes the best of repeated runs. |
| `run_euler.sh` | CPU sweep (serial/OpenMP, thread scaling, decoding). `sbatch run_euler.sh` |
| `run_compare.sh` | Same sizes for CPU, PyTorch and `memopt` v0 / v3 in one job (needs a CPU PyTorch in `~/venv-torch`). |
| `results/` | Raw outputs of the runs below and the two plots. |

The csv line has the columns of `../memopt/main.cu`:
`csv,version,Nq,Nk,d,qk_ms,softmax_ms,pv_ms,total_ms,gflops,max_err,status`
(the CPU and PyTorch versions have no per-stage times, shown as `-`).

## Results (Euler, d = 64, 8 CPU threads, RTX 4000 Ada for the CUDA rows)

Prefill, total time in ms (best of runs; the CUDA times are kernel time only, without copies):

| N | cpu_serial | cpu_omp | torch_naive | torch_sdpa | memopt v0 | memopt v3 |
|---|---|---|---|---|---|---|
| 1024 | 33.7 | 6.23 | 3.57 | 0.81 | 1.87 | 0.174 |
| 4096 | 544 | 102 | 26.6 | 10.7 | 23.6 | 2.43 |
| 8192 | 2176 | 401 | 76.6 | 44.1 | 91.7 | 9.23 |

Decoding (Nq = 1), total time in ms:

| Nk | cpu_serial | cpu_omp | torch_naive | torch_sdpa | memopt v0 | memopt v3 |
|---|---|---|---|---|---|---|
| 4096 | 0.130 | 0.151 | 0.051 | 0.058 | 0.400 | 0.207 |
| 65536 | 2.13 | 2.50 | 5.90 | 0.985 | 7.97 | 3.00 |

Notes:

1. All versions pass (max error about 1e-7, limit 1e-4); the C++ output also matches PyTorch.
2. OpenMP scales well for prefill: at N = 4096, 1/2/4/8 threads take 558 / 279 / 140 / 102 ms (5.5x with 8 threads).
3. For decoding, `cpu_omp` is not faster than `cpu_serial`. It splits over query rows and there is only one row. Splitting over the keys would be needed (partial max/sum per thread, then combine).
4. Decoding with one query is a small amount of work (about 2 ms on the CPU at Nk = 65536), and there the CPU, PyTorch and the current CUDA versions are in the same range. The CUDA gain seen in prefill does not carry over to this case yet.
5. The CPU/PyTorch numbers on the shared nodes were noisy in some runs (other jobs on the same node), so the tables use the best of three jobs (`results/`).

## Results on the RTX 5090 machine

`run_5090.sh` runs the full comparison on one machine without Slurm (CPU: Intel Core Ultra 9 275HX, 24 threads; GPU: RTX 5090 Laptop, `sm_120`; d = 64). Two runs, best of both is shown; raw output and plots are in `results/5090/`. This GPU and the Euler RTX 4000 Ada are different machines, so the numbers above and below are not mixed.

Prefill, total time in ms:

| N | cpu_serial | cpu_omp (24 thr) | torch_sdpa (CPU) | memopt v0 | v1 | v2b | v3 | torch_cuda_sdpa |
|---|---|---|---|---|---|---|---|---|
| 1024 | 18.0 | 2.30 | 1.63 | 1.03 | 0.116 | 0.102 | 0.081 | 0.116 |
| 4096 | 298 | 44.8 | 10.9 | 11.3 | 1.85 | 1.63 | 1.24 | 0.426 |
| 8192 | 1304 | 199 | 36.4 | 67.9 | 7.31 | 5.97 | 5.36 | 1.42 |

Decoding (Nq = 1), total time in ms:

| Nk | cpu_serial | cpu_omp | memopt v0 | v3 | torch_cuda_naive | torch_cuda_sdpa |
|---|---|---|---|---|---|---|
| 4096 | 0.086 | 0.315 | 0.440 | 0.161 | 0.029 | 0.413 |
| 65536 | 1.80 | 3.10 | 11.7 | 2.39 | 0.044 | 6.39 |

Notes:

1. All versions pass (max error about 1e-7, limit 1e-4).
2. OpenMP thread scaling at N = 4096 (1/2/4/8/16/24 threads): 293 / 178 / 106 / 69 / 51 / 48 ms. That is 6.1x with 24 threads; the gain flattens after 8 to 16 threads.
3. The memopt versions get faster with every step in prefill (v0 -> v3 is 9.1x at N = 4096), but at N = 4096 and 8192 `torch_cuda_sdpa` is still faster than v3 (0.43 and 1.42 ms vs 1.24 and 5.36 ms). PyTorch's numbers are the library's own kernels, so this is the gap the fused kernel (part D) is meant to close.
4. In decoding the memopt kernels are slower than `torch_cuda_naive` (0.044 ms at Nk = 65536 vs 2.39 ms for v3). `torch_cuda_sdpa` in fp32 is slow in this case, we have not checked why.
5. `cpu_omp` is slower than `cpu_serial` for decoding (one query row, 24 threads to start for no work to share).
6. `torch_naive` on the CPU with 24 threads is erratic: 24 ms at N = 256 in both runs and 3 to 20 ms at some decoding sizes where the serial C++ takes 0.1 to 0.4 ms. We did not look into it; use `torch_sdpa` (CPU) as the PyTorch CPU number.

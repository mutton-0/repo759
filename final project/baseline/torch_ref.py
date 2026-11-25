"""PyTorch reference and timing baseline for attention (CPU, float32).

  ./bench cpu_serial Nk d [Nq] --dump dump     # writes Q/K/V/O .bin
  python torch_ref.py dump Nk d [Nq] [--reps R] [--threads T] [--device cuda]

Reads the same inputs the C++ versions used (CPU by default, --device cuda runs
the same two versions on the GPU, names get a _cuda suffix), computes
softmax(Q K^T / sqrt(d)) V with PyTorch, compares it with O.bin (max abs
error, limit 1e-4) and times two PyTorch versions: the plain matmul/softmax
one and torch.nn.functional.scaled_dot_product_attention.
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

TOL = 1e-4


def timeit(fn, reps, cuda):
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    fn()  # warmup
    sync()
    best = 1e30
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def main():
    args = sys.argv[1:]
    reps, threads, device = 5, None, "cpu"
    for flag in ("--reps", "--threads", "--device"):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            del args[i:i + 2]
            if flag == "--reps":
                reps = int(val)
            elif flag == "--threads":
                threads = int(val)
            else:
                device = val
    cuda = device == "cuda"
    folder, Nk, d = args[0], int(args[1]), int(args[2])
    Nq = int(args[3]) if len(args) > 3 else Nk
    if threads:
        torch.set_num_threads(threads)

    def load(name, rows):
        a = torch.from_numpy(np.fromfile(f"{folder}/{name}", dtype=np.float32).reshape(rows, d))
        return a.to(device)

    Q, K, V, O_cpp = load("Q.bin", Nq), load("K.bin", Nk), load("V.bin", Nk), load("O.bin", Nq)

    def naive():
        return torch.softmax(Q @ K.T / d ** 0.5, dim=-1) @ V

    def sdpa():
        return F.scaled_dot_product_attention(Q[None, None], K[None, None], V[None, None])[0, 0]

    suffix = "_cuda" if cuda else ""
    for name, fn in ((f"torch{suffix}_naive", naive), (f"torch{suffix}_sdpa", sdpa)):
        err = (fn() - O_cpp).abs().max().item()
        ms = timeit(fn, reps, cuda)
        status = "PASS" if err < TOL else "FAIL"
        print(f"version {name}  Nq {Nq}  Nk {Nk}  d {d}  threads {torch.get_num_threads() if not cuda else 0}")
        print(f"time_ms total {ms:.4f}")
        print(f"max_err vs cpp {err:.3e}  {status}")
        print(f"csv,{name},{Nq},{Nk},{d},-,-,-,{ms:.4f},{4.0 * Nq * Nk * d / (ms * 1e6):.1f},{err:.3e},{status}")


if __name__ == "__main__":
    main()

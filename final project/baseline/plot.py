"""Before/after comparison plot from csv lines.

  python plot.py out1.txt out2.txt ... [--out compare]

Reads every line starting with "csv," from the given files. This is the
format printed by baseline/bench, baseline/torch_ref.py and memopt/memopt:
csv,version,Nq,Nk,d,qk_ms,softmax_ms,pv_ms,total_ms,gflops,max_err,status
Writes <out>_prefill.pdf (time vs N, Nq == Nk) and <out>_decode.pdf (Nq = 1,
time vs Nk), and prints the table. Failed rows are skipped.
"""
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(files):
    rows = []
    for fn in files:
        for line in open(fn):
            p = line.strip().split(",")
            if p[0] != "csv" or len(p) < 12 or p[11] != "PASS":
                continue
            rows.append({"version": p[1], "Nq": int(p[2]), "Nk": int(p[3]), "d": int(p[4]),
                         "ms": float(p[8])})
    return rows


def draw(rows, title, xlabel, out):
    by = defaultdict(dict)
    for r in rows:
        d_ = by[r["version"]]
        d_[r["Nk"]] = min(r["ms"], d_.get(r["Nk"], 1e30))  # best over repeated runs
    if not by:
        return
    plt.figure(figsize=(6.4, 4.6))
    for v in sorted(by):
        xs = sorted(by[v])
        plt.plot(xs, [by[v][x] for x in xs], "o-", ms=4, label=v)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("time (ms)")
    plt.title(title)
    plt.grid(True, alpha=.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    print(f"wrote {out}")
    xs = sorted({x for v in by for x in by[v]})
    print(f"{'version':14s}" + "".join(f"{x:>10d}" for x in xs))
    for v in sorted(by):
        print(f"{v:14s}" + "".join(f"{by[v][x]:10.3f}" if x in by[v] else f"{'-':>10s}" for x in xs))


def main():
    args = sys.argv[1:]
    out = "compare"
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i:i + 2]
    rows = read(args)
    d = max({r["d"] for r in rows}, key=lambda x: sum(r["d"] == x for r in rows)) if rows else 0
    draw([r for r in rows if r["Nq"] == r["Nk"] and r["d"] == d],
         f"attention, prefill (N queries, N keys, d = {d})", "N", f"{out}_prefill.pdf")
    draw([r for r in rows if r["Nq"] == 1 and r["d"] == d],
         f"attention, decoding (1 query, Nk keys, d = {d})", "Nk (KV cache length)", f"{out}_decode.pdf")


if __name__ == "__main__":
    main()

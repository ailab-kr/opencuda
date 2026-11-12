#!/usr/bin/env python3
"""Compare two llama_logits --dump-dir folders op by op, in graph order.

usage: cmp_dump.py DIR_A DIR_B [--all] [--layers N]

For each float32 tensor present in both (same name, same element count) prints
the op, the normalized max error max|a-b| / max|a| and the NMSE
sum((a-b)^2) / sum(a^2). Without --all only the first layers are printed
(--layers, default 2) plus the final tensors; the first tensor whose
normalized error exceeds 1e-5 is marked.
"""
import os
import sys

import numpy as np


def read_index(d):
    out = []
    with open(os.path.join(d, "index.txt")) as f:
        for line in f:
            name, op, *ne = line.split()
            out.append((name, op, [int(x) for x in ne]))
    return out


def main():
    a, b = sys.argv[1], sys.argv[2]
    show_all = "--all" in sys.argv
    layers = 2
    if "--layers" in sys.argv:
        layers = int(sys.argv[sys.argv.index("--layers") + 1])
    ib = {n: (op, ne) for n, op, ne in read_index(b)}
    marked = False
    rows = []
    for name, op, ne in read_index(a):
        if name not in ib or op in ("VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"):
            continue
        x = np.fromfile(os.path.join(a, name + ".bin"), dtype=np.float32).astype(np.float64)
        y = np.fromfile(os.path.join(b, name + ".bin"), dtype=np.float32).astype(np.float64)
        if x.size != y.size or x.size == 0:
            continue
        fin = np.isfinite(x) & np.isfinite(y)
        scale = np.abs(x[fin]).max() if fin.any() else 0.0
        err = np.abs(x[fin] - y[fin]).max() / scale if scale else 0.0
        den = (x[fin] ** 2).sum()
        nmse = ((x[fin] - y[fin]) ** 2).sum() / den if den else 0.0
        flag = ""
        if not marked and err > 1e-5:
            flag = "  <-- first > 1e-5"
            marked = True
        rows.append((name, op, x.size, err, nmse, flag))
    for name, op, n, err, nmse, flag in rows:
        layer = name.rsplit("-", 1)[-1].split("#")[0]
        if show_all or flag or not layer.isdigit() or int(layer) < layers:
            print(f"{name:28s} {op:10s} n={n:7d}  maxrel {err:.2e}  nmse {nmse:.2e}{flag}")
    worst = max(rows, key=lambda r: r[4])
    print(f"tensors compared {len(rows)}; worst nmse {worst[4]:.2e} at {worst[0]} ({worst[1]})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summary table: every <prefix>-<text>-<mode>-<tag>.bin against its oracle.

usage: table.py PREFIX TEXT[,TEXT...] [--oracle-suffix f32-oracle]

The oracle of a text is <prefix>-<text>-f32-oracle.bin (the same weights
dequantized to float32 and run on the stock CPU backend). For each other dump
of the text prints KL(oracle || dump) mean / p99 / max, top-1 agreement,
the number of top-1 flips where the oracle's top-1/top-2 margin is >= 0.5
(a flip there is not a near tie) and the largest margin among the flips.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmp_logits import compare  # noqa: E402


def main():
    prefix, texts = sys.argv[1], sys.argv[2].split(",")
    print(f"{'dump':24s} {'KL mean':>9s} {'KL p99':>9s} {'KL max':>9s} {'top-1':>9s} {'flips>=0.5':>10s} {'max flip margin':>15s}")
    for t in texts:
        oracle = f"{prefix}-{t}-f32-oracle.bin"
        for f in sorted(glob.glob(f"{prefix}-{t}-*.bin")):
            if f == oracle:
                continue
            r = compare(oracle, f)
            kl, agree, margin = r["kl"], r["agree"], r["margin"]
            fm = margin[~agree]
            print(f"{f[len(prefix) + 1:-4]:24s} {kl.mean():9.2e} {np.quantile(kl, .99):9.2e} {kl.max():9.2e} "
                  f"{int(agree.sum()):4d}/{len(kl):<4d} {int((fm >= 0.5).sum()):10d} {(fm.max() if fm.size else 0):15.3f}")


if __name__ == "__main__":
    main()

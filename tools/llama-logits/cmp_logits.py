#!/usr/bin/env python3
"""Compare two llama_logits dumps position by position.

usage: cmp_logits.py BASE.bin TEST.bin [--from N] [--detail] [--json]

Per position i (logits[i] predicts token i+1):
  KL(base || test) of the softmax distributions (nats, float64),
  top-1 agreement, max |delta log-prob| over tokens with base log-prob > -10,
  and the base top-1/top-2 margin (how close the position is to a tie).
"""
import json
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        n, v = np.fromfile(f, dtype=np.int32, count=2)
        toks = np.fromfile(f, dtype=np.int32, count=n)
        lg = np.fromfile(f, dtype=np.float32, count=n * v).reshape(n, v)
    return toks, lg


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def compare(base, test, start=0):
    tb, lb = load(base)
    tt, lt = load(test)
    if not np.array_equal(tb, tt):
        raise SystemExit("token sequences differ")
    lb, lt = lb[start:], lt[start:]
    pb, pt = logsoftmax(lb), logsoftmax(lt)
    kl = (np.exp(pb) * (pb - pt)).sum(axis=-1)
    top_b, top_t = lb.argmax(-1), lt.argmax(-1)
    srt = np.sort(lb, axis=-1)
    margin = srt[:, -1] - srt[:, -2]
    mask = pb > -10
    dlp = np.where(mask, np.abs(pb - pt), 0).max(axis=-1)
    return dict(kl=kl, agree=top_b == top_t, margin=margin, dlp=dlp, top_b=top_b, top_t=top_t, start=start)


def summary(r):
    kl, dlp = r["kl"], r["dlp"]
    dis = np.nonzero(~r["agree"])[0]
    return {
        "positions": int(len(kl)),
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "kl_p99": float(np.quantile(kl, 0.99)),
        "kl_max": float(kl.max()),
        "top1_agree": f"{int(r['agree'].sum())}/{len(kl)}",
        "dlogprob_median": float(np.median(dlp)),
        "dlogprob_max": float(dlp.max()),
        "flips": [(int(i + r["start"]), round(float(r["margin"][i]), 4)) for i in dis],
    }


def main():
    a = sys.argv[1:]
    start = 0
    if "--from" in a:
        start = int(a[a.index("--from") + 1])
    r = compare(a[0], a[1], start)
    s = summary(r)
    if "--json" in a:
        print(json.dumps(s))
        return
    print(f"positions {s['positions']}  KL mean {s['kl_mean']:.3e} median {s['kl_median']:.3e} "
          f"p99 {s['kl_p99']:.3e} max {s['kl_max']:.3e}  top1 {s['top1_agree']}  "
          f"|dlogp| median {s['dlogprob_median']:.4f} max {s['dlogprob_max']:.4f}")
    if s["flips"]:
        print("top-1 flips (position, base top1-top2 margin):", s["flips"])
    if "--detail" in a:
        for i in range(len(r["kl"])):
            print(f"{i + start:4d} kl {r['kl'][i]:.2e} dlp {r['dlp'][i]:.4f} margin {r['margin'][i]:.4f} "
                  f"top {r['top_b'][i]} {r['top_t'][i]}")


if __name__ == "__main__":
    main()

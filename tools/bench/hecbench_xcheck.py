#!/usr/bin/env python3
"""Compare what two runs of one program printed: the numbers in the output,
apart from timings.

A HeCBench program that prints neither PASS nor FAIL ("ran") still prints
results: checksums, sums, errors, sample values. Running it once through
OpenCuda's CPU path and once on a GPU and comparing those numbers tells
whether the GPU run computed the same thing, without knowing what the
program computes.

  hecbench_xcheck.py REF.log TEST.log [--rtol 1e-3] [--atol 1e-6] [--show N]
  hecbench_xcheck.py --dirs REF_DIR TEST_DIR   (each holds <program>/run.log)

Lines that look like timings or rates are left out (time, ms, us, sec,
GB/s, GFLOP, throughput, bandwidth, speedup, ...), as are lines with the
toolkit's own messages. The remaining numbers are compared in order: equal
counts and each pair within rtol (relative) or atol (absolute) is "same";
otherwise "differs" with the first differences. Exit status 0 when all
compared programs are "same".
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

TIMING = re.compile(
    r"time|elapsed|\bms\b|msec|\bus\b|usec|\bns\b|\bsec|seconds?\b|\bs\)|\d\s*s\b|gb/s|gib/s|mb/s|gflop|tflop|"
    r"mflop|performance|"
    r"throughput|bandwidth|speed|rate\b|per second|/s\b|fps|cycles|clock|\bdevice\b|gpu|cuda|opencl|"
    r"compute capability|driver|version|memory|warp|threads? per|block size|blocks?\b.*\bgrid|ocu",
    re.I)
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf(?:inity)?)\b", re.I)


def numbers(text: str) -> list[tuple[int, float, str]]:
    out = []
    for ln, line in enumerate(text.splitlines(), 1):
        if TIMING.search(line):
            continue
        for m in NUM.finditer(line):
            tok = m.group(0)
            try:
                out.append((ln, float(tok), line.strip()))
            except ValueError:
                pass
    return out


def same(a: float, b: float, rtol: float, atol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def compare(ref: str, test: str, rtol: float, atol: float, show: int) -> tuple[str, list[str]]:
    r, t = numbers(ref), numbers(test)
    if not r and not t:
        return "no_numbers", []
    notes = []
    if len(r) != len(t):
        notes.append(f"{len(r)} numbers against {len(t)}")
    bad = 0
    for (lr, a, la), (lt, b, lb) in zip(r, t):
        if not same(a, b, rtol, atol):
            bad += 1
            if len(notes) < show:
                notes.append(f"line {lr}: {a!r} vs {b!r}   [{la[:80]}] / [{lb[:80]}]")
    if bad == 0 and len(r) == len(t):
        return "same", [f"{len(r)} numbers"]
    return "differs", [f"{bad} of {min(len(r), len(t))} numbers differ"] + notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--dirs", action="store_true", help="REF and TEST are folders of <program>/run.log")
    ap.add_argument("--log", default="run.log")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()
    pairs = []
    if a.dirs:
        for d in sorted(Path(a.ref).iterdir()):
            if (d / a.log).is_file() and (Path(a.test) / d.name / a.log).is_file():
                pairs.append((d.name, d / a.log, Path(a.test) / d.name / a.log))
    else:
        pairs.append((Path(a.test).name, Path(a.ref), Path(a.test)))
    counts: dict[str, int] = {}
    for name, rp, tp in pairs:
        st, notes = compare(rp.read_text(errors="replace"), tp.read_text(errors="replace"), a.rtol, a.atol, a.show)
        counts[st] = counts.get(st, 0) + 1
        print(f"{name}: {st}" + (f" ({notes[0]})" if notes else ""))
        for n in notes[1:]:
            print(f"    {n}")
    print("summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0 if set(counts) <= {"same", "no_numbers"} else 1


if __name__ == "__main__":
    sys.exit(main())

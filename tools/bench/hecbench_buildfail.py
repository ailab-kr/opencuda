#!/usr/bin/env python3
"""Why HeCBench programs do not build: the first error of each build log,
sorted into causes, and how many programs each cause holds back.

  hecbench_buildfail.py RESULTS.csv LOG_DIR [--tool ocu] [--list]

RESULTS.csv is the run's results (bench,tool,status,seconds); LOG_DIR holds
<bench>.build.log for the tool. A program counts under the first cause that
matches its first error; "other" lists the rest so the rules can grow.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# (cause, pattern on the log), first match wins; order: specific before general
CAUSES = [
    ("files outside the program's folder (the harness copies one folder)", r"No rule to make target '\.\./|\.\./[\w-]+/[\w.]+: No such file"),
    ("Makefile rule the copy lacks (No rule to make target)", r"No rule to make target"),
    ("Thrust", r"thrust/|thrust::"),
    ("cuRAND", r"curand"),
    ("cuFFT", r"cufft"),
    ("cuSPARSE", r"cusparse"),
    ("cuSOLVER", r"cusolver"),
    ("cuBLAS (missing function)", r"cublas\w*'|undefined reference to `cublas|cublas\w+ was not declared"),
    ("CUB (missing part)", r"cub/|cub::"),
    ("cooperative groups", r"cooperative_groups|cg::"),
    ("MPI / NCCL", r"mpi\.h|nccl"),
    ("OpenMP / other host libraries", r"omp\.h|-fopenmp|boost/|gsl/|png\.h|opencv|zlib\.h"),
    ("inline PTX (asm)", r"\basm\b|__asm__|asm volatile|PTX"),
    ("relocatable device code (-rdc / -dc)", r"-rdc|-dc\b|relocatable|undefined reference to .*__device|extern __device__"),
    ("texture / surface objects", r"tex1D|tex2D|tex3D|texture<|cudaTextureObject|surf2D|cudaArray|cudaBindTexture"),
    ("CUDA graphs / driver API", r"cudaGraph|cuModule|cuLaunch|cuCtx|cuda\.h|nvrtc"),
    ("missing CUDA runtime function", r"(use of undeclared identifier|was not declared in this scope|undefined reference to) .?'?`?cuda\w+"),
    ("missing intrinsic / device function", r"(use of undeclared identifier|no matching function for call to|undefined reference to) .?'?`?__\w+"),
    ("missing header", r"fatal error: .*: No such file|file not found"),
    ("device code the SPIR-V path rejects", r"cu2spv:|SPIR-V back end|does not assemble|LLVM ERROR"),
    ("timeout (300 s)", r"^$TIMEOUT"),
]
ERR = re.compile(r"(error|Error|ERROR|undefined reference|fatal|No rule to make target)")


def first_error(text: str) -> str:
    for line in text.splitlines():
        if ERR.search(line) and "warning" not in line.lower()[:40]:
            return line.strip()
    return ""


def classify(text: str) -> tuple[str, str]:
    err = first_error(text)
    # look at the error line and its neighbourhood (the include chain, the call)
    lines = text.splitlines()
    ctx = err
    if err:
        i = next((k for k, l in enumerate(lines) if l.strip() == err), 0)
        ctx = "\n".join(lines[max(0, i - 3):i + 4])
    for cause, pat in CAUSES:
        if re.search(pat, ctx):
            return cause, err
    for cause, pat in CAUSES:  # anywhere in the log
        if re.search(pat, text):
            return cause, err
    return "other", err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    ap.add_argument("logs")
    ap.add_argument("--tool", default="ocu")
    ap.add_argument("--list", action="store_true", help="the programs and first errors per cause")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.results)))
    failed = [r["bench"] for r in rows if r.get("tool") == a.tool and r.get("status") == "build_fail"]
    other = {r["bench"]: r["status"] for r in rows if r.get("tool") not in (a.tool, None) and r.get("status")}
    by = defaultdict(list)
    for b in failed:
        p = Path(a.logs) / f"{b}.build.log"
        text = p.read_text(errors="replace") if p.is_file() else ""
        cause, err = classify(text) if text else ("no log", "")
        by[cause].append((b, err))
    print(f"{len(failed)} programs do not build ({a.tool})\n")
    print("| cause | programs | of them, the other toolchain builds |\n|---|---|---|")
    for cause, items in sorted(by.items(), key=lambda kv: -len(kv[1])):
        built = sum(1 for b, _ in items if other.get(b) not in (None, "build_fail"))
        print(f"| {cause} | {len(items)} | {built} |")
    if a.list:
        for cause, items in sorted(by.items(), key=lambda kv: -len(kv[1])):
            print(f"\n## {cause} ({len(items)})")
            for b, err in items:
                print(f"- {b} [other: {other.get(b, '-')}]: {err[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

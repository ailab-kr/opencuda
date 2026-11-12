#!/usr/bin/env python3
"""Count CUDA runtime and library calls in a source tree; print numbers only.

PROBE ONLY. Reads the tree as text and never executes it. Used to choose
which runtime functions to implement first (PLAN P1-03-a).

Usage: python3 -I survey_api.py <compiler/probe dir> <source tree> [--coverage]
"""
import collections, os, re, sys
sys.path.insert(0, sys.argv[1])
from cpu_rewrite import _code_mask

root = sys.argv[2]
calls = collections.Counter()
files_using = collections.Counter()
exts = (".cu", ".cuh", ".c", ".cpp", ".h", ".hpp")
for d, _, fs in os.walk(root):
    for f in fs:
        if not f.endswith(exts):
            continue
        src = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
        try:
            mask = _code_mask(src)
        except Exception:
            mask = [True] * len(src)
        seen = set()
        for m in re.finditer(r"\b(cuda[A-Z]\w*|cublas[A-Z]\w*|curand[A-Z]\w*|cufft[A-Z]\w*)\s*\(", src):
            if not mask[m.start()]:
                continue
            calls[m.group(1)] += 1
            seen.add(m.group(1))
        for name in seen:
            files_using[name] += 1
print(f"distinct functions: {len(calls)}, calls: {sum(calls.values())}")
for name, n in calls.most_common(40):
    print(f"  {name:38s} {n:5d} calls in {files_using[name]:3d} files")

# ---- coverage of a planned function list (--coverage) ----
PLAN27 = set("""cudaGetDeviceCount cudaGetDeviceProperties cudaSetDevice cudaGetDevice cudaDeviceSynchronize
cudaMalloc cudaFree cudaMemcpy cudaMemcpyAsync cudaMemset cudaMallocHost cudaFreeHost cudaMemGetInfo
cudaStreamCreate cudaStreamDestroy cudaStreamSynchronize cudaStreamQuery
cudaEventCreate cudaEventDestroy cudaEventRecord cudaEventSynchronize cudaEventQuery cudaEventElapsedTime
cudaGetLastError cudaPeekAtLastError cudaGetErrorString cudaGetErrorName""".split())
ADD = set("cudaThreadSynchronize cudaMemcpyToSymbol cudaMemsetAsync cudaThreadExit cudaEventCreateWithFlags cudaStreamWaitEvent cudaMemcpy2D cudaMemcpy2DAsync".split())
if "--coverage" in sys.argv:
    total = inplan = inplus = files_api = files_ok = files_ok_plus = 0
    for d, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(exts):
                continue
            src = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
            try:
                mask = _code_mask(src)
            except Exception:
                mask = [True] * len(src)
            names = [m.group(1) for m in re.finditer(r"\b(cuda[A-Z]\w*)\s*\(", src)
                     if mask[m.start()] and m.group(1) not in ("cudaCheckError", "cudaCheckAsyncError")]
            if not names:
                continue
            files_api += 1
            total += len(names)
            inplan += sum(n in PLAN27 for n in names)
            inplus += sum(n in PLAN27 | ADD for n in names)
            files_ok += all(n in PLAN27 for n in names)
            files_ok_plus += all(n in PLAN27 | ADD for n in names)
    if total:
        print(f"runtime calls {total}: in the planned 27 {inplan} ({100 * inplan / total:.0f}%), "
              f"with {len(ADD)} additions {inplus} ({100 * inplus / total:.0f}%)")
        print(f"files calling the runtime {files_api}: fully covered by the 27 {files_ok}, "
              f"with additions {files_ok_plus}")

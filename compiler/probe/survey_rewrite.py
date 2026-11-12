#!/usr/bin/env python3
"""Run the CPU-path rewriter over a CUDA source tree; print numbers only.

PROBE ONLY. Reads the tree as text and never executes it.
Usage: python3 -I survey_rewrite.py <compiler/probe dir> <source tree>
"""
import collections, os, re, sys
sys.path.insert(0, sys.argv[1])
from cpu_rewrite import rewrite, RewriteError, _code_mask

root = sys.argv[2]
files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith((".cu", ".cuh"))]
stats = collections.Counter()
errors = collections.Counter()
examples = {}
features = collections.Counter()
FEAT = {"warp shuffle": r"__shfl\w*", "warp vote": r"__(ballot|any|all)(_sync)?\b",
        "__syncwarp": r"__syncwarp", "cooperative groups": r"cooperative_groups",
        "inline PTX asm": r"\basm\s*(volatile)?\s*\(", "texture": r"\btex[123]D\b|texture<",
        "dynamic parallelism": r"cudaDeviceSynchronize\s*\(\s*\)\s*;[^\n]*//device",
        "__constant__": r"__constant__", "atomic*": r"\batomic[A-Z]\w*",
        "runtime API cuda*": r"\bcuda[A-Z]\w*\s*\("}
for path in files:
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    stats["files"] += 1
    mask = _code_mask(src) if True else None
    code_launch = 0
    i = 0
    while True:
        i = src.find("<<<", i)
        if i < 0: break
        if all(mask[i:i+3]): code_launch += 1
        i += 3
    stats["launches_in_code"] += code_launch
    for name, rx in FEAT.items():
        if re.search(rx, src): features[name] += 1
    try:
        out, st = rewrite(src)
        stats["launches_rewritten"] += st["launches"]
        stats["extern_shared_rewritten"] += st["extern_shared"]
        left = sum(1 for m in re.finditer("<<<", out) if all(_code_mask(out)[m.start():m.start()+3]))
        stats["launches_left"] += left
        if out.count("\n") != src.count("\n"): stats["line_count_changed"] += 1
        stats["files_ok"] += 1
    except RewriteError as e:
        key = re.sub(r"\d+", "N", str(e))
        errors[key] += 1
        examples.setdefault(key, os.path.relpath(path, root))
        stats["files_failed"] += 1
    except Exception as e:
        errors["crash " + type(e).__name__] += 1
        examples.setdefault("crash " + type(e).__name__, os.path.relpath(path, root))
        stats["files_failed"] += 1
print(dict(stats))
for k, v in errors.most_common(): print(f"  error x{v}: {k}  (e.g. {examples[k]})")
print("  files using:", dict(features))

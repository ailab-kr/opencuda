#!/usr/bin/env python3
"""Summary of a hecbench_compare.sh run (OpenCuda against chipStar) as markdown.

    tools/bench/hecbench_summary.py <OUT> [--lists]

A run that stopped because the benchmark's input data is not in the
checkout (HeCBench keeps large inputs outside the repository: "cannot
open", "no such file" ... in its log) is counted as no_data for that tool:
it says nothing about either toolchain. Speed: wall time of the timed
`make run` where both toolchains run to the end (pass or ran), as the
geometric mean of chipStar time / OpenCuda time (> 1: OpenCuda faster)."""
import csv
import math
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path.home() / "cmp" / "hec"
logs = out_dir / "logs" if (out_dir / "logs").is_dir() else out_dir / "full_logs"
rows = {}
for r in csv.DictReader(open(out_dir / "results_full.csv")):
    if r.get("tool") in ("ocu", "chip"):
        rows[(r["bench"], r["tool"])] = r
NODATA = re.compile(r"no such file|cannot open|could not open|failed to open|unable to open|error opening|"
                    r"could not find|could not stat|not opened|failed to import|cannot find|file not found|"
                    r"invalid infile|data file|fopen|could not read", re.I)


def status(b, t):
    s = rows[(b, t)]["status"]
    if s == "crash":
        try:
            text = (logs / t / f"{b}.run.log").read_text(errors="replace")[-20000:]
        except OSError:
            text = ""
        if NODATA.search(text):
            return "no_data"
    return s


done = sorted({b for b, t in rows if (b, "ocu") in rows and (b, "chip") in rows})
OK = ("pass", "ran")
order = ["pass", "ran", "fail", "crash", "timeout", "build_fail", "no_data"]
cnt = {t: dict.fromkeys(order, 0) for t in ("ocu", "chip")}
for b in done:
    for t in ("ocu", "chip"):
        cnt[t][status(b, t)] += 1
lines = [f"Programs with both results: {len(done)}\n", "| status | OpenCuda | chipStar |", "|---|---|---|"]
lines += [f"| {k} | {cnt['ocu'][k]} | {cnt['chip'][k]} |" for k in order]
both = [b for b in done if status(b, "ocu") in OK and status(b, "chip") in OK]
only_o = [b for b in done if status(b, "ocu") in OK and status(b, "chip") not in OK]
only_c = [b for b in done if status(b, "chip") in OK and status(b, "ocu") not in OK]
lines.append(f"\nRun to the end (pass or ran): OpenCuda {cnt['ocu']['pass'] + cnt['ocu']['ran']}, "
             f"chipStar {cnt['chip']['pass'] + cnt['chip']['ran']}; both {len(both)}, "
             f"only OpenCuda {len(only_o)}, only chipStar {len(only_c)}.")
ratios = []
for b in both:
    to, tc = float(rows[(b, "ocu")]["seconds"] or 0), float(rows[(b, "chip")]["seconds"] or 0)
    if to > 0.05 and tc > 0.05:
        ratios.append(tc / to)
if ratios:
    g = math.exp(sum(map(math.log, ratios)) / len(ratios))
    fast = sum(r > 1.1 for r in ratios)
    slow = sum(r < 1 / 1.1 for r in ratios)
    lines.append(f"Speed where both run to the end ({len(ratios)} programs over 0.05 s): geometric mean "
                 f"chipStar/OpenCuda {g:.2f}; OpenCuda over 10% faster on {fast}, over 10% slower on {slow}, "
                 f"within 10% on {len(ratios) - fast - slow}.")
if "--lists" in sys.argv:
    lines.append("\nOnly chipStar runs to the end: " + ", ".join(f"{b} (OpenCuda {status(b, 'ocu')})" for b in only_c))
    lines.append("\nOnly OpenCuda runs to the end: " + ", ".join(f"{b} (chipStar {status(b, 'chip')})" for b in only_o))
print("\n".join(lines))

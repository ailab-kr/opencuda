#!/usr/bin/env python3
"""Device-compile every .cu file of the repository for an AMD GPU with
tools/amdgpu/ocu_amd_cc.py and report which build (conformance level L1 for
the AMD device pass), grouped by the first error.

    build_survey.py [--arch gfx1103] [--out report.md] [paths...]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from comgr import Comgr, ComgrError  # noqa: E402
from ocu_amd_cc import compile_cu  # noqa: E402


def first_error(log: str) -> str:
    for l in log.splitlines():
        m = re.search(r"(?:fatal )?error: (.*)", l)
        if m:
            msg = m.group(1)
            msg = re.sub(r"'[^']*'", "'…'", msg)[:110]
            return msg
    return log.strip().splitlines()[-1][:110] if log.strip() else "unknown"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--arch", default="gfx1103")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    files = [Path(p) for p in a.paths] or sorted(
        p for p in REPO.rglob("*.cu") if ".git" not in p.parts and "build" not in p.parts)
    cg = Comgr()
    ok, bad, reasons = [], [], Counter()
    t0 = time.time()
    for f in files:
        rel = f.resolve().relative_to(REPO).as_posix()
        try:
            compile_cu(f.resolve(), a.arch, comgr=cg)
            ok.append(rel)
            print(f"ok    {rel}")
        except ComgrError as e:
            r = first_error(str(e))
            bad.append((rel, r))
            reasons[r] += 1
            print(f"FAIL  {rel}: {r}")
    lines = [f"# AMD device-pass build survey ({a.arch})", "",
             f"{len(ok)} / {len(files)} .cu files compile unmodified ({time.time() - t0:.0f}s).", "",
             "| first error | files |", "|---|---|"]
    lines += [f"| {r} | {n} |" for r, n in reasons.most_common()]
    lines += ["", "## Not building", ""] + [f"- `{f}`: {r}" for f, r in bad]
    lines += ["", "## Building", ""] + [f"- `{f}`" for f in ok]
    report = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(report, encoding="utf-8")
    print(f"\n{len(ok)} / {len(files)} build")
    for r, n in reasons.most_common(12):
        print(f"  {n:3d}  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

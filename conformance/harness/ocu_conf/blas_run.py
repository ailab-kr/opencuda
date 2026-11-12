"""Run the BLAS cases through an implementation's driver and grade them.

    python3 -m ocu_conf.blas_run <driver> [--out result.json] [--backend cpu]
                                 [--samples N] [--only fn[:kind],...]

Writes the cases to the driver's stdin (protocol in
tests/conformance/blas_driver.c), reads its outputs, grades every case with
blas_check, writes a result record (conformance/README.md) and prints a
summary with sample outputs. Exit status 0 only when every case passes.

--only runs a subset, for an implementation that offers only some of the
functions: "gemm,dot:f32" keeps every gemm case and the f32 dot cases. The
record then holds only those cases; it says so in run.subset.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from . import SCHEMA_VERSION
from .blas_check import check, load_tolerances
from .results import validate
from .score import summarize

ROOT = Path(__file__).resolve().parents[2]  # conformance/
sys.path.insert(0, str(ROOT))
from oracle import cases as C  # noqa: E402

CHARS = {"trans", "transa", "transb", "uplo", "diag", "side"}
REALS = {"alpha", "beta"}  # always real, even when the value is integral


def _num(v: float) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return float(v).hex()


def serialize(cases: list[dict]) -> str:
    out = []
    for c in cases:
        if any(ch.isspace() for ch in c["id"]):
            raise ValueError(f"case id with white space: {c['id']!r}")
        out.append(f"case {c['id']} {c['fn']} {c['kind']}")
        for name, v in c["args"].items():
            if name in CHARS:
                out.append(f"c {name} {v}")
            elif isinstance(v, list):
                out.append(f"v {name} {len(v)} " + " ".join(_num(float(x)) for x in v))
            elif name in REALS or isinstance(v, float):
                out.append(f"r {name} {_num(float(v))}")
            elif isinstance(v, int):
                out.append(f"i {name} {v}")
            else:
                raise ValueError(f"{c['id']}: cannot encode {name}={v!r}")
        out.append("end")
    return "\n".join(out) + "\n"


def _parse_value(tok: str) -> float:
    try:
        return float.fromhex(tok)
    except ValueError:
        return float(tok)


def parse(text: str) -> dict[str, dict]:
    results: dict[str, dict] = {}
    cur = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "out":
            cur = {"status": int(parts[2]), "outputs": {}}
            results[parts[1]] = cur
        elif tag == "end":
            cur = None
        elif cur is None:
            raise ValueError(f"driver output line outside a case: {line[:80]!r}")
        elif tag == "i":
            cur["outputs"][parts[1]] = int(parts[2])
        elif tag == "r":
            cur["outputs"][parts[1]] = _parse_value(parts[2])
        elif tag == "v":
            count = int(parts[2])
            cur["outputs"][parts[1]] = [_parse_value(t) for t in parts[3:3 + count]]
    return results


def select(cases: list[dict], only: str | None) -> list[dict]:
    """Cases whose fn (and kind, if given) is listed in `only`."""
    if not only:
        return cases
    wanted = []
    for item in only.split(","):
        fn, _, kind = item.strip().partition(":")
        if not fn:
            raise ValueError(f"--only: empty item in {only!r}")
        wanted.append((fn, kind or None))
    out = [c for c in cases
           if any(c["fn"] == fn and (kind is None or c["kind"] == kind) for fn, kind in wanted)]
    if not out:
        raise ValueError(f"--only {only!r} selects no case")
    return out


def _git_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("driver")
    ap.add_argument("--out")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--only", help="fn[:kind],... subset of the cases")
    args = ap.parse_args(argv)

    cases = select(C.all_cases(), args.only)
    proc = subprocess.run([args.driver], input=serialize(cases), capture_output=True,
                          text=True)
    if proc.returncode != 0:
        print(f"driver failed ({proc.returncode}): {proc.stderr.strip()[:500]}",
              file=sys.stderr)
        return 2
    outputs = parse(proc.stdout)
    tol = load_tolerances()

    tests = []
    for case in cases:
        got = outputs.get(case["id"])
        if got is None:
            res = {"status": "fail", "worst": math.inf, "where": "no output"}
        elif got["status"] != 0:
            res = {"status": "fail", "worst": math.inf,
                   "where": f"call returned status {got['status']}"}
        else:
            res = check(case, got["outputs"], tol)
        entry = {"id": case["id"], "card": case["card"], "level": "L4",
                 "status": res["status"]}
        if math.isfinite(res["worst"]):
            entry["error_ratio"] = round(res["worst"], 6)
        if res.get("where"):
            entry["reason"] = res["where"]
        tests.append(entry)

    record = {
        "schema": SCHEMA_VERSION,
        "run": {"date": datetime.date.today().isoformat(), "suite_version": "0.1.0",
                "set": "public", **({"subset": args.only} if args.only else {})},
        "device": {"vendor": args.backend, "model": platform.processor() or platform.machine(),
                   "driver": "none", "os": f"{platform.system()} {platform.release()}"},
        "impl": {"name": "ocu", "version": _git_version(), "backend": args.backend},
        "tests": tests,
    }
    problems = validate(record)
    if problems:
        print("result record invalid: " + "; ".join(problems), file=sys.stderr)
        return 2
    record["summary"] = summarize(tests)
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1), encoding="utf-8")

    by_card: dict[str, list[int]] = {}
    for t in tests:
        by_card.setdefault(t["card"], [0, 0])
        by_card[t["card"]][0 if t["status"] == "pass" else 1] += 1
    passed = sum(1 for t in tests if t["status"] == "pass")
    print(f"BLAS L4 on {args.backend}: {passed} / {len(tests)} pass")
    print("  per card (pass/fail): " + ", ".join(
        f"{c.removeprefix('blas-')} {p}/{f}" for c, (p, f) in sorted(by_card.items())))
    ratios = sorted(t.get("error_ratio", 0.0) for t in tests if t["status"] == "pass")
    if ratios:
        print(f"  error / bound: median {ratios[len(ratios) // 2]:.3f}, max {ratios[-1]:.3f}")
    for t in [t for t in tests if t["status"] != "pass"][:10]:
        print(f"  FAIL {t['id']}: {t.get('reason')}")
    # Sample outputs, so a reader sees values and not only counts.
    shown = 0
    seen_fn: set[str] = set()
    for case in cases:
        if shown >= args.samples:
            break
        got = outputs.get(case["id"])
        a = case["args"]
        big = a.get("n", 0) >= 16 or a.get("m", 0) >= 8
        if not got or not big or case["fn"] in seen_fn:
            continue
        seen_fn.add(case["fn"])
        key = next(iter(case["expect"]))
        exp, out = case["expect"][key], got["outputs"].get(key)
        if isinstance(exp, list):
            exp, out = exp[:3], (out or [])[:3]
        print(f"  sample {case['id']}: got {out}, want {exp}")
        shown += 1
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

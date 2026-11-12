"""Compare a BLAS implementation's outputs with a case's reference answer.

The bound for element i is

    |actual - expected| <= factor * u * terms * scale[i] + terms * tiny

with u the unit roundoff (2^-24 for f32, 2^-53 for f64) and tiny the
smallest normal number. This is the standard a-priori bound for sums of
`terms` products, so a correct implementation passes regardless of the
order it adds in, while real mistakes (wrong index, wrong operand, lost
term) do not fit inside it. Elements without a scale must match exactly,
which also checks that memory outside the result is left untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

UNIT_ROUNDOFF = {"f32": 2.0 ** -24, "f64": 2.0 ** -53}
TINY = {"f32": 1.1754943508222875e-38, "f64": 2.2250738585072014e-308}

_TOL_FILE = Path(__file__).resolve().parents[2] / "tolerances" / "blas.json"


def load_tolerances(path: Path | None = None) -> dict:
    with open(path or _TOL_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def factor_for(fn: str, tolerances: dict) -> float:
    return float(tolerances.get("per_fn", {}).get(fn, tolerances["default_factor"]))


def _same_special(a: float, e: float) -> bool | None:
    """True/False when either side is NaN or infinite, None otherwise."""
    if isinstance(a, bool) or isinstance(e, bool):
        return None
    if not isinstance(a, (int, float)) or not isinstance(e, (int, float)):
        return None
    if math.isnan(e) or math.isnan(a):
        return math.isnan(e) and math.isnan(a)
    if math.isinf(e) or math.isinf(a):
        return a == e
    return None


def check(case: dict, actual: dict, tolerances: dict) -> dict:
    """Return {"status": "pass"|"fail", "worst": ratio, "where": str|None}.

    `worst` is the largest error divided by its bound (exact elements count
    as 0 when equal); above 1 means fail.
    """
    kind = case["kind"]
    factor = factor_for(case["fn"], tolerances)
    worst = 0.0
    where = None
    for key, expected in case["expect"].items():
        if key not in actual:
            return {"status": "fail", "worst": math.inf, "where": f"{key}: missing"}
        got = actual[key]
        exp_list = expected if isinstance(expected, list) else [expected]
        got_list = got if isinstance(got, list) else [got]
        if len(exp_list) != len(got_list):
            return {"status": "fail", "worst": math.inf,
                    "where": f"{key}: length {len(got_list)} != {len(exp_list)}"}
        scale = case.get("scale", {}).get(key)
        scale_list = scale if isinstance(scale, list) else (
            [scale] if scale is not None else None)
        terms = case.get("terms", {}).get(key, 1)
        for i, (a, e) in enumerate(zip(got_list, exp_list)):
            special = _same_special(a, e)
            if special is not None:
                ratio = 0.0 if special else math.inf
            elif scale_list is None or scale_list[i] == 0:
                ratio = 0.0 if a == e else math.inf
            else:
                bound = (factor * UNIT_ROUNDOFF[kind] * terms * scale_list[i]
                         + terms * TINY[kind])
                ratio = abs(a - e) / bound
            if ratio > worst:
                worst = ratio
                where = f"{key}[{i}]: got {a!r}, want {e!r}"
    return {"status": "pass" if worst <= 1.0 else "fail", "worst": worst,
            "where": where if worst > 1.0 else None}


def run_suite(cases: list[dict], impl, tolerances: dict | None = None) -> list[dict]:
    """Run `impl(case) -> outputs dict` on every case; one result per case."""
    tol = tolerances or load_tolerances()
    results = []
    for case in cases:
        try:
            out = impl(case)
            res = check(case, out, tol)
        except Exception as error:  # an implementation crash is a failure
            res = {"status": "fail", "worst": math.inf,
                   "where": f"exception: {type(error).__name__}: {error}"}
        res["id"] = case["id"]
        res["card"] = case["card"]
        results.append(res)
    return results

"""Scoring rules shared by every suite.

Kept small on purpose: a rule that lives here is applied the same way to
every chip and every implementation.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def exact_match(actual: Sequence, expected: Sequence) -> bool:
    """Integers and control flow: every element equal, same length."""
    return len(actual) == len(expected) and all(
        a == e for a, e in zip(actual, expected)
    )


def max_relative_error(actual: Sequence[float], expected: Sequence[float],
                       floor: float = 1e-30) -> float:
    """Largest |a - e| / max(|e|, floor) over the pair of sequences.

    Returns infinity when the lengths differ, when a value is NaN on one side
    only, or when infinities do not match exactly.
    """
    if len(actual) != len(expected):
        return math.inf
    worst = 0.0
    for a, e in zip(actual, expected):
        if math.isnan(a) or math.isnan(e):
            if math.isnan(a) and math.isnan(e):
                continue
            return math.inf
        if math.isinf(a) or math.isinf(e):
            if a == e:
                continue
            return math.inf
        worst = max(worst, abs(a - e) / max(abs(e), floor))
    return worst


def within_tolerance(actual: Sequence[float], expected: Sequence[float],
                     tolerance: float) -> bool:
    return max_relative_error(actual, expected) <= tolerance


def percentile(values: Iterable[float], fraction: float) -> float:
    """Linear-interpolated percentile; `fraction` in [0, 1]."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty set")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(tests: Sequence[dict]) -> dict:
    """Counts per status plus the speed picture.

    Speed is reported as the median and the 10th percentile (the slow tail),
    never as a mean alone, together with the slowest test's id.
    """
    summary = {"pass": 0, "fail": 0, "skip": 0, "unsupported": 0}
    for test in tests:
        summary[test["status"]] += 1
    timed = [t for t in tests
             if t["status"] == "pass" and t.get("speed_ratio") is not None]
    if timed:
        ratios = [t["speed_ratio"] for t in timed]
        slowest = min(timed, key=lambda t: t["speed_ratio"])
        summary["speed_ratio_median"] = percentile(ratios, 0.5)
        summary["speed_ratio_p10"] = percentile(ratios, 0.1)
        summary["slowest_test"] = slowest["id"]
    else:
        summary["speed_ratio_median"] = None
        summary["speed_ratio_p10"] = None
        summary["slowest_test"] = None
    return summary

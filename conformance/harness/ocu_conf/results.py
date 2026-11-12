"""Shape of a result record and its validation.

A result record is one JSON file: one run of one suite on one device with one
implementation. The format is documented in conformance/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION

LEVELS = ("L1", "L2", "L3", "L4", "L5")
STATUSES = ("pass", "fail", "skip", "unsupported")
SETS = ("public", "private")

_RUN_KEYS = {"date": str, "suite_version": str, "set": str}
_DEVICE_KEYS = {"vendor": str, "model": str, "driver": str, "os": str}
_IMPL_KEYS = {"name": str, "version": str, "backend": str}
_MEASURES = ("max_rel_err", "tolerance", "time_ms_median",
             "baseline_native_ms", "speed_ratio")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_section(record: dict, name: str, keys: dict, errors: list[str]) -> None:
    section = record.get(name)
    if not isinstance(section, dict):
        errors.append(f"{name}: missing or not an object")
        return
    for key, kind in keys.items():
        value = section.get(key)
        if not isinstance(value, kind) or value == "":
            errors.append(f"{name}.{key}: missing or empty")


def _check_test(index: int, test: Any, seen: set[str], errors: list[str]) -> None:
    where = f"tests[{index}]"
    if not isinstance(test, dict):
        errors.append(f"{where}: not an object")
        return
    test_id = test.get("id")
    if not isinstance(test_id, str) or not test_id:
        errors.append(f"{where}.id: missing or empty")
    elif test_id in seen:
        errors.append(f"{where}.id: duplicate {test_id}")
    else:
        seen.add(test_id)
    if not isinstance(test.get("card"), str) or not test.get("card"):
        errors.append(f"{where}.card: missing or empty")
    if test.get("level") not in LEVELS:
        errors.append(f"{where}.level: must be one of {LEVELS}")
    status = test.get("status")
    if status not in STATUSES:
        errors.append(f"{where}.status: must be one of {STATUSES}")
    if status == "skip" and not test.get("reason"):
        errors.append(f"{where}.reason: a skipped test needs a reason")
    for key in _MEASURES:
        if key in test and test[key] is not None:
            if not _is_number(test[key]) or test[key] < 0:
                errors.append(f"{where}.{key}: must be a non-negative number")


def validate(record: Any) -> list[str]:
    """Return a list of problems; an empty list means the record is valid."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record: not an object"]
    if record.get("schema") != SCHEMA_VERSION:
        errors.append(f"schema: must be {SCHEMA_VERSION}")
    _check_section(record, "run", _RUN_KEYS, errors)
    _check_section(record, "device", _DEVICE_KEYS, errors)
    _check_section(record, "impl", _IMPL_KEYS, errors)
    run = record.get("run")
    if isinstance(run, dict) and run.get("set") not in SETS:
        errors.append(f"run.set: must be one of {SETS}")
    tests = record.get("tests")
    if not isinstance(tests, list):
        errors.append("tests: missing or not a list")
    else:
        seen: set[str] = set()
        for index, test in enumerate(tests):
            _check_test(index, test, seen, errors)
    return errors


def load(path: str | Path) -> dict:
    """Read a result record; raises ValueError when it is not valid."""
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    problems = validate(record)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return record

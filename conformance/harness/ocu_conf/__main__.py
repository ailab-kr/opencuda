"""Command line: python -m ocu_conf validate <result.json> [...]"""

from __future__ import annotations

import json
import sys

from .results import validate
from .score import summarize


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "validate":
        print("usage: python -m ocu_conf validate <result.json> [...]",
              file=sys.stderr)
        return 2
    failed = False
    for path in argv[1:]:
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            print(f"{path}: cannot read: {error}", file=sys.stderr)
            failed = True
            continue
        problems = validate(record)
        if problems:
            failed = True
            for problem in problems:
                print(f"{path}: {problem}", file=sys.stderr)
            continue
        print(f"{path}: ok {json.dumps(summarize(record['tests']))}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

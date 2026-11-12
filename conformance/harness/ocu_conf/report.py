"""Render result records as one static HTML scoreboard page.

    python3 -m ocu_conf.report <result.json> [...] --out scoreboard.html

One column per record (device + implementation), one row per card, each
cell "passed / total". The page is self-contained (no scripts, no external
files). Publishing it anywhere is a person's decision (PLAN P1-07).
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

from .results import load


def _cell(passed: int, total: int) -> str:
    if total == 0:
        return '<td class="na">-</td>'
    cls = "all" if passed == total else ("some" if passed else "none")
    return f'<td class="{cls}">{passed} / {total}</td>'


def render(records: list[dict]) -> str:
    cards = sorted({t["card"] for r in records for t in r["tests"]})
    heads = []
    for r in records:
        d, i, run = r["device"], r["impl"], r["run"]
        heads.append(
            f"<th>{html.escape(i['name'])} {html.escape(i['version'])}<br>"
            f"<small>{html.escape(i['backend'])} · {html.escape(d['vendor'])} "
            f"{html.escape(d['model'])}<br>{html.escape(d['os'])} · {html.escape(run['date'])} · "
            f"{html.escape(run['set'])}</small></th>")
    rows = []
    totals = [[0, 0] for _ in records]
    for card in cards:
        cells = []
        for k, r in enumerate(records):
            tests = [t for t in r["tests"] if t["card"] == card]
            p = sum(t["status"] == "pass" for t in tests)
            totals[k][0] += p
            totals[k][1] += len(tests)
            cells.append(_cell(p, len(tests)))
        rows.append(f"<tr><th class='card'>{html.escape(card)}</th>{''.join(cells)}</tr>")
    total_row = "".join(_cell(p, t) for p, t in totals)
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>적합성 시험 성적표</title>
<style>
body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: .35rem .7rem; text-align: center; }}
th.card {{ text-align: left; font-weight: 500; }}
td.all {{ background: #e3f4e6; }} td.some {{ background: #fdf1d6; }}
td.none {{ background: #f9dcdc; }} td.na {{ color: #999; }}
small {{ color: #555; font-weight: 400; }}
</style></head><body>
<h1>적합성 시험 성적표</h1>
<p>칸은 통과한 사례 수 / 전체 사례 수다. 기준 답과 채점 규칙은 저장소의 conformance/README.md에 있다.</p>
<table>
<tr><th>카드</th>{''.join(heads)}</tr>
{''.join(rows)}
<tr><th class="card">합계</th>{total_row}</tr>
</table>
</body></html>
"""


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        records = [load(p) for p in args.records]
    except (OSError, ValueError) as error:
        print(f"report: {error}", file=sys.stderr)
        return 2
    Path(args.out).write_text(render(records), encoding="utf-8")
    print(f"report: {len(records)} record(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

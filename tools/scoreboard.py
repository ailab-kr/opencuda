#!/usr/bin/env python3
"""Static conformance scoreboard: per backend, per area pass counts.

Reads

  * conformance result records (JSON, schema 1; conformance/README.md),
    for example <build>/tests/blas_cpu_result.json,
  * CTest output of a build directory (plain `ctest` output or the JUnit
    file written by `ctest --output-junit`), and
  * the output of conformance/external/rodinia/run.sh,

and writes one Markdown file and one self-contained HTML file (no scripts,
no external assets). Python standard library only.

    tools/scoreboard.py [records or folders ...]
                        [--ctest BACKEND=FILE ...] [--rodinia BACKEND=FILE ...]
                        [--out-dir DIR] [--title TEXT]

A folder argument is searched for files named *result*.json.
BACKEND is the backend a CTest run or a Rodinia run was made for ("cpu",
"nvidia", "metal", ...). CTest tests whose names mark another backend
(gpu_*, nvidia_*, metal_*, ocu_cc_amd_*) are counted under that backend.

Speed is reported as a median and a slow-10% value, never as a mean:
  time   median and 90th percentile of the measured time in ms
         (CTest wall time per test, or time_ms_median of a record);
  ratio  median and 10th percentile of speed_ratio (speed relative to the
         vendor's own way on the same chip; 1.0 = equal), when recorded.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "conformance" / "harness"))
from ocu_conf.results import validate  # noqa: E402
from ocu_conf.score import percentile  # noqa: E402

STATUSES = ("pass", "fail", "skip", "unsupported")
LEVEL_NAMES = {"L1": "build", "L2": "runtime", "L3": "language",
               "L4": "libs", "L5": "apps"}
SAMPLES_PER_GROUP = 3

# CTest test name -> (backend override or None, area). First match wins.
CTEST_RULES = [
    (r"^(gpu_saxpy|gpu_rt_)", "nvidia", "runtime API"),
    (r"^nvidia_backend", "nvidia", "backend interface"),
    (r"^ocu_cc_nvidia", "nvidia", "compiler path"),
    (r"^ocu_cc_amd", "amd (compile only)", "compiler path"),
    (r"^metal_(cu2msl|msl)", "metal (translation checks on host)", "compiler path"),
    (r"^metal_", "metal (translation checks on host)", "backend interface"),
    (r"^(probe_|ocu_cc)", None, "compiler path"),
    (r"^(blas_|conformance_blas)", None, "libs (BLAS)"),
    (r"^runtime_", None, "runtime API"),
    (r"^device_math", None, "device math"),
    (r"^install_", None, "packaging"),
    (r"^(smoke_|bench_|cpu_)", None, "backend interface"),
]


class Test:
    """One normalised test result."""

    __slots__ = ("backend", "area", "id", "status", "note", "time_ms",
                 "ratio", "err")

    def __init__(self, backend, area, id, status, note="", time_ms=None,
                 ratio=None, err=None):
        self.backend, self.area, self.id = backend, area, id
        self.status, self.note = status, note
        self.time_ms, self.ratio, self.err = time_ms, ratio, err


# ---- readers ------------------------------------------------------------

def read_record(path: Path) -> tuple[dict, list[Test]]:
    """A conformance result record -> (meta, tests)."""
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    problems = validate(record)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    backend = record["impl"]["backend"]
    tests = []
    for t in record["tests"]:
        parts = t["id"].split(".")
        seg = parts[1] if len(parts) >= 3 else ""
        area = f"conformance {t['level']} {LEVEL_NAMES[t['level']]}"
        if seg:
            area += f"/{seg}"
        tests.append(Test(
            backend, area, t["id"], t["status"], t.get("reason", ""),
            t.get("time_ms_median"), t.get("speed_ratio"),
            t.get("error_ratio", t.get("max_rel_err"))))
    meta = {"backend": backend, "device": record["device"],
            "impl": record["impl"], "run": record["run"],
            "source": str(path)}
    return meta, tests


_CTEST_LINE = re.compile(
    r"^\s*\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.+\s*(.*?)\s*([\d.]+)\s+sec\s*$")


def _ctest_classify(name: str, default_backend: str) -> tuple[str, str]:
    for pattern, backend, area in CTEST_RULES:
        if re.search(pattern, name):
            return backend or default_backend, "ctest: " + area
    return default_backend, "ctest: other"


def _ctest_status(word: str) -> tuple[str, str]:
    word = word.replace("*", "").strip()
    if word == "Passed":
        return "pass", ""
    if word.startswith("Skipped") or word.startswith("Not Run"):
        return "skip", word
    return "fail", word


def read_ctest(path: Path, default_backend: str) -> list[Test]:
    """Plain `ctest` output or a JUnit file from `ctest --output-junit`."""
    text = path.read_text(encoding="utf-8", errors="replace")
    tests: list[Test] = []
    if text.lstrip().startswith("<?xml") or text.lstrip().startswith("<testsuite"):
        for case in ET.fromstring(text).iter("testcase"):
            name = case.get("name", "")
            backend, area = _ctest_classify(name, default_backend)
            status, note = "pass", ""
            if case.find("failure") is not None or case.find("error") is not None:
                status, note = "fail", "failed"
            elif case.find("skipped") is not None:
                status, note = "skip", "skipped"
            seconds = float(case.get("time") or 0)
            tests.append(Test(backend, area, name, status, note, seconds * 1000))
    else:
        for line in text.splitlines():
            m = _CTEST_LINE.match(line)
            if not m:
                continue
            name, word, seconds = m.group(1), m.group(2), float(m.group(3))
            backend, area = _ctest_classify(name, default_backend)
            status, note = _ctest_status(word)
            tests.append(Test(backend, area, name, status, note, seconds * 1000))
    if not tests:
        raise ValueError(f"{path}: no CTest results found")
    return tests


_PATH = re.compile(r"(/[\w.+-]+)+/([\w.+-]+)")  # keep file names, drop folders
_RODINIA_LINE = re.compile(r"^(\S+)\s+(ok|FAIL|BUILD-FAIL|BUILT)\s+(.*)$")


def read_rodinia(path: Path, backend: str) -> list[Test]:
    """Output of conformance/external/rodinia/run.sh.

    ok          built and equal to an independent reference   -> pass
    BUILT       built (and ran, or needs input data); no independent
                check yet                                     -> skip
    FAIL / BUILD-FAIL                                         -> fail
    """
    tests = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _RODINIA_LINE.match(line)
        if not m:
            continue
        name, word, detail = m.groups()
        status = {"ok": "pass", "BUILT": "skip"}.get(word, "fail")
        note = _PATH.sub(lambda m: m.group(2), detail.strip())
        if word == "BUILT":
            note = "built, no independent check: " + note
        tests.append(Test(backend, "apps: Rodinia", name, status, note))
    if not tests:
        raise ValueError(f"{path}: no Rodinia result lines found")
    return tests


# ---- summary ------------------------------------------------------------

def _stat(values: list[float], fraction: float):
    return percentile(values, fraction) if values else None


def speed_summary(tests: list[Test]) -> dict:
    """Median and slow-10% of time, and of speed_ratio when recorded."""
    times = [t.time_ms for t in tests if t.time_ms is not None and t.status == "pass"]
    ratios = [t.ratio for t in tests if t.ratio is not None and t.status == "pass"]
    timed = [t for t in tests if t.time_ms is not None and t.status == "pass"]
    slowest = max(timed, key=lambda t: t.time_ms).id if timed else None
    return {
        "time_n": len(times),
        "time_median_ms": _stat(times, 0.5),
        "time_p90_ms": _stat(times, 0.9),
        "ratio_n": len(ratios),
        "ratio_median": _stat(ratios, 0.5),
        "ratio_p10": _stat(ratios, 0.1),
        "slowest": slowest,
    }


def counts(tests: list[Test]) -> dict:
    out = {s: 0 for s in STATUSES}
    for t in tests:
        out[t.status] += 1
    out["total"] = len(tests)
    return out


def pick_samples(tests: list[Test], n: int = SAMPLES_PER_GROUP) -> list[Test]:
    """Failures first, then tests spread evenly over the group."""
    bad = [t for t in tests if t.status in ("fail", "unsupported")]
    good = [t for t in tests if t.status == "pass"]
    chosen = bad[:n]
    room = n - len(chosen)
    if room > 0 and good:
        step = max(1, len(good) // room)
        chosen += good[::step][:room]
    return chosen


def build_model(all_tests: list[Test], metas: list[dict]) -> dict:
    backends = sorted({t.backend for t in all_tests})
    areas = sorted({t.area for t in all_tests})
    grid = {}
    for a in areas:
        for b in backends:
            grid[(a, b)] = [t for t in all_tests if t.area == a and t.backend == b]
    return {
        "backends": backends,
        "areas": areas,
        "grid": grid,
        "per_backend": {b: [t for t in all_tests if t.backend == b] for b in backends},
        "metas": metas,
    }


# ---- formatting ---------------------------------------------------------

def _fmt(value, digits=3) -> str:
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.{digits}g}"


def _cell_text(tests: list[Test]) -> str:
    if not tests:
        return "-"
    c = counts(tests)
    text = f"{c['pass']} / {c['total']}"
    extra = [f"{c[s]} {s}" for s in ("fail", "skip", "unsupported") if c[s]]
    return text + (f" ({', '.join(extra)})" if extra else "")


def _cell_class(tests: list[Test]) -> str:
    if not tests:
        return "na"
    c = counts(tests)
    if c["pass"] == c["total"]:
        return "all"
    if c["fail"] == 0:
        return "part"          # only skips / unsupported
    return "bad"


def _sample_text(t: Test) -> str:
    bits = [t.status]
    if t.err is not None:
        bits.append(f"error/bound {t.err:g}")
    if t.time_ms is not None:
        bits.append(f"{t.time_ms:g} ms")
    if t.note:
        bits.append(t.note[:100])
    return ", ".join(bits)


def render_markdown(model: dict, title: str) -> str:
    be, areas, grid = model["backends"], model["areas"], model["grid"]
    out = [f"# {title}", "",
           "Cells are passed / total. Skipped and unsupported tests are "
           "listed, never counted as passed.", ""]
    out += ["## Runs", "", "| backend | device | OS | implementation | date |",
            "|---|---|---|---|---|"]
    for m in model["metas"]:
        d, i = m["device"], m["impl"]
        out.append(f"| {m['backend']} | {d['vendor']} {d['model']} | {d['os']} | "
                   f"{i['name']} {i['version']} | {m['run']['date']} |")
    out += ["", "## Pass counts by area", "",
            "| area | " + " | ".join(be) + " |", "|---|" + "---|" * len(be)]
    for a in areas:
        out.append(f"| {a} | " + " | ".join(_cell_text(grid[(a, b)]) for b in be) + " |")
    totals = [counts(model["per_backend"][b]) for b in be]
    out.append("| **all** | " + " | ".join(f"**{c['pass']} / {c['total']}**" for c in totals) + " |")
    out += ["", "## Speed", "",
            "| backend | tests timed | time median (ms) | time slow 10% (ms) | "
            "speed ratio median | speed ratio slow 10% | slowest test |",
            "|---|---|---|---|---|---|---|"]
    for b in be:
        s = speed_summary(model["per_backend"][b])
        out.append(f"| {b} | {s['time_n']} | {_fmt(s['time_median_ms'])} | "
                   f"{_fmt(s['time_p90_ms'])} | {_fmt(s['ratio_median'])} | "
                   f"{_fmt(s['ratio_p10'])} | {s['slowest'] or '-'} |")
    out += ["", "Time is the wall time of one test run (a whole test program for "
            "CTest). Speed ratio is relative to the vendor's own way on the same "
            "chip (1.0 = equal) and is shown only where a run recorded it.", "",
            "## Sample outputs", ""]
    for a in areas:
        for b in be:
            tests = grid[(a, b)]
            if not tests:
                continue
            out.append(f"- **{a}, {b}**")
            for t in pick_samples(tests):
                out.append(f"  - `{t.id}`: {_sample_text(t)}")
    notable = [t for ts in model["per_backend"].values() for t in ts
               if t.status in ("fail", "skip", "unsupported")]
    if notable:
        out += ["", "## Not passed", ""]
        for t in notable:
            out.append(f"- {t.backend} / {t.area} / `{t.id}`: {t.status}"
                       + (f" ({t.note[:120]})" if t.note else ""))
    return "\n".join(out) + "\n"


_CSS = """
body { font: 14px/1.5 system-ui, sans-serif; margin: 2rem; color: #222; max-width: 72rem; }
table { border-collapse: collapse; margin: .5rem 0 1.5rem; }
th, td { border: 1px solid #ccc; padding: .3rem .7rem; text-align: center; }
th.l, td.l { text-align: left; }
td.all { background: #e3f4e6; } td.part { background: #fdf1d6; }
td.bad { background: #f9dcdc; } td.na { color: #999; }
code { font-size: 12px; } small { color: #555; }
@media (prefers-color-scheme: dark) {
  body { background: #181a1b; color: #ddd; } th, td { border-color: #444; }
  td.all { background: #1e3a25; } td.part { background: #46391a; }
  td.bad { background: #4a2323; } small { color: #aaa; }
}
"""


def render_html(model: dict, title: str) -> str:
    e = html.escape
    be, areas, grid = model["backends"], model["areas"], model["grid"]
    p = [f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
         f"<title>{e(title)}</title><style>{_CSS}</style></head><body>",
         f"<h1>{e(title)}</h1>",
         "<p>Cells are passed / total. Skipped and unsupported tests are "
         "listed, never counted as passed.</p>",
         "<h2>Runs</h2><table><tr><th>backend</th><th>device</th><th>OS</th>"
         "<th>implementation</th><th>date</th></tr>"]
    for m in model["metas"]:
        d, i = m["device"], m["impl"]
        p.append(f"<tr><td>{e(m['backend'])}</td><td>{e(d['vendor'])} {e(d['model'])}</td>"
                 f"<td>{e(d['os'])}</td><td>{e(i['name'])} {e(i['version'])}</td>"
                 f"<td>{e(m['run']['date'])}</td></tr>")
    p.append("</table><h2>Pass counts by area</h2><table><tr><th class=\"l\">area</th>"
             + "".join(f"<th>{e(b)}</th>" for b in be) + "</tr>")
    for a in areas:
        cells = "".join(
            f"<td class=\"{_cell_class(grid[(a, b)])}\">{e(_cell_text(grid[(a, b)]))}</td>"
            for b in be)
        p.append(f"<tr><td class=\"l\">{e(a)}</td>{cells}</tr>")
    totals = [counts(model["per_backend"][b]) for b in be]
    p.append("<tr><th class=\"l\">all</th>"
             + "".join(f"<th>{c['pass']} / {c['total']}</th>" for c in totals)
             + "</tr></table><h2>Speed</h2><table><tr><th>backend</th><th>tests timed</th>"
             "<th>time median (ms)</th><th>time slow 10% (ms)</th>"
             "<th>speed ratio median</th><th>speed ratio slow 10%</th>"
             "<th>slowest test</th></tr>")
    for b in be:
        s = speed_summary(model["per_backend"][b])
        p.append(f"<tr><td>{e(b)}</td><td>{s['time_n']}</td>"
                 f"<td>{_fmt(s['time_median_ms'])}</td><td>{_fmt(s['time_p90_ms'])}</td>"
                 f"<td>{_fmt(s['ratio_median'])}</td><td>{_fmt(s['ratio_p10'])}</td>"
                 f"<td><code>{e(s['slowest'] or '-')}</code></td></tr>")
    p.append("</table><p><small>Time is the wall time of one test run (a whole test "
             "program for CTest). Speed ratio is relative to the vendor's own way on "
             "the same chip (1.0 = equal) and is shown only where a run recorded "
             "it.</small></p><h2>Sample outputs</h2><ul>")
    for a in areas:
        for b in be:
            tests = grid[(a, b)]
            if not tests:
                continue
            items = "".join(f"<li><code>{e(t.id)}</code>: {e(_sample_text(t))}</li>"
                            for t in pick_samples(tests))
            p.append(f"<li><b>{e(a)}, {e(b)}</b><ul>{items}</ul></li>")
    p.append("</ul>")
    notable = [t for ts in model["per_backend"].values() for t in ts
               if t.status in ("fail", "skip", "unsupported")]
    if notable:
        p.append("<h2>Not passed</h2><ul>")
        for t in notable:
            note = f" ({e(t.note[:120])})" if t.note else ""
            p.append(f"<li>{e(t.backend)} / {e(t.area)} / <code>{e(t.id)}</code>: "
                     f"{t.status}{note}</li>")
        p.append("</ul>")
    p.append("</body></html>\n")
    return "\n".join(p)


# ---- command line -------------------------------------------------------

def _pair(text: str) -> tuple[str, Path]:
    backend, sep, path = text.partition("=")
    if not sep or not backend or not path:
        raise argparse.ArgumentTypeError("expected BACKEND=FILE")
    return backend, Path(path)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Static conformance scoreboard.")
    ap.add_argument("records", nargs="*", help="result JSON files or folders")
    ap.add_argument("--ctest", action="append", default=[], type=_pair,
                    metavar="BACKEND=FILE")
    ap.add_argument("--rodinia", action="append", default=[], type=_pair,
                    metavar="BACKEND=FILE")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--title", default="Conformance scoreboard")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for item in args.records:
        path = Path(item)
        files += sorted(path.rglob("*result*.json")) if path.is_dir() else [path]
    tests: list[Test] = []
    metas: list[dict] = []
    try:
        for f in files:
            meta, ts = read_record(f)
            tests += ts
            metas.append(meta)
        for backend, f in args.ctest:
            tests += read_ctest(f, backend)
        for backend, f in args.rodinia:
            tests += read_rodinia(f, backend)
    except (OSError, ValueError, ET.ParseError) as error:
        print(f"scoreboard: {error}", file=sys.stderr)
        return 2
    if not tests:
        print("scoreboard: no input", file=sys.stderr)
        return 2
    # Runs without a record (CTest, Rodinia) still get a line in "Runs".
    seen = {m["backend"] for m in metas}
    for b in sorted({t.backend for t in tests} - seen):
        metas.append({"backend": b, "device": {"vendor": "", "model": "(not recorded)", "os": "-"},
                      "impl": {"name": "-", "version": ""}, "run": {"date": "-"},
                      "source": ""})
    model = build_model(tests, metas)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scoreboard.md").write_text(render_markdown(model, args.title), encoding="utf-8")
    (out / "scoreboard.html").write_text(render_html(model, args.title), encoding="utf-8")
    c = counts(tests)
    print(f"scoreboard: {c['pass']} / {c['total']} pass, {len(model['backends'])} "
          f"backend(s), {len(model['areas'])} area(s) -> {out}/scoreboard.{{md,html}}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

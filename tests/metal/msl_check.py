#!/usr/bin/env python3
"""Check Metal images made by compiler/metal/cu2msl.py without a Metal compiler.

    msl_check.py [--clang clang++] [--expect-kernels N] <image>...

For each image: the header is well formed (param / global / init lines; every
kernel named in a param line is defined in the source; globals do not
overlap), and the MSL source compiles as C++ for OpenCL against msl_stub.h
(Clang, -fsyntax-only). Address spaces are checked as MSL checks them.

Two differences between the languages are bridged, not checked:
  - MSL vector constructors `float2(x, y)` are written `(float2)(x, y)`;
  - C++ for OpenCL cannot copy a struct between address spaces through its
    implicit copy constructor or assignment (MSL can); those diagnostics are
    dropped.
Exit status 0 when every image passes.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VEC = r"(?:u?char|u?short|u?int|u?long|float|half|bool)[234]"


def check_header(meta: list[str], source: str) -> list[str]:
    errs, spans = [], []
    for line in meta:
        f = line.split()
        if not f:
            continue
        if f[0] == "param":
            if len(f) < 2 or not all(x.isdigit() for x in f[2:]):
                errs.append(f"bad param line: {line}")
            elif not re.search(r"\bkernel\s+void\s+" + re.escape(f[1]) + r"\s*\(", source):
                errs.append(f"kernel {f[1]} is in a param line but not in the source")
        elif f[0] == "global":
            if len(f) != 4 or not f[2].isdigit() or not f[3].isdigit():
                errs.append(f"bad global line: {line}")
            else:
                spans.append((int(f[2]), int(f[2]) + int(f[3]), f[1]))
        elif f[0] == "init":
            if len(f) != 3 or not f[1].isdigit() or len(f[2]) % 2 or not re.fullmatch(r"[0-9a-f]*", f[2]):
                errs.append(f"bad init line: {line}")
            else:
                a, b = int(f[1]), int(f[1]) + len(f[2]) // 2
                if not any(s <= a and b <= e for s, e, _ in spans):
                    errs.append(f"init line outside every global: {line}")
        else:
            errs.append(f"unknown header line: {line}")
    spans.sort()
    for (a, b, n), (c, d, m) in zip(spans, spans[1:]):
        if c < b:
            errs.append(f"globals {n} and {m} overlap")
    return errs


def bridge(source: str) -> str:
    source = source.replace("#include <metal_stdlib>", '#include "msl_stub.h"')
    return re.sub(r"(?<![\w.])(" + VEC + r")\s*\(", r"(\1)(", source)


def diagnostics(text: str) -> tuple[list[str], int]:
    """Error blocks (error line + its notes), minus the struct-copy ones."""
    blocks, cur = [], None
    for line in text.splitlines():
        m = re.search(r":\d+:\d+: (error|warning|note|fatal error): ", line)
        if m and m.group(1) in ("error", "fatal error", "warning"):
            cur = [line] if m.group(1) != "warning" else None
            if cur is not None:
                blocks.append(cur)
        elif cur is not None:
            cur.append(line)
    kept, dropped = [], 0
    for b in blocks:
        notes = [l for l in b if ": note: " in l and "candidate" in l]
        if re.search(r"no matching constructor for initialization|no viable overloaded '='", b[0]) and notes and \
                all(re.search(r"implicit (copy|move) (constructor|assignment operator)", n) and
                    "address space" in n for n in notes):
            dropped += 1
            continue
        kept.append("\n".join(b))
    return kept, dropped


def check(image: Path, clang: str) -> tuple[bool, str, int]:
    text = image.read_text(encoding="utf-8")
    if not text.startswith("OCUMSL1\n") or "\n---\n" not in text:
        return False, "not an OCUMSL1 image", 0
    head, source = text[8:].split("\n---\n", 1)
    meta = [l for l in head.splitlines() if l.strip()]
    errs = check_header(meta, source)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / (image.stem + ".clcpp")
        f.write_text(bridge(source), encoding="utf-8")
        out = subprocess.run([clang, "-x", "clcpp", "-cl-std=clc++2021", "-Xclang",
                              "-cl-ext=-__opencl_c_generic_address_space,-__opencl_c_pipes,"
                              "-__opencl_c_device_enqueue", "-fsyntax-only", "-ferror-limit=0",
                              "-Wno-unknown-attributes", "-Wno-unused-value", f"-I{HERE}", str(f)],
                             capture_output=True, text=True)
    kept, dropped = diagnostics(out.stderr)
    if out.returncode != 0 and not kept and not errs:
        # Clang failed but every error was a bridged one: fine.
        pass
    errs += kept
    if out.returncode != 0 and not kept and "error" not in out.stderr:
        errs.append(out.stderr.strip()[-2000:])
    kernels = sum(1 for l in meta if l.startswith("param "))
    return not errs, "\n".join(errs), kernels


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--clang", default=os.environ.get("OCU_CLANG", "clang++"))
    ap.add_argument("--expect-kernels", type=int, default=None,
                    help="fail unless the images hold this many kernels in all")
    a = ap.parse_args(argv)
    ok_all, total = True, 0
    for img in a.images:
        ok, msg, n = check(Path(img), a.clang)
        total += n
        print(f"{img}: {'ok' if ok else 'FAIL'} ({n} kernels)")
        if not ok:
            print(msg)
            ok_all = False
    if a.expect_kernels is not None and total != a.expect_kernels:
        print(f"expected {a.expect_kernels} kernels, found {total}")
        ok_all = False
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

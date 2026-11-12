#!/usr/bin/env python3
"""Rewrite the two CUDA-only constructs that plain C++ cannot parse, so a
.cu file can be compiled as ordinary C++ for the CPU backend.

PROBE ONLY (PLAN P0-W1-11): this is a feasibility tool, not a shipped
component. It touches exactly two things and copies everything else
byte for byte, keeping line numbers:

  1. Launches:  KERNEL<<<CFG>>>(ARGS)
         ->     ::ocu_cpu::launch(KERNEL, ::ocu_cpu::cfg(CFG), ARGS)
  2. Dynamic shared memory:  extern __shared__ TYPE NAME[];
         ->     TYPE *NAME = (TYPE *)::ocu_cpu::dynamic_shared();

Comments, string and character literals (including raw strings) and
preprocessor lines are never rewritten.

Usage: cpu_rewrite.py <in.cu> <out.cpp>
"""

from __future__ import annotations

import re
import sys

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class RewriteError(Exception):
    pass


def _code_mask(src: str) -> list[bool]:
    """mask[i] is True when src[i] is code, False inside comments, string or
    character literals and preprocessor lines."""
    n = len(src)
    mask = [True] * n
    i = 0
    at_line_start = True
    while i < n:
        c = src[i]
        if at_line_start and c in " \t":
            i += 1
            continue
        if at_line_start and c == "#":
            # Preprocessor line, with backslash continuations.
            j = i
            while j < n:
                if src[j] == "\n" and (j == 0 or src[j - 1] != "\\"):
                    break
                j += 1
            for k in range(i, j):
                mask[k] = False
            i = j
            continue
        at_line_start = False
        if c == "\n":
            at_line_start = True
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                mask[k] = False
            i = j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                mask[k] = False
            i = j
            continue
        raw = re.match(r'(?:u8|u|U|L)?R"([^()\\\s]{0,16})\(', src[i:i + 24])
        if raw and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            end = src.find(")" + raw.group(1) + '"', i + raw.end())
            if end < 0:
                raise RewriteError("unterminated raw string")
            j = end + len(raw.group(1)) + 2
            for k in range(i, j):
                mask[k] = False
            i = j
            continue
        if c in "\"'":
            # A ' between digits is a digit separator (1'000), not a quote.
            if c == "'" and i > 0 and src[i - 1].isalnum() and i + 1 < n \
                    and src[i + 1].isalnum():
                i += 1
                continue
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n":
                    raise RewriteError(f"unterminated literal at offset {i}")
                j += 1
            j += 1
            for k in range(i, min(j, n)):
                mask[k] = False
            i = j
            continue
        i += 1
    return mask


def _skip_ws_back(src: str, i: int) -> int:
    while i > 0 and src[i - 1] in " \t\r\n":
        i -= 1
    return i


def _callee_start(src: str, mask: list[bool], end: int) -> int:
    """Start of the kernel expression that ends at `end` (exclusive):
    a possibly qualified name with optional template arguments."""

    def ident_char(c: str) -> bool:
        return c.isalnum() or c == "_"

    i = _skip_ws_back(src, end)
    while True:
        if i > 0 and src[i - 1] == ">":
            depth = 0
            j = i - 1
            while j >= 0:
                if mask[j] and src[j] == ">":
                    depth += 1
                elif mask[j] and src[j] == "<":
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            if j < 0:
                raise RewriteError("unbalanced template arguments before <<<")
            i = _skip_ws_back(src, j)
        j = i
        while j > 0 and ident_char(src[j - 1]):
            j -= 1
        if j == i:
            raise RewriteError("no kernel name before <<<")
        i = j
        k = _skip_ws_back(src, i)
        if k >= 2 and src[k - 2:k] == "::":
            m = _skip_ws_back(src, k - 2)
            if m > 0 and (ident_char(src[m - 1]) or src[m - 1] == ">"):
                i = m
                continue
            return k - 2  # leading '::' (global scope)
        return i


def _find_close(src: str, mask: list[bool], start: int, open_c: str,
                close_c: str) -> int:
    """Index of the bracket that closes the one at `start`."""
    depth = 0
    for i in range(start, len(src)):
        if not mask[i]:
            continue
        if src[i] == open_c:
            depth += 1
        elif src[i] == close_c:
            depth -= 1
            if depth == 0:
                return i
    raise RewriteError(f"no closing {close_c!r}")


def _find_chevron_end(src: str, mask: list[bool], start: int) -> int:
    """Index of the '>>>' closing a launch configuration that starts at
    `start` (just after '<<<'), skipping nested brackets."""
    depth = 0
    i = start
    while i < len(src):
        if mask[i]:
            c = src[i]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif depth == 0 and src.startswith(">>>", i) and all(mask[i:i + 3]):
                return i
        i += 1
    raise RewriteError("no closing >>>")


def _rewrite_launches(src: str) -> tuple[str, int]:
    count = 0
    while True:
        mask = _code_mask(src)
        pos = -1
        i = 0
        while True:
            i = src.find("<<<", i)
            if i < 0:
                break
            if all(mask[i:i + 3]):
                pos = i
                break
            i += 3
        if pos < 0:
            return src, count
        callee_from = _callee_start(src, mask, pos)
        callee = src[callee_from:_skip_ws_back(src, pos)]
        cfg_end = _find_chevron_end(src, mask, pos + 3)
        cfg = src[pos + 3:cfg_end]
        j = cfg_end + 3
        while j < len(src) and src[j] in " \t\r\n":
            j += 1
        if j >= len(src) or src[j] != "(":
            raise RewriteError("launch without an argument list")
        args_end = _find_close(src, mask, j, "(", ")")
        args = src[j + 1:args_end]
        sep = ", " if args.strip() else ""
        between = src[_skip_ws_back(src, pos):pos] + src[cfg_end + 3:j]
        newlines = "\n" * between.count("\n")  # keep line numbers
        replacement = (f"::ocu_cpu::launch({callee}, ::ocu_cpu::cfg({cfg})"
                       f"{sep}{args}){newlines}")
        src = src[:callee_from] + replacement + src[args_end + 1:]
        count += 1


_EXTERN_SHARED = re.compile(
    r"\bextern\s+__shared__\s+((?:__align__\s*\(\s*\d+\s*\)\s*)?)"
    r"([A-Za-z_][A-Za-z0-9_:<>,\s\*]*?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\]\s*;")


def _rewrite_extern_shared(src: str) -> tuple[str, int]:
    mask = _code_mask(src)
    out = []
    last = 0
    count = 0
    for m in _EXTERN_SHARED.finditer(src):
        if not all(mask[m.start():m.end()]):
            continue
        type_, name = m.group(2).strip(), m.group(3)
        newlines = "\n" * m.group(0).count("\n")
        out.append(src[last:m.start()])
        out.append(f"{type_} *{name} = ({type_} *)::ocu_cpu::dynamic_shared();"
                   f"{newlines}")
        last = m.end()
        count += 1
    out.append(src[last:])
    return "".join(out), count


def rewrite(src: str) -> tuple[str, dict]:
    src, shared = _rewrite_extern_shared(src)
    src, launches = _rewrite_launches(src)
    return src, {"launches": launches, "extern_shared": shared}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as handle:
        src = handle.read()
    try:
        out, stats = rewrite(src)
    except RewriteError as error:
        print(f"{argv[0]}: {error}", file=sys.stderr)
        return 1
    with open(argv[1], "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f'#line 1 "{argv[0]}"\n')
        handle.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

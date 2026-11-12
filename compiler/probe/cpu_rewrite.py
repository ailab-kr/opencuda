#!/usr/bin/env python3
"""Rewrite the two CUDA-only constructs that plain C++ cannot parse, so a
.cu file can be compiled as ordinary C++ for the CPU backend.

PROBE ONLY (PLAN P0-W1-11): this is a feasibility tool, not a shipped
component. It touches exactly two things and copies everything else
byte for byte, keeping line numbers:

  1. Launches:  KERNEL<<<CFG>>>(ARGS)
         ->     ::ocu_cpu::launch(::ocu_cpu::cfg(CFG),
                                  [](auto &&...ocu_a_) { KERNEL(ocu_a_...); }, ARGS)
     The kernel call inside the lambda is an ordinary C++ call, so template
     arguments, overloads and conversions resolve as in any call. A `NULL`
     argument becomes `nullptr` so it still converts to a pointer after
     being copied.
  2. Dynamic shared memory:  extern __shared__ T NAME[];
         ->     static ::ocu_cpu::dyn_shared<T> NAME;
     (qualifiers kept, __align__ dropped: the backend aligns to 64 bytes).

Comments, string and character literals (including raw strings and
prefixed literals) and preprocessor directives other than #define are never
rewritten. Bodies of #define are rewritten, with line continuations kept.

Usage: cpu_rewrite.py <in.cu> <out.cpp>
"""

from __future__ import annotations

import re
import sys

LAMBDA_ARG = "ocu_a_"


class RewriteError(Exception):
    pass


def _ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _code_mask(src: str) -> list[bool]:
    """mask[i] is True when src[i] is code, False inside comments, literals
    and preprocessor directives (except the body of a #define)."""
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
            m = re.match(r"#\s*define\s+[A-Za-z_]\w*(\([^)]*\))?", src[i:])
            if m:
                # Mask the head only; the body is code (it may hold launches).
                for k in range(i, i + m.end()):
                    mask[k] = False
                i += m.end()
                at_line_start = False
                continue
            j = i  # any other directive: skip it, with continuations
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
            j = i
            while j < n and not (src[j] == "\n" and src[j - 1] != "\\"):
                j += 1
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
        if raw and (i == 0 or not _ident_char(src[i - 1])):
            end = src.find(")" + raw.group(1) + '"', i + raw.end())
            if end < 0:
                raise RewriteError("unterminated raw string")
            j = end + len(raw.group(1)) + 2
            for k in range(i, j):
                mask[k] = False
            i = j
            continue
        if c in "\"'":
            if c == "'" and _is_digit_separator(src, i):
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
            # A prefix (L, u, U, u8) belongs to the literal.
            start = i
            while start > 0 and src[start - 1] in "LuU8" and not (
                    start > 1 and _ident_char(src[start - 2])):
                start -= 1
            for k in range(start, min(j, n)):
                mask[k] = False
            i = j
            continue
        i += 1
    return mask


def _is_digit_separator(src: str, i: int) -> bool:
    """A ' inside a number literal (1'000, 0x1'F), not a character literal."""
    if i + 1 >= len(src) or not src[i + 1].isalnum():
        return False
    j = i
    while j > 0 and (_ident_char(src[j - 1]) or src[j - 1] in ".'"):
        j -= 1
    return j < i and src[j].isdigit()


def _skip_back(src: str, mask: list[bool], i: int) -> int:
    """Move left over whitespace, comments and line continuations."""
    while i > 0 and (src[i - 1] in " \t\r\n\\" or not mask[i - 1]):
        i -= 1
    return i


def _skip_fwd(src: str, mask: list[bool], i: int) -> int:
    while i < len(src) and (src[i] in " \t\r\n\\" or not mask[i]):
        i += 1
    return i


def _callee_start(src: str, mask: list[bool], end: int) -> int:
    """Start of the kernel expression that ends at `end` (exclusive):
    a possibly qualified name with optional template arguments."""
    i = _skip_back(src, mask, end)
    while True:
        if i > 0 and src[i - 1] == ">":
            angle = 0
            paren = 0
            j = i - 1
            while j >= 0:
                if mask[j]:
                    ch = src[j]
                    if ch in ")]":
                        paren += 1
                    elif ch in "([":
                        paren -= 1
                    elif paren == 0 and ch == ">":
                        angle += 1
                    elif paren == 0 and ch == "<":
                        angle -= 1
                        if angle == 0:
                            break
                j -= 1
            if j < 0:
                raise RewriteError("unbalanced template arguments before <<<")
            i = _skip_back(src, mask, j)
        j = i
        while j > 0 and _ident_char(src[j - 1]):
            j -= 1
        if j == i:
            raise RewriteError("no kernel name before <<<")
        i = j
        k = _skip_back(src, mask, i)
        if k >= 2 and src[k - 2:k] == "::":
            m = _skip_back(src, mask, k - 2)
            if m > 0 and (_ident_char(src[m - 1]) or src[m - 1] == ">"):
                i = m
                continue
            return k - 2  # leading '::' (global scope)
        return i


def _find_close(src: str, mask: list[bool], start: int) -> int:
    """Index of the ')' that closes the '(' at `start`."""
    depth = 0
    for i in range(start, len(src)):
        if not mask[i]:
            continue
        if src[i] in "([{":
            depth += 1
        elif src[i] in ")]}":
            depth -= 1
            if depth == 0:
                return i
    raise RewriteError("no closing ')'")


def _find_chevron_end(src: str, mask: list[bool], start: int) -> int:
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


def _split_top_level(text: str) -> list[str]:
    """Split call arguments on top-level commas (text has no comments)."""
    parts, depth, cur = [], 0, []
    mask = _code_mask(text)
    for i, ch in enumerate(text):
        if mask[i] and ch in "([{":
            depth += 1
        elif mask[i] and ch in ")]}":
            depth -= 1
        if mask[i] and ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _fix_null_args(args: str) -> str:
    if not args.strip():
        return args
    out = []
    for part in _split_top_level(args):
        if part.strip() == "NULL":
            part = part.replace("NULL", "nullptr")
        out.append(part)
    return ",".join(out)


def _layout(text: str) -> str:
    """Keep only line breaks (and their continuation backslashes)."""
    return re.sub(r"[^\\\n]", "", text).replace("\\\\", "\\")


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
        # the callee text ends where the whitespace/comments before <<< start
        callee_end = _skip_back(src, mask, pos)
        callee = src[callee_from:callee_end]
        continued = callee.count("\\\n")  # line continuations in the callee
        callee = callee.replace("\\\n", " ")
        cfg_end = _find_chevron_end(src, mask, pos + 3)
        cfg = src[pos + 3:cfg_end]
        j = _skip_fwd(src, mask, cfg_end + 3)
        if j >= len(src) or src[j] != "(":
            raise RewriteError("launch without an argument list")
        args_end = _find_close(src, mask, j)
        args = _fix_null_args(src[j + 1:args_end])
        gaps = _layout(src[callee_end:pos] + src[cfg_end + 3:j]) + "\\\n" * continued
        sep = ", " if args.strip() else ""
        replacement = (f"::ocu_cpu::launch(::ocu_cpu::cfg({cfg}), "
                       f"[](auto &&...{LAMBDA_ARG}) {{ {callee}({LAMBDA_ARG}...); }}"
                       f"{sep}{args}){gaps}")
        src = src[:callee_from] + replacement + src[args_end + 1:]
        count += 1


_EXTERN_SHARED = re.compile(
    r"\bextern\b(?P<pre>[^;{}()\[\]]*?)\b__shared__\b(?P<post>[^;{}\[\]]*?)"
    r"\b(?P<name>[A-Za-z_]\w*)\s*\[\s*\]\s*;")
_ALIGN = re.compile(r"__align__\s*\(\s*\d+\s*\)")


def _rewrite_extern_shared(src: str) -> tuple[str, int]:
    mask = _code_mask(src)
    out = []
    last = 0
    count = 0
    for m in _EXTERN_SHARED.finditer(src):
        if not all(mask[m.start():m.end()]):
            continue
        type_ = _ALIGN.sub(" ", m.group("pre") + " " + m.group("post"))
        type_ = " ".join(type_.split())
        if not type_:
            raise RewriteError(f"extern __shared__ {m.group('name')} has no type")
        out.append(src[last:m.start()])
        out.append(f"static ::ocu_cpu::dyn_shared<{type_}> {m.group('name')};"
                   f"{_layout(m.group(0))}")
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
        print("usage: cpu_rewrite.py <in.cu> <out.cpp>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as handle:
        src = handle.read()
    try:
        out, _stats = rewrite(src)
    except RewriteError as error:
        print(f"{argv[0]}: {error}", file=sys.stderr)
        return 1
    with open(argv[1], "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f'#line 1 "{argv[0]}"\n')
        handle.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

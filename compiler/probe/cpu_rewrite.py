#!/usr/bin/env python3
"""Rewrite the two CUDA-only constructs that plain C++ cannot parse, so a
.cu file can be compiled as ordinary C++ for the CPU backend.

PROBE ONLY (PLAN P0-W1-11): this is a feasibility tool, not a shipped
component. It touches exactly two things and copies everything else
byte for byte, keeping line numbers:

  1. Launches:  KERNEL<<<CFG>>>(ARGS)
         ->     ::ocu_cpu::launch(::ocu_cpu::cfg(CFG),
                                  [=](auto &&...ocu_a_) { KERNEL(ocu_a_...); }, ARGS)
     The kernel call inside the lambda is an ordinary C++ call, so template
     arguments, overloads and conversions resolve as in any call; the
     lambda's return type names that call, so the launch can test which
     argument types the kernel accepts (literal 0, below). A `NULL`
     argument becomes `nullptr` and a literal `0` becomes
     ::ocu_cpu::literal_zero{}, so both still convert to a pointer after
     being copied. The lambda captures by copy, so KERNEL may also be a
     function pointer held in a variable or parameter (a launch helper's
     `Kernel kernel` argument); a named kernel function captures nothing.
  2. Dynamic shared memory:  extern __shared__ T NAME[];
         ->     T *const NAME = static_cast<T *>(::ocu_cpu::dynamic_shared());
                inside a function, or static ::ocu_cpu::dyn_shared<T> NAME;
                at namespace scope (looked up at each use)
     (qualifiers kept, __align__/alignas dropped: the backend aligns to 64
     bytes). A static `__shared__ alignas(N) T a[]` becomes
     `alignas(N) static thread_local T a[]` (the specifier may not follow
     `static thread_local`).

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


_INTEREST = re.compile(r"\n|//|/\*|\"|'")
_DEFINE_HEAD = re.compile(r"#\s*define\s+[A-Za-z_]\w*(\([^)]*\))?")
_RAW_PREFIX = re.compile(r"(?:u8|u|U|L)?R$")
_RAW_DELIM = re.compile(r'([^()\\\s]{0,16})\(')


def _line_end(src: str, i: int) -> int:
    """Index of the newline ending the logical line at i (continuations
    joined), or len(src)."""
    n = len(src)
    while True:
        j = src.find("\n", i)
        if j < 0:
            return n
        if j > 0 and src[j - 1] == "\\":
            i = j + 1
            continue
        return j


def _skip_literal(src: str, s: int, quote: str, clear) -> int:
    n = len(src)
    j = s + 1
    while j < n and src[j] != quote:
        if src[j] == "\\":
            j += 1
        elif src[j] == "\n":
            raise RewriteError(f"unterminated literal at offset {s}")
        j += 1
    j += 1
    # A prefix (L, u, U, u8) belongs to the literal.
    start = s
    while start > 0 and src[start - 1] in "LuU8" and not (
            start > 1 and _ident_char(src[start - 2])):
        start -= 1
    clear(start, min(j, n))
    return j


def _code_mask(src: str) -> bytearray:
    """mask[i] is 1 when src[i] is code, 0 inside comments, literals and
    preprocessor directives (except the body of a #define). Jumps between
    interesting characters, so a preprocessed translation unit (megabytes)
    takes a fraction of a second."""
    n = len(src)
    mask = bytearray(b"\x01") * n

    def clear(a: int, b: int) -> None:
        if b > a:
            mask[a:b] = bytes(b - a)

    def line_start(i: int) -> int:
        while i < n and src[i] in " \t":
            i += 1
        if i < n and src[i] == "#":
            m = _DEFINE_HEAD.match(src, i)
            if m:
                # Mask the head only; the body is code (it may hold launches).
                clear(i, m.end())
                return m.end()
            j = _line_end(src, i)
            clear(i, j)
            return j
        return i

    i = line_start(0)
    while True:
        m = _INTEREST.search(src, i)
        if m is None:
            break
        s, tok = m.start(), m.group()
        if tok == "\n":
            i = line_start(s + 1)
        elif tok == "//":
            j = _line_end(src, s)
            clear(s, j)
            i = j
        elif tok == "/*":
            j = src.find("*/", s + 2)
            j = n if j < 0 else j + 2
            clear(s, j)
            i = j
        elif tok == '"':
            pre = _RAW_PREFIX.search(src, max(0, s - 3), s)
            if pre and (pre.start() == 0 or not _ident_char(src[pre.start() - 1])):
                d = _RAW_DELIM.match(src, s + 1)
                if d is None:
                    raise RewriteError("bad raw string delimiter")
                end = src.find(")" + d.group(1) + '"', d.end())
                if end < 0:
                    raise RewriteError("unterminated raw string")
                j = end + len(d.group(1)) + 2
                clear(pre.start(), j)
                i = j
                continue
            i = _skip_literal(src, s, '"', clear)
        else:  # "'"
            if _is_digit_separator(src, s):
                i = s + 1
                continue
            i = _skip_literal(src, s, "'", clear)
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


_OPEN = re.compile(r"<[ \t]*<[ \t]*<")
_CLOSE = re.compile(r">[ \t]*>[ \t]*>")


def _find_chevron_end(src: str, mask, start: int) -> tuple[int, int]:
    """(start, end) of the >>> closing the configuration that begins at
    `start`. The three '>' may be separated by blanks, as nvcc accepts."""
    depth = 0
    i = start
    n = len(src)
    while i < n:
        if mask[i]:
            c = src[i]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif depth == 0 and c == ">":
                m = _CLOSE.match(src, i)
                if m and mask[m.end() - 1]:
                    return m.start(), m.end()
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


def _fix_null_args(args: str, zero: bool = True) -> str:
    if not args.strip():
        return args
    out = []
    for part in _split_top_level(args):
        # Preprocessed text may carry line markers inside an argument.
        lines = part.split("\n")
        code = [k for k, line in enumerate(lines) if not line.lstrip().startswith("#")]
        word = " ".join(lines[k] for k in code).strip()
        if word in ("NULL", "__null"):  # __null: NULL after GCC preprocessing
            new = "nullptr"
        elif word == "0" and zero:
            # A literal 0 converts to a pointer in a direct call; copied into
            # the launch it would be an int. This type converts to both.
            new = "::ocu_cpu::literal_zero{}"
        else:
            out.append(part)
            continue
        for k in code:
            if word in lines[k]:
                lines[k] = lines[k].replace(word, new, 1)
                break
        out.append("\n".join(lines))
    return ",".join(out)


def _layout(text: str) -> str:
    """Keep only line breaks (and their continuation backslashes)."""
    return re.sub(r"[^\\\n]", "", text).replace("\\\\", "\\")


def _rewrite_launches(src: str) -> tuple[str, int]:
    mask = _code_mask(src)
    opens = [m for m in _OPEN.finditer(src) if mask[m.start()] and mask[m.end() - 1]]
    count = 0
    # From the end, so earlier offsets and their mask stay valid.
    for m in reversed(opens):
        pos, open_end = m.start(), m.end()
        before = src[:_skip_back(src, mask, pos)]
        if re.search(r"\boperator$", before):
            continue  # `operator<< <>`: a template operator, not a launch
        callee_from = _callee_start(src, mask, pos)
        # the callee text ends where the whitespace/comments before <<< start
        callee_end = _skip_back(src, mask, pos)
        callee = src[callee_from:callee_end]
        continued = callee.count("\\\n")  # line continuations in the callee
        callee = callee.replace("\\\n", " ")
        cfg_end, close_end = _find_chevron_end(src, mask, open_end)
        cfg = src[open_end:cfg_end]
        j = _skip_fwd(src, mask, close_end)
        if j >= len(src) or src[j] != "(":
            raise RewriteError("launch without an argument list")
        args_end = _find_close(src, mask, j)
        args = _fix_null_args(src[j + 1:args_end])
        gaps = _layout(src[callee_end:pos] + src[close_end:j]) + "\\\n" * continued
        sep = ", " if args.strip() else ""
        replacement = (f"::ocu_cpu::launch(::ocu_cpu::cfg({_fix_null_args(cfg, zero=False)}), "
                       f"[=](auto &&...{LAMBDA_ARG}) -> decltype({callee}({LAMBDA_ARG}...)) "
                       f"{{ {callee}({LAMBDA_ARG}...); }}"
                       f"{sep}{args}){gaps}")
        src = src[:callee_from] + replacement + src[args_end + 1:]
        mask = mask[:callee_from] + bytearray(b"\x01") * len(replacement) + mask[args_end + 1:]
        count += 1
    return src, count


def normalize_chevrons(src: str) -> tuple[str, int]:
    """Spaced launch chevrons ("k << < g, b >> > (x)") as "<<<" and ">>>",
    which some compilers require; lengths are kept (blanks move to the end),
    so line and column positions stay the same. Returns (text, changes)."""
    mask = _code_mask(src)
    out = list(src)
    changes = 0
    for m in _OPEN.finditer(src):
        if not (mask[m.start()] and mask[m.end() - 1]):
            continue
        before = src[:_skip_back(src, mask, m.start())]
        if re.search(r"\boperator$", before):
            continue
        try:
            close_start, close_end = _find_chevron_end(src, mask, m.end())
        except RewriteError:
            continue
        for a, b, ch in ((m.start(), m.end(), "<"), (close_start, close_end, ">")):
            if b - a != 3:
                out[a:b] = list(ch * 3 + " " * (b - a - 3))
                changes += 1
    return "".join(out), changes


_EXTERN_SHARED = re.compile(
    r"\bextern\b(?P<pre>[^;{}()\[\]]*?)\b(?:__shared__|__ocu_shared__)\b(?P<post>[^;{}\[\]]*?)"
    r"\b(?P<name>[A-Za-z_]\w*)\s*\[\s*\]\s*;")
_ALIGN = re.compile(r"__align__\s*\(\s*\d+\s*\)")
# alignas(...) with one level of nested parentheses (alignas(sizeof(T))).
_ALIGNAS = r"\balignas\s*\((?:[^()]|\([^()]*\))*\)"


def _drop_align(text: str) -> str:
    """An extern __shared__ array's type without its alignment: __align__(n)
    as written, or alignas(...) as it is after preprocessing."""
    return re.sub(_ALIGNAS, " ", _ALIGN.sub(" ", text))


_BODY_TAIL = re.compile(r"\)\s*(?:(?:const|volatile|noexcept|override|final|mutable)\s*)*$")


def _in_function(src: str, mask, pos: int) -> bool:
    """True when `pos` is inside a function body: some enclosing '{' follows
    a ')' (a parameter list, or if/for/while inside a function)."""
    depth = 0
    i = pos - 1
    while i >= 0:
        if mask[i]:
            c = src[i]
            if c == "}":
                depth += 1
            elif c == "{":
                if depth == 0:
                    j = _skip_back(src, mask, i)
                    if _BODY_TAIL.search(src, max(0, j - 200), j):
                        return True
                else:
                    depth -= 1
        i -= 1
    return False


def _rewrite_extern_shared(src: str) -> tuple[str, int]:
    mask = _code_mask(src)
    out = []
    last = 0
    count = 0
    for m in _EXTERN_SHARED.finditer(src):
        if not all(mask[m.start():m.end()]):
            continue
        type_ = _drop_align(m.group("pre") + " " + m.group("post"))
        type_ = " ".join(type_.split())
        if not type_:
            raise RewriteError(f"extern __shared__ {m.group('name')} has no type")
        out.append(src[last:m.start()])
        if _in_function(src, mask, m.start()):
            # A pointer: casts (reinterpret_cast<U *>(name), (U *)name) work.
            out.append(f"{type_} *const {m.group('name')} = "
                       f"static_cast<{type_} *>(::ocu_cpu::dynamic_shared());"
                       f"{_layout(m.group(0))}")
        else:
            out.append(f"static ::ocu_cpu::dyn_shared<{type_}> {m.group('name')};"
                       f"{_layout(m.group(0))}")
        last = m.end()
        count += 1
    out.append(src[last:])
    return "".join(out), count


# What `__shared__` expands to during preprocessing (ocu_cuda_lang_cpu.h).
_SHARED_MARK = re.compile(r"\b__ocu_shared__\b")
# `__shared__ __align__(16) T a[n]` (alignas after preprocessing): an
# alignment specifier may not follow `static thread_local`, so it moves in
# front of them.
_SHARED_ALIGNED = re.compile(r"\b__ocu_shared__((?:\s*" + _ALIGNAS + r")+)")
# `const alignas(16) T a[n]`: the CUDA compiler accepts an alignment
# specifier after cv-qualifiers and storage classes, C++ only before them.
_LATE_ALIGNAS = re.compile(
    r"\b((?:(?:const|volatile|static|thread_local|constexpr|register)\s+)+)(" + _ALIGNAS + r")")


def rewrite(src: str, preprocessed: bool = False) -> tuple[str, dict]:
    """Rewrite one source text. With `preprocessed` the text is a whole
    preprocessed translation unit, so launches from included files and
    macros are visible; there `__shared__` appears as __ocu_shared__."""
    src, shared = _rewrite_extern_shared(src)
    src, launches = _rewrite_launches(src)
    if preprocessed:
        src = _SHARED_ALIGNED.sub(
            lambda m: " ".join(m.group(1).split()) + " static thread_local", src)
        src = _SHARED_MARK.sub("static thread_local", src)
        src = _LATE_ALIGNAS.sub(lambda m: m.group(2) + " " + m.group(1), src)
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

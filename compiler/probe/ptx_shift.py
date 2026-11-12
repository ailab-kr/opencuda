#!/usr/bin/env python3
"""PTX shift semantics for device code (ocu_cc.py --ptx-shift).

C++ leaves a shift by the operand's width or more undefined (and a shift
by a negative amount). PTX defines it (PTX ISA 8.5, 9.7.7.8 "Logic and
Shift Instructions: shl" and 9.7.7.9 "...: shr"; 9.7.9.8/9.7.9.9 in ISA
9.5; 9.4: the amount operand is always .u32): the shift amount is
unsigned and amounts larger than the register width are clamped to the
width, so `shl` gives 0, `shr.u` gives 0 and `shr.s` fills with the sign
bit. Programs written for NVIDIA GPUs sometimes depend on that. With this
option the integer shifts in device functions get those semantics on every
path. Only the left operand is wrapped; the operator stays:

    a << b    ->  __ocu_shift_lhs(a) << b
    a >> b    ->  __ocu_shift_lhs(a) >> b
    a <<= b   ->  __ocu_shift_ref(a) <<= b
    a >>= b   ->  __ocu_shift_ref(a) >>= b

(runtime/include/ocu_ptx_shift.h). The wrapper's member operators do the
PTX shift. Because the operator is still written as an operator, the
operands keep the built-in order (ISO C++17 [over.match.oper]/2: operator
notation is sequenced as the built-in operator; [expr.shift]/4: the left
operand of << and >> first; [expr.ass]/1: the right operand of <<= and >>=
first). A function call `f(a, b)` would leave that order unspecified.

What is rewritten:

  - the built-in shift operators in the bodies of functions declared
    __global__ or __device__ (also __host__ __device__: one source text,
    so their host calls get the same semantics), including lambdas and
    template instantiations inside them;
  - except shifts by an integer literal below 32, which are defined in C++
    for every promoted operand type; they stay as written (no cost);
  - in the source file and the files it includes with quotes from its
    folder (or below): those are copied into the work folder, rewritten
    there, with a #line marker naming the original. Line numbers are kept.

Macros: a shift whose operands begin and end, and whose operator is, in one
macro body (`#define SHL(a, b) ((a) << (b))`: the parentheses are the
body's; `#define W (v << n)`) is rewritten in that body. Only tokens of the
body change, so the rewrite is right for every expansion; every use of the
macro then gets the PTX semantics, host uses included. Not rewritten, and
reported: a shift whose operand begins or ends at a macro argument or
anywhere other than the operator's place (`#define SH(x) x << k`, an operand
that is itself a macro: `x << AMOUNT`), a macro body used in an #if or #elif
line (the preprocessor cannot call the wrapper), shifts in device code in
files outside the source's folder, and compound shifts of bit-fields (no
reference binds to them). Host functions are never changed.

The device functions and shifts are found in Clang's typed syntax tree
(JSON) of the translation unit, as compiler/metal/cu2msl.py does; the edits
are made in the original text at the tree's source offsets.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER = {"<<": "__ocu_shift_lhs", ">>": "__ocu_shift_lhs",
          "<<=": "__ocu_shift_ref", ">>=": "__ocu_shift_ref"}
HEADER = "ocu_ptx_shift.h"
DEVICE_ATTRS = ("CUDAGlobalAttr", "CUDADeviceAttr")
FUNCTIONS = ("FunctionDecl", "CXXMethodDecl", "CXXConstructorDecl", "CXXDestructorDecl",
             "CXXConversionDecl")
WRAPS = ("ImplicitCastExpr", "ParenExpr", "ConstantExpr")
_QUOTE_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.M)


class ShiftError(Exception):
    pass


# ---- local files and their copies -------------------------------------------------------

def local_sources(src: Path) -> list[Path]:
    """The source and the files it includes with quotes, found next to the
    including file, that lie under the source's folder."""
    root = src.parent.resolve()
    seen, todo, out = set(), [src.resolve()], []
    while todo:
        f = todo.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        try:
            f.relative_to(root)
        except ValueError:
            continue
        out.append(f)
        text = f.read_bytes().decode("latin-1")
        for name in _QUOTE_INCLUDE.findall(text):
            todo.append((f.parent / name).resolve())
    return out


def _marker(orig: Path) -> str:
    return f'#line 1 "{orig}"\n'


def make_copies(src: Path, dest: Path) -> dict[str, tuple[Path, str]]:
    """Copy the local sources under `dest` (same relative layout), with a
    #line marker and launch chevrons joined (same length; Clang rejects
    "<< <"). Returns {real path of copy: (original, copy text)}. Text is
    latin-1 so that one character is one byte, as Clang's offsets count."""
    from cpu_rewrite import RewriteError, normalize_chevrons
    root = src.parent.resolve()
    copies = {}
    for f in local_sources(src):
        text = f.read_bytes().decode("latin-1")
        try:
            text, _ = normalize_chevrons(text)
        except RewriteError:
            pass  # the compiler reports it
        out = dest / f.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = _marker(f) + text
        out.write_bytes(text.encode("latin-1"))
        copies[os.path.realpath(out)] = (f, text)
    return copies


# ---- the syntax tree -------------------------------------------------------------------

def _annotate_files(root, cur: str | None = None) -> None:
    """Clang's JSON names a location's file only when it differs from the
    location written before; fill it in everywhere (document order). `cur`
    is the file in effect before `root`."""
    stack = [iter((root,))]
    while stack:
        try:
            o = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(o, dict):
            if "offset" in o and ("col" in o or "tokLen" in o):
                if o.get("file") is None:
                    o["file"] = cur
                else:
                    cur = o["file"]
                continue
            stack.append(iter(o.values()))
        elif isinstance(o, list):
            stack.append(iter(o))


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        inner = n.get("inner")
        if inner:
            stack.extend(c for c in reversed(inner) if isinstance(c, dict))


def _kids(n) -> list:
    return [c for c in n.get("inner", []) or [] if isinstance(c, dict)]


def _is_device(fn) -> bool:
    return any(c.get("kind") in DEVICE_ATTRS and not c.get("implicit") for c in _kids(fn))


def _strip(e):
    while e.get("kind") in WRAPS and _kids(e):
        e = _kids(e)[0]
    return e


def _literal(e) -> int | None:
    e = _strip(e)
    if e.get("kind") in ("IntegerLiteral", "CharacterLiteral"):
        try:
            return int(e.get("value"))
        except (TypeError, ValueError):
            return None
    return None


def _point(loc: dict, end: bool):
    """(file, offset, in_macro, macro_arg, expansion offset) of a location;
    `end` adds the token length (an end location names the last token)."""
    if not loc:
        return None
    if "spellingLoc" in loc:
        sp, ex = loc["spellingLoc"], loc.get("expansionLoc", {})
        if "offset" not in sp:
            return None
        return (sp.get("file"), sp["offset"] + (sp.get("tokLen", 0) if end else 0), True,
                bool(ex.get("isMacroArgExpansion")), ex.get("offset"))
    if "offset" not in loc:
        return None
    return (loc.get("file"), loc["offset"] + (loc.get("tokLen", 0) if end else 0), False,
            False, None)


def _skip_blank(text: str, i: int, j: int) -> str:
    """Text between i and j without blanks, comments and line continuations."""
    s = text[i:j]
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    return re.sub(r"[ \t\r\n\\]", "", s)


def _bitfields(tu) -> set[str]:
    return {n["id"] for n in _walk(tu) if n.get("kind") == "FieldDecl" and n.get("isBitfield")}


def _is_bitfield(e, fields: set[str]) -> bool:
    e = _strip(e)
    return e.get("kind") == "MemberExpr" and e.get("referencedMemberDecl") in fields


class Shift:
    """One rewrite: the left operand [begin, lhs_end) is wrapped in helper()."""
    __slots__ = ("file", "begin", "lhs_end", "helper")

    def __init__(self, file, begin, lhs_end, helper):
        self.file, self.begin, self.lhs_end, self.helper = file, begin, lhs_end, helper


_DEFINE = re.compile(r"[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)")


def _macro_at(text: str, offset: int) -> str | None:
    """Name of the macro whose #define (with its continuation lines) holds
    `offset`, or None."""
    start = text.rfind("\n", 0, offset) + 1
    while start > 1 and text[:start - 1].rstrip("\r").endswith("\\"):  # a continuation
        start = text.rfind("\n", 0, start - 1) + 1
    m = _DEFINE.match(text, start)
    return m.group(1) if m else None


def _in_conditional(name: str, texts) -> bool:
    """True when `name` appears on an #if or #elif line of one of `texts`."""
    pat = re.compile(r"^[ \t]*#[ \t]*(?:el)?if\b[^\n]*\b" + re.escape(name) + r"\b", re.M)
    return any(pat.search(t) for t in texts)


def find_shifts(tu, copies: dict, ignored, fields: set[str] | None = None
                ) -> tuple[list[Shift], list[str]]:
    """Shifts to rewrite in the copies, and messages for those left alone.
    `tu` is a syntax tree whose locations name their files (see
    _annotate_files); `fields` the ids of bit-field declarations (default:
    those in `tu`). `ignored(file)` is true for files whose device code is
    ours or the system's (never reported)."""
    if fields is None:
        fields = _bitfields(tu)
    found: dict[tuple, Shift] = {}
    notes: set[tuple[str, int, str]] = set()
    texts: dict[str, str] = {}

    def note(f: str, offset: int, what: str):
        real = os.path.realpath(f)
        if real in copies:  # the copy starts with a #line marker
            orig, text = copies[real]
            notes.add((str(orig), text.count("\n", 0, offset), what))
            return
        if real not in texts:
            try:
                texts[real] = Path(real).read_bytes().decode("latin-1")
            except OSError:
                texts[real] = ""
        notes.add((f, texts[real].count("\n", 0, offset) + 1, what))

    for fn in _walk(tu):
        if fn.get("kind") not in FUNCTIONS or not _is_device(fn):
            continue
        for e in _walk(fn):
            k = e.get("kind")
            if k not in ("BinaryOperator", "CompoundAssignOperator") or \
                    e.get("opcode") not in HELPER:
                continue
            ops = _kids(e)
            if len(ops) != 2:
                continue
            value = _literal(ops[1])
            if value is not None and 0 <= value < 32:
                continue  # defined in C++ for every promoted type
            r0, r1 = ops[0].get("range", {}), ops[1].get("range", {})
            pts = [_point(r0.get("begin"), False), _point(r0.get("end"), True),
                   _point(r1.get("begin"), False), _point(r1.get("end"), True)]
            if any(p is None or not p[0] for p in pts):
                continue
            mine = [p for p in pts if not ignored(p[0])]
            if not mine:
                continue  # device code of our headers or the system's
            where, at = mine[0][0], mine[0][1]
            files = {os.path.realpath(p[0]) for p in pts}
            if len(files) != 1:
                note(where, at, "split between files")
                continue
            real = next(iter(files))
            if real not in copies:
                note(where, at, "file outside the source's folder")
                continue
            if any(p[2] for p in pts):
                # Only a whole expression spelled in one macro body.
                if not (all(p[2] for p in pts) and not any(p[3] for p in pts)
                        and len({p[4] for p in pts}) == 1):
                    note(where, at, "partly in a macro")
                    continue
                name = _macro_at(copies[real][1], pts[0][1])
                if name is None:
                    note(where, at, "partly in a macro")
                    continue
                if _in_conditional(name, (t for _, t in copies.values())):
                    note(where, at, f"macro {name} is also used in #if")
                    continue
            if k == "CompoundAssignOperator" and _is_bitfield(ops[0], fields):
                note(where, at, "compound shift of a bit-field")
                continue
            _, text = copies[real]
            begin, lhs_end, rhs_begin, end = (p[1] for p in pts)
            op = e["opcode"]
            ordered = begin <= lhs_end <= rhs_begin <= end
            op_at = text.find(op, lhs_end, rhs_begin) if ordered else -1
            if op_at < 0 or _skip_blank(text, lhs_end, op_at) or \
                    _skip_blank(text, op_at + len(op), rhs_begin):
                note(where, at, "operator not found between the operands")
                continue
            key = (real, begin, lhs_end)
            found.setdefault(key, Shift(real, begin, lhs_end, HELPER[op]))
    messages = [f"{f}:{line}: shift in device code left as C++ ({what})"
                for f, line, what in sorted(notes)]
    return list(found.values()), messages


def apply_edits(text: str, shifts: list[Shift]) -> str:
    """`a OP b` -> `HELPER(a) OP b` for each shift (they may nest: the left
    operand of one may hold another)."""
    edits = []  # (position, rank, tie, end, text)
    for s in shifts:
        edits.append((s.lhs_end, 0, -s.begin, s.lhs_end, ")"))         # inner closes first
        edits.append((s.begin, 1, -s.lhs_end, s.begin, s.helper + "("))  # outer opens first
    edits.sort()
    out, cur = [], 0
    for pos, _, _, end, new in edits:
        if pos < cur:
            raise ShiftError("overlapping shift edits")
        out.append(text[cur:pos])
        out.append(new)
        cur = end
    out.append(text[cur:])
    return "".join(out)


# ---- driver ------------------------------------------------------------------------------

def default_ignored(extra_dirs: list[str]):
    dirs = [os.path.realpath(d) + os.sep for d in extra_dirs] + \
        ["/usr/", "/opt/", "/Library/", "/Applications/", "/System/", "/nix/"]

    def ignored(f: str) -> bool:
        if not f or f.startswith("<"):
            return True
        p = os.path.realpath(f)
        return any(p.startswith(d) for d in dirs)
    return ignored


_DECL_START = b"\n    {\n"          # a top-level declaration (indent 4)
_FILE = re.compile(rb'"file": "((?:[^"\\]|\\.)*)"')
_BITFIELD = re.compile(rb'"isBitfield": true')
_DEVICE_ATTR = (b'"CUDAGlobalAttr"', b'"CUDADeviceAttr"')


def _json_str(raw: bytes) -> str:
    if b"\\" in raw:
        return json.loads(b'"' + raw + b'"')
    return raw.decode("utf-8", "surrogateescape")


def device_decls(stream, ignored) -> tuple[list, set[str]]:
    """Read Clang's JSON syntax tree from `stream` one top-level declaration
    at a time and parse only those that hold device functions of files not
    `ignored` (a whole tree is hundreds of megabytes, nearly all of it the
    runtime and standard headers). Returns those declarations, with every
    location's file filled in, and the ids of all bit-field declarations."""
    decls, fields = [], set()
    cur = None

    def take(chunk: bytes):
        nonlocal cur
        for m in _BITFIELD.finditer(chunk):
            i = chunk.rfind(b'"id": "', 0, m.start())
            if i >= 0:
                fields.add(chunk[i + 7:chunk.index(b'"', i + 7)].decode("ascii"))
        names = _FILE.findall(chunk)
        before = cur
        if names:
            cur = _json_str(names[-1])
        if not any(a in chunk for a in _DEVICE_ATTR):
            return
        files = {before} | {_json_str(n) for n in names}
        if not any(f and not ignored(f) for f in files):
            return
        # The declaration ends at its closing brace (indent 4); after it
        # come a comma, or for the last one the end of the tree.
        end = chunk.rfind(b"\n    }")
        if end < 0:
            raise ValueError("no end of a declaration")
        node = json.loads(chunk[:end + 6].decode("utf-8", "surrogateescape"))
        _annotate_files(node, before)
        decls.append(node)

    buf = bytearray()
    start = None   # where the current declaration begins in buf
    scan = 0       # where to look for the next start
    while True:
        block = stream.read(1 << 22)
        buf += block
        while True:
            i = buf.find(_DECL_START, max(scan, 0))
            if i < 0:
                scan = max(0, len(buf) - len(_DECL_START))
                break
            if start is not None:
                take(bytes(buf[start:i + 1]))
            del buf[:i + 1]  # buf now begins with the new declaration
            start, scan = 0, 1
        if not block:
            if start is not None and buf.strip():
                take(bytes(buf[start:]))
            break
    return decls, fields


def prepare(src: Path, dest: Path, ast_cmd: list[str], env: dict, ignored_dirs: list[str],
            warn) -> tuple[Path, int]:
    """Copy and rewrite `src` and its local includes under `dest`. `ast_cmd`
    is a Clang CUDA command line without the source file; it is run with
    -fsyntax-only on the copy. Returns (copy of src, shifts rewritten)."""
    src = src.resolve()
    copies = make_copies(src, dest)
    copy_src = dest / src.name
    cmd = [*ast_cmd, "-iquote", str(src.parent), "-w", "-fsyntax-only",
           "-Xclang", "-ast-dump=json", str(copy_src)]
    ignored = default_ignored(ignored_dirs)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) as proc:
        err = []
        reader = threading.Thread(target=lambda: err.append(proc.stderr.read()))
        reader.start()
        try:
            decls, fields = device_decls(proc.stdout, ignored)
        except ValueError as error:  # unreadable JSON: report Clang's message first
            decls, fields = None, error
            proc.stdout.read()
        proc.wait()
        reader.join()
    if proc.returncode != 0:
        msg = err[0].decode("utf-8", "replace")[-3000:] if err else ""
        raise ShiftError(f"--ptx-shift: Clang could not read the source:\n{msg}")
    if decls is None:
        raise ShiftError(f"--ptx-shift: unexpected syntax tree format ({fields})")
    shifts, messages = find_shifts({"inner": decls}, copies, ignored, fields)
    for m in messages:
        warn(m)
    by_file: dict[str, list[Shift]] = {}
    for sh in shifts:
        by_file.setdefault(sh.file, []).append(sh)
    for f, group in by_file.items():
        _, text = copies[f]
        Path(f).write_bytes(apply_edits(text, group).encode("latin-1"))
    return copy_src, len(shifts)

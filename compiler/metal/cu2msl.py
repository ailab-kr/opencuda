#!/usr/bin/env python3
"""cu2msl: CUDA kernels to Metal Shading Language, for the Metal backend.

Reads the typed syntax tree Clang produces for the device pass of the whole
translation unit (open-source compiler output) and rewrites only what MSL
needs differently, in the original source text (backends/metal/README.md,
section 4):

  - kernels (and every instantiation of a kernel template) get MSL
    signatures: arguments come from buffer 0 at the offsets in buffer 1,
    built-in indices from MSL kernel attributes;
  - device functions get the index context as a first parameter. One copy
    is emitted per combination of address spaces its pointer and reference
    parameters are called with (an overload set), and one per template
    instantiation; a pointer-returning function returns into the address
    space its return statements give;
  - local pointer and reference declarations and pointer casts get an
    address space (device, threadgroup, thread or constant), found from
    what they point at;
  - __shared__ variables become threadgroup variables; extern __shared__
    arrays point at the launch's dynamic threadgroup memory;
  - __device__ and __constant__ variables live in one per-module buffer
    the backend allocates (buffer 2); the image lists their names, offsets,
    sizes and initial values, so module_get_global can find them;
  - structs, typedefs, enums and constant variables the device code uses
    are copied (pointer fields get the device address space), with a
    static_assert that each argument struct has the size the host packs.

Kernels and device functions may be in included files (any header outside
the system and runtime directories). Builtins (threadIdx, ...), barriers,
atomics, math and warp functions come from the MSL prelude (msl_prelude.h)
under the CUDA names. Constructs not handled stop the translation with a
message (never silently wrong code).

    cu2msl.py <file.cu> -o <image> [-I dir] [-D name] [--clang clang++]
              [--keep-going] [--report]

--keep-going leaves out the kernels that cannot be translated (each is
reported on stderr) instead of failing; --report prints one line per kernel
("ok"/"skip", source file, name) on stdout.

Image format (read by backends/metal/mt_backend.m): "OCUMSL1", then lines
  param <kernel> <size>...      the kernel's argument sizes
  global <name> <offset> <size> a __device__/__constant__ variable
  init <offset> <hex bytes>     its initial value (zero otherwise)
then "---" and the MSL source.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


class TranslateError(Exception):
    pass


# ---- running clang --------------------------------------------------------------------

def cuda_path() -> Path:
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / "ocu-cuda-path"
    (cache / "include").mkdir(parents=True, exist_ok=True)
    for sub in ("bin", "lib64"):
        (cache / sub).mkdir(exist_ok=True)
    marker = cache / "include" / "cuda.h"
    if not marker.exists():
        marker.write_text("#define CUDA_VERSION 12030\n")
    return cache


def clang_env() -> dict:
    env = dict(os.environ)
    if sys.platform == "darwin" and not env.get("SDKROOT"):
        sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True)
        if sdk.returncode == 0 and sdk.stdout.strip():
            env["SDKROOT"] = sdk.stdout.strip()  # compilers called by full path need it
    return env


def device_cmd(clang: str, flags: list[str]) -> list[str]:
    rt = REPO / "runtime" / "include"
    return [clang, "-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib",
            f"--cuda-path={cuda_path()}", "--cuda-gpu-arch=sm_80", "-std=c++17",
            f"-I{rt}", "-include", str(rt / "cuda_runtime.h"), "-DOCU_METAL_PATH",
            "-Wno-unknown-cuda-version", *flags]


def run(cmd: list[str], what: str) -> str:
    out = subprocess.run(cmd, capture_output=True, encoding="latin-1", env=clang_env(),
                         )
    if out.returncode != 0:
        raise TranslateError(f"clang failed ({what}):\n{out.stderr[-2000:]}")
    return out.stdout


def annotate_files(root) -> None:
    """Clang's JSON writes a location's file only when it changes from the
    location printed before; fill it in everywhere (document order)."""
    cur = None
    stack = [iter((root,))]
    while stack:
        try:
            o = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(o, dict):
            if "offset" in o and ("col" in o or "tokLen" in o):
                f = o.get("file")
                if f is None:
                    o["file"] = cur
                else:
                    cur = f
                continue
            stack.append(iter(o.values()))
        elif isinstance(o, list):
            stack.append(iter(o))


# ---- syntax tree helpers ----------------------------------------------------------------

def walk(node, prune=()):
    """Pre-order, iterative (expressions nest deeply); does not enter the
    node kinds in `prune`."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        if n.get("kind") in prune:
            continue
        inner = n.get("inner")
        if inner:
            stack.extend(c for c in reversed(inner) if isinstance(c, dict))


def kids(n) -> list:
    return [c for c in n.get("inner", []) or [] if isinstance(c, dict)]


def bare(loc: dict) -> dict:
    return loc.get("expansionLoc", loc)


def qtype(n) -> str:
    t = n.get("type") or {}
    return t.get("qualType", "")


def dtype(n) -> str:
    t = n.get("type") or {}
    return t.get("desugaredQualType", t.get("qualType", ""))


def strip_qual(t: str) -> str:
    t = re.sub(r"\b(const|volatile|struct|class|union|enum|__restrict|restrict)\b", "", t)
    return " ".join(t.split())


def is_pointer(t: str) -> bool:
    t = re.sub(r"\b(const|volatile|__restrict|restrict)\b", "", t).rstrip()
    return t.endswith("*")


def is_ref(t: str) -> bool:
    return t.rstrip().endswith("&")


def pointee(t: str) -> str:
    t = re.sub(r"\b(const|volatile|__restrict|restrict)\b\s*$", "", t.rstrip()).rstrip()
    while True:
        t2 = re.sub(r"\b(const|volatile|__restrict|restrict)\s*$", "", t).rstrip()
        if t2 == t:
            break
        t = t2
    return t[:-1] if t.endswith("*") else t


def no_restrict(t: str) -> str:
    return " ".join(re.sub(r"\b__restrict\b|\brestrict\b", "", t).split())


def has_double(t: str) -> bool:
    return re.search(r"\bdouble\b", t) is not None


VEC3 = re.compile(r"\b(u?char|u?short|u?int|u?long|longlong|ulonglong|float|half|double)3\b")

SCALAR_SIZES = {
    "bool": 1, "char": 1, "signed char": 1, "unsigned char": 1, "short": 2,
    "unsigned short": 2, "int": 4, "unsigned int": 4, "unsigned": 4, "float": 4,
    "long": 8, "unsigned long": 8, "long long": 8, "unsigned long long": 8,
    "size_t": 8, "half": 2, "__half": 2, "int8_t": 1, "uint8_t": 1, "int16_t": 2,
    "uint16_t": 2, "int32_t": 4, "uint32_t": 4, "int64_t": 8, "uint64_t": 8,
    "std::int8_t": 1, "std::uint8_t": 1, "std::int16_t": 2, "std::uint16_t": 2,
    "std::int32_t": 4, "std::uint32_t": 4, "std::int64_t": 8, "std::uint64_t": 8,
    "short int": 2, "long int": 8, "unsigned short int": 2, "long unsigned int": 8,
    "unsigned long int": 8, "long long int": 8, "unsigned long long int": 8,
}
PACK = {"bool": "?", "char": "b", "signed char": "b", "unsigned char": "B", "short": "h",
        "unsigned short": "H", "int": "i", "unsigned int": "I", "unsigned": "I",
        "float": "f", "long": "q", "unsigned long": "Q", "long long": "q",
        "unsigned long long": "Q"}

CASTS = ("CStyleCastExpr", "CXXStaticCastExpr", "CXXReinterpretCastExpr", "CXXConstCastExpr",
         "CXXFunctionalCastExpr")
WRAPS = ("ParenExpr", "ImplicitCastExpr", "ExprWithCleanups", "ConstantExpr",
         "CXXBindTemporaryExpr", "FullExpr") + CASTS

NO_METAL = {"printf": "printf", "malloc": "device-side malloc", "free": "device-side free",
            "__assert_fail": "assert", "__assertfail": "assert", "__trap": "__trap",
            "clock": "clock", "clock64": "clock64"}


def merge(a: str | None, b: str | None, what: str) -> str | None:
    if a is None:
        return b
    if b is None or a == b:
        return a
    raise TranslateError(f"{what} points into {a} and {b} memory")


# ---- the translation ----------------------------------------------------------------------

class Instance:
    def __init__(self, fn, spaces, name, kernel, tmpl):
        self.fn = fn
        self.spaces = spaces      # per parameter: address space or None
        self.name = name
        self.kernel = kernel
        self.tmpl = tmpl          # typedef lines injected at the top of the body
        self.ret_space = None
        self.edits: list[tuple[int, int, str]] = []
        self.text = ""
        self.head = ""
        self.param_line = None
        self.busy = True


class Translator:
    def __init__(self, src: Path, flags: list[str], clang: str):
        self.src = src
        self.flags = flags
        self.clang = clang
        self.texts: dict[str, str] = {}
        self.defs: dict[str, dict] = {}       # FunctionDecl id (any redeclaration) -> definition
        self.tmpl_of: dict[str, dict] = {}    # specialization id -> FunctionTemplateDecl
        self.spec_names: dict[str, str] = {}  # specialization id -> emitted name
        self.kernels: list[dict] = []
        self.globals: dict[str, dict] = {}    # VarDecl id -> info
        self.consts: dict[str, dict] = {}     # VarDecl id -> host const variable
        self.dyn_shared: dict[str, str] = {}  # file-scope extern __shared__ array -> element type
        self.items: list[dict] = []           # type / constant declarations, TU order
        self.insts: dict[tuple, Instance] = {}
        self.order: list[Instance] = []
        self.layout: dict[str, tuple[int, int]] = {}
        self.asserts: list[str] = []
        sysdirs = ["/usr/", "/Library/", "/Applications/", "/System/", "/opt/", "/nix/",
                   str(REPO / "runtime" / "include") + os.sep, str(cuda_path()) + os.sep]
        sdk = clang_env().get("SDKROOT")
        if sdk:
            sysdirs.append(sdk.rstrip("/") + "/")
        self.sysdirs = [os.path.realpath(d) + (os.sep if d.endswith(os.sep) else "") for d in sysdirs]
        self.srcdir = os.path.realpath(str(src.parent)) + os.sep
        self._user: dict[str, bool] = {}

    # -- files ----------------------------------------------------------------------------

    def user_file(self, f: str | None) -> bool:
        if not f or f.startswith("<"):
            return False
        if f not in self._user:
            p = os.path.realpath(f)
            self._user[f] = p.startswith(self.srcdir) or \
                not any(p.startswith(d) or p + os.sep == d for d in self.sysdirs)
        return self._user[f]

    def text(self, f: str) -> str:
        p = os.path.realpath(f)
        if p not in self.texts:
            # Clang offsets count bytes: one character per byte, line ends kept
            self.texts[p] = Path(p).read_bytes().decode("latin-1")
        return self.texts[p]

    @staticmethod
    def same(a: str | None, b: str | None) -> bool:
        return a is not None and b is not None and os.path.realpath(a) == os.path.realpath(b)

    def span(self, n) -> tuple[str, int, int] | None:
        """File and [begin, end) of a node, through macro expansions."""
        r = n.get("range")
        if not r or "begin" not in r:
            return None
        b, e = bare(r["begin"]), bare(r["end"])
        if "offset" not in b or "offset" not in e:
            return None
        return b.get("file"), b["offset"], e["offset"] + e.get("tokLen", 0)

    # -- reading the translation unit -------------------------------------------------------

    def chevrons(self):
        """Launches written `k << < g, b >> > (...)` (accepted by NVIDIA's
        compiler, not by Clang): give Clang copies of the user files with
        the chevrons joined, same length, through a file-system overlay, so
        that source offsets stay those of the real files."""
        dirs = [d[2:] for d in self.flags if d.startswith("-I")]
        seen, todo, fixed = set(), [str(self.src)], []
        while todo:
            f = os.path.realpath(todo.pop())
            if f in seen or not os.path.isfile(f):
                continue
            seen.add(f)
            t = self.text(f)
            for inc in re.findall(r'^\s*#\s*include\s*"([^"]+)"', t, re.M):
                for d in [os.path.dirname(f)] + dirs:
                    cand = os.path.join(d, inc)
                    if os.path.isfile(cand):
                        todo.append(cand)
                        break
            if re.search(r"<<\s+<|>>\s+>", t):
                fixed.append((f, re.sub(r"<<(\s+)<", lambda m: "<<<" + m.group(1),
                                        re.sub(r">>(\s+)>", lambda m: m.group(1) + ">>>", t))))
        if not fixed:
            return
        self._tmp = tempfile.TemporaryDirectory()
        roots = []
        for i, (f, t) in enumerate(fixed):
            copy = Path(self._tmp.name) / f"{i}_{os.path.basename(f)}"
            copy.write_bytes(t.encode("latin-1"))
            roots.append({"name": os.path.dirname(f), "type": "directory", "contents": [
                {"name": os.path.basename(f), "type": "file", "external-contents": str(copy)}]})
        vfs = Path(self._tmp.name) / "overlay.yaml"
        vfs.write_text(json.dumps({"version": 0, "use-external-names": False, "roots": roots}))
        self.flags = [f"-ivfsoverlay{vfs}"] + self.flags

    def load(self):
        self.chevrons()
        cmd = device_cmd(self.clang, self.flags) + ["-fsyntax-only", "-Xclang", "-ast-dump=json",
                                                    str(self.src)]
        tu = json.loads(run(cmd, "syntax tree"))
        annotate_files(tu)
        self.scan(kids(tu))

    def scan(self, decls):
        for d in decls:
            k = d.get("kind")
            if d.get("isImplicit"):
                continue
            sp = self.span(d)
            f = sp[0] if sp else bare(d.get("loc", {})).get("file")
            if k == "LinkageSpecDecl":
                self.scan(kids(d))
                continue
            if not self.user_file(f):
                continue
            attrs = {c.get("kind") for c in kids(d)}
            if k == "NamespaceDecl":
                for n in walk(d):
                    if n.get("kind") in ("FunctionDecl", "VarDecl") and \
                            {c.get("kind") for c in kids(n)} & {"CUDAGlobalAttr", "CUDADeviceAttr", "CUDAConstantAttr"}:
                        raise TranslateError(f"namespace '{d.get('name', '')}': device code inside namespaces is not handled yet")
                continue
            if k in ("FunctionDecl", "CXXMethodDecl"):
                if not attrs & {"CUDAGlobalAttr", "CUDADeviceAttr"}:
                    continue
                if k == "CXXMethodDecl":
                    raise TranslateError(f"'{d.get('name')}': member functions are not handled yet")
                self.add_function(d, None)
            elif k == "FunctionTemplateDecl":
                fns = [c for c in kids(d) if c.get("kind") == "FunctionDecl"]
                if not fns or not {c.get("kind") for c in kids(fns[0])} & {"CUDAGlobalAttr", "CUDADeviceAttr"}:
                    continue
                for s in fns[1:]:
                    self.add_function(s, d)
            elif k == "VarDecl":
                written = {c.get("kind") for c in kids(d) if not c.get("implicit")}
                if written & {"CUDAConstantAttr", "CUDADeviceAttr"}:
                    self.globals[d["id"]] = {"node": d}
                elif "CUDASharedAttr" in attrs and d.get("storageClass") == "extern" and \
                        re.fullmatch(r".*\[\s*\]", qtype(d)):
                    self.dyn_shared[d["id"]] = qtype(d)[:qtype(d).rindex("[")].strip()
                elif "CUDASharedAttr" in attrs:
                    self.items.append({"node": d, "names": {d["name"]}, "error":
                                       f"'{d['name']}': __shared__ variables at file scope are not handled yet"})
                elif d.get("constexpr") or re.match(r"\s*const\b", qtype(d)) or \
                        re.search(r"\bconst\b", qtype(d).split("[")[0]):
                    if "inner" in d and d.get("init"):
                        self.consts[d["id"]] = d
                        self.items.append({"node": d, "names": {d["name"]}, "const": True})
            elif k in ("CXXRecordDecl", "RecordDecl", "TypedefDecl", "TypeAliasDecl", "EnumDecl",
                       "ClassTemplateDecl"):
                names = {d["name"]} if d.get("name") else set()
                if k == "EnumDecl":
                    names |= {c["name"] for c in kids(d) if c.get("kind") == "EnumConstantDecl"}
                if k in ("CXXRecordDecl", "RecordDecl") and not d.get("completeDefinition"):
                    continue
                item = {"node": d, "names": names}
                if k == "ClassTemplateDecl":
                    item["error"] = f"'{d.get('name')}': class templates are not handled yet"
                self.items.append(item)

    def add_function(self, d, tmpl):
        if not any(c.get("kind") == "CompoundStmt" for c in kids(d)):
            return  # a declaration; the definition comes with its body
        self.defs[d["id"]] = d
        if d.get("previousDecl"):
            self.defs[d["previousDecl"]] = d
        targs = [c for c in kids(d) if c.get("kind") == "TemplateArgument"]
        if tmpl is not None or targs:
            if tmpl is not None:
                self.tmpl_of[d["id"]] = tmpl
            self.spec_names[d["id"]] = f"{d['name']}_ocu{len(self.spec_names)}"
            if d["name"].startswith("operator"):
                raise TranslateError(f"'{d['name']}': operator templates are not handled yet")
        if any(c.get("kind") == "CUDAGlobalAttr" for c in kids(d)):
            self.kernels.append(d)

    def macros(self) -> list[str]:
        """#define/#undef lines from user files and the command line, in order,
        as the device pass saw them (inactive #if branches are not there)."""
        cmd = device_cmd(self.clang, self.flags) + ["-E", "-dD", str(self.src)]
        out = run(cmd, "macros")
        lines, cur, keep = [], None, False
        builtin_keep = {"__CUDA_ARCH__", "__CUDACC__", "__CUDA__", "__NVCC__", "CUDA_VERSION",
                        "CUDART_VERSION", "OCU_METAL_PATH"}
        skip = {"__global__", "__device__", "__host__", "__shared__", "__constant__",
                "__forceinline__", "__noinline__", "__restrict__", "__launch_bounds__",
                "__managed__", "__align__"}
        builtin: dict[str, str] = {}
        for line in out.splitlines():
            m = re.match(r'#\s+\d+\s+"([^"]*)"', line)
            if m:
                cur = m.group(1)
                keep = self.user_file(cur)
                continue
            m = re.match(r"#\s*(define|undef)\s+([A-Za-z_]\w*)", line)
            if not m or m.group(2) in skip:
                continue
            name = m.group(2)
            if name in builtin_keep and not keep:
                builtin[name] = line  # the value in force at the end
            elif keep or (cur == "<command line>" and not name.startswith("__")):
                lines.append(line)
        return list(builtin.values()) + lines

    def probe_layouts(self, types: list[str]):
        """sizeof/alignof of the given types, as the CUDA compiler lays them out."""
        types = [t for t in dict.fromkeys(types) if t not in self.layout]
        if not types:
            return
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp) / "ocu_layout.cu"
            body = [f'#include "{self.src}"']
            for i, t in enumerate(types):
                body.append(f"extern char (*ocu_layout_{i})[sizeof({t})][alignof({t})];")
            w.write_text("\n".join(body) + "\n")
            out = run(device_cmd(self.clang, self.flags) +
                      ["-fsyntax-only", "-Xclang", "-ast-dump=json", "-Xclang",
                       "-ast-dump-filter=ocu_layout_", str(w)], "layout")
        for m in re.finditer(r'"name":\s*"ocu_layout_(\d+)",\s*(?:"mangledName":\s*"[^"]*",\s*)?'
                             r'"type":\s*\{\s*"qualType":\s*"char \(\*\)\[(\d+)\]\[(\d+)\]"', out):
            self.layout[types[int(m.group(1))]] = (int(m.group(2)), int(m.group(3)))
        for t in types:
            if t not in self.layout:
                raise TranslateError(f"could not find the layout of type '{t}'")

    # -- address spaces -----------------------------------------------------------------------

    def ptr_space(self, e, st) -> str | None:
        """Address space a pointer-valued expression points into (None: null)."""
        k = e.get("kind")
        ch = kids(e)
        if k in WRAPS:
            inner = ch[-1] if ch else None
            ck = e.get("castKind")
            if ck == "ArrayToPointerDecay":
                return self.lval_space(inner, st)
            if ck == "NullToPointer":
                return None
            if ck == "IntegralToPointer":
                return "device"
            if ck == "FunctionToPointerDecay":
                raise TranslateError(f"'{st['fname']}': function pointers are not available on Metal")
            if inner is None:
                return None
            if not (is_pointer(dtype(inner)) or "[" in dtype(inner)):
                return "device" if is_pointer(dtype(e)) and ck not in ("NoOp", None) else None
            return self.ptr_space(inner, st)
        if k == "UnaryOperator":
            op = e.get("opcode")
            if op == "&":
                return self.lval_space(ch[0], st)
            if op in ("++", "--"):
                return self.ptr_space(ch[0], st)
            raise TranslateError(f"'{st['fname']}': pointers to pointers are not handled yet")
        if k == "BinaryOperator":
            op = e.get("opcode")
            if op in ("+", "-"):
                for c in ch:
                    if is_pointer(dtype(c)) or "[" in dtype(c):
                        return self.ptr_space(c, st)
            if op == ",":
                return self.ptr_space(ch[1], st)
            if op == "=":
                return self.ptr_space(ch[1], st)
        if k == "CompoundAssignOperator":
            return self.ptr_space(ch[0], st)
        if k in ("ConditionalOperator", "BinaryConditionalOperator"):
            return merge(self.ptr_space(ch[-2], st), self.ptr_space(ch[-1], st),
                         f"'{st['fname']}': a conditional pointer")
        if k == "DeclRefExpr":
            ref = e.get("referencedDecl", {})
            rid = ref.get("id")
            if rid in st["vars"]:
                isptr, sp = st["vars"][rid]
                if isptr:
                    return sp if sp is not None else st["pending"].setdefault(rid, None)
                return sp
            if rid in self.globals:
                return "device"
            if rid in self.consts:
                return "constant"
            if rid in self.dyn_shared:
                return "threadgroup"
            raise TranslateError(f"'{st['fname']}': cannot tell where '{ref.get('name')}' points")
        if k in ("ArraySubscriptExpr", "MemberExpr"):
            if k == "MemberExpr" and is_pointer(dtype(e)):
                return "device"  # pointer fields of structs are device pointers
            return self.lval_space(e, st)
        if k in ("CallExpr", "CXXOperatorCallExpr"):
            inst = self.call(e, st)
            if inst is None:
                raise TranslateError(f"'{st['fname']}': pointer returned by a function this translation does not know")
            return inst.ret_space
        if k in ("IntegerLiteral", "CXXNullPtrLiteralExpr", "GNUNullExpr", "ImplicitValueInitExpr"):
            return None
        if k == "StringLiteral":
            raise TranslateError(f"'{st['fname']}': string literals are not handled yet")
        raise TranslateError(f"'{st['fname']}': cannot tell the address space of a {k}")

    def lval_space(self, e, st) -> str:
        """Address space an lvalue lives in."""
        k = e.get("kind")
        ch = kids(e)
        if k in ("ParenExpr", "ImplicitCastExpr", "ConstantExpr", "CXXStaticCastExpr", "CStyleCastExpr") \
                and ch and e.get("valueCategory") != "prvalue":
            return self.lval_space(ch[-1], st)
        if k == "DeclRefExpr":
            ref = e.get("referencedDecl", {})
            rid = ref.get("id")
            if rid in st["vars"]:
                isptr, sp = st["vars"][rid]
                if isptr and not is_ref(ref.get("type", {}).get("qualType", "")):
                    return "thread"
                return sp or "device"
            if rid in self.globals:
                return "device"
            if rid in self.consts:
                return "constant"
            if rid in self.dyn_shared:
                return "threadgroup"
            raise TranslateError(f"'{st['fname']}': cannot tell where '{ref.get('name')}' lives")
        if k == "ArraySubscriptExpr":
            for c in ch:
                if is_pointer(dtype(c)) or "[" in dtype(c):
                    return self.ptr_space(c, st) or "device"
        if k == "UnaryOperator":
            if e.get("opcode") == "*":
                return self.ptr_space(ch[0], st) or "device"
            if e.get("opcode") in ("++", "--"):
                return self.lval_space(ch[0], st)
        if k == "MemberExpr":
            if e.get("isArrow"):
                return self.ptr_space(ch[0], st) or "device"
            return self.lval_space(ch[0], st)
        if k in ("MaterializeTemporaryExpr", "CompoundLiteralExpr", "CXXTemporaryObjectExpr",
                 "CXXConstructExpr", "InitListExpr") or e.get("valueCategory") == "prvalue":
            return "thread"
        if k in ("BinaryOperator", "CompoundAssignOperator"):
            return self.lval_space(ch[0], st)
        if k in ("ConditionalOperator",):
            return merge(self.lval_space(ch[1], st), self.lval_space(ch[2], st),
                         f"'{st['fname']}': a conditional reference") or "thread"
        raise TranslateError(f"'{st['fname']}': cannot tell the address space of a {k}")

    # -- functions ------------------------------------------------------------------------------

    def call(self, e, st) -> Instance | None:
        """The instance a call to a user device function needs (memoised per call)."""
        if e["id"] in st["calls"]:
            return st["calls"][e["id"]]
        ch = kids(e)
        callee = ch[0] if ch else None
        while callee is not None and callee.get("kind") in ("ImplicitCastExpr", "ParenExpr"):
            callee = kids(callee)[0] if kids(callee) else None
        inst = None
        if callee is not None and callee.get("kind") == "DeclRefExpr":
            ref = callee.get("referencedDecl", {})
            st["callees"].add(callee["id"])
            if ref.get("name") in NO_METAL:
                raise TranslateError(f"'{st['fname']}': {NO_METAL[ref['name']]} is not available on Metal")
            d = self.defs.get(ref.get("id"))
            if d is not None:
                params = [c for c in kids(d) if c.get("kind") == "ParmVarDecl"]
                args = ch[1:]
                spaces = []
                for i, p in enumerate(params):
                    t = dtype(p)
                    a = args[i] if i < len(args) else None
                    if is_pointer(t):
                        spaces.append((self.ptr_space(a, st) if a is not None else None) or "device")
                    elif is_ref(t):
                        spaces.append(self.lval_space(a, st) if a is not None else "thread")
                    else:
                        spaces.append(None)
                inst = self.instance(d, tuple(spaces))
                st["callees_inst"].append(inst)
                if e.get("kind") == "CallExpr":
                    lo, hi = st["body"]
                    cb, ce = self.edit_span(callee, st, "a call")
                    if d["id"] in self.spec_names:
                        st["edits"].append((cb, ce, inst.name))
                    if not d["name"].startswith("operator"):
                        t = st["text"]
                        paren = t.find("(", ce)
                        if paren < 0 or paren >= hi or t[ce:paren].strip():
                            raise TranslateError(f"'{st['fname']}': call to '{d['name']}' through a macro is not handled yet")
                        nxt = t[paren + 1:].lstrip()
                        st["edits"].append((paren + 1, paren + 1, "ocu_ctx" + ("" if nxt.startswith(")") else ", ")))
        elif callee is not None and callee.get("kind") in ("MemberExpr", "CXXDependentScopeMemberExpr"):
            raise TranslateError(f"'{st['fname']}': member function calls are not handled yet")
        elif callee is not None and callee.get("kind") not in ("DeclRefExpr",) and e.get("kind") == "CallExpr":
            if callee.get("kind") in ("UnresolvedLookupExpr",):
                pass
            else:
                raise TranslateError(f"'{st['fname']}': calls through pointers are not available on Metal")
        st["calls"][e["id"]] = inst
        return inst

    def instance(self, d, spaces: tuple) -> Instance:
        key = (d["id"], spaces)
        if key in self.insts:
            inst = self.insts[key]
            if inst.busy:
                raise TranslateError(f"'{d['name']}': recursive functions are not handled yet")
            return inst
        kernel = any(c.get("kind") == "CUDAGlobalAttr" for c in kids(d))
        name = d.get("mangledName", d["name"]) if kernel else self.spec_names.get(d["id"], d["name"])
        tmpl = self.template_lines(d)
        inst = Instance(d, spaces, name, kernel, tmpl)
        self.insts[key] = inst
        self.translate(inst)
        inst.busy = False
        self.order.append(inst)
        return inst

    def template_lines(self, d) -> list[str]:
        t = self.tmpl_of.get(d["id"])
        if t is None:
            return []
        params = [c for c in kids(t) if c.get("kind") in
                  ("TemplateTypeParmDecl", "NonTypeTemplateParmDecl", "TemplateTemplateParmDecl")]
        args = [c for c in kids(d) if c.get("kind") == "TemplateArgument"]
        if len(params) != len(args):
            raise TranslateError(f"'{d['name']}': template parameter packs are not handled yet")
        out = []
        for p, a in zip(params, args):
            name = p.get("name")
            if p["kind"] == "TemplateTemplateParmDecl" or p.get("isParameterPack"):
                raise TranslateError(f"'{d['name']}': this kind of template parameter is not handled yet")
            if not name:
                continue
            if p["kind"] == "TemplateTypeParmDecl":
                at = (a.get("type") or {}).get("qualType")
                if at is None:
                    raise TranslateError(f"'{d['name']}': template argument for '{name}' is not a type")
                if is_pointer(at) or is_ref(at) or "(" in at or "[" in at:
                    raise TranslateError(f"'{d['name']}': template argument '{at}' (pointer, array or function type) is not handled yet")
                if has_double(at):
                    raise TranslateError(f"'{d['name']}': double is not available on Apple GPUs")
                out.append(f"typedef {at} {name};")
            else:
                if "value" not in a:
                    raise TranslateError(f"'{d['name']}': template argument for '{name}' is not an integer")
                pt = qtype(p)
                v = a["value"]
                if strip_qual(pt) == "bool":
                    v = "true" if int(v) != 0 else "false"
                out.append(f"constexpr {pt} {name} = {v};")
        return out

    def edit_span(self, n, st, what, loose_begin=False) -> tuple[int, int]:
        """[begin, end) of a node inside the body text being rewritten.
        A location inside a macro expansion is usable when the token is
        spelled in the body (a macro argument); with loose_begin, a node
        that starts with a macro (such as __shared__) starts where the
        macro is used."""
        r = n.get("range") or {}
        lo, hi = st["body"]
        out = []
        for key in ("begin", "end"):
            loc = r.get(key, {})
            if "spellingLoc" in loc:
                sp = loc["spellingLoc"]
                o = sp.get("offset")
                line_start = st["text"].rfind("\n", 0, o or 0) + 1
                if self.same(sp.get("file"), st["file"]) and o is not None and lo <= o < hi and \
                        not re.match(r"\s*#\s*define", st["text"][line_start:o]):
                    loc = sp
                elif key == "begin" and loose_begin:
                    loc = dict(loc["expansionLoc"], tokLen=0)
                else:
                    raise TranslateError(f"'{st['fname']}': {what} inside a macro is not handled yet")
            if loc.get("offset") is None:
                raise TranslateError(f"'{st['fname']}': {what} without a source location")
            if not self.same(loc.get("file"), st["file"]) or not (lo <= loc["offset"] < hi):
                raise TranslateError(f"'{st['fname']}': {what} outside the function body is not handled yet")
            out.append(loc["offset"] + (loc.get("tokLen", 0) if key == "end" else 0))
        return out[0], out[1]

    def decl_begin(self, n, st) -> int:
        r = n.get("range") or {}
        loc = r.get("begin", {})
        if "spellingLoc" in loc:
            loc = loc["expansionLoc"]
        lo, hi = st["body"]
        if not self.same(loc.get("file"), st["file"]) or not (lo <= loc.get("offset", -1) < hi):
            raise TranslateError(f"'{st['fname']}': declaration of '{n.get('name')}' outside the function body")
        return loc["offset"]

    def translate(self, inst: Instance):
        fn = inst.fn
        fname = fn["name"]
        params = [c for c in kids(fn) if c.get("kind") == "ParmVarDecl"]
        body = next(c for c in kids(fn) if c.get("kind") == "CompoundStmt")
        bf, bb, be = self.span(body)
        text = self.text(bf)
        st = {"fname": fname, "vars": {}, "pending": {}, "edits": [], "calls": {}, "callees": set(),
              "callees_inst": [], "body": (bb, be), "text": text, "file": bf, "decls": []}
        ret = self.return_type(fn)
        for i, p in enumerate(params):
            t = dtype(p)
            if has_double(t) or has_double(qtype(p)):
                raise TranslateError(f"'{fname}': double is not available on Apple GPUs")
            if is_pointer(t):
                if is_pointer(pointee(t)):
                    raise TranslateError(f"'{fname}': pointers to pointers are not handled yet")
                st["vars"][p["id"]] = (True, inst.spaces[i] if i < len(inst.spaces) else "device")
            elif is_ref(t):
                if inst.kernel:
                    raise TranslateError(f"'{fname}': reference kernel parameters are not handled")
                st["vars"][p["id"]] = (True, inst.spaces[i])
            else:
                st["vars"][p["id"]] = (False, "thread")
        ret_space = None
        # PseudoObjectExpr: threadIdx.x and friends (properties in the
        # runtime headers; the prelude defines them for MSL)
        for n in walk(body, ("PseudoObjectExpr",)):
            k = n.get("kind")
            if k == "VarDecl":
                self.local_var(n, inst, st)
            elif k == "BinaryOperator" and n.get("opcode") == "=":
                lhs = kids(n)[0]
                while lhs.get("kind") == "ParenExpr":
                    lhs = kids(lhs)[0]
                rid = lhs.get("referencedDecl", {}).get("id") if lhs.get("kind") == "DeclRefExpr" else None
                if rid in st["vars"] and st["vars"][rid][0] and is_pointer(dtype(lhs)):
                    sp = self.ptr_space(kids(n)[1], st)
                    old = st["vars"][rid][1]
                    st["vars"][rid] = (True, merge(old, sp, f"'{fname}': pointer '{lhs['referencedDecl'].get('name')}'"))
            elif k in ("CallExpr", "CXXOperatorCallExpr", "CXXMemberCallExpr"):
                if k == "CXXMemberCallExpr":
                    raise TranslateError(f"'{fname}': member function calls are not handled yet")
                self.call(n, st)
            elif k in CASTS and is_pointer(dtype(n)):
                inner = kids(n)[-1]
                written = qtype(n)
                if not written.rstrip().endswith("*") and not is_pointer(written):
                    raise TranslateError(f"'{fname}': casts to pointer typedefs are not handled yet")
                if is_pointer(pointee(dtype(n))):
                    raise TranslateError(f"'{fname}': pointers to pointers are not handled yet")
                if k == "CXXFunctionalCastExpr":
                    raise TranslateError(f"'{fname}': functional casts to pointers are not handled yet")
                sp = self.ptr_space(n, st) if is_pointer(dtype(inner)) or "[" in dtype(inner) else "device"
                cb, ce = self.edit_span(n, st, "a pointer cast", True)
                if k == "CStyleCastExpr":
                    if text[cb] != "(":
                        raise TranslateError(f"'{fname}': pointer cast inside a macro is not handled yet")
                    st["edits"].append((cb + 1, cb + 1, f"{sp or 'device'} "))
                else:
                    lt = text.find("<", cb)
                    if lt < 0 or lt >= ce:
                        raise TranslateError(f"'{fname}': unexpected cast text")
                    st["edits"].append((lt + 1, lt + 1, f"{sp or 'device'} "))
            elif k == "DeclRefExpr":
                ref = n.get("referencedDecl", {})
                rid = ref.get("id")
                if rid in self.globals:
                    g = self.global_info(rid)
                    cb, ce = self.edit_span(n, st, f"use of '{ref.get('name')}'")
                    st["edits"].append((cb, ce, g["expr"]))
                elif rid in self.dyn_shared:
                    # the launch's dynamic threadgroup memory, seen as this array
                    cb, ce = self.edit_span(n, st, f"use of '{ref.get('name')}'")
                    et = self.dyn_shared[rid]
                    st["edits"].append((cb, ce, f"((threadgroup {et} *)ocu_ctx.dyn)"))
                elif ref.get("kind") == "FunctionDecl" and rid in self.defs and n["id"] not in st["callees"]:
                    raise TranslateError(f"'{fname}': taking the address of '{ref.get('name')}' is not available on Metal")
                elif ref.get("kind") == "VarDecl" and rid not in st["vars"] and rid not in self.consts \
                        and ref.get("name") not in ("threadIdx", "blockIdx", "blockDim", "gridDim", "warpSize"):
                    pass  # variables from the runtime headers: the prelude has them
            elif k == "ReturnStmt" and kids(n) and is_pointer(ret):
                ret_space = merge(ret_space, self.ptr_space(kids(n)[0], st), f"'{fname}': the returned pointer")
            elif k in ("GCCAsmStmt", "MSAsmStmt"):
                raise TranslateError(f"'{fname}': inline assembly is not available on Metal")
            elif k in ("LambdaExpr", "CXXThisExpr", "CXXNewExpr", "CXXDeleteExpr", "CXXThrowExpr",
                       "CXXTryStmt", "CXXTypeidExpr"):
                raise TranslateError(f"'{fname}': {k[3:] if k.startswith('CXX') else k} is not handled yet")
            t = qtype(n)
            if has_double(t) and k not in ("FloatingLiteral", "ImplicitCastExpr", "BinaryOperator",
                                           "ParenExpr", "UnaryOperator", "CallExpr", "DeclRefExpr",
                                           "ConditionalOperator", "CompoundAssignOperator"):
                raise TranslateError(f"'{fname}': double is not available on Apple GPUs")
            if k in ("CStyleCastExpr", "CXXStaticCastExpr", "CXXFunctionalCastExpr") and has_double(t):
                raise TranslateError(f"'{fname}': double is not available on Apple GPUs")
        # pointer and reference declarations: their address space, once known
        groups: dict[int, set] = {}
        for pos, rid, auto in st["decls"]:
            sp = st["vars"][rid][1] or "device"
            if not auto:
                groups.setdefault(pos, set()).add(sp)
        for pos, sps in groups.items():
            if len(sps) > 1:
                raise TranslateError(f"'{fname}': one declaration of pointers into different memory ({', '.join(sorted(sps))}); split it")
            st["edits"].append((pos, pos, f"{sps.pop()} "))
        inst.ret_space = ret_space or ("device" if is_pointer(ret) else None)
        out = self.apply(text[bb:be], bb, st["edits"], fname)
        inst.head, inst.param_line = self.head(inst, params, ret)
        inject = "".join(f"\n    {line}" for line in inst.tmpl)
        if inst.kernel:
            inst.text = inst.head + " {" + self.prologue(inst, params) + inject + out[1:]
        else:
            inst.text = inst.head + " {" + inject + out[1:]

    def local_var(self, n, inst, st):
        fname = st["fname"]
        t, td = qtype(n), dtype(n)
        if has_double(t) or has_double(td):
            raise TranslateError(f"'{fname}': double is not available on Apple GPUs")
        attrs = {c.get("kind") for c in kids(n)}
        text = st["text"]
        if "CUDASharedAttr" in attrs:
            st["vars"][n["id"]] = (False, "threadgroup")
            if n.get("storageClass") == "extern":
                b, e = self.edit_span(n, st, "an extern __shared__ declaration", True)
                seg = text[b:e]
                m = re.search(r"(\w+)\s*\[\s*\]\s*$", seg)
                if not m:
                    raise TranslateError(f"'{fname}': extern __shared__ '{n['name']}' is not an array")
                base = re.sub(r"\b(extern|__shared__)\b|__align__\s*\(\s*\d+\s*\)|__attribute__\s*\(\(.*?\)\)", "",
                              seg[:m.start()]).strip()
                st["edits"].append((b, e, f"threadgroup {base} *{n['name']} = (threadgroup {base} *)ocu_ctx.dyn"))
            else:
                if not inst.kernel:
                    raise TranslateError(f"'{fname}': __shared__ in a device function is not handled yet")
                b, e = self.edit_span(n, st, "a __shared__ declaration", True)
                m = re.search(r"\b__shared__\b", text[b:e])
                if not m:
                    raise TranslateError(f"'{fname}': __shared__ through a macro is not handled yet")
                st["edits"].append((b + m.start(), b + m.end(), "threadgroup"))
            return
        if n.get("storageClass") == "static":
            raise TranslateError(f"'{fname}': static local variables are not handled yet")
        init = [c for c in kids(n) if not c.get("kind", "").endswith("Attr")]
        b = self.decl_begin(n, st) if (is_pointer(td) or is_ref(td)) else None
        auto = b is not None and re.match(r"(const\s+|volatile\s+)*auto\b", text[b:]) is not None
        if is_pointer(td):
            if is_pointer(pointee(td)):
                raise TranslateError(f"'{fname}': pointers to pointers are not handled yet")
            if not auto and not t.rstrip().endswith("*") and not is_pointer(t):
                raise TranslateError(f"'{fname}': variables of pointer typedef type are not handled yet")
            sp = self.ptr_space(init[0], st) if init else None
            st["vars"][n["id"]] = (True, sp)
            st["decls"].append((b, n["id"], auto))
        elif is_ref(td):
            if not init:
                raise TranslateError(f"'{fname}': reference without initializer")
            st["vars"][n["id"]] = (True, self.lval_space(init[0], st))
            st["decls"].append((b, n["id"], auto))
        else:
            if re.search(r"\*\s*(const\s*)?\[", td) or re.search(r"\*\s*(const\s*)?\[", t):
                raise TranslateError(f"'{fname}': arrays of pointers are not handled yet")
            st["vars"][n["id"]] = (False, "thread")

    @staticmethod
    def return_type(fn) -> str:
        t = fn["type"]["qualType"]
        depth = 0
        for i in range(len(t) - 1, -1, -1):
            if t[i] == ")":
                depth += 1
            elif t[i] == "(":
                depth -= 1
                if depth == 0:
                    return t[:i].strip()
        return t

    def head(self, inst: Instance, params, ret) -> tuple[str, str | None]:
        fn = inst.fn
        fname = fn["name"]
        if has_double(ret):
            raise TranslateError(f"'{fname}': double is not available on Apple GPUs")
        if inst.kernel:
            sizes = []
            for i, p in enumerate(params):
                t = no_restrict(dtype(p))
                if is_pointer(t):
                    sizes.append(8)
                else:
                    sizes.append(self.value_size(strip_qual(t), fname))
            glob = "    device uchar *ocu_glob [[buffer(2)]],\n" if self.globals else ""
            head = (f"kernel void {inst.name}(constant uchar *ocu_args [[buffer(0)]],\n"
                    "    constant uint *ocu_offs [[buffer(1)]],\n" + glob +
                    "    threadgroup uchar *ocu_dyn [[threadgroup(0)]],\n"
                    "    uint3 ocu_tid [[thread_position_in_threadgroup]],\n"
                    "    uint3 ocu_bid [[threadgroup_position_in_grid]],\n"
                    "    uint3 ocu_bdim [[threads_per_threadgroup]],\n"
                    "    uint3 ocu_gdim [[threadgroups_per_grid]])")
            return head, f"param {inst.name} " + " ".join(str(s) for s in sizes)
        if is_ref(ret):
            raise TranslateError(f"'{fname}': functions returning references are not handled yet")
        if is_pointer(ret):
            if not ret.rstrip().endswith("*"):
                raise TranslateError(f"'{fname}': returning a pointer typedef is not handled yet")
            ret = f"{inst.ret_space} {no_restrict(ret)}"
        decls = []
        for i, p in enumerate(params):
            t = no_restrict(qtype(p))
            td = no_restrict(dtype(p))
            name = p.get("name") or f"ocu_p{i}"
            if is_pointer(td):
                if not t.rstrip().endswith("*"):
                    t = td
                decls.append(f"{inst.spaces[i]} {t} {name}")
            elif is_ref(td):
                if not t.rstrip().endswith("&"):
                    t = td
                decls.append(f"{inst.spaces[i]} {t} {name}")
            else:
                if "(" in td:
                    raise TranslateError(f"'{fname}': function pointer parameters are not available on Metal")
                decls.append(f"{t} {name}")
        if fname.startswith("operator"):
            return f"static {ret} {inst.name}(" + ", ".join(decls) + ")", None
        return (f"static {ret} {inst.name}(thread const ocu_context &ocu_ctx" +
                "".join(", " + d for d in decls) + ")"), None

    def value_size(self, t: str, fname: str) -> int:
        if t in SCALAR_SIZES:
            return SCALAR_SIZES[t]
        if has_double(t):
            raise TranslateError(f"'{fname}': double is not available on Apple GPUs")
        if t in self.layout:
            return self.layout[t][0]
        raise TranslateError(f"'{fname}': argument type '{t}' has no known layout")

    def prologue(self, inst, params) -> str:
        reads = []
        for i, p in enumerate(params):
            t = no_restrict(dtype(p))
            name = p.get("name") or f"ocu_p{i}"
            if is_pointer(t):
                reads.append(f"    device {t} {name} = ocu_ptr<device {t}>(ocu_args, ocu_offs, {i});")
            else:
                base = strip_qual(t)
                m = VEC3.fullmatch(base)
                if m:
                    reads.append(f"    {t} {name} = {base}(ocu_arg<packed_{base}>(ocu_args, ocu_offs, {i}));")
                else:
                    reads.append(f"    {t} {name} = ocu_arg<{base}>(ocu_args, ocu_offs, {i});")
                    if base not in SCALAR_SIZES:
                        self.asserts.append(base)
        glob = "ocu_glob" if self.globals else "nullptr"
        return (f"\n    ocu_context ocu_ctx{{ocu_tid, ocu_bid, ocu_bdim, ocu_gdim, ocu_dyn, {glob}}};\n"
                + "\n".join(reads))

    @staticmethod
    def apply(seg: str, base: int, edits, fname) -> str:
        edits = sorted(set(edits), key=lambda e: (e[0], e[1], e[2]))
        last_b, last_e = -1, -1
        for a, b, t in edits:
            if a < last_e and not (a == b == last_b) and not (a == last_b and last_b == last_e):
                raise TranslateError(f"'{fname}': overlapping rewrites (a construct inside a macro?)")
            if a == last_b and b == last_e and a != b:
                raise TranslateError(f"'{fname}': two rewrites of the same text")
            last_b, last_e = a, max(b, last_e)
        out = seg
        # apply from the end; at one position, a replacement before an insertion
        for a, b, t in sorted(edits, key=lambda e: (e[0], e[1] - e[0]), reverse=True):
            out = out[:a - base] + t + out[b - base:]
        return out

    # -- globals ----------------------------------------------------------------------------------

    def global_info(self, gid: str) -> dict:
        g = self.globals[gid]
        if "expr" in g:
            return g
        n = g["node"]
        name = n["name"]
        t, td = qtype(n), dtype(n)
        if has_double(t) or has_double(td):
            raise TranslateError(f"'{name}': double is not available on Apple GPUs")
        if "(" in td:
            raise TranslateError(f"'{name}': function pointer variables are not available on Metal")
        if is_pointer(td) and "[" not in td:
            if is_pointer(pointee(td)):
                raise TranslateError(f"'{name}': pointers to pointers are not handled yet")
            g["typedef"] = f"typedef device {no_restrict(td)} ocu_gt_{name};"
        elif "*" in td:
            raise TranslateError(f"'{name}': arrays of pointers are not handled yet")
        elif "[" in t:
            i = t.index("[")
            g["typedef"] = f"typedef {t[:i].strip()} ocu_gt_{name}{t[i:]};"
        else:
            g["typedef"] = f"typedef {t} ocu_gt_{name};"
        if VEC3.search(td) or VEC3.search(t):
            raise TranslateError(f"'{name}': 3-component vector variables are laid out differently on Metal")
        g["type"] = t
        g["expr"] = f"(*(device ocu_gt_{name} *)(ocu_ctx.glob + OCU_G_{name}))"
        g["name"] = name
        g["symbol"] = n.get("mangledName", name)
        return g

    def place_globals(self) -> list[str]:
        # every variable gets a place: the host may copy to any of them
        for gid in self.globals:
            self.global_info(gid)
        self.probe_layouts([g["type"] for g in self.globals.values()])
        off, lines = 0, []
        for gid, g in self.globals.items():
            size, align = self.layout[g["type"]]
            off = (off + align - 1) // align * align
            g["offset"], g["size"] = off, size
            lines.append(f"global {g['symbol']} {off} {size}")
            data = self.init_bytes(g["node"], g["type"], size)
            if data and any(data):
                lines.append(f"init {off} {data.hex()}")
            off += size
        return lines

    def init_bytes(self, n, t: str, size: int) -> bytes | None:
        init = [c for c in kids(n) if not c.get("kind", "").endswith("Attr")]
        if not init or not n.get("init"):
            return None
        out = bytearray(size)
        self.fill(init[0], t, out, 0, n["name"])
        return bytes(out)

    def fill(self, e, t: str, out: bytearray, at: int, name: str):
        t = " ".join(re.sub(r"\b(const|volatile)\b", "", t).split())
        while e.get("kind") in ("ConstantExpr", "ExprWithCleanups", "ParenExpr") and kids(e):
            e = kids(e)[0]
        m = re.fullmatch(r"(.*?)\s*\[(\d+)\](.*)", t)
        if m:
            elem = (m.group(1) + " " + m.group(3)).strip() if m.group(3) else m.group(1)
            count = int(m.group(2))
            esize = self.sizeof_scalar_array(elem, name)
            if e.get("kind") != "InitListExpr":
                raise TranslateError(f"'{name}': initializer is not handled by this translation")
            items = [c for c in kids(e) if c is not None]
            filler = e.get("array_filler")
            if filler:
                items = [c for c in filler[1:]]
            for i, c in enumerate(items[:count]):
                if c.get("kind") == "ImplicitValueInitExpr":
                    continue
                self.fill(c, elem, out, at + i * esize, name)
            return
        if t not in PACK:
            raise TranslateError(f"'{name}': initializers of type '{t}' are not handled yet")
        if e.get("kind") == "InitListExpr":
            ch = kids(e)
            if len(ch) != 1:
                raise TranslateError(f"'{name}': initializer is not handled by this translation")
            e = ch[0]
        v = self.const_value(e, name)
        code = PACK[t]
        if code in "fd":
            data = struct.pack("<" + code, float(v))
        elif code == "?":
            data = struct.pack("<?", bool(v))
        else:
            bits = struct.calcsize(code) * 8
            iv = int(v) & ((1 << bits) - 1)
            data = iv.to_bytes(bits // 8, "little")
        out[at:at + len(data)] = data

    def sizeof_scalar_array(self, t: str, name: str) -> int:
        m = re.fullmatch(r"(.*?)\s*\[(\d+)\](.*)", t)
        if m:
            elem = (m.group(1) + " " + m.group(3)).strip() if m.group(3) else m.group(1)
            return int(m.group(2)) * self.sizeof_scalar_array(elem, name)
        if t in SCALAR_SIZES and t in PACK:
            return SCALAR_SIZES[t]
        raise TranslateError(f"'{name}': initializers of type '{t}' are not handled yet")

    def const_value(self, e, name):
        k = e.get("kind")
        ch = kids(e)
        if k in ("IntegerLiteral", "CharacterLiteral"):
            return int(e["value"])
        if k == "FloatingLiteral":
            return float(e["value"])
        if k == "CXXBoolLiteralExpr":
            return 1 if e.get("value") else 0
        if k in ("ImplicitCastExpr", "CStyleCastExpr", "ParenExpr", "ConstantExpr", "CXXStaticCastExpr",
                 "CXXFunctionalCastExpr"):
            v = self.const_value(ch[-1], name)
            t = strip_qual(qtype(e))
            if t in PACK and PACK[t] not in "fd?" and isinstance(v, float):
                return int(v)  # truncation toward zero, as C converts
            if t in ("float", "double"):
                return float(v)
            return v
        if k == "UnaryOperator" and e.get("opcode") in ("-", "+", "~"):
            v = self.const_value(ch[0], name)
            return -v if e["opcode"] == "-" else (v if e["opcode"] == "+" else ~int(v))
        if k == "BinaryOperator" and e.get("opcode") in ("+", "-", "*", "/", "<<", ">>", "|", "&"):
            a, b = self.const_value(ch[0], name), self.const_value(ch[1], name)
            op = e["opcode"]
            if op == "/" and isinstance(a, int) and isinstance(b, int):
                return int(a / b)
            return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b,
                    "<<": lambda: a << b, ">>": lambda: a >> b, "|": lambda: a | b, "&": lambda: a & b}[op]()
        raise TranslateError(f"'{name}': initializer is not a constant this translation handles ({k})")

    # -- types and constants the device code uses -----------------------------------------------

    def item_text(self, item) -> str:
        n = item["node"]
        if "error" in item:
            raise TranslateError(item["error"])
        f, b, e = self.span(n)
        text = self.text(f)
        k = n["kind"]
        name = n.get("name", "")
        edits = []
        if k == "VarDecl":
            t = qtype(n)
            if has_double(t):
                raise TranslateError(f"'{name}': double is not available on Apple GPUs")
            if is_pointer(dtype(n)) or "*" in t:
                raise TranslateError(f"'{name}': pointer constants are not handled yet")
            seg = text[b:e]
            m = re.match(r"((?:\s*\b(?:static|constexpr|const|inline|extern)\b)+)", seg)
            if m:
                seg = seg[m.end():]
            return "constant " + seg.lstrip() + ";"
        for r in walk(n):
            rk = r.get("kind")
            if rk in ("CXXMethodDecl", "CXXConstructorDecl", "CXXDestructorDecl", "CXXConversionDecl") \
                    and not r.get("isImplicit"):
                raise TranslateError(f"'{name}': member functions are not handled yet")
            if rk == "VarDecl":
                raise TranslateError(f"'{name}': static data members are not handled yet")
            if rk == "FieldDecl":
                ft, fd = qtype(r), dtype(r)
                fname = f"'{name}.{r.get('name', '')}'"
                if has_double(ft) or has_double(fd):
                    raise TranslateError(f"{fname}: double is not available on Apple GPUs")
                if VEC3.search(ft) or VEC3.search(fd):
                    raise TranslateError(f"{fname}: 3-component vectors are laid out differently on Metal")
                if is_ref(fd):
                    raise TranslateError(f"{fname}: reference fields are not handled")
                if "*" in fd:
                    if not is_pointer(fd) or is_pointer(pointee(fd)) or "(" in fd:
                        raise TranslateError(f"{fname}: this kind of pointer field is not handled yet")
                    if not ft.rstrip().endswith("*") and not is_pointer(ft):
                        raise TranslateError(f"{fname}: pointer typedef fields are not handled yet")
                    fs = self.span(r)
                    if fs is None or not self.same(fs[0], f) or "spellingLoc" in r["range"]["begin"]:
                        raise TranslateError(f"{fname}: pointer field inside a macro is not handled yet")
                    edits.append((fs[1], fs[1], "device "))
        if k in ("TypedefDecl", "TypeAliasDecl"):
            td = dtype(n) if "type" in n else ""
            ut = (n.get("type") or {}).get("qualType", "")
            for c in kids(n):
                ut = ut or qtype(c)
            if has_double(td) or has_double(ut):
                raise TranslateError(f"'{name}': double is not available on Apple GPUs")
            if "*" in td and "{" not in text[b:e]:
                raise TranslateError(f"'{name}': pointer typedefs are not handled yet")
            if VEC3.fullmatch(strip_qual(td)) and "{" not in text[b:e]:
                pass  # a vector type name: same in MSL
        seg = self.apply(text[b:e], b, edits, name)
        return seg + ";"

    def provided(self) -> list[tuple[int, dict]]:
        """Items grouped by overlapping source ranges (struct + typedef)."""
        groups = []
        for item in self.items:
            sp = self.span(item["node"])
            if sp is None:
                continue
            f, b, e = sp
            if groups and self.same(groups[-1]["file"], f) and b < groups[-1]["end"]:
                g = groups[-1]
                g["items"].append(item)
                g["names"] |= item["names"]
                g["end"] = max(g["end"], e)
            else:
                groups.append({"file": f, "begin": b, "end": e, "items": [item], "names": set(item["names"])})
        return groups

    def group_text(self, g) -> str:
        """Text of declarations that share source text (typedef struct {...} T;):
        the outermost one's, with the inner records' rewrites spliced in."""
        if len(g["items"]) == 1:
            return self.item_text(g["items"][0])
        size = lambda it: self.span(it["node"])[2] - self.span(it["node"])[1]
        outer = max(g["items"], key=size)
        f, ob, oe = self.span(outer["node"])
        self.item_text(outer)  # its own checks
        full = self.text(f)[ob:oe]
        for it in sorted(g["items"], key=lambda it: -self.span(it["node"])[1]):
            if it is outer:
                continue
            k = it["node"]["kind"]
            if k not in ("CXXRecordDecl", "RecordDecl", "EnumDecl", "TypedefDecl", "TypeAliasDecl"):
                raise TranslateError(f"'{it['node'].get('name')}': a variable declared with its type is not handled yet")
            inner = self.item_text(it)[:-1]
            _, b, e = self.span(it["node"])
            if k in ("CXXRecordDecl", "RecordDecl", "EnumDecl"):
                full = full[:b - ob] + inner + full[e - ob:]
        return full + ";"

    # -- the whole module ---------------------------------------------------------------------------

    def run(self, keep_going: bool = False, report=None) -> str:
        self.load()
        if not self.kernels:
            raise TranslateError("no kernels found")
        # argument struct layouts, as the host packs them
        need = []
        for k in self.kernels:
            for p in kids(k):
                if p.get("kind") == "ParmVarDecl":
                    t = strip_qual(no_restrict(dtype(p)))
                    if not is_pointer(t) and t not in SCALAR_SIZES and not has_double(t):
                        need.append(t)
        if need:
            self.probe_layouts(need)
        kernels = []
        for k in self.kernels:
            where = f"{os.path.relpath(self.span(k)[0] or '?')}"
            name = k.get("mangledName", k["name"])
            saved = (dict(self.insts), list(self.order), list(self.asserts),
                     {gid: dict(g) for gid, g in self.globals.items()})
            try:
                spaces = tuple("device" if is_pointer(dtype(p)) else None
                               for p in kids(k) if p.get("kind") == "ParmVarDecl")
                self.instance(k, spaces)
                kernels.append(k)
                if report:
                    print(f"ok {where} {name}", file=report)
            except TranslateError as e:
                if not keep_going:
                    raise
                self.insts, self.order, self.asserts = saved[0], saved[1], saved[2]
                for gid, g in saved[3].items():
                    self.globals[gid] = g
                print(f"{self.src}: skipping kernel '{k['name']}': {e}", file=sys.stderr)
                if report:
                    print(f"skip {where} {name}: {e}", file=report)
        if not kernels:
            raise TranslateError("no kernel could be translated")
        meta = []
        glines = self.place_globals() if self.globals else []
        params = [i.param_line for i in self.order if i.param_line]
        # code, then the declarations it names (fixed point over identifiers)
        # callees before callers (an instance is complete after the ones it calls)
        code = [i.text for i in self.order]
        gdefs = [g["typedef"] for g in self.globals.values()] + \
                [f"#define OCU_G_{g['name']} {g['offset']}u" for g in self.globals.values()]
        macro_lines = self.macros()
        macro_bodies: dict[str, list[str]] = {}
        for line in macro_lines:
            m = re.match(r"#\s*define\s+([A-Za-z_]\w*)(.*)", line)
            if m:
                macro_bodies.setdefault(m.group(1), []).append(m.group(2))
        groups = self.provided()
        chosen: set[int] = set()
        tokens: set[str] = set()
        seen_macros: set[str] = set()

        def add_tokens(s: str):
            todo = [s]
            while todo:
                for tok in re.findall(r"[A-Za-z_]\w*", todo.pop()):
                    if tok not in tokens:
                        tokens.add(tok)
                        if tok in macro_bodies and tok not in seen_macros:
                            seen_macros.add(tok)
                            todo.extend(macro_bodies[tok])

        for s in code + gdefs + self.asserts:
            add_tokens(s)
        texts: dict[int, str] = {}
        changed = True
        while changed:
            changed = False
            for i, g in enumerate(groups):
                if i not in chosen and g["names"] & tokens:
                    chosen.add(i)
                    changed = True
                    try:
                        texts[i] = self.group_text(g)
                    except TranslateError as e:
                        # Found by name only: the code may not need it. If it
                        # does, the Metal compiler stops at the missing name.
                        names = ", ".join(sorted(g["names"] & tokens))
                        print(f"{self.src}: warning: left out {names}: {e}", file=sys.stderr)
                        texts[i] = f"// cu2msl: left out {names}: {e}"
                        continue
                    add_tokens(texts[i])
        decls = [texts[i] for i in sorted(chosen)]
        asserts = [f'static_assert(sizeof({t}) == {self.layout[t][0]}, '
                   f'"cu2msl: {t} is laid out differently on Metal");'
                   for t in dict.fromkeys(self.asserts) if t in self.layout]
        asserts += [f'static_assert(sizeof(ocu_gt_{g["name"]}) == {g["size"]}, '
                    f'"cu2msl: {g["name"]} is laid out differently on Metal");'
                    for g in self.globals.values()]
        protos = [i.head + ";" for i in self.order if not i.kernel]
        prelude = (HERE / "msl_prelude.h").read_text(encoding="utf-8")
        source = "\n".join([prelude, "// ---- macros", *macro_lines, "// ---- types and constants",
                            *decls, *gdefs, *asserts, "// ---- functions", *protos, *code]) + "\n"
        meta = params + glines
        return "OCUMSL1\n" + "\n".join(meta) + "\n---\n" + source


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-I", action="append", default=[])
    ap.add_argument("-D", action="append", default=[])
    ap.add_argument("--clang", default=os.environ.get("OCU_CLANG", "clang++"))
    ap.add_argument("--keep-going", action="store_true",
                    help="leave out kernels that cannot be translated instead of failing")
    ap.add_argument("--report", action="store_true", help="one line per kernel on stdout")
    a = ap.parse_args(argv)
    flags = [f"-I{d}" for d in a.I] + [f"-D{d}" for d in a.D]
    try:
        image = Translator(Path(a.source).resolve(), flags, a.clang).run(
            a.keep_going, sys.stdout if a.report else None)
    except TranslateError as e:
        print(f"{a.source}: cannot translate for Metal: {e}", file=sys.stderr)
        return 1
    Path(a.output).write_bytes(image.encode("latin-1"))  # source bytes as they were
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

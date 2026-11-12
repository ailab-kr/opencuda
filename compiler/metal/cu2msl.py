#!/usr/bin/env python3
"""cu2msl: CUDA kernels to Metal Shading Language, for the Metal backend.

FIRST CUT (backends/metal/README.md, section 4). Reads the typed syntax
tree Clang produces for the device pass (open-source compiler output) and
rewrites only what MSL needs differently, in the original source text:

  - kernel signatures: arguments come from buffer 0 at the offsets in
    buffer 1, built-in indices from MSL kernel attributes;
  - device functions get the index context as a first parameter, and
    their calls pass it;
  - pointer declarations get an address space (device, or threadgroup when
    they point into shared memory);
  - __shared__ variables become threadgroup variables; extern __shared__
    arrays point at the launch's dynamic threadgroup memory;
  - threadIdx / blockIdx / blockDim / gridDim / warpSize read the context.

Everything else (barriers, atomics, math, warp functions) is provided by
the MSL prelude (msl_prelude.h) under the CUDA names. Constructs this cut
does not handle stop the translation with a message (never silently
wrong code): templates, double, __device__/__constant__ variables,
__shared__ variables in device functions, pointer-returning functions.

    cu2msl.py <file.cu> -o <image> [-I dir] [-D name] [--clang clang++]

The image is the Metal backend's format: "OCUMSL1", one "param" line per
kernel with its argument sizes, "---", the MSL source.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


class TranslateError(Exception):
    pass


# ---- syntax tree ------------------------------------------------------------------

def clang_ast(clang: str, src: Path, flags: list[str], name: str) -> list[dict]:
    """Top-level declarations whose name contains `name`, from the device pass."""
    rt = REPO / "runtime" / "include"
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / "ocu-cuda-path"
    (cache / "include").mkdir(parents=True, exist_ok=True)
    for sub in ("bin", "lib64"):
        (cache / sub).mkdir(exist_ok=True)
    marker = cache / "include" / "cuda.h"
    if not marker.exists():
        marker.write_text("#define CUDA_VERSION 12030\n")
    cmd = [clang, "-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib",
           f"--cuda-path={cache}", "--cuda-gpu-arch=sm_80", "-std=c++17",
           f"-I{rt}", "-include", str(rt / "cuda_runtime.h"), "-DOCU_METAL_PATH",
           "-Wno-unknown-cuda-version", *flags, "-fsyntax-only",
           "-Xclang", "-ast-dump=json", "-Xclang", f"-ast-dump-filter={name}", str(src)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise TranslateError(f"clang failed:\n{out.stderr[-2000:]}")
    dec = json.JSONDecoder()
    text, i, objs = out.stdout, 0, []
    while i < len(text):
        while i < len(text) and text[i] in " \t\r\n":
            i += 1
        if i >= len(text):
            break
        if text[i] != "{":  # "Dumping <name>:" lines
            j = text.find("\n", i)
            i = len(text) if j < 0 else j + 1
            continue
        obj, i = dec.raw_decode(text, i)
        objs.append(obj)
    return objs


def walk(node):
    yield node
    for c in node.get("inner", []) or []:
        if isinstance(c, dict):
            yield from walk(c)


def loc_offset(loc: dict) -> int | None:
    """Offset in the main file of a location (expansion location for macros)."""
    if "expansionLoc" in loc:
        loc = loc["expansionLoc"]
    return loc.get("offset")


def loc_file(loc: dict, default: str | None) -> str | None:
    if "expansionLoc" in loc:
        loc = loc["expansionLoc"]
    return loc.get("file", default)


def node_range(node) -> tuple[int, int] | None:
    r = node.get("range")
    if not r:
        return None
    b, e = loc_offset(r["begin"]), loc_offset(r["end"])
    if b is None or e is None:
        return None
    end_loc = r["end"].get("expansionLoc", r["end"])
    return b, e + end_loc.get("tokLen", 0)


# ---- types --------------------------------------------------------------------------

SCALAR_SIZES = {
    "bool": 1, "char": 1, "signed char": 1, "unsigned char": 1, "short": 2,
    "unsigned short": 2, "int": 4, "unsigned int": 4, "unsigned": 4, "float": 4,
    "long": 8, "unsigned long": 8, "long long": 8, "unsigned long long": 8,
    "size_t": 8, "double": 8, "half": 2, "__half": 2,
}


def strip_qual(t: str) -> str:
    return re.sub(r"\b(const|volatile|struct|__restrict|restrict)\b", "", t).strip()


def is_pointer(t: str) -> bool:
    return t.rstrip().endswith("*") or "*__restrict" in t or "* __restrict" in t


def type_size(t: str, records: dict) -> int:
    t = strip_qual(t)
    if is_pointer(t):
        return 8
    t = " ".join(t.split())
    if t in SCALAR_SIZES:
        return SCALAR_SIZES[t]
    if t in records:
        return records[t][0]
    raise TranslateError(f"argument type '{t}' is not supported by this translation")


# ---- the translation ------------------------------------------------------------------

BUILTINS = {"threadIdx": "ocu_ctx.tid", "blockIdx": "ocu_ctx.bid", "blockDim": "ocu_ctx.bdim",
            "gridDim": "ocu_ctx.gdim"}


class Translator:
    def __init__(self, src: Path, flags: list[str], clang: str):
        self.src = src
        self.text = src.read_text(encoding="utf-8", errors="surrogateescape")
        self.flags = flags
        self.clang = clang
        self.funcs: dict[str, dict] = {}   # name -> FunctionDecl
        self.records: dict[str, tuple[int, int, str]] = {}  # name -> (size, align, text)
        self.edits: list[tuple[int, int, str]] = []

    def main_file(self, node) -> bool:
        r = node.get("range") or {}
        f = loc_file(r.get("begin", {}), None) if r else None
        return f is None or os.path.realpath(f) == os.path.realpath(str(self.src))

    def collect(self):
        names = set(re.findall(r"__(?:global|device)__[^;{(]*?\b([A-Za-z_]\w*)\s*\(", self.text))
        names |= set(re.findall(r"\bstruct\s+([A-Za-z_]\w*)\s*\{", self.text))
        for name in sorted(names):
            for obj in clang_ast(self.clang, self.src, self.flags, name):
                for n in walk(obj):
                    if n.get("name") != name or not self.main_file(n):
                        continue
                    if n.get("kind") == "FunctionDecl" and any(c.get("kind") == "CompoundStmt" for c in n.get("inner", [])):
                        attrs = {c.get("kind") for c in n.get("inner", [])}
                        if "CUDAGlobalAttr" in attrs or "CUDADeviceAttr" in attrs:
                            self.funcs[name] = n
                    elif n.get("kind") == "CXXRecordDecl" and n.get("completeDefinition"):
                        self.record(n)
                    elif n.get("kind") == "FunctionTemplateDecl":
                        raise TranslateError(f"'{name}': templates are not handled by this translation yet")

    def record(self, n):
        size, align, off = 0, 1, 0
        for f in n.get("inner", []):
            if f.get("kind") != "FieldDecl":
                continue
            t = f["type"]["qualType"]
            s = type_size(t, self.records)
            a = min(s, 8) if s in (1, 2, 4, 8) else (self.records.get(strip_qual(t), (0, 8))[1])
            off = (off + a - 1) // a * a + s
            align = max(align, a)
        size = (off + align - 1) // align * align
        rng = node_range(n)
        text = self.text[rng[0]:rng[1]] + ";" if rng else ""
        self.records[n["name"]] = (size, align, text)

    # -- address spaces -------------------------------------------------------------------

    def space_of(self, expr, spaces: dict) -> str:
        for n in walk(expr):
            if n.get("kind") == "DeclRefExpr":
                ref = n.get("referencedDecl", {})
                rid = ref.get("id")
                if rid in spaces:
                    return spaces[rid]
        return "device"

    def translate_function(self, fn) -> tuple[str, str | None]:
        """MSL text of one function and, for a kernel, its param line."""
        attrs = {c.get("kind") for c in fn.get("inner", [])}
        kernel = "CUDAGlobalAttr" in attrs
        rng = node_range(fn)
        params = [c for c in fn["inner"] if c.get("kind") == "ParmVarDecl"]
        body = next(c for c in fn["inner"] if c.get("kind") == "CompoundStmt")
        brng = node_range(body)
        ret = fn["type"]["qualType"].split("(")[0].strip()
        if is_pointer(ret):
            raise TranslateError(f"'{fn['name']}': functions returning pointers are not handled yet")
        spaces: dict[str, str] = {}
        edits: list[tuple[int, int, str]] = []
        # parameters
        decls = []
        reads = []
        sizes = []
        for i, p in enumerate(params):
            t = p["type"]["qualType"]
            if "double" in t:
                raise TranslateError(f"'{fn['name']}': double is not available on Apple GPUs")
            name = p.get("name") or f"ocu_p{i}"
            if is_pointer(t):
                spaces[p["id"]] = "device"
                if kernel:
                    reads.append(f"    device {t} {name} = ocu_ptr<device {t}>(ocu_args, ocu_offs, {i});")
                else:
                    decls.append(f"device {t} {name}")
            else:
                if kernel:
                    reads.append(f"    {t} {name} = ocu_arg<{strip_qual(t) if strip_qual(t) in self.records else t}>(ocu_args, ocu_offs, {i});")
                else:
                    decls.append(f"{t} {name}")
            if kernel:
                sizes.append(type_size(t, self.records))
        mangled = fn.get("mangledName", fn["name"])
        if kernel:
            head = (f"kernel void {mangled}(constant uchar *ocu_args [[buffer(0)]],\n"
                    "    constant uint *ocu_offs [[buffer(1)]],\n"
                    "    threadgroup uchar *ocu_dyn [[threadgroup(0)]],\n"
                    "    uint3 ocu_tid [[thread_position_in_threadgroup]],\n"
                    "    uint3 ocu_bid [[threadgroup_position_in_grid]],\n"
                    "    uint3 ocu_bdim [[threads_per_threadgroup]],\n"
                    "    uint3 ocu_gdim [[threadgroups_per_grid]])")
            prologue = ("\n    ocu_context ocu_ctx{ocu_tid, ocu_bid, ocu_bdim, ocu_gdim, ocu_dyn};\n"
                        + "\n".join(reads) + "\n")
        else:
            head = f"static {ret} {fn['name']}(thread const ocu_context &ocu_ctx" + \
                   "".join(", " + d for d in decls) + ")"
            prologue = "\n"
        # body rewrites
        for n in walk(body):
            k = n.get("kind")
            if k == "VarDecl":
                t = n["type"]["qualType"]
                if "double" in t:
                    raise TranslateError(f"'{fn['name']}': double is not available on Apple GPUs")
                vr = node_range(n)
                if vr is None:
                    continue
                inner = {c.get("kind") for c in n.get("inner", [])}
                if "CUDASharedAttr" in inner:
                    if not kernel:
                        raise TranslateError(f"'{fn['name']}': __shared__ in a device function is not handled yet")
                    seg = self.text[vr[0]:vr[1]]
                    spaces[n["id"]] = "threadgroup"
                    if n.get("storageClass") == "extern":
                        m = re.search(r"(\w+)\s*\[\s*\]\s*$", seg)
                        base = re.sub(r"\b(extern|__shared__|__align__\s*\(\s*\d+\s*\))", "", seg[:m.start()] if m else seg).strip()
                        edits.append((vr[0], vr[1], f"threadgroup {base} *{n['name']} = "
                                      f"(threadgroup {base} *)ocu_ctx.dyn"))
                    else:
                        edits.append((vr[0], vr[1], re.sub(r"\b__shared__\b", "threadgroup", seg)))
                    continue
                if is_pointer(t):
                    init = [c for c in n.get("inner", []) if not c.get("kind", "").endswith("Attr")]
                    sp = self.space_of(init[0], spaces) if init else "device"
                    spaces[n["id"]] = sp
                    # one qualifier per declaration statement start
                    edits.append((vr[0], vr[0], f"{sp} "))
            elif k == "DeclRefExpr":
                ref = n.get("referencedDecl", {})
                nm = ref.get("name")
                rr = node_range(n)
                if rr is None:
                    continue
                if nm in BUILTINS:
                    edits.append((rr[0], rr[1], BUILTINS[nm]))
                elif nm == "warpSize":
                    edits.append((rr[0], rr[1], "32"))
                elif ref.get("kind") == "VarDecl" and ref.get("id") not in spaces and \
                        nm not in {p.get("name") for p in params} and \
                        self.is_global_var(nm):
                    raise TranslateError(f"'{fn['name']}': __device__/__constant__ variable '{nm}' is not handled yet")
            elif k == "CallExpr":
                callee = next((c for c in walk(n) if c.get("kind") == "DeclRefExpr"), None)
                cname = callee.get("referencedDecl", {}).get("name") if callee else None
                if cname in self.funcs and cname is not None:
                    crng = node_range(callee)
                    if crng:
                        paren = self.text.find("(", crng[1])
                        nxt = self.text[paren + 1:].lstrip()
                        edits.append((paren + 1, paren + 1, "ocu_ctx" + ("" if nxt.startswith(")") else ", ")))
        out = self.apply(self.text[brng[0]:brng[1]], brng[0], edits)
        # the body text starts with '{'
        msl = head + " {" + prologue + out[1:]
        return msl, (f"param {mangled} " + " ".join(str(s) for s in sizes)) if kernel else None

    def is_global_var(self, name: str) -> bool:
        return re.search(r"__(device|constant)__\s+[^;(]*\b" + re.escape(name) + r"\b\s*(\[|=|;)", self.text) is not None

    @staticmethod
    def apply(seg: str, base: int, edits) -> str:
        # Edits must not overlap; inner replacements (inside a replaced
        # declaration) are dropped.
        edits = sorted(edits, key=lambda e: (e[0], -(e[1] - e[0])))
        kept, last_end = [], -1
        for a, b, t in edits:
            if a < last_end and b <= last_end and a != b:
                continue
            kept.append((a, b, t))
            last_end = max(last_end, b)
        out = seg
        for a, b, t in sorted(kept, key=lambda e: (e[0], e[1]), reverse=True):
            out = out[:a - base] + t + out[b - base:]
        return out

    def run(self) -> str:
        self.collect()
        kernels = [f for f in self.funcs.values() if any(c.get("kind") == "CUDAGlobalAttr" for c in f["inner"])]
        if not kernels:
            raise TranslateError("no kernels found")
        defines = "\n".join(l for l in self.text.splitlines() if re.match(r"\s*#\s*define\b", l))
        parts, params = [], []
        # device functions first, in source order
        ordered = sorted(self.funcs.values(), key=lambda f: node_range(f)[0])
        for f in ordered:
            msl, param = self.translate_function(f)
            parts.append(msl)
            if param:
                params.append(param)
        records = "\n".join(r[2] for r in self.records.values())
        prelude = (HERE / "msl_prelude.h").read_text(encoding="utf-8")
        source = "\n".join([prelude, defines, records, *parts]) + "\n"
        return "OCUMSL1\n" + "\n".join(params) + "\n---\n" + source


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-I", action="append", default=[])
    ap.add_argument("-D", action="append", default=[])
    ap.add_argument("--clang", default=os.environ.get("OCU_CLANG", "clang++"))
    a = ap.parse_args(argv)
    flags = [f"-I{d}" for d in a.I] + [f"-D{d}" for d in a.D]
    try:
        image = Translator(Path(a.source).resolve(), flags, a.clang).run()
    except TranslateError as e:
        print(f"{a.source}: cannot translate for Metal: {e}", file=sys.stderr)
        return 1
    Path(a.output).write_text(image, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

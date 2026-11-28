#!/usr/bin/env python3
"""ir2msl: a trial of making Metal Shading Language from LLVM IR.

Clang compiles the CUDA device code (templates, constexpr, member functions
and namespaces are gone after that); this script turns the textual IR of
each kernel into one MSL kernel function. It is a feasibility trial, not a
translator: it handles the instructions the trial kernels use and stops with
a message at anything else.

    ir2msl.py <file.cu | file.ll> -o <image> [-I dir] [-D name] [--clang clang++]

A .cu file is first compiled to IR by Clang's CUDA device pass (the same
command line cu2msl.py uses, plus -O2 without the vectorisers; the
architecture comes from OCU_CU_DEVICE_ARCH). The kernels are listed on
stdout ("ok <name>"); what cannot be converted is listed on stderr and
makes the exit status 1.

Control flow is not restructured. Every basic block is a case of a switch
inside a loop (MSL has no goto) that ends through its condition; phi nodes
become copies on the edges.
Pointers are byte pointers in one address space each (device, or
threadgroup for __shared__ variables), found by following each pointer back
to where it comes from.

Image: the format backends/metal/mt_backend.m reads ("OCUMSL1").
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

INT = {"i1": "bool", "i8": "uchar", "i16": "ushort", "i32": "uint", "i64": "ulong"}
SINT = {"i8": "char", "i16": "short", "i32": "int", "i64": "long"}
FLT = {"float": "float", "half": "half"}
SIZE = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "float": 4, "half": 2, "ptr": 8}
TYPE = r"(?:i\d+|float|double|half|ptr(?: addrspace\(\d+\))?|\[[^\]]*\]|void)"
FLAGS = r"(?:nuw|nsw|exact|inbounds|nnan|ninf|nsz|arcp|contract|afn|reassoc|fast|disjoint|nneg|samesign|tail|musttail|notail|noundef|nonnull|signext|zeroext)\b\s*"


class Unsupported(Exception):
    pass


def size_of(t: str) -> int:
    t = t.strip()
    if t.startswith("ptr"):
        return 8
    if t in SIZE:
        return SIZE[t]
    m = re.fullmatch(r"\[(\d+) x (.+)\]", t)
    if m:
        return int(m.group(1)) * size_of(m.group(2))
    raise Unsupported(f"size of type '{t}'")


def split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def take_type(s: str) -> tuple[str, str]:
    """(type, rest) from the start of an operand such as 'i32 noundef %5'."""
    s = s.strip()
    m = re.match(TYPE, s)
    if not m:
        raise Unsupported(f"operand type in '{s[:40]}'")
    rest = s[m.end():].strip()
    while True:
        m2 = re.match(FLAGS, rest) or re.match(r"(?:nocapture|readonly|writeonly|align \d+|captures\([^)]*\))\s*", rest)
        if not m2:
            break
        rest = rest[m2.end():]
    return m.group(0), rest


class Func:
    def __init__(self, name, params, blocks):
        self.name, self.params, self.blocks = name, params, blocks


def parse(ll: str):
    globs = {}
    for m in re.finditer(r"^@([\w.$]+) = .*?addrspace\((\d+)\) (?:global|constant) (\[[^\]]*\]|\w+) (\S+)", ll, re.M):
        globs[m.group(1)] = (int(m.group(2)), m.group(3), m.group(4))
    kernels = set(re.findall(r'!\{ptr @([\w.$]+), !"kernel", i32 1\}', ll))
    funcs = []
    for m in re.finditer(r"^define [^@]*@([\w.$]+)\((.*?)\)[^\n]*\{\n(.*?)^\}", ll, re.M | re.S):
        name, sig, body = m.groups()
        params = []
        for p in split_top(sig):
            t, rest = take_type(p)
            params.append((t, rest.strip()))
        blocks, cur = [], None
        for line in body.split("\n"):
            line = re.sub(r",? ![\w.]+ !\d+", "", line.split(";")[0] if not line.lstrip().startswith(";") else "").rstrip()
            if not line.strip():
                continue
            lm = re.match(r"^([\w.$]+):", line)
            if lm:
                cur = [lm.group(1), []]
                blocks.append(cur)
                continue
            if cur is None:
                cur = ["entry", []]
                blocks.append(cur)
            if cur[1] and re.match(r"^\s{4,}i\d+ ", line):   # a switch case line
                cur[1][-1] += " " + line.strip()
            elif line.strip() == "]":
                cur[1][-1] += " ]"
            else:
                cur[1].append(line.strip())
        funcs.append(Func(name, params, blocks))
    return globs, kernels, funcs


class Emit:
    def __init__(self, fn: Func, globs):
        self.fn, self.globs = fn, globs
        self.types = {}      # ssa name -> IR type
        self.space = {}      # ssa name -> "device" | "threadgroup"
        self.used_globs = []
        self.entry = fn.blocks[0][0]
        if self.entry == "entry" and not any(b[0] == "entry" for b in fn.blocks[1:]):
            pass
        # the entry block of an unnamed-value function is numbered after the parameters
        self.label = {b[0]: i for i, b in enumerate(fn.blocks)}
        if fn.blocks[0][0] == "entry":
            self.label[str(len(fn.params))] = 0

    def var(self, name: str) -> str:
        return "v_" + re.sub(r"\W", "_", name)

    def mtype(self, t: str, name=None) -> str:
        if t in INT:
            return INT[t]
        if t in FLT:
            return FLT[t]
        if t.startswith("ptr"):
            return f"{self.space.get(name, 'device')} uchar *"
        if t == "double":
            raise Unsupported("double is not available on Apple GPUs")
        raise Unsupported(f"type '{t}'")

    # -- operands -------------------------------------------------------------------
    def const_ptr(self, v: str):
        """(expression, space) for a constant pointer expression, or None."""
        m = re.fullmatch(r"addrspacecast \(ptr addrspace\((\d+)\) @([\w.$]+) to ptr\)", v)
        if m or re.fullmatch(r"@([\w.$]+)", v):
            g = m.group(2) if m else v[1:]
            if g not in self.globs:
                raise Unsupported(f"global '{g}'")
            sp, _, _ = self.globs[g]
            if sp != 3:
                raise Unsupported(f"global '{g}' in address space {sp} (only __shared__ in this trial)")
            if g not in self.used_globs:
                self.used_globs.append(g)
            return f"((threadgroup uchar *)g_{re.sub(r'[^A-Za-z0-9_]', '_', g)})", "threadgroup"
        return None

    def val(self, t: str, v: str) -> str:
        v = v.strip()
        if v.startswith("%"):
            return self.var(v[1:])
        if t.startswith("ptr"):
            c = self.const_ptr(v)
            if c:
                return c[0]
            if v == "null":
                return "((device uchar *)nullptr)"
            raise Unsupported(f"pointer constant '{v[:50]}'")
        if v in ("undef", "poison", "zeroinitializer"):
            v = "0" if t in INT else "0.0"
        if t == "i1":
            return {"true": "true", "false": "false", "0": "false", "1": "true"}[v]
        if t in INT:
            bits = int(t[1:])
            return f"{INT[t]}({int(v) & ((1 << bits) - 1)}u{'l' if bits == 64 else ''})"
        if t == "float":
            if v.startswith("0x"):
                d = struct.unpack("<d", struct.pack("<Q", int(v, 16)))[0]
            else:
                d = float(v)
            return f"as_type<float>(0x{struct.unpack('<I', struct.pack('<f', d))[0]:08x}u)"
        raise Unsupported(f"constant '{v}' of type {t}")

    def sval(self, t: str, v: str) -> str:
        return f"{SINT[t]}({self.val(t, v)})"

    # -- pointer address spaces ---------------------------------------------------------
    def infer_spaces(self):
        src = {}   # ptr ssa -> list of sources (ssa names or literal space)
        for t, n in self.fn.params:
            if t.startswith("ptr"):
                self.types[n[1:]] = t
                src[n[1:]] = ["device"]
            else:
                self.types[n[1:]] = t
        for _, insts in self.fn.blocks:
            for ins in insts:
                m = re.match(r"%([\w.$]+) = (.*)", ins)
                if not m:
                    continue
                name, rhs = m.groups()
                op = re.sub(FLAGS, "", rhs).split()[0] if rhs.split()[0] in ("tail", "musttail", "notail") else rhs.split()[0]
                rt = self.result_type(rhs)
                self.types[name] = rt
                if not rt.startswith("ptr"):
                    continue
                def base(tok):
                    tok = tok.strip()
                    if tok.startswith("%"):
                        return tok[1:]
                    c = self.const_ptr(tok)
                    return c[1] if c else "device"
                if op == "getelementptr":
                    args = split_top(re.sub(r"^getelementptr\s+(?:" + FLAGS + ")*", "", rhs))
                    src[name] = [base(take_type(args[1])[1])]
                elif op == "phi":
                    src[name] = [base(x.strip()[1:].split(",")[0]) for x in re.findall(r"\[[^\]]*\]", rhs)]
                elif op == "select":
                    a = split_top(re.sub(r"^select\s+(?:" + FLAGS + ")*", "", rhs))
                    src[name] = [base(take_type(a[1])[1]), base(take_type(a[2])[1])]
                elif op in ("bitcast", "addrspacecast"):
                    src[name] = [base(take_type(rhs.split(None, 1)[1].rsplit(" to ", 1)[0])[1])]
                else:
                    src[name] = ["device"]   # loaded pointers, inttoptr, calls
        changed = True
        while changed:
            changed = False
            for n, ss in src.items():
                got = {s if s in ("device", "threadgroup") else self.space.get(s) for s in ss} - {None}
                if len(got) > 1:
                    raise Unsupported(f"pointer %{n} may point into device and threadgroup memory")
                if got and self.space.get(n) != next(iter(got)):
                    self.space[n] = next(iter(got))
                    changed = True

    def result_type(self, rhs: str) -> str:
        rhs = re.sub(r"^(?:tail |musttail |notail )", "", rhs)
        op = rhs.split()[0]
        rest = re.sub(r"^\S+\s+(?:" + FLAGS + ")*", "", rhs)
        if op in ("icmp", "fcmp"):
            return "i1"
        if op in ("zext", "sext", "trunc", "sitofp", "uitofp", "fptosi", "fptoui", "fpext", "fptrunc", "bitcast",
                  "ptrtoint", "inttoptr", "addrspacecast"):
            return rest.rsplit(" to ", 1)[1].strip()
        if op == "getelementptr":
            return "ptr"
        if op == "load":
            return take_type(rest)[0]
        if op == "select":
            return take_type(split_top(rest)[1])[0]
        if op == "call":
            return take_type(rest)[0]
        if op == "alloca":
            raise Unsupported("alloca (a local the optimiser kept in memory)")
        return take_type(rest)[0]

    # -- instructions -----------------------------------------------------------------------
    BIN = {"add": "+", "sub": "-", "mul": "*", "and": "&", "or": "|", "xor": "^", "shl": "<<", "lshr": ">>",
           "udiv": "/", "urem": "%", "fadd": "+", "fsub": "-", "fmul": "*", "fdiv": "/"}
    ICMP = {"eq": "==", "ne": "!=", "ugt": ">", "uge": ">=", "ult": "<", "ule": "<=",
            "sgt": ">", "sge": ">=", "slt": "<", "sle": "<="}
    FCMP = {"oeq": "==", "ogt": ">", "oge": ">=", "olt": "<", "ole": "<=", "une": "!="}
    SREG = {"tid": "ocu_tid", "ntid": "ocu_bdim", "ctaid": "ocu_bid", "nctaid": "ocu_gdim"}

    def expr(self, name, rhs: str) -> str:
        rhs = re.sub(r"^(?:tail |musttail |notail )", "", rhs)
        op = rhs.split()[0]
        rest = re.sub(r"^\S+\s+(?:" + FLAGS + ")*", "", rhs)
        if op in self.BIN:
            t, r = take_type(rest)
            a, b = split_top(r)
            if t == "i1" and op in ("and", "or", "xor"):
                return f"({self.val(t, a)} {self.BIN[op]}{'' if op == 'xor' else self.BIN[op]} {self.val(t, b)})".replace("^^", "!=")
            mt = self.mtype(t)
            return f"{mt}({self.val(t, a)} {self.BIN[op]} {self.val(t, b)})"
        if op in ("sdiv", "srem", "ashr"):
            t, r = take_type(rest)
            a, b = split_top(r)
            c = {"sdiv": "/", "srem": "%", "ashr": ">>"}[op]
            rhs_b = self.val(t, b) if op == "ashr" else self.sval(t, b)
            return f"{INT[t]}({self.sval(t, a)} {c} {rhs_b})"
        if op == "fneg":
            t, r = take_type(rest)
            return f"(-{self.val(t, r)})"
        if op == "icmp":
            pred, r = rest.split(None, 1)
            t, r = take_type(r)
            a, b = split_top(r)
            f = self.sval if pred.startswith("s") and t in SINT else self.val
            return f"({f(t, a)} {self.ICMP[pred]} {f(t, b)})"
        if op == "fcmp":
            pred, r = rest.split(None, 1)
            t, r = take_type(r)
            a, b = split_top(r)
            if pred not in self.FCMP:
                raise Unsupported(f"fcmp {pred}")
            return f"({self.val(t, a)} {self.FCMP[pred]} {self.val(t, b)})"
        if op == "select":
            c, a, b = split_top(rest)
            (ct, cv), (at, av), (bt, bv) = take_type(c), take_type(a), take_type(b)
            return f"({self.val(ct, cv)} ? {self.val(at, av)} : {self.val(bt, bv)})"
        if op in ("zext", "trunc", "uitofp", "fptoui", "fpext", "fptrunc"):
            src, to = rest.rsplit(" to ", 1)
            t, v = take_type(src)
            return f"{self.mtype(to.strip())}({self.val(t, v)})"
        if op == "sext":
            src, to = rest.rsplit(" to ", 1)
            t, v = take_type(src)
            if t == "i1":
                return f"({self.val(t, v)} ? {self.val(to.strip(), '-1')} : {self.val(to.strip(), '0')})"
            return f"{INT[to.strip()]}({SINT[to.strip()]}({self.sval(t, v)}))"
        if op == "sitofp":
            src, to = rest.rsplit(" to ", 1)
            t, v = take_type(src)
            return f"{self.mtype(to.strip())}({self.sval(t, v)})"
        if op == "fptosi":
            src, to = rest.rsplit(" to ", 1)
            t, v = take_type(src)
            return f"{INT[to.strip()]}({SINT[to.strip()]}({self.val(t, v)}))"
        if op == "bitcast":
            src, to = rest.rsplit(" to ", 1)
            t, v = take_type(src)
            if t.startswith("ptr"):
                return self.val(t, v)
            return f"as_type<{self.mtype(to.strip())}>({self.val(t, v)})"
        if op == "freeze":
            t, v = take_type(rest)
            return self.val(t, v)
        if op == "getelementptr":
            args = split_top(rest)
            elem = args[0]
            bt, bv = take_type(args[1])
            off, cur = [], elem
            for k, a in enumerate(args[2:]):
                it, iv = take_type(a)
                if k > 0:
                    m = re.fullmatch(r"\[(\d+) x (.+)\]", cur.strip())
                    if not m:
                        raise Unsupported(f"getelementptr into '{cur}'")
                    cur = m.group(2)
                sz = size_of(cur)
                idx = self.sval(it, iv) if it in SINT else self.val(it, iv)
                off.append(f"long({idx}) * {sz}")
            base = self.val(bt, bv)
            return f"({base} + ({' + '.join(off) or '0'}))"
        if op == "load":
            t, r = take_type(rest)
            pt, pv = take_type(split_top(r.lstrip(", "))[0])
            sp = self.ptr_space(pv)
            return f"(*({sp} {self.mtype(t)} *)({self.val(pt, pv)}))"
        if op == "call":
            return self.call(rest)
        raise Unsupported(f"instruction '{op}'")

    def ptr_space(self, v: str) -> str:
        v = v.strip()
        if v.startswith("%"):
            return self.space.get(v[1:], "device")
        c = self.const_ptr(v)
        return c[1] if c else "device"

    def call(self, rest: str) -> str:
        m = re.match(r"(" + TYPE + r")\s+@([\w.$]+)\((.*)\)", rest)
        if not m:
            raise Unsupported(f"call '{rest[:60]}'")
        rt, fn, a = m.groups()
        args = [take_type(x) for x in split_top(a)]
        av = [self.val(t, v) for t, v in args]
        m2 = re.fullmatch(r"llvm\.nvvm\.read\.ptx\.sreg\.(\w+)\.([xyz])", fn)
        if m2:
            return f"{self.SREG[m2.group(1)]}.{m2.group(2)}"
        if fn == "llvm.nvvm.barrier0":
            return "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)"
        m2 = re.fullmatch(r"llvm\.nvvm\.shfl\.sync\.(bfly|up|down|idx)\.(i32|f32)", fn)
        if m2:
            f = {"bfly": "simd_shuffle_xor", "up": "simd_shuffle_up", "down": "simd_shuffle_down", "idx": "simd_shuffle"}[m2.group(1)]
            return f"{f}({av[1]}, ushort({av[2]}))"
        simple = {"llvm.fma.f32": "fma", "llvm.fmuladd.f32": "fma", "llvm.sqrt.f32": "sqrt", "llvm.fabs.f32": "fabs",
                  "llvm.smax.i32": None, "llvm.smin.i32": None, "llvm.umax.i32": "max", "llvm.umin.i32": "min",
                  "llvm.maxnum.f32": "fmax", "llvm.minnum.f32": "fmin", "llvm.floor.f32": "floor", "llvm.ceil.f32": "ceil",
                  "llvm.rint.f32": "rint", "llvm.trunc.f32": "trunc", "llvm.ctpop.i32": "popcount"}
        if fn in ("llvm.smax.i32", "llvm.smin.i32"):
            return f"uint({'max' if 'smax' in fn else 'min'}(int({av[0]}), int({av[1]})))"
        if fn in simple:
            return f"{simple[fn]}({', '.join(av)})"
        if fn.startswith("llvm.lifetime."):
            return ""
        raise Unsupported(f"call to '{fn}'")

    # -- the function ----------------------------------------------------------------------------
    def run(self) -> tuple[str, str]:
        fn = self.fn
        self.infer_spaces()
        phis = {}   # block -> [(name, type, {pred: value})]
        for b, insts in fn.blocks:
            for ins in insts:
                m = re.match(r"%([\w.$]+) = phi (?:" + FLAGS + r")*(" + TYPE + r") (.*)", ins)
                if m:
                    inc = {}
                    for x in re.findall(r"\[([^\]]*)\]", m.group(3)):
                        v, p = x.rsplit(",", 1)
                        inc[p.strip()[1:]] = v.strip()
                    phis.setdefault(b, []).append((m.group(1), m.group(2), inc))
        first = fn.blocks[0][0]

        def goto(pred, target, ind):
            lines = []
            ps = phis.get(target, [])
            pkeys = [pred] + ([str(len(fn.params))] if pred == first == "entry" else [])
            tmp = []
            for k, (n, t, inc) in enumerate(ps):
                v = next((inc[p] for p in pkeys if p in inc), None)
                if v is None:
                    raise Unsupported(f"phi %{n} has no value from block {pred}")
                lines.append(f"{ind}{self.mtype(t, n)} ir_t{k} = {self.val(t, v)};")
                tmp.append((n, k))
            lines += [f"{ind}{self.var(n)} = ir_t{k};" for n, k in tmp]
            lines.append(f"{ind}ir_bb = {self.label[target]}u;")
            return lines

        body = []
        for b, insts in fn.blocks:
            body.append(f"        case {self.label[b]}u: {{")
            for ins in insts:
                ind = "            "
                m = re.match(r"%([\w.$]+) = (.*)", ins)
                if m:
                    name, rhs = m.groups()
                    if rhs.startswith("phi "):
                        continue
                    e = self.expr(name, rhs)
                    body.append(f"{ind}{self.var(name)} = {e};")
                    continue
                op = re.sub(r"^(?:tail |musttail |notail )", "", ins)
                if op.startswith("store "):
                    r = re.sub(r"^store\s+(?:volatile\s+)?", "", op)
                    v, p = split_top(r)[:2]
                    vt, vv = take_type(v)
                    pt, pv = take_type(p)
                    body.append(f"{ind}*({self.ptr_space(pv)} {self.mtype(vt)} *)({self.val(pt, pv)}) = {self.val(vt, vv)};")
                elif op.startswith("call "):
                    e = self.call(re.sub(r"^call\s+(?:" + FLAGS + ")*", "", op))
                    if e:
                        body.append(f"{ind}{e};")
                elif op in ("ret void", "unreachable"):
                    body.append(f"{ind}ir_bb = IR_DONE;")
                elif op.startswith("br label "):
                    body += goto(b, op.split("%")[1].strip(), ind)
                elif op.startswith("br i1 "):
                    m2 = re.match(r"br i1 (\S+), label %([\w.$]+), label %([\w.$]+)", op)
                    c, t1, t2 = m2.groups()
                    body.append(f"{ind}if ({self.val('i1', c)}) {{")
                    body += goto(b, t1, ind + "    ")
                    body.append(f"{ind}}} else {{")
                    body += goto(b, t2, ind + "    ")
                    body.append(f"{ind}}}")
                elif op.startswith("switch "):
                    m2 = re.match(r"switch (i\d+) (\S+), label %([\w.$]+) \[(.*)\]", op)
                    t, v, dflt, cases = m2.groups()
                    first_case = True
                    for cv, cl in re.findall(r"i\d+ (-?\d+), label %([\w.$]+)", cases):
                        body.append(f"{ind}{'if' if first_case else '} else if'} ({self.val(t, v)} == {self.val(t, cv)}) {{")
                        body += goto(b, cl, ind + "    ")
                        first_case = False
                    body.append(f"{ind}}} else {{" if not first_case else f"{ind}{{")
                    body += goto(b, dflt, ind + "    ")
                    body.append(f"{ind}}}")
                else:
                    raise Unsupported(f"instruction '{op[:50]}'")
            body.append("            break; }")

        params = [(t, n[1:]) for t, n in fn.params]
        head = (f"kernel void {fn.name}(constant uchar *ocu_args [[buffer(0)]],\n"
                "    constant uint *ocu_offs [[buffer(1)]],\n"
                "    threadgroup uchar *ocu_dyn [[threadgroup(0)]],\n"
                "    uint3 ocu_tid [[thread_position_in_threadgroup]],\n"
                "    uint3 ocu_bid [[threadgroup_position_in_grid]],\n"
                "    uint3 ocu_bdim [[threads_per_threadgroup]],\n"
                "    uint3 ocu_gdim [[threadgroups_per_grid]])\n{")
        out = [head]
        for g in self.used_globs:
            _, t, _ = self.globs[g]
            m = re.fullmatch(r"\[(\d+) x (.+)\]", t)
            et, cnt = (m.group(2), int(m.group(1))) if m else (t, 1)
            out.append(f"    threadgroup {self.mtype(et)} g_{re.sub(r'[^A-Za-z0-9_]', '_', g)}[{cnt}];")
        sizes = []
        for i, (t, n) in enumerate(params):
            if t.startswith("ptr"):
                out.append(f"    device uchar *{self.var(n)} = ocu_ptr<device uchar *>(ocu_args, ocu_offs, {i});")
                sizes.append(8)
            else:
                out.append(f"    {self.mtype(t)} {self.var(n)} = ocu_arg<{self.mtype(t)}>(ocu_args, ocu_offs, {i});")
                sizes.append(SIZE[t])
        pnames = {n for _, n in params}
        for n, t in self.types.items():
            if n not in pnames and t != "void":
                out.append(f"    {self.mtype(t, n)} {self.var(n)};")
        # The loop ends through its condition. A `return` inside the loop
        # compiled, but on an Apple M3 only the first thread then stored its
        # result (kernel d4 of the trial); with the exit in the condition
        # every thread is right.
        out.append("    const uint IR_DONE = 0xffffffffu;")
        out.append("    uint ir_bb = 0u;")
        out.append("    while (ir_bb != IR_DONE) {")
        out.append("        switch (ir_bb) {")
        out += body
        out.append("        default: ir_bb = IR_DONE; break;")
        out.append("        }")
        out.append("    }")
        out.append("}")
        return "\n".join(out), f"param {fn.name} " + " ".join(map(str, sizes))


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-I", action="append", default=[])
    ap.add_argument("-D", action="append", default=[])
    ap.add_argument("--clang", default=os.environ.get("OCU_CLANG", "clang++"))
    ap.add_argument("--prelude", default=str(HERE / "msl_prelude.h"))
    a = ap.parse_args(argv)
    if a.source.endswith(".ll"):
        ll = open(a.source).read()
    else:
        sys.path.insert(0, str(HERE))
        import cu2msl
        flags = [f"-I{d}" for d in a.I] + [f"-D{d}" for d in a.D]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "device.ll")
            cmd = cu2msl.device_cmd(a.clang, flags) + ["-O2", "-fno-vectorize", "-fno-slp-vectorize",
                                                       "-emit-llvm", "-S", a.source, "-o", out]
            done = subprocess.run(cmd, capture_output=True, text=True, env=cu2msl.clang_env())
            if done.returncode != 0:
                print(f"{a.source}: clang failed (device IR):\n{done.stderr[-2000:]}", file=sys.stderr)
                return 1
            ll = open(out).read()
    globs, kernels, funcs = parse(ll)
    code, meta, bad = [], [], 0
    for fn in funcs:
        if fn.name not in kernels:
            print(f"skip {fn.name}: a device function that was not inlined", file=sys.stderr)
            bad += 1
            continue
        try:
            text, param = Emit(fn, globs).run()
        except Unsupported as e:
            print(f"skip {fn.name}: {e}", file=sys.stderr)
            bad += 1
            continue
        code.append(text)
        meta.append(param)
        print(f"ok {fn.name}")
    prelude = open(a.prelude).read()
    open(a.output, "w").write("OCUMSL1\n" + "\n".join(meta) + "\n---\n" + prelude + "\n// ---- kernels from LLVM IR\n" + "\n\n".join(code) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Region-loop census (analysis only): which CUDA kernels the planned
region-loop pass (docs/cpu-region-loops.md) could convert at stage 2, and
why the others are refused.

    region_census.py --corpus NAME --out DIR [-I dir] [-D X] [-j 2] [-O 2] FILE.cu...
    region_census.py --corpus NAME --ll FILE.ll...      (analyse existing IR)

Each .cu is compiled to NVPTX device IR (the nvidia target's device step
of compiler/probe/ocu_cc.py, plus -emit-llvm, sm_50 so __CUDA_ARCH__=500
as on the CPU path); the IR is cached in DIR/ll. The analysis runs in
Python over the textual IR and prints one TSV row per kernel (deduplicated
by mangled name and body) to stdout and DIR/census.tsv; a summary goes to
stderr.

Columns: corpus file kernel n_barriers n_warp_ops verdict primary reasons
mangled. verdict: no-cuts | convertible | refused.

Rules (after inlining, i.e. on -O2 IR):
 R1 every cut (barrier, warp op) is in the kernel body; refused: a call to
    a defined function that may synchronise, an indirect call, an unknown
    external call.
 R2 every cut is reached uniformly: each branch it is (transitively)
    control dependent on has a uniform condition. Uniform: kernel
    arguments, ctaid/ntid/nctaid, constants, loads through byval kernel
    parameters or from constant memory (addrspace 4 / immutable globals),
    and values computed only from these. Every other load, atomics, warp
    op results, inline asm and calls are divergent. Exception: a
    divergent branch where at most one successor can still reach a cut
    (the others only reach the exit: "inactive from here").
 R3 R2 for the exit branches of loops that contain a cut. A divergent exit
    that only reaches the function exit (early return inside a loop with a
    barrier) is refused too, as "R3 divergent early exit from cut loop".
 R4 warp ops use the constant full mask (-1); legacy, asm and implicit-
    mask warp ops are refused.
 R5 no dynamic alloca (any in a kernel with cuts is counted, liveness is
    not checked).
 R6 no FP-control change, no call that looks up the current thread.

Limits: this is the NVPTX device IR of our headers, not the CPU-path IR the
real pass will see (different inlining, ocu_cpu_* calls instead of
intrinsics); divergence is a simple forward analysis (sync dependence of
phis approximated by the region between a divergent branch and its
post-dominator, temporal divergence by loop live-outs); loads are never
uniform, so kernels whose barriers depend on a value loaded once from
global memory are refused although they would pass a run-time check;
uniformity is per block, so warp ops under a warp-uniform but not
block-uniform branch (`if (tid < 32) warp_reduce`) are refused; code that
a source compiles only for newer architectures (NO_DEVICE_CODE stubs at
sm_50) shows up as the stub; inlining decisions follow -O2 for NVPTX.
No code is copied from llama.cpp or Rodinia.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "compiler" / "probe"))
import ocu_cc  # noqa: E402

# ---------------------------------------------------------------- compile


def compile_ll(src: Path, out: Path, inc: list[str], defs: list[str], opt: str, std: str,
               arch: str = "sm_50") -> tuple[Path, str]:
    """Device IR of one .cu through ocu_cc's nvidia device step."""
    tag = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:8]
    ll = out / "ll" / f"{src.stem}.{tag}.ll"
    if ll.exists() and ll.stat().st_mtime > src.stat().st_mtime:
        return ll, ""
    work = out / "work" / tag
    work.mkdir(parents=True, exist_ok=True)
    ll.parent.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(target="nvidia", arch=arch, I=[str(src.parent), *inc], D=defs, O=opt,
                           g=False, cache=str(out / "cache"), std=std, pre=[], host=[],
                           device=["-emit-llvm", "-Wno-everything"], deps=[], ptx_shift=False)
    cmds = ocu_cc.build_commands(args, src, work / (src.stem + ".o"), work)
    dev = next(c for c in cmds if "--cuda-device-only" in c)
    dev = list(dev)
    dev[dev.index("-o") + 1] = str(ll)
    r = subprocess.run(dev, capture_output=True, text=True, env=ocu_cc.clean_env())
    if r.returncode:
        ll.unlink(missing_ok=True)
        (out / "ll" / f"{src.stem}.{tag}.err").write_text(" ".join(dev) + "\n" + r.stderr)
        return None, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "failed"
    return ll, ""


# ---------------------------------------------------------------- IR model

STR = re.compile(r'c?"(?:[^"\\]|\\.)*"')
VAL = re.compile(r'%[-\w.$]+')
LABEL_REF = re.compile(r'label (%[-\w.$]+)')
CALLEE = re.compile(r'\bcall\b[^@%"]*?(?:(asm)\b|@([-\w.$]+)\(|(%[-\w.$]+)\()')

BARRIERS = re.compile(r'llvm\.nvvm\.(barrier0|barrier\.sync|barrier\.n|barrier$|barrier\.cta|bar\.sync)')
WARP = re.compile(r'llvm\.nvvm\.(bar\.warp\.sync|shfl\.|vote\.|match\.|redux\.sync|wmma\.|mma\.|ldmatrix)')
UNIFORM_SREG = re.compile(r'llvm\.nvvm\.read\.ptx\.sreg\.(ctaid|ntid|nctaid|nctaid|warpsize)')
MEM_INTRIN = re.compile(r'(ldg|ldu|load|atom|cp\.async|read\.ptx\.sreg|shfl|vote|match|redux)')
SAFE_EXTERN = re.compile(r'^(vprintf|printf|malloc|free|__assertfail|__nv_\w+|memcpy|memset|memmove)$')
FP_CONTROL = re.compile(r'(fesetround|fesetenv|feupdateenv|fesetexceptflag|_controlfp|llvm\.set\.rounding'
                        r'|llvm\.set\.fpenv|llvm\.reset\.fpenv)')
THREAD_LOOKUP = re.compile(r'(pthread_self|omp_get_thread_num|ocu_cpu_self|ocu_cpu_current|gettid)')
PTR_OPND = re.compile(r'\bptr(addrspace\(\d+\))? ([%@][-\w.$]+)')
ASM_SYNC = re.compile(r'\b(bar\.sync|barrier|bar\.warp|shfl|vote\.|match\.|redux\.)')


class Inst:
    __slots__ = ("res", "op", "text", "ops", "phi", "callee", "kind", "block")

    def __init__(self, line: str, block: str):
        self.block = block
        m = re.match(r'\s*(%[-\w.$]+)\s*=\s*(.*)$', line)
        self.res, body = (m.group(1), m.group(2)) if m else (None, line.strip())
        self.text = body
        clean = STR.sub('""', body.split(", !")[0] if "!" in body else body)
        self.op = re.sub(r'^(tail|musttail|notail)\s+', '', clean).split(" ", 1)[0]
        self.phi = None
        if self.op == "phi":
            self.phi = re.findall(r'\[\s*([^,\]]+?),\s*(%[-\w.$]+)\s*\]', clean)
            self.ops = [v.strip() for v, _ in self.phi if v.strip().startswith("%")]
        else:
            labels = set(LABEL_REF.findall(clean))
            self.ops = [v for v in VAL.findall(clean) if v not in labels]
        self.callee, self.kind = None, None
        if self.op == "call":
            c = CALLEE.search(body)
            if c and c.group(1):
                self.kind, self.callee = "asm", body
            elif c and c.group(2):
                self.kind, self.callee = "direct", c.group(2)
            else:
                self.kind = "indirect"


class Func:
    def __init__(self, name: str, header: str):
        self.name, self.header = name, header
        self.blocks: dict[str, list[Inst]] = {}
        self.order: list[str] = []
        self.byval = set()
        self.args = set()
        params = header[header.index("(") + 1: header.rindex(")")]
        depth, cur, parts = 0, "", []
        for ch in params:
            depth += ch in "([{<"
            depth -= ch in ")]}>"
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        parts.append(cur)
        for p in parts:
            names = VAL.findall(STR.sub("", p))
            if names:
                self.args.add(names[-1])
                if "byval" in p:
                    self.byval.add(names[-1])
        self.body_hash = ""

    def succs(self, b):
        t = self.blocks[b][-1]
        if t.op in ("br", "switch", "indirectbr", "invoke", "callbr"):
            return [x for x in dict.fromkeys(LABEL_REF.findall(t.text)) if x in self.blocks]
        return []


def parse(text: str):
    funcs, kernels, consts = {}, set(), set()
    for m in re.finditer(r'^@([-\w.$]+) = [^\n]*?\bconstant\b', text, re.M):
        consts.add(m.group(1))
    for m in re.finditer(r'^@([-\w.$]+) = [^\n]*addrspace\(4\)', text, re.M):
        consts.add(m.group(1))
    for m in re.finditer(r'!\{ptr @([-\w.$]+), !"kernel", i32 1\}', text):
        kernels.add(m.group(1))
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("define "):
            name = re.search(r'@([-\w.$]+|"[^"]+")\(', ln).group(1)
            if "ptx_kernel" in ln.split("@")[0]:
                kernels.add(name)
            f = Func(name, ln)
            cur = "%entry.0"
            f.blocks[cur], f.order = [], [cur]
            body = []
            i += 1
            while not lines[i].startswith("}"):
                s = lines[i]
                body.append(s)
                lm = re.match(r'^([-\w.$]+):', s)
                if lm:
                    cur = "%" + lm.group(1)
                    f.blocks[cur] = []
                    f.order.append(cur)
                elif s.startswith("  ") and not s.strip().startswith(";"):
                    if s.rstrip().endswith("[") or s.strip().startswith("switch"):  # multi-line switch
                        while "]" not in s.split("[", 1)[-1]:
                            i += 1
                            s += " " + lines[i].strip()
                    f.blocks[cur].append(Inst(s, cur))
                i += 1
            if not f.blocks["%entry.0"]:
                del f.blocks["%entry.0"]
                f.order.pop(0)
            f.body_hash = hashlib.sha1("\n".join(body).encode()).hexdigest()[:12]
            funcs[name] = f
        i += 1
    return funcs, kernels, consts


# ---------------------------------------------------------------- graphs


def idoms(nodes, entry, succ, pred):
    """Cooper-Harvey-Kennedy immediate dominators over the nodes reachable from entry."""
    order, seen, stack = [], {entry}, [(entry, iter(succ(entry)))]
    while stack:
        n, it = stack[-1]
        for s in it:
            if s not in seen:
                seen.add(s)
                stack.append((s, iter(succ(s))))
                break
        else:
            order.append(n)
            stack.pop()
    order.reverse()
    num = {n: k for k, n in enumerate(order)}
    idom = {entry: entry}
    changed = True
    while changed:
        changed = False
        for n in order[1:]:
            ps = [p for p in pred(n) if p in idom]
            if not ps:
                continue
            new = ps[0]
            for p in ps[1:]:
                a, b = p, new
                while a != b:
                    while num[a] > num[b]:
                        a = idom[a]
                    while num[b] > num[a]:
                        b = idom[b]
                new = a
            if idom.get(n) != new:
                idom[n] = new
                changed = True
    return idom


def reach(start, succ, stop=None):
    seen, todo = set(), list(start)
    while todo:
        n = todo.pop()
        if n in seen or n == stop:
            continue
        seen.add(n)
        todo.extend(succ(n))
    return seen


class Graph:
    EXIT = "<exit>"

    def __init__(self, f: Func):
        self.f = f
        self.succ = {b: f.succs(b) for b in f.order}
        self.pred = {b: [] for b in f.order}
        for b, ss in self.succ.items():
            for s in ss:
                self.pred[s].append(b)
        entry = f.order[0]
        self.dom = idoms(f.order, entry, lambda n: self.succ[n], lambda n: self.pred[n])
        # Post-dominators: exits are blocks without successors; blocks that cannot
        # reach an exit (endless loops) are tied to the virtual exit too.
        exits = [b for b in f.order if not self.succ[b]]
        can = reach(exits, lambda n: self.pred[n])
        self.rsucc = {b: list(self.succ[b]) or [self.EXIT] for b in f.order}
        for b in f.order:
            if b not in can:
                self.rsucc[b] = self.rsucc[b] + [self.EXIT]
        rpred = {self.EXIT: [b for b in f.order if self.EXIT in self.rsucc[b]]}
        for b in f.order:
            rpred[b] = list(self.pred[b])
        self.pdom = idoms(f.order + [self.EXIT], self.EXIT, lambda n: rpred[n], lambda n: self.rsucc.get(n, []))
        # Control dependence (Ferrante et al.): edge A->B, walk B up the post-dominator tree to ipdom(A).
        self.cd = {b: set() for b in f.order}
        for a in f.order:
            if len(self.succ[a]) < 2:
                continue
            for b in self.succ[a]:
                n = b
                while n != self.pdom.get(a) and n != self.EXIT and n in self.pdom:
                    self.cd[n].add(a)
                    n = self.pdom[n]
        # Natural loops from back edges.
        self.loops = []
        for b in f.order:
            for h in self.succ[b]:
                if self.dominates(h, b):
                    body = {h} | reach([b], lambda n: self.pred[n], stop=h)
                    self.loops.append((h, body))

    def dominates(self, a, b):
        while True:
            if a == b:
                return True
            if b not in self.dom or self.dom[b] == b:
                return False
            b = self.dom[b]

    def cd_closure(self, b):
        return reach(self.cd[b], lambda n: self.cd[n])


# ---------------------------------------------------------------- analysis


def cut_kind(i: Inst):
    """'barrier' | 'warp' | None for one instruction."""
    if i.kind == "direct":
        if BARRIERS.match(i.callee):
            return "barrier"
        if WARP.match(i.callee):
            return "warp"
    if i.kind == "asm" and ASM_SYNC.search(i.text.split('"')[1] if '"' in i.text else ""):
        return "barrier" if re.search(r'bar\.sync|barrier', i.text) and "warp" not in i.text else "warp"
    return None


def mask_reason(i: Inst):
    """R4 reason for a warp op, or None for a constant full mask."""
    if i.kind == "asm":
        return "R4 warp op in inline asm"
    name = i.callee
    if not re.search(r'\.sync', name):
        return f"R4 implicit-mask warp op: {name}"
    args = re.search(r'\(([^()]*)\)', i.text[i.text.index(name) + len(name):])
    vals = [a.strip() for a in args.group(1).split(",")] if args else []
    k = 1 if "redux.sync" in name else 0
    m = vals[k].split()[-1] if k < len(vals) else ""
    if m == "-1":
        return None
    return "R4 computed mask" if m.startswith("%") else f"R4 partial mask: {m}"


def may_sync_funcs(funcs):
    """Defined functions that may synchronise (cut, indirect or unknown call), transitively."""
    calls = {n: set() for n in funcs}
    bad = set()
    for n, f in funcs.items():
        for b in f.order:
            for i in f.blocks[b]:
                if cut_kind(i) or i.kind == "indirect":
                    bad.add(n)
                elif i.kind == "direct":
                    if i.callee in funcs:
                        calls[n].add(i.callee)
                    elif not i.callee.startswith("llvm.") and not SAFE_EXTERN.match(i.callee):
                        bad.add(n)
    changed = True
    while changed:
        changed = False
        for n in funcs:
            if n not in bad and calls[n] & bad:
                bad.add(n)
                changed = True
    return bad


def divergence(f: Func, g: Graph, consts):
    insts = [i for b in f.order for i in f.blocks[b]]
    defs = {i.res: i for i in insts if i.res}
    users = {}
    for i in insts:
        for o in i.ops:
            users.setdefault(o, []).append(i)
    # Pointers into uniform memory: byval parameters, constant memory, immutable globals.
    uptr = set(f.byval)

    def uniform_base(i: Inst):
        m = PTR_OPND.search(STR.sub('""', i.text))
        if not m:
            return False
        base = m.group(2)
        return "addrspace(4)" in (m.group(1) or "") or base in uptr or base[1:] in consts

    for i in insts:
        if i.res and i.op in ("getelementptr", "bitcast", "addrspacecast") and uniform_base(i):
            uptr.add(i.res)

    def seed(i: Inst):
        if i.op == "load":
            return not uniform_base(i)
        if i.op in ("atomicrmw", "cmpxchg", "va_arg", "landingpad"):
            return True
        if i.op == "call" and i.res:
            if i.kind != "direct":
                return True
            if UNIFORM_SREG.match(i.callee) or re.match(r'llvm\.nvvm\.barrier0\.(popc|and|or)', i.callee):
                return False
            if i.callee.startswith("llvm.") and not MEM_INTRIN.search(i.callee):
                return False  # pure intrinsic: operand rule
            return True
        return False

    div = set()
    work = []

    def mark(v):
        if v and v not in div:
            div.add(v)
            work.append(v)

    for i in insts:
        if i.res and seed(i):
            mark(i.res)
    done_branches = set()
    while True:
        while work:
            v = work.pop()
            for u in users.get(v, []):
                if u.res and u.res not in div:
                    if u.op == "call" and u.kind == "direct" and (UNIFORM_SREG.match(u.callee) or BARRIERS.match(u.callee)):
                        continue
                    mark(u.res)
        new = False
        for b in f.order:
            t = f.blocks[b][-1]
            if b in done_branches or t.op not in ("br", "switch") or len(g.succ[b]) < 2:
                continue
            if not any(o in div for o in t.ops):
                continue
            done_branches.add(b)
            new = True
            # Sync dependence: phis between the branch and its post-dominator.
            p = g.pdom.get(b)
            region = reach(g.succ[b], lambda n: g.succ[n], stop=p)
            if p in f.blocks:
                region.add(p)
            for r in region:
                for i in f.blocks[r]:
                    if i.op == "phi":
                        mark(i.res)
            # Temporal divergence: values of a loop the branch can leave, used outside it.
            for h, body in g.loops:
                if b in body and region - body:
                    for r in body:
                        for i in f.blocks[r]:
                            if i.res and any(u.block not in body for u in users.get(i.res, [])):
                                mark(i.res)
        if not new and not work:
            return div


def classify(f: Func, funcs, maysync, consts):
    nb = nw = 0
    reasons = []
    cut_blocks = set()
    dyn_alloca = False
    for b in f.order:
        for i in f.blocks[b]:
            k = cut_kind(i)
            if k:
                cut_blocks.add(b)
                nb += k == "barrier"
                nw += k == "warp"
                if k == "warp":
                    r = mask_reason(i)
                    if r:
                        reasons.append(r)
            if i.op == "call":
                if i.kind == "indirect":
                    reasons.append("R1 indirect call")
                elif i.kind == "direct":
                    c = i.callee
                    if FP_CONTROL.search(c):
                        reasons.append(f"R6 FP control: {c}")
                    elif THREAD_LOOKUP.search(c):
                        reasons.append(f"R6 thread lookup: {c}")
                    elif c in funcs:
                        if c in maysync:
                            reasons.append(f"R1 may-sync call: {c}")
                    elif not c.startswith("llvm.") and not SAFE_EXTERN.match(c):
                        reasons.append(f"R1 unknown external: {c}")
                    if c.startswith("llvm.stacksave"):
                        dyn_alloca = True
            if i.op == "alloca" and (b != f.order[0] or re.search(r',\s*i\d+\s+%', i.text)):
                dyn_alloca = True
    has_cut = bool(cut_blocks) or any(r.startswith("R1 may-sync") for r in reasons)
    if not has_cut:
        return nb, nw, ("no-cuts" if not reasons else "refused"), sorted(set(reasons))
    if dyn_alloca:
        reasons.append("R5 dynamic alloca")
    if cut_blocks:
        g = Graph(f)
        div = divergence(f, g, consts)
        reach_cut = {}

        def side_has_cut(s):
            if s not in reach_cut:
                reach_cut[s] = bool(reach([s], lambda n: g.succ[n]) & cut_blocks)
            return reach_cut[s]

        def divergent(d):
            return any(o in div for o in f.blocks[d][-1].ops)

        # R3: every exit branch of a loop that contains a cut is uniform. A divergent
        # exit that only reaches the function exit (early return inside the loop) is
        # still refused, but reported apart: the per-thread trip count it makes needs
        # the active-set loop of a later stage.
        loop_exits = set()
        for _, body in g.loops:
            if not body & cut_blocks:
                continue
            for d in body:
                outs = [s for s in g.succ[d] if s not in body]
                if not outs:
                    continue
                loop_exits.add(d)
                if divergent(d):
                    reasons.append("R3 divergent early exit from cut loop"
                                   if not any(side_has_cut(s) for s in outs)
                                   else "R3 divergent loop trip count")
        # R2: the branches each cut is control dependent on (transitively).
        for c in sorted(cut_blocks, key=f.order.index):
            for d in g.cd_closure(c):
                if d in loop_exits or not divergent(d):
                    continue
                if sum(side_has_cut(s) for s in g.succ[d]) <= 1:
                    continue  # divergent side only reaches the exit: inactive from here
                reasons.append("R2 divergent branch before cut")
    order = {f"R{k}": k for k in range(1, 7)}
    reasons = sorted(set(reasons), key=lambda r: (order[r[:2]], r))
    return nb, nw, ("refused" if reasons else "convertible"), reasons


def demangle(names):
    if not names:
        return {}
    try:
        r = subprocess.run(["llvm-cxxfilt"], input="\n".join(names), capture_output=True, text=True)
        out = r.stdout.splitlines()
        if len(out) == len(names):
            return dict(zip(names, out))
    except OSError:
        pass
    return {n: n for n in names}


def analyse(ll: Path, label: str):
    funcs, kernels, consts = parse(ll.read_text(errors="replace"))
    maysync = may_sync_funcs(funcs)
    rows = []
    for k in sorted(kernels):
        if k not in funcs:
            continue
        nb, nw, verdict, reasons = classify(funcs[k], funcs, maysync, consts)
        rows.append([label, k, funcs[k].body_hash, nb, nw, verdict, reasons])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", help="scratch folder (IR cache, census.tsv)")
    ap.add_argument("--ll", action="store_true", help="inputs are .ll files")
    ap.add_argument("-I", action="append", default=[])
    ap.add_argument("-D", action="append", default=[])
    ap.add_argument("-O", default="2")
    ap.add_argument("--std", default="c++17")
    ap.add_argument("--arch", default="sm_50", help="sm_50: __CUDA_ARCH__ 500, as on the CPU path")
    ap.add_argument("-j", type=int, default=2)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    out = Path(a.out or ".").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    srcs = [Path(x) for x in a.files]
    lls, fails = [], []
    if a.ll:
        lls = [(s, s) for s in srcs]
    else:
        ocu_cc.pinned_cuda_path(out / "cache")  # once: its temp name is per process
        with cf.ThreadPoolExecutor(a.j) as ex:
            futs = {ex.submit(compile_ll, s, out, a.I, a.D, a.O, a.std, a.arch): s for s in srcs}
            for fu in cf.as_completed(futs):
                ll, err = fu.result()
                if ll:
                    lls.append((futs[fu], ll))
                else:
                    fails.append((futs[fu], err))
                    print(f"compile failed: {futs[fu]}: {err}", file=sys.stderr)
    common = os.path.commonpath([str(s.resolve().parent) for s, _ in lls]) if lls else ""
    seen, rows = set(), []
    for s, ll in sorted(lls):
        for r in analyse(ll, os.path.relpath(s.resolve(), common)):
            key = (r[1], r[2])
            if key not in seen:
                seen.add(key)
                rows.append(r)
    names = demangle([r[1] for r in rows])
    lines = ["corpus\tfile\tkernel\tn_barriers\tn_warp_ops\tverdict\tprimary\treasons\tmangled"]
    for file, m, _, nb, nw, v, rs in rows:
        lines.append("\t".join([a.corpus, file, names.get(m, m), str(nb), str(nw), v,
                                rs[0] if rs else "", "; ".join(rs), m]))
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    (out / "census.tsv").write_text(text)
    # Summary
    from collections import Counter
    verdicts = Counter(r[5] for r in rows)
    prim = Counter(r[6][0].split(" ")[0] for r in rows if r[6])
    allr = Counter(c for r in rows for c in {x.split(": ")[0] for x in r[6]})
    print(f"{a.corpus}: files {len(srcs)} (compile failed {len(fails)}), kernels {len(rows)}: "
          f"no-cuts {verdicts['no-cuts']}, convertible {verdicts['convertible']}, "
          f"refused {verdicts['refused']}", file=sys.stderr)
    print(f"  primary: {dict(prim)}", file=sys.stderr)
    print(f"  all reasons: {dict(allr)}", file=sys.stderr)


if __name__ == "__main__":
    main()

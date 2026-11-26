#!/usr/bin/env python3
"""cu2spv: CUDA device code to SPIR-V by compilation (not translation).

Clang compiles the .cu file in HIP mode for spirv64 (--offload=spirv64),
with OpenCuda's headers (runtime/include, ocu_cuda_lang_spirv.h): the whole
C++ language works (templates, namespaces, member functions, lambdas,
function-pointer template arguments), because Clang compiles it and SPIR-V
has a generic address space. Then the LLVM IR is adjusted for the OpenCL
backend and LLVM's SPIR-V back end (llc) writes SPIR-V:

  - extern __shared__ arrays (the launch's dynamic shared memory) become one
    extra kernel parameter in local memory, set by the backend;
  - kernels that use sub-group operations get intel_reqd_sub_group_size(32),
    so a sub-group is a CUDA warp;
  - each kernel's parameters are listed (pointer or value) for the backend.

    cu2spv.py <file.cu> -o <image> [-I dir] [-D name] [--arch sm_XY]

Image (read by backends/opencl/ocl_backend.c): "OCUSPV1", then one line per
kernel "sparam <name> <p8|vN>... [dyn]" (pointer, or value of N bytes), "spirv <bytes>", "---", the module.
"""
import argparse, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RT = REPO / "runtime" / "include"


def tool(name, env):
    p = os.environ.get(env)
    if p:
        return p
    found = shutil.which(name)
    if not found:
        sys.exit(f"cu2spv: {name} not found (set {env})")
    return found


def split_args(sig: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in sig:
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


class Layout:
    """Sizes and alignments of IR types (spirv64: natural alignment)."""
    def __init__(self, ll: str):
        self.defs = {}
        for m in re.finditer(r'^(%(?:"[^"]+"|[\w.$]+)) = type (<?\{.*\}>?)$', ll, re.M):
            self.defs[m.group(1)] = m.group(2)

    def split(self, body: str) -> list[str]:
        return split_args(body)

    def size_align(self, t: str) -> tuple[int, int]:
        t = t.strip()
        prim = {"i1": (1, 1), "i8": (1, 1), "i16": (2, 2), "i32": (4, 4), "i64": (8, 8), "half": (2, 2),
                "bfloat": (2, 2), "float": (4, 4), "double": (8, 8)}
        if t in prim:
            return prim[t]
        if t.startswith("ptr"):
            return 8, 8
        if t in self.defs:
            return self.size_align(self.defs[t])
        m = re.fullmatch(r"\[(\d+) x (.+)\]", t)
        if m:
            sz, al = self.size_align(m.group(2))
            return int(m.group(1)) * sz, al
        m = re.fullmatch(r"<(\d+) x (.+)>", t)
        if m:
            sz, al = self.size_align(m.group(2))
            n = int(m.group(1))
            v = n * sz
            p2 = 1
            while p2 < v:
                p2 *= 2
            return p2, p2
        packed = t.startswith("<{")
        m = re.fullmatch(r"<?\{(.*)\}>?", t)
        if m:
            off, al = 0, 1
            for f in self.split(m.group(1)) if m.group(1).strip() else []:
                fs, fa = self.size_align(f)
                if packed:
                    fa = 1
                off = (off + fa - 1) // fa * fa + fs
                al = max(al, fa)
            return (off + al - 1) // al * al, al
        raise SystemExit(f"cu2spv: no layout for parameter type {t}")


def param_info(p: str, lay: Layout) -> str:
    """'p8' for a global pointer, 'vN' for a value of N bytes (byval structs too)."""
    if p.startswith("ptr addrspace(1)") and "byval" not in p:
        return "p8"
    m = re.search(r"byval\((.+?)\)(?:\s|$)", p)
    if m:
        return f"v{lay.size_align(m.group(1))[0]}"
    t = p.split(" ")[0]
    if t == "ptr":
        return "v8"
    return f"v{lay.size_align(t)[0]}"


SG_CALL = re.compile(r"@_Z\d+(sub_group_\w+|get_sub_group_local_id\w*|intel_sub_group\w+)")


def rewrite(ll: str) -> tuple[str, list[str]]:
    # the ThinLTO summary (lines "^N = ...") is optional and names globals this rewrite removes
    lines = [l for l in ll.split("\n") if not l.startswith("^")]
    lay = Layout(ll)
    dyn = set()
    for l in lines:
        m = re.match(r"^(@[\w.$\"]+) = external (?:\w+ )*addrspace\(3\) global \[0 x ", l)
        if m:
            dyn.add(m.group(1))
    # functions: name -> (start, end) line indices
    funcs, i = {}, 0
    while i < len(lines):
        m = re.match(r'^define .*?@("[^"]+"|[\w.$]+)\(', lines[i])
        if m:
            j = i
            while not lines[j].startswith("}"):
                j += 1
            funcs[m.group(1)] = (i, j)
            i = j
        i += 1
    calls = {}
    for f, (a, b) in funcs.items():
        body = "\n".join(lines[a + 1:b])
        calls[f] = set(re.findall(r"call [^@]*@([\w.$\"]+)\(", body))
    direct_sg = {f for f, (a, b) in funcs.items() if SG_CALL.search("\n".join(lines[a + 1:b]))}
    def uses_sg(f, seen):
        if f in seen:
            return False
        seen.add(f)
        return f in direct_sg or any(uses_sg(c, seen) for c in calls.get(f, ()) if c in funcs)
    meta, sg_md, sg16 = [], False, False
    other_sg = os.environ.get("OCU_SPV_SUBGROUP_OTHER", "16") == "16"

    def body_of(f):
        a, b = funcs[f]
        return "\n".join(lines[a + 1:b])
    # functions that use the dynamic shared memory, and every function on a call
    # path from a kernel to one of them: each gets the pointer as a parameter
    direct = {f for f in funcs if any(re.search(re.escape(g) + r"\b", body_of(f)) for g in dyn)}
    need = set(direct)
    changed = True
    while changed:
        changed = False
        for f in funcs:
            if f not in need and calls.get(f, set()) & need:
                need.add(f)
                changed = True

    def close_paren(text, lo):
        depth, k = 1, lo
        while depth:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        return k - 1  # index of the matching ')'

    for f in need:
        a, b = funcs[f]
        head = lines[a]
        lo = head.index("(", head.index("@" + f)) + 1
        hi = close_paren(head, lo)
        empty = head[lo:hi].strip() == ""
        lines[a] = head[:hi] + ("" if empty else ", ") + "ptr addrspace(3) noundef %ocu_dyn" + head[hi:]
        casts, geps, geps2 = set(), {}, {}
        for k in range(a + 1, b):
            # calls to functions that now take the pointer: pass it on
            for callee in need:
                tag = "@" + callee + "("
                at = 0
                while True:
                    c = lines[k].find(tag, at)
                    if c < 0:
                        break
                    q = c + len(tag)
                    e = close_paren(lines[k], q)
                    sep = "" if lines[k][q:e].strip() == "" else ", "
                    lines[k] = lines[k][:e] + sep + "ptr addrspace(3) %ocu_dyn" + lines[k][e:]
                    at = e
            for g in dyn:
                eg = re.escape(g)
                def as_cast(m):
                    casts.add(m.group(1))
                    return f"%ocu_dyn_as{m.group(1)}"
                lines[k] = re.sub(r"addrspacecast \(ptr addrspace\(3\) " + eg + r" to ptr addrspace\((\d+)\)\)", as_cast, lines[k])
                def as_gep(m):
                    key = m.group(0)
                    if key not in geps:
                        geps[key] = (f"%ocu_dyn_gep{len(geps)}", m.group(1), m.group(2), m.group(3))
                    return geps[key][0]
                lines[k] = re.sub(r"getelementptr (inbounds |inbounds nuw |nuw )?\(([^(),]+), ptr addrspace\(3\) " + eg + r", ([^()]+)\)", as_gep, lines[k])
                if re.search(r"\(\s*ptr addrspace\(3\) " + eg + r"\b", lines[k]):
                    raise SystemExit(f"cu2spv: extern __shared__ {g} inside a constant expression not handled yet: {lines[k].strip()[:160]}")
                lines[k] = re.sub(eg + r"\b", "%ocu_dyn", lines[k])
            while True:
                def as_gep2(m):
                    key = m.group(0)
                    if key not in geps2:
                        geps2[key] = (f"%ocu_dyn_g{len(geps2)}", m.group(1) or "", m.group(2), m.group(3), m.group(4), m.group(5))
                    return geps2[key][0]
                new = re.sub(r"getelementptr (inbounds nuw |inbounds |nuw )?\(([^(),]+), ptr addrspace\((\d+)\) (%ocu_dyn\w*), ([^()]+)\)", as_gep2, lines[k])
                if new == lines[k]:
                    break
                lines[k] = new
        entry = [f"  %ocu_dyn_as{n} = addrspacecast ptr addrspace(3) %ocu_dyn to ptr addrspace({n})" for n in sorted(casts)]
        entry += [f"  {v} = getelementptr {ib or ''}{ty}, ptr addrspace(3) %ocu_dyn, {idx}" for v, ib, ty, idx in geps.values()]
        entry += [f"  {v} = getelementptr {ib}{ty}, ptr addrspace({asp}) {base}, {idx}" for v, ib, ty, asp, base, idx in geps2.values()]
        if entry:
            lines[a + 1] = "\n".join(entry) + "\n" + lines[a + 1]

    for f, (a, b) in funcs.items():
        head = lines[a]
        if " spir_kernel " not in head:
            continue
        lo = head.index("(", head.index("@" + f)) + 1
        hi = close_paren(head, lo)
        params = split_args(head[lo:hi])
        uses_dyn = f in need
        if uses_dyn:
            params = params[:-1]  # the added pointer is not a CUDA parameter
        kinds = [param_info(p, lay) for p in params]
        if uses_sg(f, set()):
            head = re.sub(r"\s*\{\s*$", " !intel_reqd_sub_group_size !900001 {", head)
            sg_md = True
        elif other_sg:
            # no warp-wide operation: 16-wide sub-groups keep twice the registers per work-item
            head = re.sub(r"\s*\{\s*$", " !intel_reqd_sub_group_size !900002 {", head)
            sg16 = True
        lines[a] = head
        meta.append(f"sparam {f} " + " ".join(kinds) + (" dyn" if uses_dyn else ""))
    out = "\n".join(lines)
    # the dynamic-shared globals must no longer be used outside kernels
    for g in dyn:
        rest = [l for l in out.split("\n") if re.search(re.escape(g) + r"\b", l) and not l.startswith(g + " =")]
        if rest:
            raise SystemExit(f"cu2spv: extern __shared__ {g} is still used outside the functions given its pointer")
        out = "\n".join(l for l in out.split("\n") if not l.startswith(g + " ="))
    if sg_md:
        out += "\n!900001 = !{i32 32}\n"
    if sg16:
        out += "\n!900002 = !{i32 16}\n"
    return out, meta


GLOBAL_P2I = re.compile(r"^\s*(%[\w.]+) = ptrtoint ptr addrspace\(1\) (%[\w.]+) to i64$", re.M)
GENERIC_I2P = re.compile(r"^\s*(%[\w.]+) = inttoptr i64 (%[\w.]+) to ptr addrspace\(4\)$", re.M)


def global_pointers(ll: str) -> str:
    """Clang's HIP mode turns each global pointer argument into a generic one
    through an integer (ptrtoint + inttoptr). That hides the address space:
    every load and store becomes a generic one, which the driver must test
    for local memory at run time. As an addrspacecast instead, LLVM's
    InferAddressSpaces gives the accesses back their global address space."""
    # Value names are local to a function: one function at a time.
    parts = re.split(r"(?m)^(?=define )", ll)
    out = []
    for part in parts:
        p2i = dict(GLOBAL_P2I.findall(part))

        def cast(m):
            dst, src = m.group(1), m.group(2)
            if src not in p2i:
                return m.group(0)
            return f"  {dst} = addrspacecast ptr addrspace(1) {p2i[src]} to ptr addrspace(4)"
        out.append(GENERIC_I2P.sub(cast, part) if p2i else part)
    return "".join(out)


DIV64 = re.compile(r"^(\s*)(%[\w.]+) = (sdiv|udiv|srem|urem)(?: exact)? i64 ([^,]+), (.+)$", re.M)


def _div_helper(op: str) -> str:
    # 32-bit fast path when both operands are small and non-negative (the
    # signed ops need < 2^31), as NVPTX's BypassSlowDivision does: Intel GPUs
    # have no 64-bit divide, the driver emulates it.
    u = op[0] == "u"
    shift = 32 if u else 31
    op32 = "udiv" if op.endswith("div") else "urem"
    return (f"define internal i64 @__ocu_{op}64(i64 %a, i64 %b) alwaysinline {{\n"
            f"  %o = or i64 %a, %b\n  %h = lshr i64 %o, {shift}\n  %s = icmp eq i64 %h, 0\n"
            f"  br i1 %s, label %f, label %l\nf:\n  %a32 = trunc i64 %a to i32\n  %b32 = trunc i64 %b to i32\n"
            f"  %q32 = {op32} i32 %a32, %b32\n  %q = zext i32 %q32 to i64\n  ret i64 %q\n"
            f"l:\n  %q2 = {op} i64 %a, %b\n  ret i64 %q2\n}}\n")


def bypass_div64(ll: str) -> str:
    used = set()

    def call(m):
        used.add(m.group(3))
        return f"{m.group(1)}{m.group(2)} = call i64 @__ocu_{m.group(3)}64(i64 {m.group(4)}, i64 {m.group(5)})"
    ll = DIV64.sub(call, ll)
    return ll + "\n" + "".join(_div_helper(op) for op in sorted(used))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", required=True)
    ap.add_argument("-I", action="append", default=[])
    ap.add_argument("-D", action="append", default=[])
    ap.add_argument("--arch", default=os.environ.get("OCU_CU_DEVICE_ARCH", "sm_80"))
    ap.add_argument("--keep", help="folder to keep the .ll files in")
    a = ap.parse_args(argv)
    clang = tool("clang++", "OCU_SPV_CLANG")
    llvm_as = tool("llvm-as", "OCU_SPV_LLVM_AS")
    llc = tool("llc", "OCU_SPV_LLC")
    # SPIR-V extensions the OpenCL drivers take (all of them would add ones a driver refuses)
    ext = os.environ.get("OCU_SPV_EXT", "+SPV_EXT_shader_atomic_float_add,+SPV_KHR_no_integer_wrap_decoration,+SPV_INTEL_subgroups,+SPV_KHR_integer_dot_product")
    arch = int(re.sub(r"\D", "", a.arch) or 80) * 10
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(a.keep) if a.keep else Path(tmp)
        t.mkdir(parents=True, exist_ok=True)
        bc0, ll, ll2, bc, spv = t / "dev0.bc", t / "dev.ll", t / "dev2.ll", t / "dev.bc", t / "dev.spv"
        cmd = [clang, "-x", "hip", "--offload=spirv64", "-nogpulib", "-nogpuinc", "--offload-device-only",
               "-std=c++17", "-O2", "-S", "-emit-llvm", "-ffp-contract=fast",
               # HIP mode does not define __CUDA__, which OpenCuda's headers test for device code
               "-DOCU_SPIRV_PATH=1", "-D__CUDACC__", "-D__CUDA__=1", f"-DOCU_SPV_CUDA_ARCH={arch}", f"-D__CUDA_ARCH_LIST__={arch}",
               f"-I{RT}", "-include", str(RT / "cuda_runtime.h"), "-Wno-unknown-cuda-version",
               *[f"-I{i}" for i in a.I], *[f"-D{d}" for d in a.D], a.src, "-o", str(ll)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stderr[-6000:])
            if os.environ.get("OCU_SPV_KEEP_GOING") == "1":
                # like cu2ocl --keep-going: the file's kernels are left out; launching
                # one fails at run time ("not found"), the build goes on
                sys.stderr.write(f"cu2spv: {a.src}: device code left out (OCU_SPV_KEEP_GOING=1)\n")
                Path(a.o).write_bytes(b"OCUSPV1\n---\n")
                return 0
            return 1

        text, meta = rewrite(ll.read_text())
        if not meta:
            Path(a.o).write_bytes(b"OCUSPV1\n---\n")
            return 0
        infer = os.environ.get("OCU_SPV_INFER", "1") != "0"
        if infer:
            text = global_pointers(text)
        div = infer and os.environ.get("OCU_SPV_DIV64", "1") != "0"
        if div:
            text = bypass_div64(text)
        ll2.write_text(text)
        opt = os.environ.get("OCU_SPV_OPT") or str(Path(llc).with_name("opt"))
        if infer and Path(opt).exists():
            r = subprocess.run([opt, "-passes=always-inline,infer-address-spaces,instcombine,simplifycfg", str(ll2), "-o", str(bc)],
                               capture_output=True, text=True)
        else:
            r = subprocess.run([llvm_as, str(ll2), "-o", str(bc)], capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write("cu2spv: the adjusted IR does not assemble:\n" + r.stderr[-3000:])
            return 1
        r = subprocess.run([llc, "-mtriple=spirv64-unknown-unknown", "-O2", "-filetype=obj", f"--spirv-ext={ext}",
                            str(bc), "-o", str(spv)], capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write("cu2spv: the SPIR-V back end failed:\n" + r.stderr[-3000:])
            return 1
        blob = spv.read_bytes()
        head = "OCUSPV1\n" + "\n".join(meta) + f"\nspirv {len(blob)}\n---\n"
        Path(a.o).write_bytes(head.encode() + blob)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

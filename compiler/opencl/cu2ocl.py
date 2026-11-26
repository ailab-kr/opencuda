#!/usr/bin/env python3
"""cu2ocl: CUDA kernels to C++ for OpenCL (and SPIR-V), for the OpenCL backend.

The translation is cu2msl.py's (compiler/metal): the same typed syntax
tree, the same address-space analysis, the same handling of templates,
device functions, __shared__ and module variables. OpenCL's kernel
language has the same address-space model as MSL (global = device,
local = threadgroup, constant, private = thread), so only the edges
differ, and they are the hooks this file overrides:

  - kernel signature: (constant uchar *ocu_args, constant uint *ocu_offs,
    [device uchar *ocu_glob,] threadgroup uchar *ocu_dyn); arguments are
    read from ocu_args at the offsets in ocu_offs (backends/opencl/README.md);
  - built-in indices from get_local_id / get_group_id / get_local_size /
    get_num_groups;
  - 3-component vector arguments read as three scalars (CUDA packs them in
    12 bytes, OpenCL's are 16);
  - double is allowed (the device needs cl_khr_fp64; the build fails on a
    device without it);
  - the prelude (ocl_prelude.h) maps the MSL address-space words and the
    CUDA built-in names to OpenCL.

The generated source is compiled here with Clang as C++ for OpenCL 2021
(errors stop the translation) and, when llvm-spirv is found, turned into
SPIR-V, which drivers with cl_khr_il_program / OpenCL 2.1 take directly.
Drivers that compile C++ for OpenCL source (cl_ext_cxx_for_opencl) use the
source.

    cu2ocl.py <file.cu> -o <image> [-I dir] [-D name] [--clang clang++]
              [--keep-going] [--report]

OCU_LLVM_SPIRV names the SPIR-V translator (default: llvm-spirv on PATH,
"none" for source-only images).

Image format (read by backends/opencl/ocl_backend.c): "OCUOCL1", then the
param / global / init lines of cu2msl's format, then
  source <bytes>                the C++ for OpenCL source length
  spirv <bytes>                 the SPIR-V module length (when present)
then "---", the source, and the SPIR-V module right after it.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cu2msl", HERE.parent / "metal" / "cu2msl.py")
cu2msl = importlib.util.module_from_spec(_spec)
sys.dont_write_bytecode = True  # no __pycache__ in the source tree
_spec.loader.exec_module(cu2msl)
TranslateError = cu2msl.TranslateError


def spirv_tool() -> str | None:
    t = os.environ.get("OCU_LLVM_SPIRV")
    if t:
        return None if t == "none" else t
    for name in ("llvm-spirv", "llvm-spirv-18", "llvm-spirv-19", "llvm-spirv-20"):
        p = shutil.which(name)
        if p:
            return p
    return None


class OclTranslator(cu2msl.Translator):
    TARGET = "OpenCL"
    CHIP = "this device"
    DOUBLE_OK = True
    PATH_DEFINE = "OCU_OPENCL_PATH"
    MAGIC = "OCUOCL1"
    GLOB_PARAM = "device uchar *ocu_glob"
    # printf exists in OpenCL; assert() is checked by the prelude's macro
    UNAVAILABLE = {k: v for k, v in cu2msl.NO_METAL.items()
                   if k not in ("printf", "__assert_fail", "__assertfail")}
    LAMBDA_OK = True
    WRAP_CLASS_TERNARY = True
    PTR_ARRAY_FIELDS = True

    WARP_RE = re.compile(r"\b(__shfl\w*|__ballot\w*|__any_sync|__all_sync|__syncwarp|warpSize|__activemask|"
                         r"__match_\w+|__reduce_\w+_sync|thread_block_tile|tiled_partition|coalesced_threads|"
                         r"cooperative_groups|__lane_id)\b")

    def uses_warp(self, inst, seen=None) -> bool:
        """Whether a kernel (or a device function it reaches) uses warp-wide
        operations: only those need the 32-wide sub-group; the others are left
        to the compiler, which can then pick 16 and keep twice the registers
        per work-item."""
        seen = set() if seen is None else seen
        if id(inst) in seen:
            return False
        seen.add(id(inst))
        body = next(c for c in cu2msl.kids(inst.fn) if c.get("kind") == "CompoundStmt")
        sp = self.span(body)
        text = self.text(sp[0])[sp[1]:sp[2]] if sp else ""
        if self.WARP_RE.search(text):
            return True
        return any(self.uses_warp(c, seen) for c in getattr(inst, "callees_inst", None) or ())

    BYVAL_LIMIT = 1024  # bytes of packed arguments passed by value (devices allow >= 1024 per argument)

    def head(self, inst, params, ret):
        head, line = super().head(inst, params, ret)
        if not inst.kernel or line is None or os.environ.get("OCU_OPENCL_BYVAL") == "0":
            return head, line
        sizes = [int(x) for x in line.split()[2:]]
        offs, at = [], 0
        for s in sizes:  # the backend's packing (ocl_backend.c arg_align)
            a = 1
            while a < 16 and s % (a * 2) == 0:
                a *= 2
            at = (at + a - 1) // a * a
            offs.append(at)
            at += s
        total = max(16, (at + 15) // 16 * 16)
        if total > self.BYVAL_LIMIT:
            return head, line
        old = "(constant uchar *ocu_args,\n    constant uint *ocu_offs,\n"
        assert old in head, head
        inst.byval = (total, offs)
        head = head.replace(old, f"(ocu_pack_{total} ocu_pk,\n")
        # a plain struct per size (kernel parameters cannot have template types)
        head = (f"#ifndef OCU_PACK_{total}\n#define OCU_PACK_{total}\n"
                f"struct ocu_pack_{total} {{ uchar b[{total}]; }};\n#endif\n") + head
        return head, line + f"\nbyval {inst.name} {total}"

    def prologue(self, inst, params) -> str:
        text = self.prologue_sh(inst, params)
        bv = getattr(inst, "byval", None)
        if bv is None:
            return text
        offs = ", ".join(str(o) for o in bv[1]) or "0"
        return (f"\n    const uchar *ocu_args = ocu_pk.b;\n    const uint ocu_offs[] = {{{offs}}};" + text)

    def device_shared(self, inst, st, n):
        """__shared__ in a device function: OpenCL allows local memory only
        at kernel scope, so each kernel that reaches such a function holds one
        local buffer (ocu_ctx.sh) and every such variable has a fixed offset
        in it, the same in all kernels (offsets only grow, so a kernel's
        buffer ends at its last variable)."""
        if not hasattr(self, "dev_sh"):
            self.dev_sh = {}  # VarDecl id -> (index, type)
        td = cu2msl.no_restrict(cu2msl.dtype(n)).strip()
        if n["id"] not in self.dev_sh:
            self.dev_sh[n["id"]] = (len(self.dev_sh), td)
        if not hasattr(inst, "sh_keys"):
            inst.sh_keys = set()
        inst.sh_keys.add(n["id"])
        cut = td.find("[")
        base, dims = (td, "") if cut < 0 else (td[:cut].strip(), td[cut:])
        off = self.sh_offset(self.dev_sh[n["id"]][0])
        b, e = self.edit_span(n, st, "a __shared__ declaration", True)
        name = n["name"]
        st["edits"].append((b, e, f"threadgroup {base} (&{name}){dims} = "
                                  f"*(threadgroup {base} (*){dims})(ocu_ctx.sh + {off})"))

    def sh_offset(self, idx: int) -> str:
        types = [t for i, t in sorted(self.dev_sh.values()) if i < idx]
        return "(" + "".join(f"OCU_SH_ALIGN(sizeof({t})) + " for t in types) + "0)"

    def sh_reach(self, inst, seen=None) -> set:
        seen = set() if seen is None else seen
        if id(inst) in seen:
            return set()
        seen.add(id(inst))
        out = set(getattr(inst, "sh_keys", ()))
        for c in getattr(inst, "callees_inst", None) or ():
            out |= self.sh_reach(c, seen)
        return out

    def kernel_head(self, inst) -> str:
        glob = "    device uchar *ocu_glob,\n" if self.globals else ""
        sg = ""
        if os.environ.get("OCU_OPENCL_SUBGROUP") == "32":
            if self.uses_warp(inst):
                sg = "__attribute__((intel_reqd_sub_group_size(32))) "
            elif os.environ.get("OCU_OPENCL_SUBGROUP_OTHER") in ("8", "16", "32"):
                sg = f"__attribute__((intel_reqd_sub_group_size({os.environ['OCU_OPENCL_SUBGROUP_OTHER']}))) "
        return (f"{sg}kernel void {inst.name}(constant uchar *ocu_args,\n"
                "    constant uint *ocu_offs,\n" + glob +
                "    threadgroup uchar *ocu_dyn)")

    def context_init(self, glob: str) -> str:
        sh = getattr(self, "_kernel_sh", None)
        buf = ""
        if sh:
            buf = f"\n    threadgroup uchar ocu_sh_buf[{sh}] __attribute__((aligned(16)));"
        return (buf + "\n    ocu_context ocu_ctx{OCU_U3(get_local_id), OCU_U3(get_group_id), "
                f"OCU_U3(get_local_size), OCU_U3(get_num_groups), ocu_dyn, {glob}, "
                f"{'ocu_sh_buf' if sh else 'nullptr'}}};\n")

    def prologue_sh(self, inst, params) -> str:
        keys = self.sh_reach(inst) if hasattr(self, "dev_sh") else set()
        if keys:
            last = max(self.dev_sh[k][0] for k in keys)
            t = [tt for i, tt in self.dev_sh.values() if i == last][0]
            self._kernel_sh = f"{self.sh_offset(last)} + OCU_SH_ALIGN(sizeof({t}))"
        else:
            self._kernel_sh = None
        try:
            return super().prologue(inst, params)
        finally:
            self._kernel_sh = None

    def vec3_read(self, t: str, base: str, name: str, i: int) -> str:
        scalar = re.sub(r"3$", "", base)
        scalar = {"longlong": "long", "ulonglong": "ulong"}.get(scalar, scalar)
        return f"    {t} {name} = ocu_arg3<{base}, {scalar}>(ocu_args, ocu_offs, {i});"

    def prelude(self) -> str:
        text = (HERE / "ocl_prelude.h").read_text(encoding="utf-8")
        if os.environ.get("OCU_OPENCL_SUBGROUP") == "32":
            # Every kernel carries intel_reqd_sub_group_size(32) (kernel_head),
            # so a sub-group is a CUDA warp: the warp functions are turned on.
            text = "#define OCU_SUBGROUP_32 1\n" + text
            try:
                user = Path(self.src).read_text(encoding="latin-1")
            except (OSError, AttributeError):
                user = ""
            aliases = sorted(set(re.findall(r"^\s*namespace\s+\w+\s*=\s*cooperative_groups\s*;", user, re.M)))
            if aliases:
                text += "\n" + "\n".join(a.strip() for a in aliases) + "\n"
        return text

    def shared_decl(self, inst, st, n, b: int, e: int, m) -> None:
        """OpenCL allows local-memory variables only in the outermost scope
        of a kernel: one declared in an inner block moves to the top of the
        kernel (CUDA __shared__ variables have no initializer, and live for
        the whole block anyway)."""
        if not hasattr(inst, "ocl_nested"):
            body = next(c for c in cu2msl.kids(inst.fn) if c.get("kind") == "CompoundStmt")
            inst.ocl_nested = [self.span(c) for c in cu2msl.walk(body)
                               if c.get("kind") == "CompoundStmt" and c is not body and self.span(c)]
            inst.ocl_hoisted = set()
        if not any(self.same(f, st["file"]) and lo < b < hi for f, lo, hi in inst.ocl_nested):
            return super().shared_decl(inst, st, n, b, e, m)
        name = n.get("name")
        if name in inst.ocl_hoisted:
            raise TranslateError(f"'{st['fname']}': two inner-block __shared__ variables named '{name}'")
        inst.ocl_hoisted.add(name)
        decl = st["text"][b:e]
        inst.tmpl.append(decl[:m.start()] + "threadgroup" + decl[m.end():] + ";")
        st["edits"].append((b, e, ""))

    def run(self, keep_going: bool = False, report=None) -> str:
        try:
            image = super().run(keep_going, report)
        except TranslateError as e:
            # A file with host code only (whole-program builds compile every
            # .cu): with OCU_CU2OCL_EMPTY_OK=1 it gets an image without code.
            if str(e) == "no kernels found" and os.environ.get("OCU_CU2OCL_EMPTY_OK") == "1":
                return self.MAGIC + "\n---\n"
            raise
        head, source = image[len(self.MAGIC) + 1:].split("\n---\n", 1)
        # OpenCL reserves `long long` for a 128-bit type; CUDA's is 64 bits,
        # which is OpenCL's `long`. (After the prelude: the user's code and
        # what the translation wrote for it.)
        cut = source.index("// ---- macros")
        source = source[:cut] + re.sub(r"\blong(\s+)long\b", r"long", source[cut:])
        spirv = self.compile(source)
        meta = [l for l in head.splitlines() if l.strip()] + [f"source {len(source)}"]
        if spirv is not None:
            meta.append(f"spirv {len(spirv)}")
        return self.MAGIC + "\n" + "\n".join(meta) + "\n---\n" + source + \
            (spirv.decode("latin-1") if spirv is not None else "")

    def compile(self, source: str) -> bytes | None:
        try:
            return self.compile_at(source, os.environ.get("OCU_OPENCL_OPT", "-O2"))
        except TranslateError as e:
            if "SPIR-V translation failed" not in str(e) or os.environ.get("OCU_OPENCL_OPT"):
                raise
            # Intel's llvm-spirv (2026.2) aborts in its bit-cast lowering on
            # some -O2 modules; the driver optimises the SPIR-V again anyway.
            return self.compile_at(source, "-O1")

    def compile_at(self, source: str, opt: str) -> bytes | None:
        """Compile the generated source as C++ for OpenCL (its errors stop the
        translation); SPIR-V when the translator is there."""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "module.clcpp"
            src.write_bytes(source.encode("latin-1"))
            bc = Path(tmp) / "module.bc"
            cmd = [self.clang, "-x", "clcpp", "-cl-std=clc++2021", "-target", "spir64",
                   "-Xclang", "-finclude-default-header", opt, "-emit-llvm", "-c",
                   "-Wno-unused-value", "-ferror-limit=20", str(src), "-o", str(bc)]
            out = subprocess.run(cmd, capture_output=True, encoding="latin-1")
            if out.returncode != 0:
                raise TranslateError("the generated C++ for OpenCL does not compile:\n" +
                                     out.stderr[-3000:])
            tool = spirv_tool()
            if tool is None:
                return None
            spv = Path(tmp) / "module.spv"
            out = subprocess.run([tool, str(bc), "-o", str(spv)], capture_output=True,
                                 encoding="latin-1")
            if out.returncode != 0:
                raise TranslateError("SPIR-V translation failed:\n" + out.stderr[-2000:])
            return spv.read_bytes()


if __name__ == "__main__":
    sys.exit(cu2msl.main(sys.argv[1:], OclTranslator))

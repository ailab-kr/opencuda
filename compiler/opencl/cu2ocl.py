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
    UNAVAILABLE = {k: v for k, v in cu2msl.NO_METAL.items() if k != "printf"}

    def kernel_head(self, inst) -> str:
        glob = "    device uchar *ocu_glob,\n" if self.globals else ""
        return (f"kernel void {inst.name}(constant uchar *ocu_args,\n"
                "    constant uint *ocu_offs,\n" + glob +
                "    threadgroup uchar *ocu_dyn)")

    def context_init(self, glob: str) -> str:
        return ("\n    ocu_context ocu_ctx{OCU_U3(get_local_id), OCU_U3(get_group_id), "
                f"OCU_U3(get_local_size), OCU_U3(get_num_groups), ocu_dyn, {glob}}};\n")

    def vec3_read(self, t: str, base: str, name: str, i: int) -> str:
        scalar = re.sub(r"3$", "", base)
        scalar = {"longlong": "long", "ulonglong": "ulong"}.get(scalar, scalar)
        return f"    {t} {name} = ocu_arg3<{base}, {scalar}>(ocu_args, ocu_offs, {i});"

    def prelude(self) -> str:
        return (HERE / "ocl_prelude.h").read_text(encoding="utf-8")

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
        spirv = self.compile(source)
        meta = [l for l in head.splitlines() if l.strip()] + [f"source {len(source)}"]
        if spirv is not None:
            meta.append(f"spirv {len(spirv)}")
        return self.MAGIC + "\n" + "\n".join(meta) + "\n---\n" + source + \
            (spirv.decode("latin-1") if spirv is not None else "")

    def compile(self, source: str) -> bytes | None:
        """Compile the generated source as C++ for OpenCL (its errors stop the
        translation); SPIR-V when the translator is there."""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "module.clcpp"
            src.write_bytes(source.encode("latin-1"))
            bc = Path(tmp) / "module.bc"
            cmd = [self.clang, "-x", "clcpp", "-cl-std=clc++2021", "-target", "spir64",
                   "-Xclang", "-finclude-default-header", "-O2", "-emit-llvm", "-c",
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

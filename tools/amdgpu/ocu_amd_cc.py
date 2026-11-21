#!/usr/bin/env python3
"""ocu_amd_cc: compile an unmodified .cu file for an AMD GPU with the
driver's own compiler library (amd_comgr), no Clang executable needed.

    ocu_amd_cc.py k.cu -o k.co [--arch gfx1103] [-I dir] [-D name[=v]] [-O3]

The device pass is Clang's HIP mode with OpenCuda's headers
(runtime/include/cuda_runtime.h -> ocu_cuda_lang_amdgcn.h) force-included,
-nogpuinc -nogpulib: no AMD HIP headers and no AMD device library, the same
rule as the NVPTX path. The output is an amdgcn code object (ELF) that the
HIP runtime loads with hipModuleLoadData, the image the AMD backend takes
(backends/amd/README.md).

--arch defaults to the first HIP device's architecture, or OCU_AMDGPU_ARCH.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from comgr import Comgr, ComgrError  # noqa: E402

RT = REPO / "runtime" / "include"


def device_options(src: Path, includes: list[str], defines: list[str], opt: str) -> list[str]:
    return ["-std=c++17", "-nogpuinc", "-nogpulib", opt, "-ffp-contract=fast-honor-pragmas",
            "-fno-gpu-rdc", "-DOCU_AMD_PATH=1",
            f"-I{RT}", f"-I{src.parent}", *[f"-I{d}" for d in includes], *[f"-D{d}" for d in defines],
            "-include", str(RT / "cuda_runtime.h"),
            # last in the search: limits.h, float.h, stdarg.h, stdbool.h for a
            # machine without C library headers (comgr carries only stddef.h
            # and stdint.h of Clang's built-in headers)
            "-idirafter", str(RT / "freestanding"),
            "-Wno-unused-command-line-argument"]


def default_arch() -> str:
    a = os.environ.get("OCU_AMDGPU_ARCH")
    if a:
        return a
    try:
        import oclpy
        return oclpy.Device().arch            # the OpenCL device name is the gfx name
    except Exception:
        return "gfx1103"


def compile_cu(src: Path, arch: str, includes=(), defines=(), opt="-O3", comgr: Comgr | None = None) -> bytes:
    cg = comgr or Comgr()
    return cg.hip_to_executable(src.read_bytes(), arch, device_options(src, list(includes), list(defines), opt),
                                name=src.name.replace(".cu", ".hip"))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--arch")
    ap.add_argument("-I", action="append", default=[])
    ap.add_argument("-D", action="append", default=[])
    ap.add_argument("-O", default="3")
    a = ap.parse_args(argv)
    arch = a.arch or default_arch()
    t = time.time()
    try:
        co = compile_cu(Path(a.source).resolve(), arch, a.I, a.D, f"-O{a.O}")
    except ComgrError as e:
        print(f"{a.source}: cannot compile for {arch}:\n{e}", file=sys.stderr)
        return 1
    Path(a.output).write_bytes(co)
    print(f"{a.source} -> {a.output} ({arch}, {len(co)} bytes, {time.time() - t:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

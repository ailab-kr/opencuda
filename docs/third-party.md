# Third-party components

Everything this repository uses but does not own, with its licence.
Add a row before the first use.

| Component | Used for | Licence | How it is used | Added |
|---|---|---|---|---|
| Clang / LLVM 18.1.3 | Compiling `.cu` sources for each target | Apache-2.0 WITH LLVM-exception | External tool, not vendored | 2026-11-12 |
| CMake | Build | BSD-3-Clause | External tool | 2026-11-12 |
| Python 3.10+ standard library | Conformance harness | PSF | External runtime | 2026-11-12 |
| OpenCL ICD loader / drivers (any) | Run time of the OpenCL backend | per driver | Loaded with dlopen; nothing vendored, no SDK header used (`backends/opencl/ocl_api.h` written from the Khronos specification) | 2026-11-13 |
| PoCL 6.0 (CPU driver, built from source) | Test device for the OpenCL backend | MIT | External library loaded at test time (`OCU_OPENCL_TEST_LIBRARY`), not vendored | 2026-11-13 |
| Khronos SPIRV-LLVM-Translator 18.1 (`llvm-spirv`) | C++ for OpenCL bitcode to SPIR-V in `compiler/opencl/cu2ocl.py` | Apache-2.0 WITH LLVM-exception | External tool, optional | 2026-11-13 |
| GCC aarch64 cross compiler, qemu-user | ARM64 build and test in tools/check.sh | GPL (tools only, not linked into our code) | External tools | 2026-11-12 |

Not used, by rule (see `ALLOWLIST.md`): vendor CUDA SDK headers, libraries,
samples and tools.

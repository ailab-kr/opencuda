# Third-party components

Everything this repository uses but does not own, with its licence.
Add a row before the first use.

| Component | Used for | Licence | How it is used | Added |
|---|---|---|---|---|
| Clang / LLVM 18.1.3 | Compiling `.cu` sources for each target | Apache-2.0 WITH LLVM-exception | External tool, not vendored | 2026-11-12 |
| LLVM 18 C++ API headers and CMake package (`llvm-18-dev`) | Building the CPU path's pass plugin `compiler/cpu_loops` | Apache-2.0 WITH LLVM-exception | Headers at build time; the plugin is loaded into Clang and links no LLVM library itself; nothing vendored | 2026-11-13 |
| CMake | Build | BSD-3-Clause | External tool | 2026-11-12 |
| Python 3.10+ standard library | Conformance harness | PSF | External runtime | 2026-11-12 |
| OpenCL ICD loader / drivers (any) | Run time of the OpenCL backend | per driver | Loaded with dlopen; nothing vendored, no SDK header used (`backends/opencl/ocl_api.h` written from the Khronos specification) | 2026-11-13 |
| PoCL 6.0 (CPU driver, built from source) | Test device for the OpenCL backend | MIT | External library loaded at test time (`OCU_OPENCL_TEST_LIBRARY`), not vendored | 2026-11-13 |
| Khronos SPIRV-LLVM-Translator 18.1 (`llvm-spirv`) | C++ for OpenCL bitcode to SPIR-V in `compiler/opencl/cu2ocl.py` | Apache-2.0 WITH LLVM-exception | External tool, optional | 2026-11-13 |
| GCC aarch64 cross compiler, qemu-user | ARM64 build and test in tools/check.sh | GPL (tools only, not linked into our code) | External tools | 2026-11-12 |
| Clang / LLVM 23 (upstream; the build shipped inside Intel oneAPI also works) | `compiler/spirv/cu2spv.py`: HIP-mode device compile for spirv64, `opt`, LLVM's SPIR-V back end (`llc`) | Apache-2.0 WITH LLVM-exception | External tools, not vendored | 2026-11-26 |
| Intel compute runtime (NEO) and Intel Graphics Compiler | OpenCL driver on Intel Arc GPUs | MIT | External driver loaded through the ICD loader, not vendored | 2026-11-26 |
| bitsandbytes (CUDA kernels, built from source) | Test program for the SPIR-V path and the NF4 GEMM (`ocublasGemmNF4` reads its NF4 storage format) | MIT | Fetched and built outside the repository, not vendored; the NF4 code values are those of the QLoRA paper | 2026-11-26 |
| `compiler/spirv/clang-hipspv-kernel-address.patch` (a change to Clang, also proposed upstream) | Lets host code store a HIP kernel in a function pointer on spirv64 (llama.cpp FlashAttention); applied by the user to their own Clang build | Apache-2.0 WITH LLVM-exception (it changes LLVM source) | A patch file only; no LLVM source vendored | 2026-11-27 |

Not used, by rule (see `ALLOWLIST.md`): vendor CUDA SDK headers, libraries,
samples and tools.

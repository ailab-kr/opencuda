# Source Allowlist

**English** · [한국어](ALLOWLIST.ko.md)

The code in this repository is written independently, using only public material. All contributors follow this list.

Status: draft.

## What you may look at

- Documents published by NVIDIA: function descriptions, programming guides, published specification documents
- Source released under a permissive license: the base compiler, and the public runtimes and libraries of chip vendors
- Standards documents, papers, textbooks
- Inputs, outputs, and timings obtained by running programs we wrote on NVIDIA hardware

If you need a new source, add it to this list first, then use it.

## What you must not do

- Take apart programs and libraries distributed by NVIDIA: disassembly, decompilation, tracing internal behavior
- Copy NVIDIA headers, examples, or source, or transcribe them while looking at them
- Analyze output produced by NVIDIA tools and convert it for other chips
- Copy code under a license with strong copying restrictions
- Add code of unknown origin

## Records

- For an implementation change, record the public document it is based on. Put it on the `Spec-Source:` line of the commit message.
- List the external components we use and their licenses in `docs/third-party.md`.

## Allowed sources

| Source | Address | Date added |
|---|---|---|
| Khronos OpenCL 3.0 API and OpenCL C specifications, SPIR-V specification | https://registry.khronos.org/OpenCL/ , https://registry.khronos.org/SPIR-V/ | 2026-11-26 |
| Khronos OpenCL extension specifications: cl_intel_subgroups, cl_intel_subgroup_2d_block_io, cl_intel_subgroup_matrix_multiply_accumulate, cl_intel_unified_shared_memory, cl_khr_integer_dot_product | https://registry.khronos.org/OpenCL/extensions/ | 2026-11-26 |
| LLVM / Clang documentation (HIP language mode, SPIR-V back end, InferAddressSpaces, BypassSlowDivision) | https://llvm.org/docs/ , https://clang.llvm.org/docs/ | 2026-11-26 |
| CUB public API documentation (BlockLoad, BlockStore, BlockReduce, WarpReduce) | https://nvidia.github.io/cccl/cub/ | 2026-11-26 |
| cuBLAS / cuBLASLt API reference (function descriptions only) | https://docs.nvidia.com/cuda/cublas/ | 2026-11-26 |
| bitsandbytes source, MIT licence: the NF4 code table and block layout (block 64, f32 absmax, high nibble first), read to match its storage format; built unmodified as a test program | https://github.com/bitsandbytes-foundation/bitsandbytes | 2026-11-26 |
| QLoRA paper (Dettmers et al. 2023): the NF4 data type | https://arxiv.org/abs/2305.14314 | 2026-11-26 |
| GNU ld manual (-rpath, --disable-new-dtags) and the ld.so(8) manual page (DT_RPATH and DT_RUNPATH search order) | https://sourceware.org/binutils/docs/ld/ , https://man7.org/linux/man-pages/man8/ld.so.8.html | 2026-11-27 |
| Intel oneAPI DPC++/C++ Compiler documentation (OpenMP options, libiomp5) | https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/ | 2026-11-27 |
| LLVM / Clang source (Apache-2.0 WITH LLVM-exception): `clang/lib/Sema/SemaType.cpp` (calling convention of CUDA / HIP kernels for SPIR-V), `clang/lib/CodeGen/Targets/SPIR.cpp` and `clang/lib/CodeGen/CGCall.cpp` (`setCUDAKernelCallingConvention`), and the commits that wrote them (D114407, #110447); read to find why host code cannot store a kernel in a function pointer on spirv64, and to write `compiler/spirv/clang-hipspv-kernel-address.patch` | https://github.com/llvm/llvm-project | 2026-11-27 |
| Khronos OpenCL extension specification cl_intel_device_attribute_query (CL_DEVICE_NUM_SLICES_INTEL, CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL): the multiprocessor count on Intel GPUs | https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_device_attribute_query.html | 2026-11-27 |

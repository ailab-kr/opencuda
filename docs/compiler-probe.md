# Compiler probe (PLAN P0-W1-11)

Date: 2026-11-12 · Toolchain: Ubuntu clang 18.1.3 (WSL2, Ubuntu 24.04) · Script: `compiler/probe/run_probe.sh`

Question: can the open-source compiler build an unmodified `.cu` file for NVIDIA, AMD and the CPU using only headers we write, with no vendor SDK headers or libraries?

Input: `compiler/probe/probe_kernel.cu`. Two kernels (`saxpy`, and a block reduction using shared memory and `__syncthreads`) plus one host function that launches with `<<<...>>>`. Headers: `compiler/probe/probe_defs.h`, written for the probe.

## Results

| # | Check | Result |
|---|---|---|
| 1 | NVIDIA device code (sm_80), CUDA mode | OK: PTX with 2 kernels |
| 2 | Host pass, CUDA mode | OK; the launch needs 3 runtime symbols (below) |
| 3 | AMD device code, HIP mode, same `.cu` | OK for gfx1100 (RDNA3) and gfx1201 (RDNA4); ELF code objects with both kernels |
| 4 | Host pass, HIP mode | OK; the launch needs 7 runtime symbols (below) |
| 5 | Plain C++ (for a CPU backend) | Fails: `<<<...>>>` does not parse |
| 6 | Kernel bodies in the host pass | Not emitted; only `__device_stub__*` |

Launch symbols the compiler emits calls to:

- CUDA mode, when clang assumes an old CUDA version (the empty `--cuda-path`): `cudaConfigureCall`, `cudaSetupArgument`, `cudaLaunch`.
- HIP mode: `__hipPushCallConfiguration`, `__hipPopCallConfiguration`, `hipLaunchKernel`, `__hipRegisterFatBinary`, `__hipRegisterFunction`, `__hipUnregisterFatBinary`, `__hip_fatbin`.

## What this means for the design

1. **The base plan holds.** The same unmodified `.cu` source compiled to NVIDIA and AMD (RDNA3 and RDNA4) device code with no vendor headers. A first cut of our headers needs about 80 lines for this subset.
2. **The header set must be per target.** Clang's own builtin-variable header (`__clang_cuda_builtin_vars.h`) is NVIDIA-only. For AMD we map `threadIdx`/`blockIdx`/`blockDim` and the barrier to AMDGPU builtins ourselves. This belongs in `compiler/`, under `#if` per target, and nowhere else.
3. **AMD goes through HIP mode.** Clang has no "CUDA source for AMD" mode. We compile `.cu` with `-x hip`. The source stays unchanged; the driver picks the mode.
4. **The launch ABI is a contract between the compiler and our runtime.** Which symbols clang calls depends on the mode and on the CUDA version clang believes it targets. We should pin it on purpose rather than inherit it:
   - CUDA mode: point `--cuda-path` at a folder we own, with a version file that selects the newer launch ABI. To be checked; see open questions.
   - HIP mode: the `__hip*` registration and launch symbols are implemented by our runtime as internal entry points, then forwarded to the backend.
   - Whether those symbol names may be used is an open question.
5. **The CPU backend cannot reuse the host pass.** Plain C++ rejects `<<<...>>>`, and the CUDA host pass emits only stubs, not kernel bodies. Options, to decide when the CPU backend starts:
   - a. Our driver rewrites only the launch syntax into a function call, then compiles the file as plain C++ with CPU definitions of the index variables and barrier.
   - b. Compile the device side for a CPU target through an offload toolchain (needs research, may not exist for this use).
   - c. Build a small clang plugin that emits host bodies for `__global__` functions.

   Option a is the smallest and is the current lean. It needs a reliable tokenizer, not a regex, to avoid touching `<<<` inside strings, comments or templates.
6. **Clean-room guard is required.** This machine has a vendor CUDA installation at `/usr/local/cuda` (13.3), and clang finds it automatically. `-nocudainc` alone stops header inclusion, but our driver and build must always pass `--cuda-path=<a folder we own>` and `-nocudainc -nocudalib`. A CI check should fail if any vendor path appears in compiler command lines or dependency files.
7. **Toolchain gap.** Ubuntu's clang-18 package does not ship `clang-offload-bundler`, so a multi-target fat binary did not build. We can embed per-target code objects ourselves (we own the registration format), or install a fuller LLVM. Decide in P0-W4.

## Open questions

- Which version file and contents make clang 18 use the newer launch ABI (`__cudaPushCallConfiguration` / `cudaLaunchKernel`) with a `--cuda-path` we own?
- Does clang 18 accept everything the five Rodinia programs use (P0-W1-07)? The probe covered only indexing, shared memory and a barrier.
- Newer LLVM (19, 20) changes: worth testing before fixing the toolchain version.
- How much of `__clang_cuda_*` (Apache-2.0 with LLVM exception) can we reuse for the NVIDIA target versus writing our own? Licence allows it; add it to `docs/third-party.md` if used.

## Reproduce

```sh
sh compiler/probe/run_probe.sh /tmp/ocu-probe
```

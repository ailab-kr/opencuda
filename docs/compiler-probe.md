# Compiler probe (PLAN P0-W1-11)

Date: 2026-11-12 · Toolchain: Ubuntu clang 18.1.3 (WSL2, Ubuntu 24.04) · Script: `compiler/probe/run_probe.sh`

Question: can the open-source compiler build an unmodified `.cu` file for NVIDIA, AMD and the CPU using only headers we write, with no vendor SDK headers or libraries?

Inputs, compiled unchanged on every path:

- `compiler/probe/probe_kernel.cu`: two kernels (`saxpy`, and a block reduction using a static `__shared__` array and `__syncthreads`) plus one host function that launches with `<<<...>>>`.
- `compiler/probe/probe_features.cu`: a template kernel with two instantiations and a `__device__` helper, 2-D grid and block, `extern __shared__` with a barrier, `atomicAdd` across blocks, a struct argument by value, a globally qualified kernel name, a launch spread over several lines, and `<<<...>>>` inside a comment and a string.

Headers: `compiler/probe/probe_defs.h` (GPU targets) and `compiler/probe/probe_cpu.h` (CPU path), written for the probe.

## Results

| # | Check | Result |
|---|---|---|
| 1 | NVIDIA device code (sm_80), CUDA mode | OK: PTX with 2 kernels |
| 2 | Host pass, CUDA mode | OK; the launch needs 3 runtime symbols (below) |
| 3 | AMD device code, HIP mode, same `.cu` | OK for gfx1100 (RDNA3) and gfx1201 (RDNA4); ELF code objects with both kernels |
| 4 | Host pass, HIP mode | OK; the launch needs 7 runtime symbols (below) |
| 5 | Plain C++ (for a CPU backend) | Fails: `<<<...>>>` does not parse |
| 6 | Kernel bodies in the host pass | Not emitted; only `__device_stub__*` |
| 7 | `probe_features.cu` for NVIDIA, gfx1100, gfx1201 | OK: 5 kernels on each target |
| 8 | CPU path: rewrite, compile as C++17, run on the CPU backend | OK: all three files give correct results in fixed and random thread order (test `probe_cpu`) |
| 9 | `probe_review.cu` (patterns from the independent review) for sm_80 and gfx1201 | OK: 4 kernels on each |
| 10 | `probe_warp.cu` (shuffle up/down/xor/idx, ballot, any, all, `__syncwarp`) for sm_80, gfx1100, gfx1201, and on the CPU path with warp size 32 and 64 | OK: 3 kernels on each GPU target, correct results on the CPU |
| 11 | CUDA version pinned with our own one-line version marker | OK: clang reports CUDA 12.3, emits PTX 8.3, and the launch uses `__cudaPushCallConfiguration`, `__cudaPopCallConfiguration`, `cudaLaunchKernel` |

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

   Option a is implemented in the probe (`compiler/probe/cpu_rewrite.py`, 18 unit tests). It rewrites only the launch syntax and `extern __shared__` declarations, never inside comments, strings, raw or prefixed literals or directives other than `#define` (macro bodies are rewritten, continuations kept), and keeps line numbers. A launch becomes `launch(cfg, [](auto &&...a) { K(a...); }, args)`: the kernel call is an ordinary C++ call, so template deduction, overloads and conversions work as written. It goes through the backend interface: one module per launch site and argument types. Static `__shared__` arrays become `static thread_local` (one block's threads share one OS thread); `extern __shared__` becomes an accessor that looks the block's memory up at each use, so file-scope declarations work.
6. **Clean-room guard is required.** This machine has a vendor CUDA installation at `/usr/local/cuda` (13.3), and clang finds it automatically. `-nocudainc` alone stops header inclusion, but our driver and build must always pass `--cuda-path=<a folder we own>` and `-nocudainc -nocudalib`. A CI check should fail if any vendor path appears in compiler command lines or dependency files.
7. **Toolchain gap.** Ubuntu's clang-18 package does not ship `clang-offload-bundler`, so a multi-target fat binary did not build. We can embed per-target code objects ourselves (we own the registration format), or install a fuller LLVM. Decide in P0-W4.

## Rewriter on real code (2026-11-12)

`compiler/probe/survey_rewrite.py` ran the rewriter over two public CUDA code bases, read as text only:

| Code base | Files | Launches in code | Rewritten | Left | Files failed | `extern __shared__` rewritten |
|---|---|---|---|---|---|---|
| Rodinia (CUDA part, public mirror `yuhc/gpu-rodinia`) | 89 | 81 | 81 | 0 | 0 | 2 |
| llama.cpp `ggml/src/ggml-cuda` (MIT) | 303 | 192 | 192 | 0 | 0 | 20 |

Line counts were preserved in every file. Whether the rewritten files then compile was not checked: they also need the CUDA runtime API, which waits for legal review.

Features those files use that the CPU path does not support yet (number of files):

| Feature | Rodinia | llama.cpp |
|---|---|---|
| CUDA runtime API calls | 42 | 26 |
| `__constant__` memory | 9 | 0 |
| Textures | 5 | 0 |
| Atomics | 6 | 3 |
| Warp shuffles | 0 | 12 |
| Warp votes | 0 | 4 |
| Inline PTX assembly | 0 | 4 |

## Runtime functions real code calls (2026-11-12)

`compiler/probe/survey_api.py` counted calls to the CUDA runtime in the same two code bases (text only):

| Code base | Distinct runtime/library functions | Runtime calls | Covered by the 27 planned for phase 0 | Covered with 8 additions | Files fully covered: 27 / with additions |
|---|---|---|---|---|---|
| Rodinia | 39 | 908 | 87 % | 95 % | 21 / 40 of 49 |
| llama.cpp ggml-cuda | 82 | 232 | 60 % | 76 % | 10 / 15 of 26 |

The 8 additions: `cudaThreadSynchronize` (an old alias, 36 calls in 23 Rodinia files), `cudaMemcpyToSymbol` (34 calls; needs `__constant__`), `cudaMemsetAsync`, `cudaThreadExit`, `cudaEventCreateWithFlags`, `cudaStreamWaitEvent`, `cudaMemcpy2D`, `cudaMemcpy2DAsync`. Still outside after that: textures (Rodinia; the old texture-reference API they use is gone from current CUDA versions), graphs, peer copies and `cudaFuncSetAttribute` (llama.cpp), and the extended BLAS calls `cublasGemmEx`, `cublasGemmStridedBatchedEx` and the `cublasLt*` family (llama.cpp).

## CPU path limits found so far

- Kernels launched from inside a kernel (dynamic parallelism) are refused with "not supported".
- Warp-level operations work through a warp barrier and a tagged gather in the backend (`ocu_cpu_warp_gather_tagged`). Every live lane must take part, as code written with a full mask expects. Partial-mask code (lanes of one warp at different calls, or at different occurrences of the same call) is detected through the tag and a per-lane sequence number and fails the launch instead of giving wrong values. A lane named in the mask that has already returned is undefined in CUDA; it is tolerated by default and fails the launch under `OCU_CPU_STRICT=1`.
- `__shfl_xor_sync` with a width below the warp size may read an earlier group and returns its own value for a later one. `__ballot_sync` with a 64-lane warp returns the bits of the caller's own 32-lane half, since CUDA's result has 32 bits; `__any_sync` and `__all_sync` cover the whole warp.
- A literal `0` passed for a pointer parameter does not convert after being copied (an int is passed); `NULL` is rewritten to `nullptr` and works.
- The stream argument of a launch is accepted and ignored: the probe runs every launch synchronously.

## Open questions

- Answered: clang 18 reads `CUDA_VERSION` from `<cuda-path>/include/cuda.h` (`version.txt` and `version.json` are ignored). A folder we own with a one-line marker (`#define CUDA_VERSION 12030`) pins the newer launch ABI and PTX 8.3; nothing else in it is read under `-nocudainc -nocudalib`. `run_probe.sh` generates it at run time. Whether the file name may be used this way is an open question.
- Does clang 18 accept everything the five Rodinia programs use (P0-W1-07)? The probe covered only indexing, shared memory and a barrier.
- Newer LLVM (19, 20) changes: worth testing before fixing the toolchain version.
- How much of `__clang_cuda_*` (Apache-2.0 with LLVM exception) can we reuse for the NVIDIA target versus writing our own? Licence allows it; add it to `docs/third-party.md` if used.

## Reproduce

```sh
sh compiler/probe/run_probe.sh /tmp/ocu-probe
```

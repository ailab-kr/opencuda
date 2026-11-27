# Results on an Intel Arc Pro B70

**English** · [한국어](arc-pro-b70.ko.md)

What unmodified CUDA programs do on an Intel Arc Pro B70 with OpenCuda's
SPIR-V path (`ocu-nvcc --ocu-target=spirv`), measured 2026-11-27/28. Every
number here was measured on the machine below; where a result is partial
or a run is still going, it says so.

## Machine and software

| | |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB (OpenCL reports 256 compute units, 30.3 GiB) |
| Driver | Intel compute-runtime (NEO) 26.35.39758.21, IGC 2.41.16 (IGC 2.42 where noted) |
| Host | Intel Xeon Silver 4510, 61 GB, Ubuntu 26.04.1, Linux 7.0.0-30 |
| OpenCuda | branch `intel-arc-train` (PR #2); Clang/LLVM 23.1.6 for the device code |
| Compared with | chipStar 1.3.0 (`cucc`), the other open CUDA-on-Intel toolchain |

The GPU is shared with other work, so each measurement ran with nothing else on it.

## HeCBench: every CUDA program, OpenCuda against chipStar

[HeCBench](https://github.com/zjin-lcf/HeCBench) (commit 4426c9f) has 570
`*-cuda` programs. Each one is built and run **unmodified** by both
toolchains under the same rules (`tools/bench/hecbench_compare.sh`, summary
by `tools/bench/hecbench_summary.py`):

- `make` with the toolchain's compiler (300 s limit), then `make run` twice
  (180 s limit each); the second run is the one timed.
- Status: `build_fail`, `timeout`, `crash` (non-zero exit), `fail` (the
  program printed FAIL), `pass` (it printed PASS), `ran` (it finished and
  printed neither: many HeCBench programs only print timings or a checksum).
- `no_data`: the program stopped because its input files are not in the
  repository (HeCBench keeps large inputs elsewhere). This says nothing about
  either toolchain and is counted apart.

The run used the OpenCuda toolkit as it was at its start (commit 4f9930a)
for every program, so that all of them are measured with one toolkit.

**Interim, 277 of 570 programs done (the run is still going):**

| status | OpenCuda | chipStar 1.3.0 |
|---|---|---|
| pass | 104 | 80 |
| ran | 53 | 50 |
| fail | 10 | 5 |
| crash | 13 | 8 |
| timeout | 12 | 12 |
| build_fail | 62 | 104 |
| no_data | 23 | 18 |

- Programs that run to the end (pass or ran): OpenCuda 157, chipStar 130.
  Both: 117; only OpenCuda: 40; only chipStar: 13.
- Speed where both run to the end (115 programs above 0.05 s): the geometric
  mean of chipStar time / OpenCuda time is 1.00. OpenCuda is more than 10%
  faster on 30, more than 10% slower on 44, within 10% on 41.
- Of the 13 programs only chipStar finished, the toolkit as of this writing
  (later than the one the run uses) fixes four, checked on the same GPU:
  axpby-cuda (FAIL → PASS, kernel arguments over the driver's 2 KB limit),
  cmembench-cuda (FAIL → PASS, `cudaMemcpyToSymbol`), intrinsics-cast-cuda
  (build → runs, conversion intrinsics), ert-cuda (builds, `__half2`
  operators). Also collision-cuda (crash → runs) and convolution1D-cuda
  (crash → 180 s timeout, as chipStar), both symbol copies.

Where OpenCuda does not build, the main causes are libraries it does not
have yet (Thrust, cuSPARSE, cuSOLVER, cuRAND, cuFFT, NCCL/MPI) and inline
PTX assembly; the per-program first errors are in the run's logs.

## llama.cpp

Unmodified llama.cpp, CUDA backend, qwen3-0.6B Q4_K, `-fa 1`,
`OCU_COMPUTE_CAPABILITY=7.0`, `llama-bench -p 512 -n 128 -r 3`, two
alternating rounds:

| | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| OpenCuda, this branch | 11,223 / 11,649 | 243.3 / 245.8 |
| llama.cpp's SYCL backend (reference, measured 2026-11-27 on this machine) | 25,964 | 403 |

`test-backend-ops -b CUDA0` with the same build: SOFT_MAX 215/215, RMS_NORM
121/121, MUL_MAT 2724/2724, GET_ROWS 228/228, CPY 286/286, ROPE 514/514,
ADD 173/173, MUL 143/143, SUM_ROWS 11/11, ARGMAX 7/7, FLASH_ATTN_EXT with
head size 128 859/859. FlashAttention builds with a Clang carrying the fix
proposed upstream (llvm/llvm-project#246644, `compiler/spirv/clang-hipspv-kernel-address.patch`).
Over all FLASH_ATTN_EXT cases, 4178/4370 pass with IGC 2.41; the 192 others
were traced to IGC 2.41's automatic large-register-file mode (the cases
checked with IGC 2.42 compile correctly; forcing 256 registers per thread
still fails some, nondeterministically).

## Runtime and library tests on the GPU

Written for OpenCuda, run on the B70 (SPIR-V path): symbol copies
(`rt_symbols` 32/32), managed memory (`rt_managed` 45/45), page-locked host
memory used by kernels (`rt_pinned_mapped` 21/21), memory API (`rt_mem`
49/49), warp shuffles and votes with short last warps (`rt_warp_lang`
346/346), `__half2` comparisons (`rt_half2_cmp` 7073/7073), conversion and
math intrinsics (`rt_math_ext_gpu` 71/71), large kernel arguments
(`rt_big_args` 8/8), cuBLAS accuracy (`cublas_accuracy` 1959/1959, int8
GEMM exact). The CPU path suite (`ctest`) passes 197/197.

## Known limits

- A kernel waiting on a flag the host sets in page-locked memory may not see
  the write (it did once per process in our tests, not reliably). Programs
  that hand off work between host and a running kernel through host memory
  can hang on this path.
- Device variables shared between translation units (relocatable device
  code, `-rdc`) are separate per file (HeCBench egs-cuda crashes).
- No Thrust, cuSPARSE, cuSOLVER, cuRAND, cuFFT, CUDA graphs, inline PTX.
- Token generation in llama.cpp is ~60% of the SYCL backend's speed; SYCL
  uses Intel-specific weight layouts that a CUDA compatibility layer does
  not reproduce.
- The tests were written by the same people (and AI sessions) as the code.
  Independent review is welcome.

# Rodinia programs on the CPU path

`run.sh` fetches the public Rodinia benchmark suite (CUDA and OpenMP
versions, a fixed revision), builds each CUDA program unmodified with
`compiler/probe/ocu_cc.py --target cpu`, runs it and checks the result
against something that does not depend on this project:

| check | programs |
|---|---|
| the OpenMP version of the same program, same inputs | pathfinder, nw, bfs, lavaMD, streamcluster, backprop, srad_v2, particlefilter, b+tree |
| a serial reference written from the kernel's formula (`hotspot_ref.c`) | hotspot |
| a double-precision solver from the input formula (`gaussian_ref.py`) | gaussian |
| the program's own check | lud, hotspot3D, huffman, hybridsort, mummergpu (its `-C` CPU version) |
| an independent computation (`nn_check.py`) | nn |
| the serial version of the suite (`openmp/kmeans/kmeans_serial`) | kmeans |
| the OpenMP version, on a generated video (`leukocyte_avi.c`) | leukocyte |

Programs that print nothing about their results get a small driver
(`gaussian_dump.cpp`, `backprop_dump.cpp`). It is built with
`-Dmain=ocu_prog_main` for the program's sources, so the driver's own `main`
is used, and it calls the same functions the program's `main` calls, then
prints the results. Nothing in the Rodinia sources is changed.

Inputs the script generates (fixed seeds): `gaussian_ref.py` (gaussian),
`bpt_gen.py` (b+tree tree and batch queries), `huffman_gen.py` (huffman),
`graphgen.py` (bfs), `hurricane_gen.c` (nn).

Known results on the CPU path:

- huffman: FAILS its own check by default. The GPU bitstream differs from
  the program's CPU encoder (`cpuencode.cpp`, also built by `ocu_cc`): 5,627
  differing elements for the 1 MiB input, 1,360 for 256 KiB, 376 for 64 KiB.
  The cause is in the program (see below, "huffman"); by default the CPU
  path follows C++ here and the program does not. With PTX shift semantics
  (`OCU_PTX_SHIFT=1`, see "compat" below) it passes.
  At 4 blocks (16 KiB input or less) the Pack2 kernel fails with "invalid
  launch configuration"; that is the program too (see below).

compat (`OCU_PTX_SHIFT=1 run.sh ...`): every program is built with
`ocu_cc.py --ptx-shift`, which gives shifts in device code the semantics of
the PTX `shl`/`shr` instructions (amounts at or above the width clamp to
it; `compiler/probe/ptx_shift.py`, card `cards/compiler/ptx-shift.md`).
Default runs stay standard C++. Results on 2026-11-13 (CPU path, and the
NVIDIA path on an RTX 5070 Ti):

| program | cpu, standard | cpu, ptx-shift | nvidia, standard | nvidia, ptx-shift |
|---|---|---|---|---|
| huffman 64 KiB | 376 differing | PASS | PASS | PASS |
| huffman 128 KiB | 766 differing | PASS | PASS | PASS |
| huffman 256 KiB | 1,360 differing | PASS | PASS | PASS |
| huffman 512 KiB | 3,836 differing | PASS | PASS | PASS |
| huffman 1 MiB (5 runs) | 5,627-5,628 differing | PASS (5/5) | PASS (5/5) | PASS (5/5) |
| the other 17 (before textures, below) | 14 equal, 2 built only, kmeans does not build | same | same | same |

The NVIDIA standard column passes by luck, not by rule: Clang turns a C++
shift into an LLVM shift, whose result is poison at or above the width, and
the optimizer may use that (`b == 32 ? a << b : 5` compiles to the constant
5; a store of `dw << (32 - bit)` under `if (bit == 0)` is deleted). The
huffman kernels happen to compile to plain `shl`. The option makes the
NVIDIA path follow PTX too.

Cost of the option, on a kernel that does nothing but shift by amounts read
from memory (none can be folded): NVIDIA, median 7.60 vs 7.60 ms and 7.11 vs
6.95 ms (off vs on; the device pass uses the clamping `shl`/`shr` directly);
CPU path (4 threads, three rounds of 41 runs), medians 39.4/36.1/39.4 ms off
and 39.1/42.4/40.2 ms on, about 6% (an earlier round of 21 runs: 15%); the
slow 10% move by the same order. Shifts by integer literals below 32 are
not changed, so ordinary code pays less. Compile time: one Clang syntax pass
per .cu file (huffman's main file: CPU 4.1 s to 4.9 s, NVIDIA 1.4 s to 4.0 s).

Differences that come from the programs themselves, not from this
project (checked by reproducing the kernels as written):

- nn (CUDA): the selection loop's record swap repeats records; the check
  reproduces that loop.
- hotspot: the CUDA and OpenMP versions treat chip edges differently, so
  the CUDA result is checked against the CUDA formula instead.
- srad_v1 (CUDA): the last block of the reduce kernel adds `d_sums[i]`
  where `d_sums[i * d_mul]` is meant, so the image statistics are wrong
  on any correct implementation (here 225,637 for a sum whose exact value
  is 395,467; a serial run of the kernel as written gives 225,732, the
  same within float rounding). Its output therefore differs from the
  OpenMP version; it is listed as built only.
- particlefilter (CUDA vs OpenMP): with one filter step the estimates are
  identical (checked). Over more steps the two versions sum the weights in
  a different order; a resampling index then flips and the later estimates
  move (about 0.004 after 3 frames, about 0.2 after 10). Only one step is
  compared.
- backprop (CUDA): the input-layer weights are never copied back into the
  network, so they are not compared; the hidden and output layers are.
- huffman (CUDA): two kernels shift a 32-bit value by 32, which C++ leaves
  undefined. `pack_kernels.cu` lines 35, 40 and 50 compute `dw << 32-bit`;
  when a block's bitstream starts on a word boundary (`bit == 0`) the
  program needs 0 there. `vlc_kernel_sm64huff.cu` line 141 masks with
  `(1<<wrbits)-1`; when four codewords together are longer than 32 bits
  after the first word (`wrbits == 32`) it needs all ones. The NVIDIA
  `shl` instruction clamps the shift amount to the register width (PTX ISA,
  `shl`), which gives those values; the x86 shift instruction uses the
  amount modulo 32 (so `dw` and 0 instead). Evidence: a build with
  `-fsanitize=shift` reports "shift exponent 32 is too large" at exactly
  these four places on the 1 MiB input; a copy of the sources with only
  those shifts made explicit (`bit ? dw << (32-bit) : 0`, and all ones for
  `wrbits >= 32`) passes the program's own check on the CPU path for six
  inputs (64 KiB to 1 MiB, five repeated runs at 1 MiB), with the backend
  unchanged. Fixing only `pack_kernels.cu` leaves 5,368 differing elements
  at 1 MiB, fixing only line 141 leaves 264. The unmodified program passes
  on the RTX 5070 Ti for the same six inputs. There is no warp-synchronous
  code in the program (no shuffles, votes, `volatile` shared memory or
  `tid < 32` sections), so warp lockstep is not involved.
  Pack2 at 4 blocks: `main_test_cu.cu` launches `pack2<<<num_blocks/16, 16>>>`
  with `num_blocks` = elements / 256, so fewer than 16 blocks (an input under
  16 KiB) gives a grid of 0 blocks, which CUDA rejects as an invalid launch
  configuration (each grid dimension must be at least 1). Our CPU and NVIDIA
  backends both reject it before launching; the limit is the documented one,
  not one of ours.
  The project's decision: the default stays standard C++; the opt-in
  `--ptx-shift` gives the PTX-defined result (the published instruction
  semantics, not an observation), see "compat" above.

Not checked:

- myocyte: the program prints only timings and keeps the solver's results in
  arrays that `work()` frees before it returns. Checking it needs a copy of
  `work()`, so it is built only.
- srad_v1: see above.

Other programs are built only when they need input files the script does
not generate.

textures (card `cards/runtime/texture.md`): hybridsort, kmeans, leukocyte
and mummergpu use texture references (removed from the CUDA runtime in
12.0); on the CPU path they are provided and sampled in software. Results on
2026-11-13 (CPU path):

| program | before | now |
|---|---|---|
| hybridsort | not in the script (did not build) | PASSED, own check (4,194,304 floats against qsort) |
| kmeans | did not build | builds; crashes at run time (below) |
| leukocyte | not in the script | 4 cells as the OpenMP version; 118 candidate positions identical; after 3 frames 3 of 4 cell positions within 3e-5, one 0.09 px apart (below): reported FAIL at tolerance 1e-5 |
| mummergpu | not in the script | output identical to its own CPU version (2,000 reads, 3,048 matches) |

Whole script: 21 programs, 21 built, 16 checked equal (the failures:
huffman with standard shifts, kmeans, leukocyte). With `OCU_PTX_SHIFT=1`:
19 built, 16 equal (huffman passes; kmeans and mummergpu stop in the Clang
syntax pass of `--ptx-shift`: GCC's `omp.h` uses an attribute form Clang
rejects, and mummergpu's kernel file writes `"..."NAME` without a space,
which C++11 reads as a literal suffix).

What the script does for them, without changing their sources:

- hybridsort includes `GL/glew.h`, `GL/glut.h` and `cuda_gl_interop.h` and
  uses none of them: empty files are supplied where the system has none.
  `checkCudaErrors` comes from the program's bundled helper header only under
  a condition our headers do not meet; it is defined on the command line as
  the call itself.
- leukocyte: the suite's sample video is not in its source tree.
  `leukocyte_avi.c` writes a 640x480 grey video (dark rings on a textured
  background) with the program's own `avilib.c`.
- mummergpu: its host-only C++ files are built with the host compiler, as its
  Makefile does, at -O0 and with `-include unistd.h` (they call `read` and
  `lseek` without including it). With `g++ -O2` they crash while flattening
  the suffix tree, in the program's `-C` CPU version too.

Differences that come from the programs:

- kmeans passes the name of its `__constant__` array as a string,
  `cudaMemcpyToSymbol("c_clusters", ...)`, a form the runtime API removed in
  CUDA 5.0. On the CPU path the string's address is taken for the symbol and
  the copy writes into the string literal, which crashes.
- leukocyte: the texture kernels (GICOV, dilation) give the same candidate
  cells as the OpenMP version (checked with a debug copy that prints them).
  The difference starts after them: one cell's detected centre is 0.25 px
  apart, and 0.09 px after tracking. The CUDA version computes GICOV in
  float with float sine and cosine tables, the OpenMP version in double;
  that this is the cause is likely but not shown.

    conformance/external/rodinia/run.sh <cmake build dir> [work dir]

Sources are fetched into the work dir (default `~/.cache/ocu-rodinia`),
never into this repository. Rodinia is distributed by the University of
Virginia under its own licence terms; nothing from it is copied here.

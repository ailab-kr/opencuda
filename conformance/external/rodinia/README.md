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
| the program's own check | lud, hotspot3D, huffman |
| an independent computation (`nn_check.py`) | nn |

Programs that print nothing about their results get a small driver
(`gaussian_dump.cpp`, `backprop_dump.cpp`). It is built with
`-Dmain=ocu_prog_main` for the program's sources, so the driver's own `main`
is used, and it calls the same functions the program's `main` calls, then
prints the results. Nothing in the Rodinia sources is changed.

Inputs the script generates (fixed seeds): `gaussian_ref.py` (gaussian),
`bpt_gen.py` (b+tree tree and batch queries), `huffman_gen.py` (huffman),
`graphgen.py` (bfs), `hurricane_gen.c` (nn).

Known results on the CPU path:

- huffman: FAILS its own check. The GPU bitstream differs from the program's
  CPU encoder (`cpuencode.cpp`, also built by `ocu_cc`); for the 1 MiB input
  the first differing element is 189,425 and there are 5,628 differing
  elements in all; for 64 KiB there are 376.
  At 4 blocks the Pack2 kernel fails with "invalid launch configuration".
  Not yet known whether the cause is in the CPU path or in the program; it is
  left as FAIL on purpose.

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

Not checked:

- myocyte: the program prints only timings and keeps the solver's results in
  arrays that `work()` frees before it returns. Checking it needs a copy of
  `work()`, so it is built only.
- srad_v1: see above. kmeans: `texture` references; it does not build.

Other programs are built only when they need input files the script does
not generate. Programs that use texture references (removed from the CUDA
runtime in 12.0) are expected not to build.

    conformance/external/rodinia/run.sh <cmake build dir> [work dir]

Sources are fetched into the work dir (default `~/.cache/ocu-rodinia`),
never into this repository. Rodinia is distributed by the University of
Virginia under its own licence terms; nothing from it is copied here.

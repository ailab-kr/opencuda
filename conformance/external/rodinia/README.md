# Rodinia programs on the CPU path

`run.sh` fetches the public Rodinia benchmark suite (CUDA and OpenMP
versions, a fixed revision), builds each CUDA program unmodified with
`compiler/probe/ocu_cc.py --target cpu`, runs it and checks the result
against something that does not depend on this project:

| check | programs |
|---|---|
| the OpenMP version of the same program, same inputs | pathfinder, nw, bfs, lavaMD, streamcluster |
| a serial reference written from the kernel's formula (`hotspot_ref.c`) | hotspot |
| the program's own check | lud, hotspot3D |
| an independent computation (`nn_check.py`) | nn |

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

Other programs are built and run to completion, or only built when they
need input files the script does not generate. Programs that use texture
references (removed from the CUDA runtime in 12.0) are expected not to
build.

    conformance/external/rodinia/run.sh <cmake build dir> [work dir]

Sources are fetched into the work dir (default `~/.cache/ocu-rodinia`),
never into this repository. Rodinia is distributed by the University of
Virginia under its own licence terms; nothing from it is copied here.

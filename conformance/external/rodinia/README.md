# Rodinia programs on the CPU path

`run.sh` fetches the public Rodinia benchmark suite (CUDA and OpenMP
versions, a fixed revision), builds each CUDA program unmodified with
`compiler/probe/ocu_cc.py --target cpu`, runs it and checks the result
against something that does not depend on this project:

| check | programs |
|---|---|
| the OpenMP version of the same program, same inputs | pathfinder, nw, bfs, lavaMD |
| a serial reference written from the kernel's formula (`hotspot_ref.c`) | hotspot |
| the program's own check | lud, hotspot3D |
| an independent computation (`nn_check.py`) | nn |

Other programs are built and run to completion, or only built when they
need input files the script does not generate. Programs that use texture
references (removed from the CUDA runtime in 12.0) are expected not to
build.

    conformance/external/rodinia/run.sh <cmake build dir> [work dir]

Sources are fetched into the work dir (default `~/.cache/ocu-rodinia`),
never into this repository. Rodinia is distributed by the University of
Virginia under its own licence terms; nothing from it is copied here.

# Experiment scripts, 2026-11-28

The scripts behind [the day's log](../../arc-pro-b70-log-2026-11-28.md) that are
not part of the test suite. They were run by hand on one machine and assume
its layout (`~/opencuda-dev` the source tree, `~/oc-repro` a scratch folder,
`~/llama-spvdev` a llama.cpp build, `~/cmp/HeCBench` HeCBench, `~/qlora/.venv`
a Python environment with PyTorch XPU); they are kept as a record of what
was measured and how, not as tools.

| file | what it did |
|---|---|
| `bisect16.sh` | llama.cpp builds each with one part of the native binary16 change, then test-backend-ops and llama-bench |
| `segbis.sh`, `optbis.sh`, `fnbis.sh` | segsort-cuda: which OpenCuda step, which opt pass, which function (opt-bisect) makes the GPU result wrong |
| `spv_constgep.ll` | a small module with constant getelementptrs on shared memory; LLVM 23's SPIR-V back end compiles this one right, so it is not yet a reproducer |
| `genlocal.cu` | shared memory written through a generic pointer in a function that is not inlined, then read as shared memory (right on the B70) |
| `jitcache.sh` | llama.cpp start-up with the driver's compiler cache empty, 1 GB-limited and unlimited |
| `prebuild.sh` | llama.cpp start-up with and without building the other modules ahead |
| `hec_rerun.sh` | the OpenCuda side of HeCBench again with the development toolkit (same rules as `tools/bench/hecbench_compare.sh`) |
| `midm_fullft.py` | full fine-tuning of Mi:dm 2.0 with PyTorch XPU: mixed precision with 8-bit AdamW, or pure bf16; loss, peak memory, tokens/s, share of weights a step changed |
| `midm_lora.py` | LoRA fine-tuning of Mi:dm 2.0 with PyTorch XPU on a small Korean instruction set |
| `xpu_probe.py` | device-only compute and host-to-device copies by size (used after the GPU hang) |

`midm_fullft.py` and `midm_lora.py` did not produce results: the GPU stopped
working before they could (see section 8 of the log).

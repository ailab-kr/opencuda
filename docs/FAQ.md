# Frequently asked questions

**English** · [한국어](FAQ.ko.md)

These are the questions developers and researchers most often ask about a project like this one. The answers state where OpenCuda is today, including what does not work yet.

- [Positioning](#positioning)
- [Correctness](#correctness)
- [Performance](#performance)
- [Coverage](#coverage)
- [Using it](#using-it)
- [Project](#project)

---

## Positioning

### How is this different from HIP/HIPIFY, SYCLomatic, ZLUDA or chipStar?

Those tools are good at one job: moving CUDA code onto one vendor's stack, or onto one portability layer. OpenCuda does two additional things:

1. It publishes a **reference implementation** and a **conformance suite**, so any implementation can be measured, including those tools.
2. It treats **accelerators without SIMT threads** (NPUs) as a first-class target, through an operator table, instead of only GPUs.

Other compatibility tools are not rivals to the conformance suite. They can use it to measure themselves.

### Most AI work goes through PyTorch, JAX or Triton. Why target CUDA C++ at all?

Frameworks hide the hardware from model authors. Underneath, though, a large amount of performance-critical code is still written in CUDA C++. That includes inference engines, custom kernels, scientific codes and the libraries the frameworks call. Each new chip has to support that code before it can run real software. OpenCuda works at that layer, so it complements framework backends rather than replacing them.

### Why source compatibility and not binary compatibility?

Source is portable across instruction sets. CPU, GPU and NPU targets can all compile it. It also keeps the project clean-room: no vendor binaries are read or run. The cost is that you need the source. Prebuilt binaries, such as framework wheels built for NVIDIA GPUs, are out of scope.

### Is the NVIDIA backend the reference?

No. It is one backend among several, used as a cross-check. Reference answers come from documented behaviour, then from mathematics computed in high precision, and only then from reviewed observations.

---

## Correctness

### How do you define "correct" without copying NVIDIA's behaviour?

Each test has an answer source, in this order:

1. what the public documentation states;
2. what mathematics fixes, computed in high precision;
3. where neither decides, a reviewed observation, recorded as such.

Integer and control-flow results must match exactly. Floating-point results must be within a per-operation tolerance, such as ULP bounds for math functions or ROW and DOT classes for reductions.

### Floating point is not associative. How can an application "match"?

It cannot match bit for bit in general, and the suite does not require that. For applications (level L5), outputs are compared with a high-precision oracle using a distribution distance (KL divergence). The allowed band is derived from how much the application's own official backends differ from each other.

For llama.cpp, our paths are as close to the oracle as llama.cpp's own CPU backend, and much closer when the prompt is processed in one batch. Different top tokens appear only at near-ties. See [the study](llama-cpp-logit-agreement.md).

### How do you know the tests themselves are right?

- **Mutants.** Deliberately broken implementations must fail the suite. When a seeded defect goes undetected, the test is fixed.
- **Separate authors.** Tests and implementation of the same feature are written separately, by different people. These independent tests have found real defects in this project, such as double rounding in bf16 fma.

---

## Performance

### Will code tuned for NVIDIA GPUs be fast elsewhere?

Not automatically. Portability of results and portability of performance are different goals. Code tuned for 32-wide warps, a specific shared-memory size or tensor-core instructions may run correctly but slowly on another chip. OpenCuda's first goal is correctness. Speed is measured and reported against each chip's native software. We publish the numbers as they are, and do not claim parity.

### The CPU path is about 60× slower than llama.cpp's own CPU backend. What is it for?

The CPU path is the **reference**. It is deterministic, can run threads in random order to expose ordering bugs, and runs on any machine. It is not meant to replace a tuned CPU backend. Making it faster is still on the roadmap, because a faster reference makes the suite cheaper to run. That work includes sending matrix work to a fast CPU library and compiling barrier-free regions as vector loops.

### Are there results on non-NVIDIA GPUs or NPUs?

Partly:

- **Apple GPU (Metal):** the backend and translator pass their tests on M3 and M1 Pro.
- **OpenCL:** passes on an open implementation (PoCL).
- **NPUs:** the operator table is exercised by an NPU simulator.

End-to-end application numbers on AMD, Intel and NPU hardware are the top roadmap item.

---

## Coverage

### Which CUDA version do you target?

The headers declare API level 12.3. llama.cpp builds with its default settings (`CUDART_VERSION` 12030). An older profile (11060) is kept for comparison. Features are added in the order real applications need them, not by version number.

### What is not supported yet?

| Area | State |
|---|---|
| CUDA graphs | stream capture returns "not supported" |
| Textures: hardware texture units, mipmapped arrays, sRGB, cubemap and gather fetches; surfaces | texture objects, CUDA arrays and (CPU path) legacy texture references work, sampled in software (card `runtime-texture`). Mipmaps, sRGB and format-changing views return "not supported"; cubemap and gather fetches and the surface API are not declared. On the NVIDIA path texture references stop the build; the Metal translator reports texture kernels per kernel |
| cuBLASLt matmul | headers exist, `cublasLtMatmul` returns "not supported" |
| cuDNN, cuFFT, cuRAND, cuSPARSE, NCCL | not provided |
| Inline PTX assembly | passes through on the NVIDIA path; other paths need a C++ equivalent or report the kernel as unsupported |
| Thrust | not provided yet; the CUB subset it would build on exists |
| Native MSVC build | not supported; Windows builds use MinGW-w64 |

Every unsupported call returns a "not supported" error rather than pretending to succeed.

### Isn't the project focused on llama.cpp?

So far, yes. llama.cpp was chosen because it is large, widely used, and ships its own operator-level reference test. That made it a strong first proof. To avoid tuning the layer to one program, the next targets cover other kinds of code:

- broad benchmark collections, for the build rate across hundreds of programs;
- a training code base, for backward passes;
- a Thrust/CUB-heavy library;
- classic HPC suites.

### What about PyTorch?

PyTorch needs cuDNN-, cuBLASLt- and NCCL-level libraries and a large part of the runtime. It is a long-term goal that we will reach library by library. Each library will get its own conformance suite.

---

## Using it

### Do I have to change my build files?

Usually not:

- **Makefile projects:** set `NVCC` to the bundled `ocu-nvcc` driver.
- **CMake projects:** use the bundled toolchain file. `enable_language(CUDA)` and `find_package(CUDAToolkit)` then work unchanged.

See the [README quick start](../README.md#quick-start).

### Can I mix OpenCuda with NVIDIA's own toolkit in one program?

No. A program is built either with OpenCuda's headers and libraries or with the vendor's. Mixing the two in one binary is not supported.

### How do I reproduce the numbers in the README?

Each result names its feature card in `cards/`. The card lists the commands, the inputs, the machine and the date. `tools/scoreboard.py` turns result records into the scoreboard. Speed is always reported as median and slow 10 %.

---

## Project

### Is it production-ready?

Not yet. OpenCuda is pre-release (0.x). Interfaces may change, and the backend ABI is versioned (`OCU_BACKEND_ABI`).

### How can a chip vendor or NPU team add support?

Read the [backend guide](public/BACKEND_GUIDE.md):

- **GPU-style chips** implement one C interface of 26 functions.
- **NPUs** implement the [operator table](spec/op-table.md).

Conformance is reported by level, B0 to B3.

### Who runs the project, and how is neutrality kept?

The project is meant to be run by a non-profit steward, and no chip vendor has special rights. The rules are in [GOVERNANCE](public/GOVERNANCE.md). The code is under the Apache License 2.0.

### Why "OpenCuda"? Is it affiliated with NVIDIA?

No. OpenCuda is an independent project and is not affiliated with or endorsed by NVIDIA. CUDA is a trademark of NVIDIA Corporation. The name describes the kind of source the project accepts. Code identifiers use the short prefix `ocu`.

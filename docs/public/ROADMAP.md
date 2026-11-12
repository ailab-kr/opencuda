# Roadmap

**English** · [한국어](ROADMAP.ko.md)

> DRAFT for review. No dates are promised; items are ordered by intent, and
> each has an exit measure so progress can be checked.

## Where we are

Measured so far (details in [README.md](../../README.md)):

- llama.cpp (about 150 CUDA source files) builds unmodified; its operator test
  passes 15,894 / 15,894 on the CPU path and on the NVIDIA GPU path.
- The BLAS-compatible library passes 1,395 / 1,395 cases on CPU and GPU, and
  the CUB subset passes 5,500 independent checks.
- Backends: CPU (reference) and NVIDIA GPU work; the Apple GPU (Metal)
  backend is a first cut (backend tests 53 / 53; the CUDA to Metal Shading
  Language translator handles 35 of 79 Rodinia kernels); the OpenCL backend
  works on PoCL (103 / 103 tests); the NPU simulator is a design target. AMD
  and Intel GPUs have device code that compiles but no runtime backend yet.
- Rodinia (18 CUDA programs, unmodified): 17 build and 14 match an
  independent reference.
- Check suites: Linux 204 / 204, macOS 116 / 116, Windows 10 37 / 38
  (1 skipped).
- The conformance harness, result format and scoreboard exist.

## Next

| Item | Exit measure |
|---|---|
| Conformance suites L1 to L3 (build, runtime, language) | each documented API call and language feature has a test; mutants show the tests catch seeded defects |
| More Rodinia programs checked against independent references; remaining programs that need input data | every buildable program has a reference check |
| Metal: templates, global device variables, unified memory, programs built end to end | Rodinia and the runtime test programs pass on Apple GPUs |
| OpenCL backend | backend level B0 and B1 on at least one open OpenCL implementation |
| AMD backend (runtime on top of the vendor's public runtime layers) | B0 and B1 on real hardware |
| Command-line driver that accepts the usual compiler flags | existing build files compile unmodified |
| Kernel-time comparison against vendor runtimes on the same chip | speed ratio recorded in result records |

## Later

| Item | Exit measure |
|---|---|
| Library layer beyond BLAS (FFT, random numbers, sparse, neural-network primitives) | per-library L4 suites |
| Texture API, graphs, peer access, cooperative groups | each passes its L2 tests or reports "not supported" |
| Library and operator-level backends for NPUs without SIMT cores (planned, see BACKEND_GUIDE.md) | RFC accepted, one reference provider, B3 reported without B2 |
| More operating systems and architectures (native Windows build, RISC-V) | check suite green on each |
| Public scoreboard with submitted result records from other parties | a documented submission and verification path |

## How the plan changes

Roadmap changes go through the pull-request process; items that alter the
backend interface need an RFC (GOVERNANCE.md).

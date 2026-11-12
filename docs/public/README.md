# PROJECT_NAME

> DRAFT for review. `PROJECT_NAME` is a placeholder; the public name is not decided.

PROJECT_NAME builds and runs unmodified CUDA C++ source on chips that are not
NVIDIA GPUs, and ships a public conformance test suite that measures how
closely each chip and each implementation follows the documented behaviour of
the CUDA Runtime API and the CUDA C++ language.

It is a compatibility layer plus a yardstick:

- **Layer**: a runtime API with CUDA-compatible names and headers, a library
  layer (BLAS first), and a compiler path that takes `.cu` files as they are.
  Everything chip-specific sits behind one small C interface
  (`backends/interface/ocu_backend.h`). Adding a chip means implementing that
  interface and nothing else.
- **Yardstick**: a conformance suite with exact reference answers, tolerances,
  a result-record format, and a static scoreboard. The suite uses only what a
  user program sees, so any other compatibility approach can be measured by
  the same tests.

> CUDA is a trademark of NVIDIA Corporation. This project is not affiliated
> with or endorsed by NVIDIA. "CUDA C++ source" and "CUDA Runtime API" are
> used here only to describe what the software accepts. The headers and
> implementation are written from publicly available documentation.

## Backends

| Backend | State | What runs |
|---|---|---|
| CPU (x86-64, ARM64) | working, reference implementation | all interface entries; GPU threads run as fibers, deterministic and random orderings, 32- and 64-wide warps |
| NVIDIA GPU (reference) | working | the installed driver's public API is loaded at run time; no vendor files are needed to build; device code is PTX produced by an open-source compiler |
| Apple GPU (Metal) | working, first cut | backend plus a CUDA-to-MSL translator; tested on Apple M3 and M1 Pro |
| OpenCL | in progress | not in the tree yet |
| AMD, Intel | device code compiles; no runtime backend yet | gfx1100, gfx1201 and sm_80 compile checks only |

## What works today

Measured on one Linux machine (WSL2, x86-64) with an NVIDIA RTX 5070 Ti; Apple
numbers were measured on an M3 and an M1 Pro. See the scoreboard for the
current figures; the table is a snapshot.

| Area | Result |
|---|---|
| Test suite (CTest) | 66 of 66 pass (CPU, NVIDIA GPU and translator host checks) |
| BLAS layer, 20 functions | 537 of 537 reference cases pass on the CPU backend; 15 deliberately broken implementations are all detected by the suite |
| Runtime API | more than 80 functions (devices, errors, memory incl. managed memory, streams, events, launch, occupancy), vector types, half precision, device built-ins, atomics, warp functions |
| API coverage on real programs | 87 percent of the Runtime API call names used by llama.cpp, 90 percent for Rodinia (by name) |
| Unmodified programs | 17 of 18 Rodinia CUDA programs build without source changes; in the latest run 13 (CPU path) and 14 (NVIDIA backend) match an independent reference (the OpenMP version of the same program, a serial reference, or the program's own check) |
| NVIDIA backend (RTX 5070 Ti) | backend test 50 of 50; 9 runtime test programs pass |
| Apple backend | backend test 53 of 53 on M3 and on M1 Pro; translated kernels 23 of 23; first-run 1M-element saxpy 2.3 ms (M3), 1.4 ms (M1 Pro) |
| Compiler path | unmodified `.cu` files compile for sm_80, gfx1100 and gfx1201 targets; the CPU path preprocesses, rewrites launch syntax, and compiles as C++ |
| CPU backend speed | 4M-thread saxpy 19 ms; 64x64x64 sgemm 0.17 ms (24-core machine) |

Not supported yet: texture references and the texture API, graphs, peer
access, a `nvcc`-style command-line driver, kernel-time comparison against
vendor runtimes. Features that are not supported return a "not supported"
error; the project does not report success for work it did not do.

## Quick start

Requirements: CMake 3.20 or newer, a C11/C++17 compiler, Python 3.9 or newer.
For the compiler paths: an LLVM Clang with the target you need.

### Linux

```sh
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

The NVIDIA backend is built by default and finds the driver at run time; with
no driver the tests that need it are skipped. Select a backend at run time with
`OCU_BACKEND=cpu` (or `nvidia`).

Full check with compilers and sanitizers: `sh tools/check.sh`.

### macOS (Apple silicon)

```sh
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

The Metal backend is on by default on Apple platforms. Metal tests skip when
no GPU is available (for example on CI runners without one).

### Windows

The CPU path cross-builds with MinGW-w64 (POSIX threads) from Linux and runs
under Wine or on a Windows machine:

```sh
cmake -S . -B build-win -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/x86_64-w64-mingw32.cmake
cmake --build build-win -j4
ctest --test-dir build-win
python3 tools/make_test_bundle.py build-win tests.zip   # runnable on a plain Windows machine
```

A native MSVC build is not supported yet.

### Build and run a CUDA C++ file on the CPU path

After `cmake --install build --prefix /opt/project`, another CMake project uses
`find_package(Ocu)`, links `ocu::cudart`, and calls
`ocu_add_cpu_cu_executable()` on its `.cu` files. (The package and function
names carry the working prefix and will change with the public name.)

## How conformance is measured

The suite has five levels, from "does it compile" to "does a real program give
the same answer":

| Level | Question |
|---|---|
| L1 build | does the source compile unmodified |
| L2 runtime | does each API call behave as documented |
| L3 language | do device-code semantics match |
| L4 libraries | are numerical results within the stated tolerance of an exact reference |
| L5 applications | does a real program produce the same result as an independent reference |

Reference answers come from, in order: documented behaviour, mathematically
determined results (computed in high precision on the CPU), and, only where
those do not decide, reviewed observations. Integer and control-flow results
must match exactly; floating-point results must be within a per-operation
tolerance; NaN and infinity must match position for position. Speed is
reported as a median and a slow-10 percent value with the slowest test named,
never as a mean alone.

Every run produces one JSON result record (format in `conformance/README.md`).
`tools/scoreboard.py` turns result records, CTest output and Rodinia output
into a Markdown file and one self-contained HTML page:

```sh
python3 tools/scoreboard.py build/tests/blas_cpu_result.json \
    --ctest cpu=ctest.txt --rodinia cpu=rodinia.txt --out-dir scoreboard/
```

The suite checks itself: a correct implementation passes, and deliberately
broken ones (mutants) must be caught.

## Adding a chip

See [BACKEND_GUIDE.md](BACKEND_GUIDE.md). Governance is described in
[GOVERNANCE.md](GOVERNANCE.md), the plan in [ROADMAP.md](ROADMAP.md), the
security policy in [SECURITY.md](SECURITY.md).

## License

Not yet chosen. See [LICENSING-OPTIONS.md](LICENSING-OPTIONS.md).

[한국어](README.ko.md)

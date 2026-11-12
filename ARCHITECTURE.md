# Architecture

**English** · [한국어](ARCHITECTURE.ko.md)

This document defines the parts of the repository, the boundaries between them, and the rules that protect those boundaries.

Current results by area and platform are in [README.md](README.md#results-so-far).

## 1. Goal and design principles

Build and run source written for CUDA on other GPUs without changing it, and measure the level of compatibility with the same tests.

1. **Keep chip differences in one place.** Everything that differs between chips lives below the device interface layer.
2. **Tests are the standard.** An implementation is complete when it passes the conformance tests.
3. **Use what exists.** Use public compilers and per-chip foundations. Build only the layer that connects them.
4. **No false success.** A feature that does not work returns a "not supported" error.
5. **Verify without a GPU.** The same code runs on the CPU backend.

## 2. Layers and parts

```
User's CUDA source and build settings (unchanged)
        │
   ┌────┴─────────┬──────────────┐
compiler/         runtime/         libs/
command compat,   execution API,   matrix, FFT, random,
code generation   headers          neural-network op APIs
   └────┬─────────┴──────────────┘
        │
   runtime/core        backend registration, chip-independent common functions
        │
   backends/interface  device interface layer spec (one header)
        │
   ┌────┼──────────┬──────────┐
 backends/cpu  backends/amd  backends/intel  ...

 conformance/  checks every layer and every chip with the same tests
 tuning/       speed measurement and per-chip settings
```

| Folder | Role | Status |
|---|---|---|
| `backends/interface/` | Device interface layer spec `ocu_backend.h` | Draft exists |
| `backends/cpu/` | CPU reference implementation | All 26 spec functions |
| `backends/amd/`, `backends/intel/` | Per-chip implementations | None |
| `backends/opencl/` | Devices with OpenCL 2.0 or later (shared virtual memory, SPIR-V or C++ for OpenCL). Device code goes through `compiler/opencl/cu2ocl.py` | First implementation; tests pass on OpenCL for CPU (PoCL) |
| `backends/npusim/` | Backend that imitates NPU constraints: no kernels, op table only, device memory the host cannot read, on-chip memory limit | 5 ops |
| `backends/ops_ref/` | Shared checks and reference computation for the op table (not a backend; backends that provide an op table link it) | Exists |
| `runtime/core/` | Backend registration and selection | Exists |
| `runtime/include/` | Headers that user programs see: `cuda_runtime_api.h`, `cuda_runtime.h`, `cuda.h`, `cuda_fp16.h`, vector types, CPU-path language support `ocu_cuda_lang_cpu.h`, and a header-only subset of parallel primitives `cub/cub.cuh` (compiled per target together with the program, `cards/libs/cub-subset.md`) | About 80 functions |
| `runtime/cudart/` | Execution API implementation (`ocu_cudart`): device, error, memory, stream, event, launch. Calls only the backend spec | Exists |
| `libs/` | Per-library compatibility layers | `libs/cublas`: a subset of `cublas_v2.h` (handle, stream, GEMM family, part of levels 1 and 2, batched strsm). A single CUDA C source passes on both CPU and NVIDIA. `libs/blas`: 20 functions (internal names); 526 reference cases pass on the CPU |
| `compiler/` | Command compatibility, build tool integration, device math functions | Experimental driver `compiler/probe/ocu_cc.py` (the CPU path uses it to build real programs), CMake function `ocu_add_cpu_cu_executable`, nvcc-compatible command driver `compiler/driver/ocu-nvcc` and toolkit folder (CMake toolchain file `OcuCUDA.cmake`, `docs/compiler-driver.md`) |
| `conformance/` | Conformance tests, scorer, result format | 526 reference answers for the 20 BLAS functions, error bounds, self-verification |
| `tuning/` | Speed measurement, auto-tuning, per-chip setting tables | Benchmark tool `ocu_bench`, which follows the measurement rules (median and slow 10 %), and BLAS benchmark `bench_blas` |
| `cards/` | Feature cards | Template, 2 part cards, 20 BLAS cards |
| `tests/` | Basic behavior tests of the skeleton itself | Exists |

## 3. Dependency rules

An arrow means "may use." Any direction not in the table is forbidden.

| Part | May use |
|---|---|
| User program | Public headers of `runtime/include` and `libs` |
| `libs/*` | `runtime/core`, `backends/interface` |
| `runtime` (execution API) | `runtime/core`, `backends/interface` |
| `runtime/core` | `backends/interface`. Registration functions of the backends included in the build |
| `backends/*` | `backends/interface`, and the external libraries of that chip |
| `conformance` | Only what a user program also uses |

Rules to keep

- Do not put code that branches on chip name in `runtime` or `libs`. The only place that knows chip names is `runtime/core/registry.c`.
- Backends do not use each other.
- `conformance` does not use internal headers. It sees only the same surface as the user. This lets other compatibility approaches be measured with the same tests.

## 4. Device interface layer

The spec is one file, `backends/interface/ocu_backend.h`. A backend fills in and exports one function table (`ocu_backend`).

| Group | Functions |
|---|---|
| Device | `dev_count`, `dev_info`, `dev_open`, `dev_close` |
| Memory | `mem_alloc`, `mem_free`, `mem_copy`, `mem_set`, `mem_info`, `host_alloc_pinned`, `host_free_pinned` |
| Queue | `queue_create`, `queue_destroy`, `queue_sync`, `queue_query` |
| Code | `module_load`, `module_unload`, `module_get_kernel` |
| Launch | `kernel_launch` |
| Marker | `marker_create`, `marker_destroy`, `marker_record`, `marker_wait`, `marker_query`, `marker_elapsed_ms` |
| Selection | `get_provider` |

Contract

- **A device pointer is an address that supports addition.** User code adds values to pointers, so `ptr + n` must be valid inside an allocation.
- **Work put in one queue runs in the order it was put in.** Order across different queues is not guaranteed. If `NULL` is given in place of a queue, the call waits until the work is done.
- **Report the real warp size.** Do not fake 32.
- **A free first waits for the remaining work in the queues.** `mem_free` and `host_free_pinned` wait until pending work has finished using that memory.
- **The total of launch arguments is at most `OCU_MAX_ARG_BYTES` (32768 bytes).**
- **Calling the backend again from inside a kernel returns "not supported."** This covers launch, queue sync and destroy, marker wait, and memory free.
- **A marker wait waits only for the record that existed when the wait was called.** Recording again during the wait does not extend it.
- **Errors are returned as a single `ocu_status`.** Converting it to the error code that users see is the job of `runtime`.
- **Distinguish `NOT_SUPPORTED` from `NOT_IMPLEMENTED`.** The first means the chip cannot do it; the second means it is not built yet. They are shown differently in the report card.
- **When the spec changes, raise `OCU_BACKEND_ABI`.** Registration is rejected if the number differs.

### Library providers

For libraries such as matrix operations, using a version tuned by the chip vendor is usually faster. But if `libs` called the vendor library directly, it would break the rules in section 3. So the backend exports a "provider."

```
libs/blas  ──(get_provider("blas"))──▶  backend
   │                                      │
   │  if a provider and the function      ▼
   │  exist                          vendor library
   ├────────────────────────────────▶
   │
   └─ if not ─▶ common implementation inside libs (runs via kernel_launch)
```

- `libs` decides the shape of the provider table. Both sides use it, so the header goes in `backends/interface/` (one per library, for example `ocu_blas_provider.h`).
- What exists now: the BLAS provider table (14 operations, f32 and f64) and its CPU backend implementation (`backends/cpu/cpu_blas.c`). `libs/blas` exports 20 functions on top of it.
- `libs` handles CUDA-side behavior such as argument checks, error codes, and indexing conventions. It hands only the actual computation to the provider.
- A chip without a provider still works through the common implementation. It is slower, not wrong.

### Libraries with CUDA names (`libs/cublas`)

Functions in `cublas_v2.h` do not go through the provider table. The host side is chip-neutral C++ that calls only the execution API. The computation is done by CUDA C kernels in the same file (shared-memory tiled GEMM, level 1 and 2 kernels). The project compiler (`ocu_cc`) compiles this one `.cu` separately for each target and builds a static library per target.

| Target | Library | How it is built |
|---|---|---|
| CPU path | `ocu_cublas_cpu` | `ocu_add_cu_library(... cpu ...)`: CPU-path rewrite, then C++ compile |
| NVIDIA | `ocu_cublas_nvidia` | `ocu_add_cu_library(... nvidia ...)`: device code is put into a container as PTX and embedded in the host object |
| Metal | (next) | Same source through the `cu2msl` path. There is no double precision, so build with `OCU_CUBLAS_NO_FP64` and 64F returns "not supported" |

- A program links the variant for the target it is built for: `ocu-nvcc -lcublas` (the variant that matches the target), `CUDA::cublas` in CMake (the toolkit's FindCUDAToolkit, the variant that matches `OCU_TARGET`), and inside this repository `ocu_add_cpu_cu_executable(t a.cu LIBRARIES ocu_cublas_cpu)` and `ocu_add_nvidia_cu_executable(t a.cu LIBRARIES ocu_cublas_nvidia)`.
- NVIDIA's own library is not called, even on the NVIDIA target. The point is that the same kernel serves every target.
- If a fast path that uses a vendor library becomes necessary, add it through the provider mechanism above (even then, `libs` does not branch on chip name).

### Op table (`ocu_ops`)

A chip without SIMT threads (an NPU) cannot run kernels, but it can run operations such as matrix multiply, normalization, and softmax. The op table is the second way of working for such chips. The spec is `docs/spec/op-table.md`, and the header is `backends/interface/ocu_ops.h`.

- A backend exports the table through `get_provider("ops")`. The `ocu_backend` table and `OCU_BACKEND_ABI` do not change. The op table has its own version number and size.
- Sequence: support query (op × data type) → exact query (one descriptor) → plan (fixed shape, can be precompiled, may return "not supported") → run (in a queue) → release. Calls where only the row count changes use one plan with `OCU_OP_FLAG_VAR_ROWS`.
- Tolerance is set by four classes (exact, ulp, row reduction, dot product), and passes even when the accumulation order differs.
- Upper layers (`libs/blas` and others) choose a path from the query result. They do not look at chip names.
- `backends/ops_ref` is a shared component, not a backend. The rule that backends do not use each other still holds.

## 5. The path of one run

This is the flow when a GPU program is built and run. Steps 2 to 4 for GPU chips are not implemented yet. The CPU path works as described in "CPU path" below.

1. **Build**: the compiler layer compiles the source along two branches: the host-side code, and one device code bundle per target chip.
2. **Bundle**: put the device code bundles into the executable, and attach a registration record that says which chip each bundle is for and what the function names are. One executable can hold bundles for several chips.
3. **Start**: when the program starts, the registration record is registered with `runtime`. `runtime/core` selects a backend.
4. **Launch request**: the compiler turns the launch statement in user code into a call to the launch entry point of `runtime`. `runtime` loads the bundle for the active backend with `module_load` (once only) and calls `kernel_launch`.
5. **Synchronize**: wait with `queue_sync` or a marker before reading results.

On the CPU backend, the "device code bundle" is a table of functions compiled for the CPU. The rest of the flow is the same.

To confirm: check on a real installation which function call the base compiler turns a launch statement into, and match the entry point in step 4 to it.

### CPU path (works now)

1. **Preprocess**: preprocess the `.cu` as C++17. As nvcc does, include `cuda_runtime.h` first and define `__CUDACC__`. This way launch statements inside included files and macros are seen as part of the same unit.
2. **Rewrite** (`cpu_rewrite.py`): turn `K<<<cfg>>>(args)` into `::ocu_cpu::launch(cfg, lambda, args)`, and turn `extern __shared__ T a[]` inside a function into a pointer to per-block shared memory. Everything else stays byte for byte, and the line count is kept (so error locations point to the original file).
3. **Compile and run**: compile the result as C++ and link it with `ocu_cudart`. `launch` copies the arguments and passes them to `ocu_rt_launch`. The execution API applies stream ordering rules and calls the backend's `kernel_launch`. `threadIdx` and the like inside a kernel are thread-local variables that the backend fills in each time it switches a GPU thread.

Shifts: the default is standard C++ (shifting by the width or more is undefined). With the optional `--ptx-shift` (CMake `PTX_SHIFT`), shifts in device functions are found in the Clang syntax tree and replaced with a helper call that follows the PTX definition of `shl`/`shr` (the amount is clamped to the width). This applies to both the CPU path and the NVIDIA path (`compiler/probe/ptx_shift.py`, card `compiler-ptx-shift`).

Execution API rules (behavior from the public documents):
- Legacy default stream: work on the default stream first finishes the blocking streams, and work on a blocking stream first finishes the default stream. A per-device lock is held from the ordering check until the work is put in the queue (this holds with several host threads too).
- An asynchronous copy to or from ordinary (unpinned) host memory finishes before the call returns. Only pinned memory that the runtime allocated or registered is read later. The runtime records the ranges it allocated, and uses them to decide the pointer kind and the direction of `cudaMemcpyDefault`.
- A kernel failure stays on the device. Until `cudaDeviceReset`, later sync and submit calls return the same error.
- Features that are not implemented (graphs, textures, peer access, unified memory when the device does not share host memory, and so on) are not declared, or return "not supported."

## 6. State and threads

- There is one active backend per process. It is chosen at first use, and the design assumes it does not change after that.
- The "current device" and the "last error" are kept per thread. `runtime` manages them.
- Backend functions can be called from several threads at the same time. The backend must be safe by itself.
- `registry.c` registers the built-in backends once (`pthread_once`) and guards registration, selection, and lookup with one lock. It is safe even when several threads use it for the first time at once.
- The selection at first use (`OCU_BACKEND`, or the first registered backend) is made only once. If the name is unknown, the result stays `NULL` afterward too.

## 7. CPU reference implementation

It runs GPU code as is on the CPU. The goal is to check correctness, not speed. Even so, it must be fast enough to run the tests in bulk.

- **Fibers**: one GPU thread is one fiber. The fibers of one block run in turn on one OS thread and switch only at synchronization points. So shared memory within a block needs no locks.
- **Switching**: x86-64 and ARM64 use a hand-written register switch (`cpu_fiber_switch.S`). It makes no operating system calls. Other chips use `ucontext`. The build option `-DOCU_CPU_UCONTEXT=ON` lets you compare the two.
    - Measured (24 cores): saxpy with 4 million threads 105 ms → 19 ms. Block sum with 9 syncs per thread 684 ms → 76 ms. A simple loop takes 2.6 ms.
- **Synchronization**: in one round, run all ready fibers. When all of them have reached the sync point or finished, release them together. A thread that finished before the sync counts as arrived. With `OCU_CPU_STRICT=1` it is an error.
    - Warp operations: the last lane named by the mask to arrive checks and releases the others, and keeps running without a switch. Values alternate between two slots by sequence parity, so each operation waits once.
    - In fixed order, a waiting fiber hands over directly to the next ready fiber in number order, without going through the scheduler. A finished thread starts the next not-yet-started thread on the same stack. Stacks are taken on first run and returned when done (last in, first out), so a block with no waits runs on one stack. Random order and sanitizer builds always go through the scheduler.
    - Measured (x86-64, one OS thread, per GPU thread, before → after, same load): no sync 62 → 5 ns; one block sync 92 → 43 ns; five warp shuffles and one block sync 650 → 146 ns. A small launch (8 blocks × 128 threads, 8 OS threads) averages 350 → 98 µs: pool threads watch for new launches briefly (`OCU_CPU_SPIN_US`) before sleeping.
- **Block distribution**: blocks are spread over several OS threads. Sharing between blocks needs real atomic operations, as on a GPU.
- **Order**: there is a fixed order (for reproduction) and a random order (for finding code that depends on order). The random order is reproducible with a seed value.
- **Stack**: each worker thread takes one large mapping, with one guard page at the very bottom. At the bottom of each fiber stack there is a canary, which is checked when the block ends, and an overflow is reported as an error. Giving each fiber a guard page would double the kernel memory regions to twice the number of GPU threads and hit the operating system limit (`vm.max_map_count`).
- **Sanitizers**: fiber switches are reported to the address sanitizer and the thread sanitizer. The whole test suite is clean under both.
- **Worker thread pool**: the threads that split and run blocks are created once and reused. The launching thread (the caller or the queue thread) also runs blocks itself. Each thread reuses its fiber stacks while the block shape stays the same, and releases them when the thread ends. Small launches are 5 to 9 times faster than when threads were created on every launch (saxpy with 60 thousand elements 3.3 → 0.66 ms, 64³ sgemm 1.5 → 0.17 ms). A helper thread that fails to get a stack drops out of that launch (it does not rejoin). After `fork()`, the child starts with an empty pool and creates threads again at the next launch. Queues do not cross `fork()`: the child forgets the parent's queues (a free does not wait for them), and returns "not supported" for inherited queue handles. Queues the child creates anew work normally.
- **Left as is**: a pool thread keeps holding the stacks for the block shape it used last (about 8 MB per thread for a block of 1024 threads).
- **Limits**: race conditions that occur only under truly concurrent execution inside a block are not all caught even by the random order. Code that spins waiting for another thread without synchronization never finishes.
- **Warp operations**: there are warp-level synchronization and value gathering (`ocu_cpu_warp_gather`), and shuffle and vote are built on top. If part of a warp waits in a warp operation and part waits in a block sync, each waiting for the other, it does not hang but is reported as a launch failure. Each warp operation carries a tag (operation, mask, width) and a sequence number. If the lanes of one warp meet in different calls or at different sequence numbers (partial-mask code), it does not silently give wrong values but is reported as a launch failure. Only the lanes named by the mask are waited for and checked. So partial-mask code that is legal in CUDA runs as is: disjoint masks (one half `0x0000ffff`, the other half `0xffff0000`), or code where only half the warp syncs at warp level and the rest goes to a block sync. If a lane named in the mask has already finished (undefined behavior in CUDA), it is allowed by default, as with block sync, and reported as a failure with `OCU_CPU_STRICT=1`. Warp sizes 32 and 64 are both tested.

## 8. Environment variables

| Name | Value | Meaning | Status |
|---|---|---|---|
| `OCU_BACKEND` | `cpu`, `amd`, ... | Backend to use. If unset, the first registered one | Works |
| `OCU_CPU_WARP` | `32`, `64` | Warp size reported by the CPU backend. Default 32 | Works |
| `OCU_CPU_ORDER` | `fixed`, `random` | Execution order of the CPU backend | Works |
| `OCU_CPU_SEED` | integer | Seed value for the random order. Default 1 | Works |
| `OCU_CPU_THREADS` | integer | Number of OS threads used for one launch. Default is the core count | Works |
| `OCU_CPU_STACK_KB` | integer | Fiber stack size. Default 128 | Works |
| `OCU_CPU_STRICT` | `1` | Treat a thread that finished before a sync as an error | Works |
| `OCU_CPU_SPIN_US` | integer | How long an idle pool thread watches for a new launch before sleeping (µs). Default 50; 0 sleeps at once | Works |
| `OCU_CPU_STATS` | `1` | At exit, print per-kernel launches, blocks, fiber switches, waits and time, plus copy time, to standard error | Works |
| `OCU_OPENCL_LIBRARY` | file path | Library the OpenCL backend opens. If unset, `libOpenCL.so.1` | Works |
| `OCU_OPENCL_VERBOSE` | `1` | Reasons for missing devices, build failure logs | Works |
| `OCU_NPUSIM_SCRATCH_KB` | integer | npusim on-chip memory size. Default 256 | Works |
| `OCU_NPUSIM_TILING` | `0`, `1` | npusim matrix multiply tiling. Default 1 | Works |
| `OCU_NPUSIM_DTYPES` | subset of `f32,f16,bf16` | Input data types npusim accepts. Default all three | Works |
| `OCU_NPUSIM_PLAN_MS` | integer | Time npusim waits per plan (imitates compilation). Default 0 | Works |
| `OCU_NPUSIM_MEM_MB` | integer | npusim device memory size. Default 256 | Works |
| `OCU_NPUSIM_REPORT` | `1` | Print npusim totals to standard error when the device closes | Works |
| `OCU_LOG` | `0` to `3` | Log level | Planned |

## 9. Build targets

| Target | Kind | Content |
|---|---|---|
| `ocu_backend_interface` | Header only | Device interface layer spec |
| `ocu_backend_cpu` | Static library | CPU backend |
| `ocu_core` | Static library | Backend registration and selection |
| `smoke_registry` | Test | Registration |
| `test_cpu_backend` | Test | CPU backend behavior. Run in four modes: default, random, strict, 64 threads |
| `test_cpu_regressions` | Test | Regression tests for problems found in review |
| `test_probe_cpu` | Test | Runs three unmodified `.cu` files on the CPU |
| `ocu_cudart` | Static library | Execution API |
| `rt_api`, `rt_lang`, `rt_mem`, `rt_misc`, `rt_fp16`, `rt_review` | Test | Execution API, language, and half-precision tests written as ordinary CUDA programs |
| `install_package` | Test | Builds and runs an outside project with the installed package |
| `ocu_cublas_cpu`, `ocu_cublas_nvidia` | Static library | `cublas_v2.h` implementation, per-target variants (section 4) |
| `cublas_cpu*`, `gpu_cublas_*`, `conformance_cublas_cpu*`, `gpu_conformance_cublas` | Test | Accuracy of `cublas_v2.h` (high-precision reference) and reference cases, on the CPU and on an NVIDIA GPU (skipped if there is no device) |
| `bench_cublas` | Benchmark | GEMM throughput on an NVIDIA GPU (median and 90% range) |

Install: `cmake --install <build folder> --prefix <location>` places the libraries, the public headers (gathered in one folder so that `<cuda_runtime.h>` finds ours), the CPU-path driver, and the CMake package. Other projects use `ocu::cudart` and `ocu_add_cpu_cu_executable()` after `find_package(Ocu)`.

Build options: `OCU_BACKEND_CPU` (default on), `OCU_BACKEND_NPUSIM` (default on; it is registered last, so it is never the default choice), `OCU_BACKEND_AMD`, `OCU_BACKEND_INTEL` (do not exist yet; turning them on is an error), `OCU_BACKEND_OPENCL` (default on for Linux), `OCU_BUILD_TESTS` (default on), `OCU_CPU_UCONTEXT` (default off).

Run the full verification with the single script `tools/check.sh`. It runs gcc and clang builds (warnings fail), address, undefined-behavior, and thread sanitizer runs, an ARM64 cross build with emulator (qemu-user) runs (when the tools are present), the scorer tests, the compiler experiments, and a clean-room check.

## 10. Conformance tests

- `conformance/suites/` has one folder per level: L1 build, L2 execution API, L3 language features, L4 libraries, L5 applications.
- `conformance/harness/ocu_conf/` holds the scoring rules and the result record format. It uses only the Python standard library.
- A result record is one JSON file per run. The format is in `conformance/README.md`.
- Speed is reported as the median, the slowest-10% range, and the name of the slowest test. We do not report the mean alone.
- Tests use only the same surface as users (section 3).

## 11. Steps to add a new chip

1. Put an implementation that fills in the `ocu_backend` table in `backends/<name>/`.
2. Add the option and the folder to `backends/CMakeLists.txt`.
3. Add one line to the registration list in `runtime/core/registry.c`.
4. Add a way to produce device code in `compiler/`.
5. Run the conformance tests on that chip.

If you had to change `runtime`, `libs`, or `conformance`, the spec has a gap. Fix the spec and raise `OCU_BACKEND_ABI`.

## 12. Decisions and open questions

| Item | State |
|---|---|
| Project name and code prefix | Decided: OpenCuda; identifiers keep the prefix `ocu` |
| How the headers users see are written | Decided: CUDA-compatible names, written from public documentation only |
| Execution unit of the CPU backend | Decided: fibers, deterministic or randomised order, warps of 32 or 64 lanes |
| Language of docs and comments | Decided: English first; Korean companions (`*.ko.md`) for the main documents |
| Format of the device code bundle and shape of the registration record | Open: settle after more compilers and chips are covered |
| Which layer of a vendor runtime the AMD backend builds on | Open: settle on real hardware |

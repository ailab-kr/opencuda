# ocu-nvcc and the toolkit folder

Date: 2026-11-13 · Card: `cards/compiler/compiler-driver.md` · Checked with CMake 3.28.3, Ubuntu clang 18.1.3 (WSL2)

`compiler/driver/ocu-nvcc` takes nvcc's documented command line, so build systems that call nvcc can call it instead. `.cu` files are compiled by the same steps as `compiler/probe/ocu_cc.py` (imported, not copied); `.c`/`.cpp` go to the host compiler with our headers; linking adds our runtime libraries.

## Target

| How | Example |
|---|---|
| option | `ocu-nvcc --ocu-target=nvidia ...` |
| name the driver is called by | `<toolkit>/bin/ocu-nvcc-nvidia`, `ocu-nvcc-cpu` (links to `ocu-nvcc`) |
| environment | `OCU_TARGET=nvidia` |
| default | `cpu` |

`cpu`: the CPU path (preprocess with our `cuda_runtime.h`, rewrite launches, compile as C++); runs on the CPU backend. `nvidia`: Clang CUDA mode, device code as PTX in our container; runs through the installed NVIDIA driver.

## Options

| Group | Options | Notes |
|---|---|---|
| phases | `-c`, `-E`, `-M`, `-MM`, link (default), `-shared`, `-lib`, `-ptx`, `-dc`, `-dw`, `-dlink`, `-rdc=`, `-dryrun`, `-v`, `-keep`, `-keep-dir` | `-ptx` on nvidia only. Relocatable device code and `-dlink` work on cpu (device code is host code there; `-dlink` writes an empty object); on nvidia they are refused as not supported |
| dependencies | `-MD`, `-MMD`, `-MF`, `-MT`, `-MP` | default file: object name with `.d` |
| input/output | `-o`, `-x cu|c|c++`, `@file`, `-optf` | `-x` applies to every input, as in nvcc |
| paths, macros | `-I`, `-isystem`, `-include`, `-D`, `-U`, `-L`, `-l` | vendor SDK paths refused. `-D` is one macro per option (values may hold commas) |
| code | `-O0..3`, `-g`, `-G`, `-lineinfo`, `-std=`, `--use_fast_math`, `--expt-relaxed-constexpr`, `--extended-lambda` | `.cu` without `-O` gets `-O3` (device code is optimised by default with nvcc) |
| GPU architectures | `-arch`, `-code`, `-gencode`, `--cuda-gpu-arch` | recorded; nvidia builds PTX for the lowest one LLVM 18 accepts (above sm_90: sm_90 PTX, which the driver compiles for the GPU) |
| host tools | `-ccbin`, `-Xcompiler`, `-Xlinker`, `-f*`, `-W*`, `-m*`, `-Wl,`, `-Xclang`, `-pthread` | `-Xcompiler` values are split on commas and spaces |
| accepted, ignored | `-Xptxas`, `-Xnvlink`, `-Xfatbin`, `-Xcudafe`, `-compress-mode`, `--threads`, `-diag-suppress`, `-maxrregcount`, `-ftz=`, `-prec-div=` ... | `-v` lists them |
| refused | unknown options, `-cubin`, `-fatbin`, `-cuda`, `-run` | "not supported" |
| libraries | `-lcudart`, `-lcudart_static`, `-lcudadevrt`, `-lcuda` → our runtime; `-lcublas` → `ocu_blas`; `-lcublasLt` → nothing yet (noted) | `-cudart none` links no runtime |
| other | `--version`, `--help`, `--list-gpu-arch` | |

`--version` prints our name, the host compiler's own version line, and the CUDA API level our headers declare:

```
ocu-nvcc 0.0.1: CUDA compiler driver of the ocu project (target cpu)
Ubuntu clang version 18.1.3 (1ubuntu1)  [host and device compiler: /usr/lib/llvm-18/bin/clang++]
CUDA API level 12.3 (V12.3.0), runtime and headers from the ocu project
```

## Toolkit folder

Built by default in the build tree (`<build>/toolkit`, target `ocu_toolkit`) and installed as `<prefix>/<libdir>/ocu/toolkit` (links to the installed driver, headers and libraries):

```
bin/ocu-nvcc  bin/ocu-nvcc-cpu  bin/ocu-nvcc-nvidia  bin/nvcc
include/      cuda_runtime.h and the rest of our headers
lib/          libocu_cudart.a libocu_core.a libocu_backend_*.a libocu_blas.a
nvvm/libdevice/   (empty)
cmake/OcuCUDA.cmake  cmake/OcuCUDARules.cmake  cmake/modules/FindCUDAToolkit.cmake
```

A Makefile project sets `NVCC=<toolkit>/bin/ocu-nvcc`. A CMake project:

```
cmake -S <project> -B <build> -DCMAKE_TOOLCHAIN_FILE=<toolkit>/cmake/OcuCUDA.cmake \
      [-DOCU_TARGET=cpu|nvidia] [-DCMAKE_CUDA_ARCHITECTURES=80]
cmake --build <build>
```

(or `include(<toolkit>/cmake/OcuCUDA.cmake)` before `enable_language(CUDA)`.)

### How CMake sees it (and why this is honest)

- CMake identifies a CUDA compiler from its `--version` output: nvcc's banner, or "clang version". Ours prints Clang's real version line because LLVM Clang does compile the host and device code. CMake then uses its Clang CUDA rules (`-x cuda`, `--cuda-gpu-arch=sm_NN`, `--cuda-path=`, `-MD -MT -MF`), which the driver accepts next to nvcc's options that projects add themselves.
- For a Clang CUDA compiler CMake looks for a toolkit: a program named `nvcc` under `CUDAToolkit_ROOT/bin`, `include/cuda_runtime.h`, a `lib` folder and a `nvvm/libdevice` folder. `OcuCUDA.cmake` sets `CUDAToolkit_ROOT` to the toolkit folder, whose `bin/nvcc` is a link to `ocu-nvcc` (it prints nothing of NVIDIA; the name only satisfies the search), and whose `nvvm/libdevice` is empty and never read. Avoiding the name would mean replacing CMake's whole identification step (forced compiler variables), which breaks between CMake versions.
- `find_package(CUDAToolkit)` loads our `cmake/modules/FindCUDAToolkit.cmake` (first on `CMAKE_MODULE_PATH`): `CUDA::cudart`, `CUDA::cudart_static`, `CUDA::cuda_driver`, `CUDA::cudadevrt` name our runtime, `CUDA::cublas(_static)` names `ocu_blas`. Libraries without an implementation get no target, except names in `OCU_CUDA_PLACEHOLDER_LIBS` (default `cublasLt;cublasLt_static`), which get an empty target and a status line.
- `OcuCUDARules.cmake` (`CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA`) makes the CUDA runtime that CMake links by itself (`CMAKE_CUDA_RUNTIME_LIBRARY`) ours.
- Nothing outside the toolkit folder is searched or run: the configured test project's cache is checked for vendor paths.

## Tests

`ctest -R ocu_nvcc`: 20 command-line checks (`tests/driver/test_ocu_nvcc.py`), the runtime test programs and `main_saxpy` built file by file like a Makefile would (cpu, and nvidia when a device exists), and `tests/driver/cmake_project` (shared library with kernels, C++ host program, single-file `.cu` program) built with the build-tree toolkit for cpu and nvidia and with the installed toolkit.

## llama.cpp (ggml CUDA backend), first try

Configure (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_TOOLCHAIN_FILE=.../OcuCUDA.cmake`, cpu target) succeeds: CUDA compiler identified as Clang 18.1.3, toolkit 12.3.0, host compiler detection through `-Xcompiler --version` works. Building `ggml-cuda` stops in every one of its 150 CUDA sources on missing headers (`cublas_v2.h`, also `cublasLt.h`, `cuda_bf16.h`, `cuda_fp8.h`, `cub/cub.cuh`). With empty placeholder headers outside the repository (diagnosis only), the next errors are API gaps (bf16/fp8 types, `make_half2`, `__byte_perm`/`__vsub4`/`__vcmpne4`/`__umulhi`/`__trap`, cuBLAS and cuBLASLt types, `CUresult`, `cudaLaunchConfig_t`/`cudaLaunchAttribute`, `cudaDataType` values, `INT_MAX` without `<climits>`) and two CPU-path rewrite limits: a launch through a function-pointer parameter (`kernel<<<...>>>(...)` inside a template function) makes a lambda that would capture `kernel`, and `__shared__ alignas(16) T a[]` becomes an ill-formed declaration.

# llama.cpp ggml-cuda: requirements and gaps

Date: 2026-11-13 · Subject: llama.cpp `0fb8f15` (2026-11-12, MIT), clone at WSL `~/ocu-ext/llama.cpp` · Scope: `ggml/src/ggml-cuda` (74 `.cu` + 96 `.cuh` + 135 generated `template-instances/*.cu`, about 50 000 lines in the top folder), its `CMakeLists.txt`, `ggml/CMakeLists.txt`, `ggml/cmake/common.cmake`.

Method: text search over llama.cpp, compared by name with our public surface (`runtime/include/*.h`, `backends/interface/ocu_backend.h`, `compiler/probe/ocu_cc.py`). Clean-room: no NVIDIA header, SDK file or source was opened; `/usr/local/cuda` was not read. Everything below about CUDA behaviour comes from how llama.cpp uses it, from CMake's own modules (Kitware, BSD) and from the public documentation names. "Exists" means the name is declared in one of our headers; whether it is implemented correctly on every backend was not tested here.

Line references are `file:line` in `ggml/src/ggml-cuda/` unless a path is given.

## 0. Summary

| Item | Number |
|---|---|
| Distinct runtime/driver functions called (`cuda*` 56, `cu*` 11) | 67 (245 call sites) |
| of which declared in our headers | 39 |
| of which missing | 28 (41 call sites) |
| Missing, in the proposed minimal configuration | 3 functions (`cudaDeviceGetPCIBusId` needed at run time; `cudaDeviceEnablePeerAccess`, `cudaLaunchCooperativeKernel` compile-only) + 1 enumerator (`cudaDevAttrCooperativeLaunch`) |
| cuBLAS functions, minimal configuration | 12 (11 if the workspace call is switched off); cublasLt: 0 in minimal, 12 in the default profile |
| cuRAND / cuFFT / thrust / cutlass | 0 / 0 / 0 / 0 (cutlass appears in one comment) |
| NCCL | 8 functions, optional (`GGML_CUDA_NCCL`, default ON but falls back if not found) |
| CUB | 6 files; 5 distinct entities (`DeviceScan`, `DeviceReduce`, `DeviceRadixSort`/`DeviceSegmentedSort`, `BlockScan`, `BlockLoad`/`BlockStore`), plus `DeviceTopK` and libcu++ iterators in newer CCCL; all switch off with a lower `CUDART_VERSION` |
| Inline `asm` statements | 53: 51 PTX (1 + 7 + 43), 2 AMD-only |
| Inline PTX that cannot be turned off by a build option | 1 (`prefetch.global.L2`, `common.cuh:137`); the other 50 compile out below sm_70/sm_80/sm_120 |
| Kernels (`__global__` lines) / `<<<>>>` launches | 172 / 192 (all 192 already rewritten by `cpu_rewrite.py` per `docs/compiler-probe.md`) |
| `__launch_bounds__` / `__shared__` / `extern __shared__` | 31 / 83 / 21 lines |
| Device lambdas (`-extended-lambda`) | 4 sites, all in flash attention (`fattn-mma-f16.cuh:433,480`, `fattn-tile.cuh:383,433`) |
| Compile units | 74 + 135 = 209 `.cu` (the 135 are one-line instantiations) |

Important structural facts:

1. `GGML_CUDA_F16` no longer exists. FP16 use is decided per architecture at compile time (`FP16_AVAILABLE`, `common.cuh:277`) and per device at run time (`fp16_available(cc)`, `common.cuh:326`). The knob for "old/generic" code is therefore the architecture, not a CMake option.
2. Host code picks kernels from the compute capability the runtime reports (`100*major + 10*minor`, `ggml-cuda.cu:~344`). Device code picks paths from `__CUDA_ARCH__`. If they disagree, a kernel whose body was compiled out (it expands to `NO_DEVICE_CODE`) gets launched: on NVIDIA that is a trap, on the CPU path `NO_DEVICE_CODE` expands to nothing, so the result is silent garbage. Section 7 turns this into a rule.
3. All 209 compile units are built regardless of options. Options change what the bodies contain, not how many files there are.

## 1. Runtime and driver API

Totals come from `grep` of `cuda[A-Z]*(` and `cu[A-Z]*(` call forms over `*.cu *.cuh` (comments included, so counts can be a little high).

### 1.1 Exists in our headers (39)

`cudaGetLastError` (40 call sites), `cudaMemcpyAsync` (21), `cudaStreamSynchronize` (17), `cudaMemsetAsync` (12), `cudaEventRecord` (11), `cudaEventDestroy` (10), `cudaEventCreateWithFlags` (10), `cudaStreamWaitEvent` (9), `cudaMemcpy2DAsync` (6), `cudaGetErrorString` (6), `cudaFuncSetAttribute` (6), `cudaFree` (6), `cudaGetDeviceProperties` (5), `cudaMalloc` (4), `cudaSetDevice`, `cudaGetDevice`, `cudaFreeHost`, `cudaStreamIsCapturing`, `cudaStreamDestroy`, `cudaStreamCreateWithFlags`, `cudaStreamBeginCapture`, `cudaMemGetInfo`, `cudaHostUnregister`, `cudaHostRegister`, `cudaHostGetDevicePointer`, `cudaEventSynchronize`, `cudaDeviceCanAccessPeer`, `cudaSetDeviceFlags`, `cudaRuntimeGetVersion`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaMemset`, `cudaMallocManaged`, `cudaMallocHost`, `cudaLaunchHostFunc`, `cudaHostAlloc`, `cudaGetDeviceCount`, `cudaFuncGetAttributes`, `cudaDeviceSynchronize`, `cudaDeviceGetAttribute`. Also used: `cudaStreamPerThread` (29 uses; every host-to-device tensor copy goes through it, e.g. `ggml-cuda.cu:1090`), `cudaStreamNonBlocking`, `cudaEventDisableTiming`, `cudaHostAllocPortable|Mapped`, `cudaMemcpy*` kinds, `cudaDeviceScheduleSpin`, `cudaFuncAttributeMaxDynamicSharedMemorySize`, `cudaFuncAttributes.ptxVersion`, `cudaDeviceProp` fields `name, major, minor, totalGlobalMem, multiProcessorCount, sharedMemPerBlock, sharedMemPerBlockOptin, warpSize` (all declared in `cuda_runtime_api.h`).

Points to verify even though the names exist:

- `cudaFuncSetAttribute`, `cudaFuncGetAttributes`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` are declared with `const void *func`. llama.cpp passes template kernel function pointers directly (`ggml-cuda.cu:897`, `fattn-common.cuh:1199`, `common.cuh:~250` macro `CUDA_SET_SHARED_MEMORY_LIMIT`). Implicit function-pointer to `void *` conversion is accepted by clang with a warning only; check it builds with `-Werror` off.
- `cudaFuncGetAttributes(...).ptxVersion` decides PDL use (`ggml-cuda.cu:844-905`): `>= 90` enables it. Our backends must report a value below 90.
- `cudaDeviceProp.sharedMemPerBlockOptin` drives kernel choice (`mmq.cu:443` requires at least 48 KB; `softmax.cu:353` compares the dynamic shared size to it).
- `cudaDeviceGetAttribute` is called with `cudaDevAttrCooperativeLaunch` at start-up (`ggml-cuda.cu:352`).

### 1.2 Missing (28), by area and centrality

| Area | Functions (call sites) | Where | Needed for a minimal run? |
|---|---|---|---|
| Device info | `cudaDeviceGetPCIBusId` (1) | `ggml-cuda.cu:6918`, runs when each device is registered | Yes (real implementation; any stable string works) |
| Device info | enumerator `cudaDevAttrCooperativeLaunch` | `ggml-cuda.cu:352` | Yes (returns 0) |
| Cooperative launch | `cudaLaunchCooperativeKernel` (1) | `softmax.cu:363`, only if `supports_cooperative_launch` | Compile only (stub) |
| Peer access | `cudaDeviceEnablePeerAccess` (1) | `ggml-cuda.cu:434`, only with env `GGML_CUDA_P2P` | Compile only (stub) |
| Peer copy | `cudaMemcpyPeerAsync` (3) | `ggml-cuda.cu:1099,1149,3171` | No: removed by `GGML_CUDA_NO_PEER_COPY` |
| Programmatic launch (PDL) | `cudaLaunchKernelEx` (1), `cudaGridDependencySynchronize` (1), `cudaTriggerProgrammaticLaunchCompletion` (1); types `cudaLaunchConfig_t`, `cudaLaunchAttribute`, `cudaLaunchAttributeValue`; enumerators `cudaLaunchAttributeProgrammaticStreamSerialization`, `cudaLaunchAttributeClusterDimension` | `common.cuh:143-153, 1497-1538` | Only if `CUDART_VERSION >= 12030` (which our `cuda_runtime_api.h:25` defines today). The two device-side calls are compiled only for `__CUDA_ARCH__ >= 900`. |
| Graphs | `cudaGraphInstantiate` (3), `cudaGraphExecDestroy` (4), `cudaGraphLaunch` (2), `cudaGraphGetNodes` (2), `cudaGraphExecUpdate` (2), `cudaGraphDestroy`, `cudaGraphNodeGetType`, `cudaGraphKernelNodeGetAttribute`, `cudaStreamEndCapture`; types `cudaGraph_t`, `cudaGraphExec_t`, `cudaGraphNode_t`, `cudaGraphExecUpdateResult(Info)`, `cudaErrorGraphExecUpdateFailure` | `ggml-cuda.cu:3205-3393, 5374-5471`, `graph.cuh` | No: all under `USE_CUDA_GRAPH` (= `GGML_CUDA_USE_GRAPHS`, default OFF, `ggml/CMakeLists.txt:117`). One caveat: `cudaStreamBeginCapture` at `ggml-cuda.cu:5480` is compiled unconditionally but runs only when graphs are on; it exists in our header. |
| Virtual memory pool (driver API) | `cuDeviceGet`, `cuDeviceGetAttribute`, `cuGetErrorString`, `cuMemAddressReserve` (2), `cuMemAddressFree`, `cuMemCreate`, `cuMemMap`, `cuMemUnmap` (2), `cuMemRelease`, `cuMemSetAccess` (2), `cuMemGetAllocationGranularity` (11 functions, 17 sites); types `CUdevice`, `CUresult`, `CUmemAllocationProp`, `CUmemAccessDesc`, `CUmemGenericAllocationHandle`, constants `CU_MEM_*`, `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`, `CUDA_SUCCESS` | `ggml-cuda.cu:319-331, 585-743`, `common.cuh:239-246` | No: removed by `GGML_CUDA_NO_VMM` (also drops the `CUDA::cuda_driver` link). The legacy pool (`ggml_cuda_pool_leg`, `ggml-cuda.cu:~460`) uses plain `cudaMalloc`. |
| Unified memory | `cudaMemAdvise` (1) | `ggml-cuda.cu:204` | No: inside `#if defined(GGML_USE_HIP)` |

Also missing and needed (types, not functions): `cudaDataType_t` and `CUDA_R_32F`, `CUDA_R_16F`, `CUDA_R_16BF`, `CUDA_R_8F_E4M3` (library-types names used by every cuBLAS call), `CUDA_SUCCESS`/`CUresult` (only with VMM).

Centrality summary:

- Core, all present: allocation (`cudaMalloc/Free`, `cudaMallocHost/FreeHost` for the pinned host buffer type), streams and events, `cudaMemcpyAsync`/`cudaMemcpy2DAsync`/`cudaMemsetAsync`, `cudaGetLastError`, device properties, `cudaFuncSetAttribute`.
- Core, missing: `cudaDeviceGetPCIBusId`, `cudaDevAttrCooperativeLaunch`.
- Optional: graphs (9), VMM (11), peer (2), NCCL (8), PDL (3 + 3 types), cooperative launch (1), fp8/fp4 conversions (headers only).

## 2. Library APIs

### 2.1 cuBLAS (and cublasLt)

Always included: `cublas_v2.h`, `cublasLt.h` (`vendors/cuda.h:4-5`, unconditional, so the file must exist even when unused). Handle and workspace setup at `common.cuh:1405-1420`.

Functions (non-Lt), 13 including the optional status-string helper:

| Function | Call sites | Notes |
|---|---|---|
| `cublasCreate`, `cublasDestroy`, `cublasSetStream` | `common.cuh:1407,1409`; `ggml-cuda.cu:928` | one handle per device and stream (up to 8 streams) |
| `cublasSetMathMode` | `common.cuh:1408`; `solve_tri.cu:69,74` | `CUBLAS_TF32_TENSOR_OP_MATH` set by default; `CUBLAS_DEFAULT_MATH` for the triangular solve. Accept and ignore. |
| `cublasSetWorkspace` | `common.cuh:1420` | only if `CUBLAS_VER_MAJOR > 11 \|\| (== 11 && MINOR >= 2)` (`common.cuh:1410`); 4 MiB workspace via `cudaMalloc` (32 MiB from cc 9.0) |
| `cublasSgemm` | `ggml-cuda.cu:1968`, `out-prod.cu:119`, `ssm-scan.cu:688` | main F32 matrix multiply |
| `cublasGemmEx` | `ggml-cuda.cu:1975`, `conv2d.cu:411`, `conv3d.cu:330` | see 2.2 |
| `cublasGemmStridedBatchedEx` | `ggml-cuda.cu:1990`, `ssm-scan.cu:693,733,753,783` | attention score and value products |
| `cublasGemmBatchedEx` | `ggml-cuda.cu:2028` | pointer arrays built by a kernel (`k_compute_batched_ptrs`) when K/V are broadcast |
| `cublasSgemmStridedBatched`, `cublasSgemmBatched` | `out-prod.cu:83,110` | `OUT_PROD`, only exercised by `test-backend-ops` and training |
| `cublasStrsmBatched` | `solve_tri.cu:70` | `SOLVE_TRI`, only exercised by `test-backend-ops` and some architectures |
| `cublasGetStatusString` | `common.cuh:216` | only if `CUDART_VERSION >= 12000`; otherwise a 9-case `switch` over `CUBLAS_STATUS_*` |

Existing internals (`libs/blas/ocu_blas.h`): `sgemm`, `dgemm`, `strsm`, `strsv`, level 1/2 routines. Missing there: batched and strided-batched forms, pointer-array batched forms, `GemmEx` with type arguments, `strsm` batched, half/bf16 input.

Types and constants used: `cublasHandle_t`, `cublasStatus_t` and the 9 `CUBLAS_STATUS_*` values, `cublasOperation_t` (`CUBLAS_OP_N`, `CUBLAS_OP_T`), `cublasComputeType_t` (`CUBLAS_COMPUTE_32F`, `CUBLAS_COMPUTE_16F`), `cublasGemmAlgo_t` (`CUBLAS_GEMM_DEFAULT`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`), `cublasMath_t` (`CUBLAS_DEFAULT_MATH`, `CUBLAS_TF32_TENSOR_OP_MATH`), `CUBLAS_SIDE_RIGHT`, `CUBLAS_FILL_MODE_UPPER`, `CUBLAS_DIAG_NON_UNIT`, macros `CUBLAS_VER_MAJOR`, `CUBLAS_VER_MINOR`. Data types: `CUDA_R_32F`, `CUDA_R_16F`, `CUDA_R_16BF`, `CUDA_R_8F_E4M3`.

### 2.2 Which type combinations are really reached

`ggml_cuda_mul_mat_cublas` (`ggml-cuda.cu:~2050`) chooses the compute type from the device:

- `compute_type` becomes F16 only if `fast_fp16_hardware_available(cc)` (cc >= 6.0 and != 6.1, `common.cuh:339`); then `GemmEx` uses `CUDA_R_16F` with `CUBLAS_COMPUTE_16F` (`ggml-cuda.cu:1820-1833`), and for Volta/AMD CDNA the output is F32.
- Otherwise inputs are converted to F32 by kernels and the call is `CUDA_R_32F` with `CUBLAS_COMPUTE_32F`.
- BF16 weights stay BF16 (`CUDA_R_16BF`, compute 32F) when the batch is at most 128 columns on a pre-Volta device (`ggml-cuda.cu:~2290`).
- The `conv2d`/`conv3d` GEMM path (`CUDA_R_16F` in, `CUDA_R_32F` out) runs only if `turing_mma_available(cc) || amd_*` (`conv2d.cu:377`).

So a device that reports cc 5.0 only needs the F32/F32/32F combination of `Sgemm`, `GemmEx`, `GemmStridedBatchedEx`, `GemmBatchedEx` (plus BF16 input for bf16 weight tests).

### 2.3 cublasLt

12 functions, all in `fp8.cu:74-190` (`cublasLtCreate/Destroy`, `MatmulDescCreate/SetAttribute/Destroy`, `MatrixLayoutCreate/SetAttribute/Destroy`, `MatmulPreferenceCreate/Destroy`, `MatmulAlgoGetHeuristic`, `Matmul`); handle in `common.cuh:1361,1426`. Compiled only if `CUDART_VERSION >= 11080`. Used only for FP8 weights.

### 2.4 Other libraries

| Library | Use | Gate |
|---|---|---|
| cuRAND, cuFFT, thrust, cutlass, nvJPEG | none | n/a |
| NCCL | `ncclCommInitAll`, `ncclCommInitRankConfig`, `ncclGetUniqueId`, `ncclAllReduce` (`ncclFloat`, `ncclBfloat16`, `ncclSum`), `ncclGroupStart/End`, `ncclCommDestroy`, `ncclGetErrorString`; `ggml-cuda.cu:1314-1653` | `GGML_USE_NCCL`, set only if `find_package(NCCL)` succeeds |
| CUB | `cub::DeviceScan::InclusiveSum` (`cumsum.cu:198,209`), `DeviceReduce::Sum` (`sum.cu:14,16`, `mean.cu:50,53`), `DeviceRadixSort`/`DeviceSegmentedSort` (`argsort.cu:92-154`), `cub::BlockScan` (`cumsum.cu:19`, `ssm-scan.cu:426`), `BlockLoad`/`BlockStore` with `BLOCK_*_WARP_TRANSPOSE` (`ssm-scan.cu:75-76`), `DeviceTopK::MaxPairs` (`top-k.cu:65,71`, CCCL >= 3.4.3) | `GGML_CUDA_USE_CUB` = `CUDART_VERSION >= 11070` (`common.cuh:116`); `ssm-scan.cu:1` repeats the test. Every use has a non-CUB fallback. |
| libcu++ (`<cuda/iterator>`, `cuda::stream_ref`, `cuda::execution::*`) | `argsort.cu:6-8,66`, `top-k.cu:31-62` | CUB branch and `CCCL_MAJOR_VERSION >= 3.1/3.4` macros; absent macros mean off |
| cooperative_groups | `softmax.cu:6-9` includes `cooperative_groups.h` and `cooperative_groups/reduce.h`; `cg::this_grid()` and `grid_group::sync()` at `softmax.cu:153-155` and its use after | **Not gated**: compiled for every build and architecture. |
| `nvcuda::wmma`, `<mma.h>` | none (`mma.cuh` has its own tile layer; "nvcuda::wmma" appears in comments only) | n/a |

## 3. Headers included

Counted by `#include <...>` in `*.cu *.cuh`; the vendor switch is `vendors/cuda.h`.

| Header | Where | Condition | Exists with us |
|---|---|---|---|
| `cuda_runtime.h` | `vendors/cuda.h:3` | always | yes |
| `cuda.h` | `vendors/cuda.h:4` | always | yes (only `CUDA_VERSION`) |
| `cublas_v2.h` | `vendors/cuda.h:5` | always | **no** |
| `cublasLt.h` | `vendors/cuda.h:6` | always (include unconditional) | **no** |
| `cuda_bf16.h` | `vendors/cuda.h:7` | always; 145 uses of `nv_bfloat16`/`nv_bfloat162` | **no** |
| `cuda_fp16.h` | `vendors/cuda.h:8` | always; 782 uses of `half`/`half2` | yes |
| `nccl.h` | `vendors/cuda.h:11` | `GGML_USE_NCCL` | no (off) |
| `cuda_fp8.h` | `vendors/cuda.h:15` | `CUDART_VERSION >= 11080` | no |
| `cuda_fp4.h` | `vendors/cuda.h:19` | `CUDART_VERSION >= 12080` | no |
| `cooperative_groups.h`, `cooperative_groups/reduce.h` | `softmax.cu:8-9` | always | **no** |
| `cub/cub.cuh` | `argsort.cu:4`, `cumsum.cu:8`, `mean.cu:5`, `ssm-scan.cu:6`, `sum.cu:5`, `top-k.cu:28` | `GGML_CUDA_USE_CUB` | no |
| `cuda/iterator` | `argsort.cu:8`, `top-k.cu:35` | CCCL macros | no |
| C++ standard headers | `<cstdint>` (22 files), `<algorithm>`, `<vector>`, `<unordered_map>`, `<mutex>`, `<atomic>`, `<charconv>`, `<cinttypes>`, `<condition_variable>`, others | always | host toolchain |

The language is C++17 (`ggml/src/CMakeLists.txt:619`, `cxx_std_17`), with `std::is_same_v`, `if constexpr`, fold-style template recursion in device code (`convert.cuh`, `common.cuh`, `mmf.cuh`, `fattn-common.cuh`).

## 4. Device-side features

### 4.1 Types and vector helpers

- `half`/`half2` everywhere (450 `half2`, 332 `half`), including loads through `reinterpret_cast<half&>` (22). `make_half2` is used 60 times (`common.cuh:634,656,811`) and is **not provided** by our `cuda_fp16.h` (and not defined by llama.cpp's vendor header either).
- `nv_bfloat16` (101), `nv_bfloat162` (44): conversions `__bfloat162float`, `__float2bfloat16`, `__float22bfloat162_rn`, `__bfloat1622float2`, `__low2bfloat16`, `__high2bfloat16`, `make_bfloat162`, brace-init of `nv_bfloat162` (`convert.cuh:70-95`, `cpy.cu`, `mmvf.cu`, `unary.cu`, `binbcast.cu`). All are compiled for every architecture; none is gated by `FP16_AVAILABLE`.
- Vector types: `float2` (225), `uint3` (183), `float4` (79), `int4` (43), `uint2` (40), `int2` (35), `char2` (17), `char4` (6), `dim3` (244); `make_float2` (38), `make_uint3` (23), `make_int2` (17), `make_float4` (16), `make_int4`, `make_char4`, `make_uint2`. All exist in `ocu_vector_types.h`.
- FP8/FP4 types (`__nv_fp8_e4m3`, `__nv_fp4x2_storage_t`, `__nv_fp4x4_e2m1`, `__NV_E4M3`, `__NV_SATFINITE`) and conversion functions (`__nv_cvt_fp8_to_halfraw`, `__nv_cvt_e8m0_to_bf16raw`, ...): only with `CUDART_VERSION >= 11080` / `12080` (`common.cuh:852-956`).

### 4.2 Half and bf16 intrinsics used (by name)

`__half2float` (38), `__half22float2` (64), `__float2half`, `__float22half2_rn`, `__half2half2` (10), `__halves2half2` (4), `__high2half`, `__low2half`, `__high2float`, `__low2float`, `__highs2half2`, `__lows2half2`, `__hmul2` (7), `__hadd2` (2), `__hmax`, `__hmax2`, `__hgt2_mask` (only `CUDART_VERSION >= 12000` expects the library one; below that `common.cuh:726` defines its own). Operators on `half2` (`+ - *`) are used in the FP16 paths. All except `make_half2`, `__hgt2_mask` and the bf16 family are in our `cuda_fp16.h`.

### 4.3 Warp, block and integer intrinsics (with counts)

| Intrinsic | Uses | Gate | In our `ocu_cuda_lang_cpu.h` | In `ocu_cuda_lang_nvptx.h` |
|---|---|---|---|---|
| `__syncthreads` | 133 | none | yes | yes |
| `__shfl_xor_sync` | 36 | none (also on `half2`, `common.cuh:~715`) | yes | yes |
| `__shfl_sync` / `__shfl_up_sync` | 14 / 6 | none | yes | yes |
| `__ballot_sync`, `__any_sync`, `__all_sync`, `__syncwarp` | 3, 1, 1, 1 | none | yes | yes |
| `__syncthreads_or` | 1 | none | **no** | yes |
| `__threadfence_system` | 5 (`allreduce.cu`, always compiled) | none | yes | yes |
| `__popc`, `__popcll` | 3, 2 | none | yes | yes |
| `__byte_perm` | 37 (`vecdotq.cuh`, `mmq-load-tiles.cuh`) | none | **no** | yes |
| `__vsub4` | 38 (`vecdotq.cuh` 20, `mmq-load-tiles.cuh` 30 incl. `__vcmpne4`, `fattn-common.cuh:468,560`) | none | **no** | **no** |
| `__vcmpne4` | 22 | none | **no** | **no** |
| `__umulhi` | 1 (`common.cuh:1021`, fast division) | none | **no** | yes |
| `__float2int_rn` | 4 (`quantize.cu:114`, `common.cuh:941,947` FP8) | partly | **no** | yes |
| `__isnanf` | 2 (`topk-moe.cu:201,220`) | none | no | no |
| `__ldg` | 2 (`pool1d.cu:33`, `pool2d.cu:39`) | `__CUDA_ARCH__ >= 350` (true for sm_50) | no | **no** |
| `__trap` | 1 (`no_device_code`, `common.cuh:~440`) | none | **no** | yes |
| `printf` in device code | 9 sites (`no_device_code`, others) | none | host `printf` | needs device printf support |
| `assert` in device code | 10 | none | host assert | needs device assert |
| `__dp4a` | 1 (`common.cuh:763`) | `__CUDA_ARCH__ >= 610`; else a C fallback in the next lines | no | no |
| `__reduce_add_sync` | 1 (`common.cuh:479`) | `__CUDA_ARCH__ >= 800` | no | no |
| `__cvta_generic_to_shared` | 2 (`cp-async.cuh:8`) | `CP_ASYNC_AVAILABLE` (>= 800) | no | no |
| `__nanosleep` | 1 (`allreduce.cu:169`) | `__CUDA_ARCH__ >= 700` | no | no |
| `__frcp_rn`, `__nv_cvt_*` | quantize.cu | `CUDART_VERSION >= 12080` | n/a | n/a |
| `__builtin_assume` | 3 (`GGML_CUDA_ASSUME`) | `CUDART_VERSION >= 11010` | clang builtin | clang builtin |

Atomics: `atomicAdd` on `int` only, 3 sites (`count-equal.cu:25` global, `top-k.cu:132,218` shared). No float atomics, no `atomicCAS`/`atomicMax`, no `__half` atomics. `volatile` is used in 23 places (the all-reduce flags and a few shared-memory tricks). `__constant__`: 0. `__managed__`: 0. Textures and surfaces: 0. Dynamic parallelism: 0. One `__device__ const float` global (`upscale.cu:156`).

Math: `fmaxf` 61, `expf` 51, `fabsf` 48, `powf` 23, `roundf` 12, `fminf` 11, `rsqrtf` 8, `fmaf` 8, `floorf` 7, `tanhf` 6, `sqrtf` 6, `log1pf` 5, `sinf` 4, `logf` 4, `ldexpf` 4, `log2f`, `cosf`, `expm1f`, `erff`, `frexpf`, `isinf`, `isfinite`, `isnan`. `-use_fast_math` is passed by CMake, so `expf` and friends may be approximate on NVIDIA, but they must not be built with finite-math-only semantics: masks are `-INFINITY` and tests call `isnan`/`isinf` (`topk-moe.cu`, `softmax.cu:~170`).

### 4.4 Tensor-core and asynchronous-copy paths

No `nvcuda::wmma`. All tensor-core work goes through the project's own `mma.cuh` (inline PTX), used by `mmq`, `mmf`, `fattn-mma-f16`, `gated-delta-net-mma`, `lightning-indexer`, `conv2d/3d` and `cp-async.cuh`. Every one of these is under an architecture macro (section 4.7), so compiling for an old architecture removes them.

### 4.5 Every inline PTX statement

53 `asm` statements in `*.cu *.cuh` (51 PTX, 2 AMD ISA). All are in 3 files.

| File | Count | Instruction | Gate macro | Compiled when `__CUDA_ARCH__` is |
|---|---|---|---|---|
| `common.cuh:137` | 1 | `prefetch.global.L2` | `!GGML_USE_HIP && !GGML_USE_MUSA` only | any NVIDIA arch (no option removes it) |
| `common.cuh:742`, `:788` | 2 | AMD `v_mul_i32_i24...`, `v_dot2_f32_f16` | `GGML_USE_HIP` | never on NVIDIA |
| `cp-async.cuh:27,30,33,38,53,62,73` | 7 | `cp.async.cg.shared.global` (4 forms), `cp.async.wait_all`, `commit_group`, `wait_group` | `CP_ASYNC_AVAILABLE` | >= 800 |
| `mma.cuh:32` | 1 | `movmatrix.sync.aligned.m8n8.trans.b16` | `TURING_MMA_AVAILABLE`, `CUDART_VERSION >= 11080` | >= 750 |
| `mma.cuh:814,829,858,920,960,964,1028,1032` | 8 | `ldmatrix.sync.aligned.m8n8.x2/x4(.trans).b16` | `TURING_MMA_AVAILABLE` | >= 750 |
| `mma.cuh:1074-1110` | 8 | `mma.sync.aligned.m16n8k16/m8n8k16/m16n8k32 ... s32.s8.s8.s32` | `TURING_MMA_AVAILABLE` (k32 and k16 forms by `>= 800`) | >= 750 |
| `mma.cuh:1127-1169` | 9 | `mma.sync ... f16.f16.f16.f16` (m16n8k8, m16n8k16) | `TURING_MMA_AVAILABLE` | >= 750 |
| `mma.cuh:1239` | 1 | `mma.sync ... f32.tf32.tf32.f32` | `AMPERE_MMA_AVAILABLE` | >= 800 |
| `mma.cuh:1287,1294` | 2 | block-scaled FP4 `mma` | `BLACKWELL_MMA_AVAILABLE` | 1200 to 1299 |
| `mma.cuh:1313-1371` | 10 | `mma.sync ... f32.f16.f16.f32`, `f32.bf16.bf16.f32` | `TURING_MMA_AVAILABLE` | >= 750 |
| `mma.cuh:1522,1526,1542,1546` | 4 | `mma.sync.aligned.m8n8k4` (Volta) | `VOLTA_MMA_AVAILABLE` (exactly 700) | 700 only |
| Total NVIDIA PTX | 51 | | | |

Which can be avoided by build options: all 50 in `cp-async.cuh` and `mma.cuh`, by building for `__CUDA_ARCH__ < 700` (the macros are `__CUDA_ARCH__` comparisons, so `CMAKE_CUDA_ARCHITECTURES=50` or `61` removes them; nothing else in CMake does). Unavoidable: **1** (`prefetch.global.L2`), reached from `mmvq.cu:756,759` and `mmq-few.cu:572`. It is a pure hint. The NVIDIA backend accepts it as PTX. For the CPU and Metal paths the driver has to drop or replace that one statement (a pattern match on `prefetch.*` is enough).

### 4.6 Launch bounds, templates, shared memory

- `__launch_bounds__`: 31 lines, many with calls to `constexpr __device__` functions as arguments (`ggml_cuda_get_physical_warp_size()`, `ggml_cuda_mmq_get_nthreads(...)`, `calc_nwarps(...)`). The compiler must evaluate them in a host context (clang does; the CPU path ignores the attribute).
- Templates: 172 kernels, heavy use of non-type template parameters, `if constexpr`, variadic launch helper `ggml_cuda_kernel_launch(Kernel kernel, params, Args&&...)` (`common.cuh:1534`) that launches through a function-pointer argument. The CPU rewrite handles this (`launch(cfg, [](auto&&...a){ K(a...); }, args)`).
- Shared memory: 83 `__shared__` lines and 21 `extern __shared__`. Dynamic sizes above 48 KB are requested with `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, n)` (`common.cuh:~255`; `fattn-common.cuh`). The code reads `sharedMemPerBlockOptin` and chooses accordingly; MMQ is switched off below 48 KB (`mmq.cu:443`). Apple GPUs allow 32 KB of threadgroup memory, so the Metal backend must report a value under 48 KB and the reachable kernels must still fit.
- Warp size 32 is a compile-time constant (`WARP_SIZE`, `common.cuh:42`; `ggml_cuda_get_physical_warp_size()` returns 32 for NVIDIA). The runtime must report 32.
- Block sizes up to 1024 threads (`CUDA_SOFT_MAX_BLOCK_SIZE`), 3-D grids with `gridDim.y/z` set from tensor dimensions.
- Device lambdas (4, FA only).

### 4.7 `__CUDA_ARCH__` thresholds and what "old/generic" selects

`common.cuh:277-323` defines the feature macros from `__CUDA_ARCH__`:

| Macro | Condition | Effect |
|---|---|---|
| `FP16_AVAILABLE` | `>= 600` | half2 arithmetic kernels (soft-max in half, some mul_mat_vec, `convert.cu:45` `dequantize_block_q8_0_f16`) |
| `FAST_FP16_AVAILABLE` | `FP16_AVAILABLE && != 610` | tile and vector flash-attention in half |
| `__dp4a` use | `>= 610` (`common.cuh:762`) | else a C loop over 4 bytes |
| `VOLTA_MMA_AVAILABLE` | `== 700` | Volta `m8n8k4` |
| `TURING_MMA_AVAILABLE` | `>= 750` | `ldmatrix`, int8 and f16 `mma` |
| `AMPERE_MMA_AVAILABLE`, `CP_ASYNC_AVAILABLE` | `>= 800` | tf32 `mma`, `cp.async` |
| `BLACKWELL_MMA_AVAILABLE` | `1200..1299` | FP4 `mma` |
| PDL device calls | `>= 900` and `GGML_CUDA_USE_PDL` | Hopper only |
| `__reduce_add_sync`, `__nanosleep` | `>= 800`, `>= 700` | warp reduce, spin sleep |
| `__ldg` | `>= 350` | `pool1d.cu:32`, `pool2d.cu:38` |

For an architecture below 600 (the "old" tier; llama.cpp's CMake default list starts at 50, `ggml-cuda/CMakeLists.txt:43-44`):

- No half arithmetic: kernels guarded by `FP16_AVAILABLE` become `NO_DEVICE_CODE`; the host must not select them, and does not (`fp16_available(cc)` uses `ggml_cuda_highest_compiled_arch(cc)`).
- No `dp4a`: `ggml_cuda_dp4a` uses a scalar loop. MMQ for dense matrices is off (`mmq.cu:452-455`: `highest_compiled_arch < 610` returns true only for MoE on cc >= 600). Quantised weights use dequantise + cuBLAS (F32) or MMVQ (batch up to 8; config `get_mmvq_mmid_max_batch_pascal_older`, `mmvq.cu:~300`; MMQ config `ggml_cuda_mmq_get_config_pascal_older`, `mmq.cuh:~268`).
- `fast_fp16_hardware_available` false: F16 and quantised weights go through F32 cuBLAS.
- No tensor-core and no async-copy code: flash attention falls to the tile/vec kernels, `mmf` is off, no `cp.async`.
- Everything else (norms, rope, soft-max, binary ops, get_rows, cpy, concat, ...) is plain float code and works at any architecture.

For a CPU path where `__CUDA_ARCH__` is simply undefined the preprocessor sees 0, which behaves like the "old" tier, with one difference: `NO_DEVICE_CODE` expands to nothing instead of a trap (`common.cuh:451-455`).

## 5. Build

### 5.1 What CMake needs from the toolkit

`ggml-cuda/CMakeLists.txt` (268 lines) does:

1. `find_package(CUDAToolkit)` (line 16). CMake's `FindCUDAToolkit.cmake` locates `nvcc` (or the compiler), reads the version from `nvcc --version` or from `version.txt`/`version.json` in the root (lines 501-528 of the module), checks `include/cuda_runtime.h` (line 952), finds `libcudart` (line 986) and builds imported targets. llama.cpp uses `CUDA::cudart`, `CUDA::cublas`, `CUDA::cublasLt` (line 175; the `_static` variants with `GGML_STATIC`), and `CUDA::cuda_driver` unless `GGML_CUDA_NO_VMM` (line 181). `CUDAToolkit_VERSION` is compared at lines 40, 43, 49, 53, 66, 168, 203.
2. `enable_language(CUDA)` (line 72). CMake detects the CUDA compiler by id: `NVIDIA` (banner regex `nvcc: NVIDIA (R) Cuda compiler driver`, then compiles a test file with `--keep --keep-dir`, and parses `nvcc -v` output for implicit include and link settings) or `Clang` (`clang version`, gives `--cuda-gpu-arch=sm_XX`, `-x cuda`, no SDK libraries; `Compiler/Clang-CUDA.cmake`). Clang with CUDA is rejected on Windows by CMake.
3. `CMAKE_CUDA_ARCHITECTURES`: if not set by the user, the file builds a list (50, 61, 70 virtual; 75, 80 virtual; 86 real; 89, 90; 120a, 121a real) (lines 21-70). With `GGML_NATIVE` and CMake >= 3.24 it uses `native`, which asks the GPU driver at configure time.
4. Global definitions: `GGML_CUDA_USE_GRAPHS`, `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_FORCE_CUBLAS`, `GGML_CUDA_NO_VMM`, `GGML_CUDA_NO_FA`, `GGML_CUDA_NO_PEER_COPY`, `GGML_USE_NCCL` (lines 135-157, 187); and from `cmake/common.cmake:60-120` the fattn-vec instance selection (`GGML_CUDA_FA_QUANTS`, `GGML_CUDA_FA_<K>_<V>=0/1`).
5. `target_compile_features(... cxx_std_17)` for every target.

### 5.2 Flags passed to the CUDA compiler

From `ggml-cuda/CMakeLists.txt:194-265`, applied as `$<$<COMPILE_LANGUAGE:CUDA>:...>`:

| Flag | Condition | Meaning for our driver |
|---|---|---|
| `-use_fast_math` | always | approximate math; must **not** imply finite-math-only |
| `-extended-lambda` | always | `__device__` lambdas in `[&]` (4 sites); clang accepts natively |
| `-compress-mode=size` | `CUDAToolkit_VERSION >= 12.8` | accept and ignore (not reached for a toolkit that reports 12.3 or lower) |
| `-lineinfo` | `GGML_CUDA_DEBUG` | accept and ignore |
| `-Werror all-warnings` | `GGML_FATAL_WARNINGS` | accept |
| `-Xcompiler <flags>` | always (`-Wno-pedantic` on non-MSVC; plus warnings when `GGML_ALL_WARNINGS`) | pass to the host compiler; also `-ccbin` |
| `GGML_ALL_WARNINGS` probe | when ON and not MSVC | runs `<cuda compiler> .c -ccbin <cxx> -Xcompiler --version` and `-Xcompiler "-dumpfullversion -dumpversion"` (`CMakeLists.txt:217-247`): the driver must forward these and print the host compiler's answer |
| `--generate-code=arch=compute_XX,code=[...]`, `-gencode`, `-arch`, `-std=c++17`, `-x cu`, `-MD -MT -MF`, `-isystem`, `-D`, `-I` | added by CMake itself for the NVIDIA compiler id | accept; map to our per-target output |

Today `compiler/probe/ocu_cc.py` takes `--target`, `--arch`, `-c`, `-o`, `-I`, `-D`, `-O`, `-g` only. It is a probe, not an nvcc-compatible command line; none of the above is accepted yet.

### 5.3 Options that switch features off

| Option | Default | Effect | Needed for first run |
|---|---|---|---|
| `GGML_CUDA` | OFF | enable backend | ON |
| `GGML_CUDA_FA` | ON | OFF defines `GGML_CUDA_NO_FA`: `FLASH_ATTN_AVAILABLE` is not defined (`common.cuh:322`), all fattn bodies are empty (`fattn-vec.cuh:45`, `fattn-tile.cuh:816`, `fattn-mma-f16.cuh:1935`), `ggml_cuda_get_best_fattn_kernel` returns NONE (`fattn.cu:588`). The instance files are still compiled but trivial. | OFF |
| `GGML_CUDA_FA_QUANTS` / `_ALL_QUANTS` | `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;f8_e4m3-f8_e4m3` | selects which fattn-vec instance files are built (8 to 64) | irrelevant when FA is off |
| `GGML_CUDA_GRAPHS` | OFF (`GGML_CUDA_GRAPHS_DEFAULT`, `ggml/CMakeLists.txt:117`) | removes 9 graph functions | OFF (default) |
| `GGML_CUDA_NO_VMM` | OFF | removes the `cuMem*` pool (11 driver functions) and the `CUDA::cuda_driver` link | ON |
| `GGML_CUDA_NO_PEER_COPY` | OFF | removes `cudaMemcpyPeerAsync` | ON |
| `GGML_CUDA_NCCL` | ON (soft: skipped if NCCL is not found) | NCCL all-reduce | OFF |
| `GGML_CUDA_FORCE_MMQ` | OFF | MMQ for every quantised matmul where the architecture allows (`mmq.cu:458`) | OFF |
| `GGML_CUDA_FORCE_CUBLAS` | OFF | never use MMQ (`mmq.cu:396`, `mmq-few.cu:815`); quantised dense matmul then goes through dequantise + cuBLAS | optional; no effect below cc 6.1 where MMQ is already off |
| `GGML_CUDA_COMPRESSION_MODE` | `size` | only with toolkit >= 12.8 | n/a |
| `GGML_CUDA_CCCL_VERSION` | empty | fetches NVIDIA's CCCL from git | leave empty |
| `GGML_CUDA_DEBUG` | OFF | `-lineinfo` | OFF |
| `GGML_NATIVE` | ON | `CMAKE_CUDA_ARCHITECTURES=native` (needs a GPU at configure time) | OFF, give the architecture explicitly |
| `GGML_CUDA_F16` | **does not exist any more** | | |
| no option | | PDL (`GGML_CUDA_USE_PDL`, `common.cuh:124`), CUB (`GGML_CUDA_USE_CUB`, `common.cuh:116`), fp8, cublasLt: all derived from `CUDART_VERSION` | see section 6 |

Run-time environment switches seen: `GGML_CUDA_DISABLE_GRAPHS`, `GGML_CUDA_PDL=0`, `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_NO_PINNED`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY`, `GGML_CUDA_P2P`, `GGML_CUDA_REGISTER_HOST`, `GGML_CUDA_DEVICES` (virtual devices), `GGML_CUDA_CUBLAS_COMPUTE_TYPE`, `GGML_CUDA_MMQ_PREC`.

## 6. Proposed minimal configuration

Goal: smallest set of NVIDIA-specific features that still builds all 209 units and runs `test-backend-ops` and `llama-cli` on a small quantised model.

### 6.1 Settings

CMake (llama.cpp, no source change):

```
-DGGML_CUDA=ON
-DGGML_NATIVE=OFF
-DCMAKE_CUDA_ARCHITECTURES=50          # generic tier, see below
-DGGML_CUDA_FA=OFF                     # no tensor-core / cp.async / device-lambda code
-DGGML_CUDA_GRAPHS=OFF                 # default
-DGGML_CUDA_NO_VMM=ON
-DGGML_CUDA_NO_PEER_COPY=ON
-DGGML_CUDA_NCCL=OFF
-DGGML_CUDA_FA_QUANTS=f16-f16
-DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
-DLLAMA_BUILD_TESTS=ON
-DCMAKE_BUILD_TYPE=Release
```

Architecture choice, `50` (cc 5.0):

- Compiled PTX asm statements: **1** (the prefetch). With `61` it is also 1, but `61` adds `__dp4a`, half arithmetic kernels and MMQ (more to implement and verify).
- Reported compute capability must be 5.0 (`prop.major=5, minor=0`), `warpSize=32`, `sharedMemPerBlockOptin` at least 48 KB on CPU/NVIDIA backends (Metal: the real value, under 48 KB, is fine because MMQ is off at 5.0).
- Our driver must define `__CUDA_ARCH__=500` during the device pass and `__CUDA_ARCH_LIST__=500` in every pass (that is what the CUDA compiler does). `ggml_cuda_highest_compiled_arch(cc)` (`common.cuh:155-192`) then caps whatever cc the device reports at 500, which makes the host/device disagreement of section 0 impossible even if a backend reports a higher cc. Without `__CUDA_ARCH_LIST__` the function returns the reported cc unchanged.
- NVIDIA comparison backend: build sm_50 PTX and let the driver JIT it; the reported cc of the real GPU is capped by the list above.

`CUDART_VERSION` profile. Today `cuda_runtime_api.h:25` defines 12030, which switches on four things that this configuration does not need: PDL (`cudaLaunchKernelEx` and 3 types), fp8 (`cuda_fp8.h`, 6 names), cublasLt (12 functions and the always-included header), CUB (about 8 entities), and it makes llama.cpp expect `__hgt2_mask` and `cublasGetStatusString` from the library. Proposal: make the value a profile of our header (for example `-DOCU_CUDART_VERSION=11060`, default unchanged) and keep `CUDA_VERSION 12030` in the `--cuda-path` marker used by clang, which is independent. At 11060:

| Removed | Because |
|---|---|
| PDL: 1 function + 3 types | needs `>= 12030` (or `>= 11080` off Windows) |
| fp8 and cublasLt: 12 functions, 6 fp8 names, `cuda_fp8.h` | `>= 11080` |
| CUB and libcu++ (5 CUB entities + top-k) | `>= 11070`; fallbacks exist |
| `cublasGetStatusString`, `__hgt2_mask` | `>= 12000`; fallbacks exist |
| `movmatrix` PTX | `>= 11080`; not reached below sm_75 anyway |

This path (`CUDART_VERSION` between 11020 and 11070) was supported by llama.cpp historically but is not exercised by its CI any more; expect small compile fixes to be found by the first build. If it proves brittle, the fallback is to keep 12030 and provide the PDL declarations plus a trivial CUB subset instead (section 8, items 11-12).

Also set `CUBLAS_VER_MAJOR=11`, `CUBLAS_VER_MINOR=1` in our `cublas_v2.h` to skip `cublasSetWorkspace` (`common.cuh:1410`), or implement it as a no-op.

### 6.2 Build driver contract (nvcc compatibility)

CMake must be satisfied with one of two shapes:

- A. `CMAKE_CUDA_COMPILER` = an nvcc-compatible driver (`ocu-nvcc`) inside a toolkit-shaped root we own: `bin/ocu-nvcc` (named `nvcc` for CMake's lookup), `include/` (our headers), `lib64/libcudart.*`, `libcublas.*`, `libcublasLt.*`, a `version.json` or a `--version` text saying 12.3 or lower, and the `CMAKE_CUDA_COMPILER_ID`. CMake's detection by banner string `nvcc: NVIDIA (R) Cuda compiler driver` is a vendor identity string: whether we may print it is an open question. The alternative that avoids it is a toolchain file that sets `CMAKE_CUDA_COMPILER_ID`, `CMAKE_CUDA_COMPILER_VERSION` and `CMAKE_CUDA_COMPILER_FORCED` itself.
- B. `CMAKE_CUDA_COMPILER=clang++` (CMake's Clang CUDA mode). This fails on `-use_fast_math -extended-lambda -compress-mode` unless the compiler accepts them, so a wrapper is needed anyway.

Either way the driver must: accept the flags in 5.2; implement `-Xcompiler`, `-ccbin`, `--generate-code`, `-MD/-MT/-MF`; define `__CUDACC__`, `__CUDA_ARCH__`, `__CUDA_ARCH_LIST__`; force-include `cuda_runtime.h` (the probe already does); keep every `--cuda-path` / `-nocudainc` guard; map `-use_fast_math` to approximate transcendentals and flush-to-zero **without** `-ffinite-math-only`; compile `.cu` for each target (CPU rewrite, NVIDIA PTX, Metal).

### 6.3 Must-have list for a first successful build and run

Counts are for this configuration (arch 50, `OCU_CUDART_VERSION=11060`, options of 6.1).

Runtime API: all 39 existing functions behave correctly, plus 3 new (`cudaDeviceGetPCIBusId` real; `cudaDeviceEnablePeerAccess` and `cudaLaunchCooperativeKernel` may return `cudaErrorNotSupported`), the enumerator `cudaDevAttrCooperativeLaunch` (value 0), `cudaDataType_t` with `CUDA_R_32F/16F/16BF`.

Device properties: `major=5, minor=0`, `warpSize=32`, `multiProcessorCount>0`, `totalGlobalMem`, `sharedMemPerBlock`, `sharedMemPerBlockOptin >= 48 KB` (CPU/NVIDIA), `name`; `cudaFuncGetAttributes().ptxVersion < 90`; `cudaStreamPerThread` fully working (every `set/get_tensor` uses it); pinned host buffer (`cudaMallocHost`).

cuBLAS (12): `cublasCreate`, `cublasDestroy`, `cublasSetStream`, `cublasSetMathMode`, `cublasSetWorkspace` (or avoided by version macros), `cublasSgemm`, `cublasGemmEx`, `cublasGemmStridedBatchedEx`, `cublasGemmBatchedEx`, `cublasSgemmStridedBatched`, `cublasSgemmBatched`, `cublasStrsmBatched`; headers `cublas_v2.h`, `cublasLt.h` (may be empty at this version); F32 data type for first run, BF16 input for bf16 tests.

Headers to add: `cublas_v2.h`, `cublasLt.h`, `cuda_bf16.h` (type, `nv_bfloat162`, the 8 conversions in 4.1, operators), `cooperative_groups.h` + `cooperative_groups/reduce.h` (grid group whose `sync()` fails; never launched because `supports_cooperative_launch` is 0).

Device functions to add:

- Both paths: `__vsub4` (38), `__vcmpne4` (22), `make_half2` (60), `__isnanf` (2).
- CPU path only (present for NVPTX): `__byte_perm` (37), `__umulhi`, `__float2int_rn`, `__syncthreads_or`, `__trap`.
- NVPTX path only: `__ldg` (2).

Inline PTX: **1** statement unavoidable (`prefetch.global.L2`). Zero for mma/ldmatrix/cp.async/movmatrix.

Language and compiler: C++17 device code with `if constexpr`, variadic templates, device lambdas (none reached with FA off), device `printf`/`assert`, `__launch_bounds__` with constexpr calls, `extern __shared__` (21), static `__shared__` (83), 192 launches (already rewritten).

Tests to run, in order: (1) configure only; (2) build `ggml-cuda` and fix compile errors by class; (3) `test-backend-ops` filtered to `ADD`, `MUL`, `SCALE`, `RMS_NORM`, `SOFT_MAX`, `ROPE`, `GET_ROWS`, `CPY`, `MUL_MAT` (f32 then q4_0/q8_0, each batch size 1 and 8 and 512), then the full list; (4) `llama-cli -m <small q4_0 or q8_0 gguf> -ngl 99 -fa off -n 16 -no-cnv -s 1` compared token by token with the CPU reference backend; (5) repeat with the CPU backend, the NVIDIA backend and the Metal backend; report per axis.

## 7. Risks to watch

1. Host/device architecture mismatch (section 0, point 2): enforce through `__CUDA_ARCH_LIST__` and a conformance check that `ggml_cuda_info().devices[i].cc` is at most the compiled architecture.
2. CPU path: `NO_DEVICE_CODE` is empty when `__CUDA_ARCH__` is undefined. A wrongly selected kernel returns garbage, not an error. Define `__CUDA_ARCH__=500` for the whole translation unit on the CPU path so that `no_device_code()` (trap) is reached instead, or make `NO_DEVICE_CODE` fail loudly in `ocu_cuda_lang_cpu.h`.
3. `-use_fast_math` mapping (4.3).
4. `shfl` on `half2` is used outside the FP16 gate (`common.cuh:~715`): check that both language headers overload it for 4-byte structs.
5. Reductions assume a full warp of 32 lanes with a full mask in nearly all uses (`0xffffffff`); the CPU path detects partial-mask code and fails the launch, which is the right behaviour here.
6. `cudaStreamPerThread` as the default stream for tensor copies mixes with ordinary streams created with `cudaStreamNonBlocking`; the CPU backend that runs launches synchronously hides ordering mistakes. The `test-backend-ops` run with several streams (`GGML_CUDA_MAX_STREAMS` is 8) should be repeated on the NVIDIA backend.
7. 209 compile units of heavy templates: the CPU path rewrite plus a C++17 compile per unit; measure build time early.

## 8. Gaps in priority order

| # | Gap | Size | Blocks |
|---|---|---|---|
| 1 | nvcc-compatible driver and toolkit-shaped root so CMake's CUDA language and `find_package(CUDAToolkit)` work; define `__CUDA_ARCH__`, `__CUDA_ARCH_LIST__`; flag handling (`-use_fast_math` without finite-math, `-extended-lambda`, `-Xcompiler`, `-ccbin`, `-gencode`, `-MD`) | large | configure and every compile |
| 2 | cuBLAS front end: `cublas_v2.h`, `cublasLt.h`, 12 functions, 4 GEMM forms with pointer-array and strided batching, type and compute-type arguments, status codes | medium | `MUL_MAT` for anything but the vector kernels, attention, `llama-cli` prompt processing |
| 3 | `cuda_bf16.h` (type, `nv_bfloat162`, ~8 conversions) | small-medium | compile of 12 files (145 uses) |
| 4 | Library-type names (`cudaDataType_t`, `CUDA_R_*`) and the capability policy (cc 5.0, warp 32, shared memory, `ptxVersion`) | small | compile, kernel selection |
| 5 | Device intrinsics: `__vsub4`, `__vcmpne4`, `make_half2`, `__isnanf` (both paths); `__byte_perm`, `__umulhi`, `__float2int_rn`, `__syncthreads_or`, `__trap` (CPU path); `__ldg` (NVPTX) | small | compile of `vecdotq`, `mmq`, quantisation, soft-max, pool |
| 6 | `cooperative_groups.h`, `reduce.h` stubs and `cudaLaunchCooperativeKernel` | small | compile of `softmax.cu` |
| 7 | Runtime additions: `cudaDeviceGetPCIBusId` (real), `cudaDevAttrCooperativeLaunch`, `cudaDeviceEnablePeerAccess` stub | small | start-up and registration |
| 8 | `CUDART_VERSION` profile (11060) or, alternatively, PDL declarations + fp8 header + cublasLt + CUB subset | small vs medium | removes 4 feature groups |
| 9 | Handling of the one unavoidable PTX statement on CPU and Metal; test that a wrongly selected kernel fails loudly | small | CPU/Metal compile of `mmvq.cu`, `mmq-few.cu` |
| 10 | Tier 2 and beyond: `__dp4a` + half arithmetic (arch 61), 51 mma/ldmatrix/cp.async PTX statements or a translation (arch 75/80), graphs (9 functions), VMM (11), peer (2), NCCL (8), CUB or CCCL subset, fp8/fp4 | large | performance tiers, not first run |

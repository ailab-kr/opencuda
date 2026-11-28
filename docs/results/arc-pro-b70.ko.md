# Intel Arc Pro B70에서의 결과

[English](arc-pro-b70.md) · **한국어**

고치지 않은 CUDA 프로그램이 Intel Arc Pro B70에서 OpenCuda의 SPIR-V 경로(`ocu-nvcc --ocu-target=spirv`)로 어떻게 도는지, 2026-11-27/28에 잰 것. 여기 있는 숫자는 모두 아래 기계에서 잰 것이고, 결과가 부분적이거나 실행이 아직 도는 중이면 그렇다고 적었다.

## 기계와 소프트웨어

| | |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB(OpenCL 보고: 계산 단위 256, 30.3 GiB) |
| 드라이버 | Intel compute-runtime(NEO) 26.35.39758.21, IGC 2.41.16(적은 곳만 IGC 2.42) |
| 호스트 | Intel Xeon Silver 4510, 61 GB, Ubuntu 26.04.1, Linux 7.0.0-30 |
| OpenCuda | `main`(HeCBench 실행: 커밋 4f9930a), 장치 코드는 Clang/LLVM 23.1.6 |
| 비교 대상 | chipStar 1.3.0(`cucc`): Intel에서 CUDA를 돌리는 다른 공개 도구 |

GPU는 다른 작업과 함께 쓰는 장비라, 측정은 모두 GPU에 다른 일이 없을 때 했다.

## HeCBench: CUDA 프로그램 전부, OpenCuda와 chipStar

[HeCBench](https://github.com/zjin-lcf/HeCBench)(커밋 4426c9f)의 `*-cuda` 프로그램 570개를 두 도구가 **고치지 않고** 같은 규칙으로 빌드하고 실행한다(`tools/bench/hecbench_compare.sh`, 요약 `tools/bench/hecbench_summary.py`).

- 그 도구의 컴파일러로 `make`(300초 제한), 이어서 `make run` 두 번(각 180초 제한). 두 번째를 잰다.
- 상태: `build_fail`, `timeout`, `crash`(0이 아닌 종료), `fail`(FAIL 출력), `pass`(PASS 출력), `ran`(끝까지 돌았고 둘 다 출력하지 않음: HeCBench 프로그램 다수는 시간이나 검사합만 찍는다).
- `no_data`: 입력 파일이 저장소에 없어 멈춘 경우(HeCBench는 큰 입력을 따로 둔다). 어느 도구에 대해서도 말해 주는 것이 없어 따로 센다.

실행 전체에 실행을 시작할 때의 OpenCuda 툴킷(커밋 4f9930a)을 썼다. 모든 프로그램을 한 툴킷으로 재기 위해서다.

**570개 전부(실행 끝: 2026-11-28 04:29 UTC):**

| 상태 | OpenCuda | chipStar 1.3.0 |
|---|---|---|
| pass | **206** | 161 |
| ran | 118 | 111 |
| fail | 17 | 9 |
| crash | 25 | 18 |
| timeout | 16 | 15 |
| build_fail | **153** | 228 |
| no_data | 35 | 28 |

- 끝까지 도는 프로그램(pass 또는 ran): **OpenCuda 324, chipStar 272**. 둘 다 252, OpenCuda만 72, chipStar만 20.
- 둘 다 끝까지 도는 프로그램의 속도(0.05초 넘는 247개): chipStar 시간 / OpenCuda 시간의 기하평균 0.98(1보다 크면 OpenCuda가 빠름). OpenCuda가 10% 넘게 빠른 것 71, 10% 넘게 느린 것 90, 10% 안 86. 실행 시작 때의 툴킷이다. 그 뒤 바뀐 것은 다음 두 절에 있다.

### 실행 뒤의 성능 작업

같은 GPU에서 같은 세션 A/B(설정 켬·끔)로, 실행과 같이 `make run`을 쟀다. 두 변경 모두 `main`에 있다.

**장치 메모리를 Intel USM 장치 할당으로**(`cudaMalloc`. 전에는 거친 단위 SVM 버퍼. `OCU_OPENCL_DEVICE_USM=0`이면 예전 동작). B70에서 대역폭이 두 배쯤 된다:

| | 전 | 지금 | chipStar |
|---|---|---|---|
| BabelStream Copy(GB/s) | 547 | 906 | 891 |
| BabelStream Triad(GB/s) | 535 | 1101 | 1114 |
| divergence-cuda(초) | 38.4 | **1.2** | 1.34 |
| collision-cuda(초) | 33.9 | 2.4 | build_fail |
| nonzero-cuda(초) | 65.8 | 39.6 | |
| memtest-cuda(초) | 53.9 | 25.3 | |
| perlin-cuda(초) | 18.2 | 12.5 | |
| streamOrderedAllocation-cuda(초) | 41.4 | 28.2 | |

divergence-cuda는 실행에서 격차가 가장 컸던 프로그램이다(chipStar보다 28.8배 느림). 반복마다 버퍼 다섯 개를 할당하고 해제한다.

**`-use_fast_math`**가 SPIR-V 경로에서 nvcc 문서대로 동작한다(단정밀도 `expf`, `sinf`, `powf` ...가 내장 함수로). 전에는 플래그가 장치 컴파일러에 닿지 않았다. 이 플래그를 쓰고 실행된 HeCBench 17개 모두 통과 유지. aidw-cuda 50.7 → 8.9초, minkowski-cuda 27.4 → 20.6초(chipStar 23.2초), aobench-cuda 0.08 → 0.05초, 나머지 같음.

재고 채택하지 않은 것: IGC 내장 수학 함수는 CUDA Math API 오차 한도를 넘고(expf 2.54 ulp, sin 3.38 ulp, tanhf 4.84 ulp, 한도 2·2·2) 대부분 OpenCuda 자체 함수보다 느리다.

### 나중 커밋으로 고친 프로그램

실행은 모든 프로그램을 시작 때의 툴킷으로 잰다. 이 브랜치의 나중 커밋을 같은 GPU, 같은 `make run`(잰 실행 한 번)으로 확인한 결과:

| 프로그램 | 실행에서(OpenCuda / chipStar) | 지금 툴킷 | 없던 것 |
|---|---|---|---|
| axpby-cuda | fail / pass | **pass**(39.6초, chipStar 33.8초) | 드라이버의 2KB 한도를 넘는 커널 인자 |
| cmembench-cuda | fail / pass | **pass** | SPIR-V 경로의 `cudaMemcpyToSymbol` |
| collision-cuda | crash / build_fail | 실행 | 호스트에서 쓰는 장치 변수 |
| intrinsics-cast-cuda | build_fail / ran | 실행 | 변환 내장 함수 |
| mcmd-cuda | build_fail / ran | 실행 | double `erf`·`erfc` |
| asyncAllocation-cuda | build_fail / build_fail | **pass**(cudaMallocAsync 215ms, cudaMalloc 1592ms) | 스트림 순서 할당 |
| ccsd-trpdrv-cuda | build_fail / build_fail | **pass** | 헤더 없이 쓰는 `std::chrono` |
| relu-cuda, mlp-cuda, streamOrderedAllocation-cuda, streamUM-cuda, moe-cuda | 아직 안 돎(이전 툴킷 빌드 점검에서 실패) | **pass** | `__half2` 비교, cuBLASLt, 메모리 풀, `cublasDgemv`, `cub::KeyValuePair` |
| qkv-cuda, rayleighBenardConvection-cuda | 아직 안 돎 | 실행 | cuBLASLt 에필로그, `cublasScopy` |

실행 정정: HeCBench Makefile 16개는 `EXTRA_CFLAGS`를 컴파일러에 넘기지 않아, OpenCuda 빌드가 GPU 커널 없는 CPU 대상 빌드였다(colorwheel-cuda가 그렇게 실패. SPIR-V로 빌드하면 통과). 실행은 이제 `OCU_TARGET=spirv`(ocu-nvcc의 기본 대상 변수. chipStar는 플래그가 필요 없음)도 주고, 그 16개의 OpenCuda 쪽을 같은 툴킷으로 다시 잰다.
OpenCuda가 빌드하지 못하는 주된 이유는 아직 없는 라이브러리(Thrust, cuSPARSE, cuSOLVER, cuRAND, cuFFT, NCCL/MPI)와 인라인 PTX 어셈블리다. 프로그램마다 첫 오류는 실행 기록에 있다.

## llama.cpp

고치지 않은 llama.cpp, CUDA 백엔드, qwen3-0.6B Q4_K, `-fa 1`, `OCU_COMPUTE_CAPABILITY=7.0`, `llama-bench -p 512 -n 128 -r 3`, 번갈아 두 번:

| | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| OpenCuda, 이 브랜치 | 11,521 / 11,555 | 245.4 / 244.6 |
| llama.cpp SYCL 백엔드(기준, 2026-11-27 같은 기계에서 잼) | 25,964 | 403 |

같은 빌드의 `test-backend-ops -b CUDA0`: SOFT_MAX 215/215, RMS_NORM 121/121, MUL_MAT 2724/2724, GET_ROWS 228/228, CPY 286/286, ROPE 514/514, ADD 173/173, MUL 143/143, SUM_ROWS 11/11, SUM 10/10, ARGMAX 7/7, ARGSORT 98/98, 머리 크기 128의 FLASH_ATTN_EXT 859/859. FlashAttention은 상류에 제안한 수정(llvm/llvm-project#246644, `compiler/spirv/clang-hipspv-kernel-address.patch`)을 넣은 Clang으로 빌드된다. FLASH_ATTN_EXT 전체로는 IGC 2.41에서 4178/4370이 통과하고, 나머지 192개는 IGC 2.41의 자동 큰 레지스터 파일 모드 문제로 좁혀졌다(IGC 2.42로 확인한 경우는 맞게 컴파일된다. 스레드당 레지스터 256개로 강제하면 여전히 일부가 비결정적으로 틀린다).

## GPU에서의 실행 라이브러리·라이브러리 시험

OpenCuda용으로 쓴 시험을 B70에서(SPIR-V 경로): 기호 복사(`rt_symbols` 32/32), 통합 메모리(`rt_managed` 45/45), 커널이 직접 쓰는 고정 호스트 메모리(`rt_pinned_mapped` 21/21), 메모리 API(`rt_mem` 49/49), 짧은 마지막 워프를 포함한 워프 셔플·투표(`rt_warp_lang` 346/346), `__half2` 비교(`rt_half2_cmp` 7073/7073), 변환·수학 내장 함수(`rt_math_ext_gpu` 71/71), double erf·erfc(`rt_erf_gpu`: 최대 0.58·3.25ulp, 문서 한도 2·5), 큰 커널 인자(`rt_big_args` 8/8), 메모리 풀(`rt_mempool` 275/275), CUB 쌍·반복자(`rt_cub_kvp` 120/120, `rt_cub_iter` 13/13), cuBLAS 정확도(`cublas_accuracy` 1959/1959, int8 GEMM 정확), cuBLASLt(`cublaslt_matmul` 101/101), cuBLAS 1·2단계(`cublas_level1` 301/301). CPU 경로 묶음(`ctest`) 209/209.

## 알려진 한계

- 커널이 호스트가 고정 메모리에 쓰는 깃발을 기다리면 그 쓰기를 못 볼 수 있다(시험에서 한 프로세스에 한 번은 보였지만 믿을 만하지 않다). 호스트 메모리로 실행 중인 커널과 일을 주고받는 프로그램은 이 경로에서 멈출 수 있다.
- 재배치 가능 장치 코드(`-dc`·`-rdc`: 파일 사이에 나누는 장치 함수·변수)는 이 경로에서 링크하지 않는다(HeCBench egs-cuda, fresnel-cuda. HeCBench 11개가 이렇게 빌드).
- Thrust, cuSPARSE, cuSOLVER, cuRAND, cuFFT, CUDA 그래프, 인라인 PTX는 없다.
- llama.cpp 토큰 생성은 SYCL 백엔드의 약 60%다. SYCL은 Intel 전용 가중치 배치를 쓰고, CUDA 호환 층은 그것을 재현하지 않는다.
- 시험은 코드와 같은 사람(과 AI 세션)이 썼다. 독립적인 검토를 환영한다.

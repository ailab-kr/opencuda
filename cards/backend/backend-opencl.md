---
id: backend-opencl
kind: component
name: OpenCL backend and CUDA-to-OpenCL translation
spec_sources:
  - backends/interface/ocu_backend.h (ABI 2)
  - Khronos OpenCL 3.0 API and OpenCL C 3.0 specifications; C++ for OpenCL 2021 documentation
  - backends/opencl/README.md (design)
backend_map:
  opencl: backends/opencl/, compiler/opencl/
state: cpu-pass on PoCL 6.0 (CPU, SPIR-V input); no GPU driver tried yet
owner_spec: AI Lab 2026-11-13
owner_tests: AI Lab 2026-11-13 (same author as impl: see note)
owner_impl: AI Lab 2026-11-13
cost: { hours: 3, ai_usage: 0 }
---

## 동작

- 백엔드(`backends/opencl/ocl_backend.c`): OpenCL 라이브러리를 실행 중에 연다(`libOpenCL.so.1`, `OCU_OPENCL_LIBRARY`). 장치 포인터 = 거친 단위 공유 가상 메모리(SVM) 주소: 더하기·메모리에 저장·구조체로 넘기기가 그대로 된다. 실행마다 모든 할당을 커널에 알린다. 인자는 Metal과 같은 값 덩어리 + 위치표. 장치 코드는 SPIR-V, C++ for OpenCL 소스, OpenCL C 소스 중 장치가 받는 것.
- 변환(`compiler/opencl/cu2ocl.py`): cu2msl의 하위 클래스. 주소 공간 분석·템플릿·장치 함수·모듈 변수를 그대로 쓰고, 커널 머리·인덱스·3성분 벡터 인자·`long long`(OpenCL에서는 128비트라 `long`으로)·안쪽 블록의 `__shared__`(커널 맨 위로)만 다르다. 생성한 소스는 변환 때 Clang으로 C++ for OpenCL 2021로 컴파일해 보고, `llvm-spirv`가 있으면 SPIR-V로.
- 전체 프로그램: `ocu_cc.py --target opencl`, `OCU_BACKEND=opencl`로 실행 API가 그대로 OpenCL에서 돈다.

- 공유 할당(`mem_alloc_shared`, `cudaMallocManaged`): 고운 단위 SVM 버퍼가 있으면 그것, 없고 `cl_intel_unified_shared_memory`의 단일 장치 공유 메모리 접근이 되면 USM 공유 할당(`clSharedMemAllocINTEL`, 256바이트 정렬, 해제는 `clMemBlockingFreeINTEL`). 호스트가 직접 읽고 쓰고 드라이버가 페이지를 옮긴다. 커널은 이미 켠 간접 접근 플래그로 닿는다. 둘 다 없으면 "지원 안 함". Arc Pro B70은 고운 단위 SVM이 없어 USM.
- 고정 호스트 메모리(`host_alloc_pinned`, `cudaMallocHost`·`cudaHostAlloc`): 장치가 USM 호스트 할당 접근을 알리면(`CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL`) `clHostMemAllocINTEL`(4096바이트 정렬). 커널이 같은 주소로 바로 읽고 쓴다(`pinned_is_mapped` = 1). 할당 표에 들어가 복사·채우기가 받지만 장치 메모리 사용량에는 안 센다. `cudaFree`로는 안 풀린다. 없으면(또는 `OCU_OPENCL_PINNED_USM=0`) 예전처럼 `posix_memalign`, 복사만.
- 모듈 변수(SPIR-V 경로): cu2spv가 `__device__`·`__constant__` 변수(addrspace(1)에서 바뀔 수 있거나 호스트가 쓰는 것)를 모듈 블록 하나로 옮긴다. 이미지 줄 `global <이름> <위치> <크기>`(cu2ocl 이미지와 같은 줄)로 `module_get_global`이 주소를 준다. 쓰는 커널은 `sparam` 줄 끝의 `glob` 표시: 블록 주소가 마지막(동적 공유 메모리 다음) 인자. 초기값은 원래 변수에 남고, 모듈을 올릴 때 `ginit <커널> <위치> <크기>` 커널이 블록으로 복사한다. 함수 주소를 쓰는 모듈처럼 다루지 못하는 꼴이거나 바꾼 IR이 맞지 않으면 그대로 둔다(그때 기호 복사는 예전처럼 "기호 없음"). `OCU_SPV_GLOBALS_BLOCK=0`이면 끈다.
- 기다리는 반복(SPIR-V 경로): 분기를 정하는 전역·일반 주소의 volatile 정수 읽기는 seq_cst 원자 읽기로 바꾼다(cu2spv `spin_loads`, `OCU_SPV_SPIN_LOADS=0`이면 끔). Arc Pro B70에서 호스트가 고정 메모리 깃발을 바꿔도 volatile 읽기·relaxed/acquire 원자 읽기·seq_cst 울타리·더하기 0 원자 연산은 30초 동안 옛 값을 봤고, seq_cst 원자 읽기만 바로 봤다(한 번에 약 1µs). 연산에 쓰는 volatile 읽기와 공유 메모리 읽기는 그대로.

## 경계 조건

- SVM이 없는 장치(OpenCL 1.2, SVM을 빼는 3.0 드라이버)는 목록에 없다(`OCU_OPENCL_VERBOSE=1`로 이유).
- 워프 함수(셔플·투표)는 "변환 안 됨": 하위 묶음 폭이 32로 정해지지 않는다(PoCL CPU는 블록 전체). `warpSize`도 없다.
- double은 `cl_khr_fp64` 장치에서만(없으면 모듈 빌드 실패).
- `mem_info`의 남은 양은 이 장치 핸들이 잡은 양을 뺀 값(OpenCL에는 질의가 없다).
- PoCL 6.0 CPU의 기본 작업 묶음 방식(loopvec)은 장벽 뒤 `break`가 있는 반복문(Rodinia pathfinder·hotspot)을 틀리게 돈다. `POCL_WORK_GROUP_METHOD=cbs`면 맞다. 우리 변환이 아니라 드라이버 문제로 보고, cbs로 같은 SPIR-V가 기준과 같은 답을 내는 것으로 확인했다.

## 시험 목록 (PoCL 6.0, Intel Core Ultra 7 270K Plus, 2026-11-13)

- [x] `opencl_backend`(손으로 쓴 OpenCL C): 171/171. 100만 값 saxpy 정답, 블록 합(동적 공유 메모리·장벽, 할당 안쪽 주소에서 시작), 전역 원자 히스토그램, 장치 메모리에 저장한 포인터 읽기, 2차원 격자, 모듈 변수(초기값 7 → 커널 뒤 12), 고운 단위 공유 메모리, 범위·실행 오류
    - 속도(100만 값 saxpy, 동기 실행 50회, 실행부터 끝까지): 중앙값 0.311ms, 느린 10% 0.451ms부터, 최대 0.507ms. 첫 실행은 JIT 때문에 61.8ms
- [x] `opencl_run_probe_kernel`: `probe_kernel.cu` 변환 실행 23/23
- [x] `opencl_run_cu2msl_features`: `cu2msl_features.cu` 변환 실행 90/90(주소 공간 오버로드, 커널·장치 함수 템플릿, `__constant__`/`__device__` 변수, 구조체 인자, 동적 공유 메모리)
- [x] `opencl_cu2ocl_host`: 변환기 9건(double, float3 인자, 이미지 모양, SPIR-V 매직, long long, 안쪽 블록 공유 메모리, 거부 3건)
- [x] 전체 프로그램: `tests/probe/main_saxpy.cu` + `probe_kernel.cu`를 `--target opencl`로: `main_saxpy: ok (0 bad)`
- [x] Rodinia 18개(`OCU_TARGET=opencl conformance/external/rodinia/run.sh`): `POCL_WORK_GROUP_METHOD=cbs`에서 17개 빌드, 14개 기준과 같음. PoCL 기본 방식에서는 17개 빌드, 12개 같음(pathfinder·hotspot 다름, 위 경계 조건). 실패: huffman(프로그램이 32비트 값을 32만큼 민다: CPU 경로와 같은 실패, OpenCL은 민 양을 폭으로 나눈 나머지로 정의), kmeans(빌드: Clang에 `omp.h` 없음)

- [x] Arc Pro B70(2026-11-28, SPIR-V 경로): `rt_symbols` 32/32(상수 표·오프셋·비동기 기호 복사, 구조체, 초기값, 원자 카운터, 커널 사이 변수, 장치 함수를 거친 접근, `cudaGetSymbolAddress`·크기, 범위 밖 오류), `rt_pinned_mapped` 21/21, `rt_mem` 49/49(전에는 GPU에서 멈춤), `rt_managed` 45/45. `rt_api`는 호스트가 고정 메모리 깃발로 커널을 푸는 첫 단계를 지나고, 두 번째 단계(예전 기본 스트림 복사가 막는 스트림을 기다림)에서 아직 멈춘다.
- [x] `spirv_cu2spv_host`: 19/19(변수 블록 4건, 기다리는 반복 6건 포함, llc·llvm-as로 바꾼 IR 확인)

## 메모

- 레지스터 파일(2026-11-27, Arc Pro B70): `OCU_OPENCL_BUILD_OPTIONS`가 없으면 모듈은 드라이버 기본 레지스터 파일로 빌드된다. 기본에서 스필하는 커널만 큰 레지스터 파일(`-cl-intel-enable-auto-large-GRF-mode`)로 한 번 더 만들어(두 번째 프로그램은 필요할 때 빌드), 처음 12번 실행을 번갈아 돌리고 작업 항목당 GPU 시간이 짧은 쪽을 남긴다. 스필이 있어도 기본 쪽이 빠른 커널이 있어서(llama.cpp의 mul_mat_vec_q) 스필 여부만으로 고르지 않는다. 바뀔 수 있는 프로그램 범위 변수가 있는 모듈(cu2spv가 `mutable_globals`로 알림, 그 밖의 이미지는 `CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE` > 0)은 변수가 두 벌이 되므로 예전처럼 한 번 빌드한다. 측정: llama.cpp qwen3-0.6B Q4_K tg128 220.3/219.7 → 235.4/234.8 t/s(번갈아 두 번), pp512 같음, 27B QLoRA 학습기 1.98 → 1.99 s/step(이 모듈은 한 번 빌드).

시험과 구현을 같은 세션이 썼다. 실제 GPU 드라이버(인텔·AMD·Mali)에서는 아직 돌리지 않았다. 시험 장치 PoCL과 SPIR-V 번역기는 공개 소스에서 사용자 폴더에 빌드했다(각 1분 안팎, 설치 크기 12MB·22MB).

## 다음

1. GPU 드라이버(인텔 NEO, Mesa rusticl, ROCm)에서 시험과 Rodinia
2. SVM 없는 장치용 `cl_ext_buffer_device_address`
3. 하위 묶음 폭을 정할 수 있는 장치에서 워프 함수

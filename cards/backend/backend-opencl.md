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
- 장치 메모리(`mem_alloc`, `cudaMalloc`, 2026-11-28부터 기본): 드라이버에 `cl_intel_unified_shared_memory`가 있으면 USM 장치 할당(`clDeviceMemAllocINTEL`, 256바이트 정렬). 그때 포인터 인자·복사·채우기·간접 접근 목록은 같은 인자 순서의 Intel USM 진입점(`clSetKernelArgMemPointerINTEL`, `clEnqueueMemcpyINTEL`, `clEnqueueMemFillINTEL`, `CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL`)으로 간다(`ocl_api_use_usm`). `OCU_OPENCL_DEVICE_USM=0`이면 예전처럼 거친 단위 SVM 버퍼. 잰 것(Arc Pro B70, 같은 회차 A/B): BabelStream Copy 547 → 906, Triad 535 → 1101 GB/s(chipStar 891·1114), HeCBench collision 33.9 → 2.4초, nonzero 65.8 → 39.6초, memtest 53.9 → 25.3초, perlin 18.2 → 12.5초, streamOrderedAllocation 41.4 → 28.2초. 느려진 것: cmembench 0.2 → 1.9초. llama.cpp는 측정 편차 안. GPU 시험 18종과 llama.cpp test-backend-ops 13종은 켬·끔 모두 같은 결과.
- 고정 호스트 메모리(`host_alloc_pinned`, `cudaMallocHost`·`cudaHostAlloc`): 장치가 USM 호스트 할당 접근을 알리면(`CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL`) `clHostMemAllocINTEL`(4096바이트 정렬). 커널이 같은 주소로 바로 읽고 쓴다(`pinned_is_mapped` = 1). 할당 표에 들어가 복사·채우기가 받지만 장치 메모리 사용량에는 안 센다. `cudaFree`로는 안 풀린다. 없으면(또는 `OCU_OPENCL_PINNED_USM=0`) 예전처럼 `posix_memalign`, 복사만.
- 모듈 변수(SPIR-V 경로): cu2spv가 `__device__`·`__constant__` 변수(addrspace(1)에서 바뀔 수 있거나 호스트가 쓰는 것)를 모듈 블록 하나로 옮긴다. 이미지 줄 `global <이름> <위치> <크기>`(cu2ocl 이미지와 같은 줄)로 `module_get_global`이 주소를 준다. 쓰는 커널은 `sparam` 줄 끝의 `glob` 표시: 블록 주소가 마지막(동적 공유 메모리 다음) 인자. 초기값은 원래 변수에 남고, 모듈을 올릴 때 `ginit <커널> <위치> <크기>` 커널이 블록으로 복사한다. 함수 주소를 쓰는 모듈처럼 다루지 못하는 꼴이거나 바꾼 IR이 맞지 않으면 그대로 둔다(그때 기호 복사는 예전처럼 "기호 없음"). `OCU_SPV_GLOBALS_BLOCK=0`이면 끈다.
- 기다리는 반복(SPIR-V 경로): 분기를 정하는 전역·일반 주소의 volatile 정수 읽기는 seq_cst 원자 읽기로 바꾼다(cu2spv `spin_loads`, `OCU_SPV_SPIN_LOADS=0`이면 끔). Arc Pro B70에서 호스트가 고정 메모리 깃발을 바꿔도 volatile 읽기·relaxed/acquire 원자 읽기·seq_cst 울타리·더하기 0 원자 연산은 30초 동안 옛 값을 봤고, seq_cst 원자 읽기만 바로 봤다(한 번에 약 1µs). 연산에 쓰는 volatile 읽기와 공유 메모리 읽기는 그대로.
  - 한계(아직 풀지 못함): 이것으로 깃발 하나를 한 번 기다리는 경우(시험 프로그램, `rt_api` 첫 단계)는 되지만 믿을 만하지 않다. 한 프로세스에서 두 번 기다리면(같은 깃발, 새로 할당한 깃발, 다른 커널, 루프 안 시스템 울타리 모두) 두 번째가 끝나지 않았고, 같은 프로그램의 첫 번째도 실행에 따라 끝나지 않았다(15초 제한). 원인은 모른다. 2026-11-28 더 잰 것: 별도 스트림·기본 스트림, 1~3회 반복, 레지스터 파일 고정(`-cl-intel-128-GRF-per-thread`), IGC 2.41·2.42 어느 조합에서도 결과가 실행마다 달랐다(같은 설정이 3회 모두 되기도, 1회차에 멈추기도). 될 때는 늘 약 32만 번 읽은 뒤(300ms) 깃발을 봐서 커널은 돌고 있었다. 호스트 쪽 호출(조회는 `clFlush`, 동기화는 `clFinish`)은 막지 않는다. 그래서 런타임 논리보다 드라이버·하드웨어의 호스트 메모리 일관성 쪽으로 본다(재현 프로그램: 서버 /tmp/sym/p4.cu). 호스트 고정 메모리로 실행 중인 커널과 주고받는 프로그램은 이 경로에서 멈출 수 있다.
- 큰 값 인자(SPIR-V 경로): OpenCL 드라이버가 받는 커널 인자는 `CL_DEVICE_MAX_PARAMETER_SIZE`까지(Arc Pro B70 2048바이트), CUDA는 32764(12.1 전에는 4096). 값 인자 합이 1024바이트(`OCU_SPV_ARG_LIMIT`)를 넘는 커널은 cu2spv가 가장 큰 byval 매개변수부터 `noalias readonly` 전역 포인터로 바꾸고(`sparam`의 `b<크기>`), 커널이 들어갈 때 사적 사본으로 복사한다(byval과 같은 뜻). opt 뒤 그 사본을 주소 계산(getelementptr)과 읽기로만 쓰면 사본을 없애고 버퍼를 그 자리에서 읽는다(`byval_direct`): 작업 항목마다 3KB를 복사하면 HeCBench axpby-cuda의 실행 약 1만 3천 번(블록 320 × 512)이 실행마다 약 0.5GB를 옮겨 180초 안에 끝나지 않았다. 다른 쓰임(phi, select, 호출, 비교, 포인터 저장)이 있으면 사본을 둔다(맞지만 느림). 백엔드는 실행마다 SVM 버퍼를 만들어 커널 앞에 비동기 복사(호스트 사본은 복사가 끝나면 이벤트 콜백이 해제)하고, 커널 뒤에 `clEnqueueSVMFree`를 넣는다(포인터 목록은 힙 사본: 스택 배열을 넘기면 axpby가 호스트에서 세그폴트). 콜백·`clEnqueueSVMFree`가 없거나 `OCU_OPENCL_STAGE_SYNC=1`이면 동기 복사, 커널이 끝난 뒤 해제. PyTorch·Apex의 `multi_tensor_apply` 꼴(주소·크기·블록 표 구조체 약 3KB). llama.cpp 커널 7814개 중 가장 큰 인자는 408바이트라 바뀌지 않는다. Arc Pro B70: `rt_big_args` 8/8, axpby-cuda 13/13 PASS 39초(전에는 FAIL; chipStar 33.8초 PASS).
- 워프 셔플(SPIR-V 경로): 짧은 마지막 워프(블록 크기가 32의 배수가 아님)에서 없는 레인을 읽으면 자기 값(다른 경로와 같은 규칙; Arc Pro B70은 쓰레기 값을 줬다). 셔플은 원래 레인 식 그대로 하고 결과만 바꾼다. 하위 묶음 질의(`get_sub_group_local_id`·`get_sub_group_size`)는 `const, noconvergent`로 선언해 셔플마다 다시 묻지 않는다(HIP 모드 Clang은 장치 호출을 모두 convergent로 만들고 LLVM은 convergent 호출을 합치지 않는다). 비용: llama.cpp qwen3-0.6B Q4_K tg128 245 t/s, 확인 없이 251(`OCU_SPV_SHFL_ANY_LANE`), noconvergent 전에는 238. `threadIdx` 등 색인 질의에도 붙여 봤지만 차이가 없어 두지 않았다.

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

- [x] Arc Pro B70(2026-11-28, SPIR-V 경로): `rt_symbols` 32/32(상수 표·오프셋·비동기 기호 복사, 구조체, 초기값, 원자 카운터, 커널 사이 변수, 장치 함수를 거친 접근, `cudaGetSymbolAddress`·크기, 범위 밖 오류), `rt_pinned_mapped` 21/21, `rt_mem` 49/49(전에는 GPU에서 멈춤), `rt_managed` 45/45. `rt_api`는 호스트가 고정 메모리 깃발로 커널을 푸는 첫 단계를 지나고, 두 번째 단계(예전 기본 스트림 복사가 막는 스트림을 기다림)에서 아직 멈춘다(위 한계). `rt_warp_lang` 346/346(전에는 짧은 마지막 워프 9건 실패). llama.cpp `test-backend-ops` SOFT_MAX 215·RMS_NORM 121·SUM_ROWS 11·ARGMAX 7·MUL_MAT 2724·FLASH_ATTN_EXT(hsk=128) 859 모두 통과.
- [x] `spirv_cu2spv_host`: 26/26(변수 블록 4건, 기다리는 반복 6건, 큰 값 인자 7건 포함, llc·llvm-as로 바꾼 IR 확인)

## 메모

- 레지스터 파일(2026-11-27, Arc Pro B70): `OCU_OPENCL_BUILD_OPTIONS`가 없으면 모듈은 드라이버 기본 레지스터 파일로 빌드된다. 기본에서 스필하는 커널만 큰 레지스터 파일(`-cl-intel-enable-auto-large-GRF-mode`)로 한 번 더 만들어(두 번째 프로그램은 필요할 때 빌드), 처음 12번 실행을 번갈아 돌리고 작업 항목당 GPU 시간이 짧은 쪽을 남긴다. 스필이 있어도 기본 쪽이 빠른 커널이 있어서(llama.cpp의 mul_mat_vec_q) 스필 여부만으로 고르지 않는다. 바뀔 수 있는 프로그램 범위 변수가 있는 모듈(cu2spv가 `mutable_globals`로 알림, 그 밖의 이미지는 `CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE` > 0)은 변수가 두 벌이 되므로 예전처럼 한 번 빌드한다. 측정: llama.cpp qwen3-0.6B Q4_K tg128 220.3/219.7 → 235.4/234.8 t/s(번갈아 두 번), pp512 같음, 27B QLoRA 학습기 1.98 → 1.99 s/step(이 모듈은 한 번 빌드).

시험과 구현을 같은 세션이 썼다. 실제 GPU 드라이버(인텔·AMD·Mali)에서는 아직 돌리지 않았다. 시험 장치 PoCL과 SPIR-V 번역기는 공개 소스에서 사용자 폴더에 빌드했다(각 1분 안팎, 설치 크기 12MB·22MB).

## 다음

1. GPU 드라이버(인텔 NEO, Mesa rusticl, ROCm)에서 시험과 Rodinia
2. SVM 없는 장치용 `cl_ext_buffer_device_address`
3. 하위 묶음 폭을 정할 수 있는 장치에서 워프 함수

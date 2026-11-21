# AMD GPU 백엔드 설계

상태: 1차 구현(2026-11-21). 장치 코드 경로는 실제 GPU(Radeon 780M, gfx1103, Windows 11, 드라이버 3686.0, HIP 런타임 7.1)에서 시험 47건을 통과했다(7장). C 백엔드(`amd_backend.c`, `amd_hip.c`, `amd_meta.c`)는 이 문서의 2장대로 작성했고 C11 `-Wall -Wextra -Wpedantic` 타입 검사를 경고 없이 통과했지만, **이 PC에서 빌드·실행하지는 못했다**: 스마트 앱 컨트롤이 서명 없는 실행 파일(직접 빌드한 것 포함)을 막는다. 같은 호출 순서를 서명된 Python에서 HIP 런타임으로 그대로 실행해 확인했다(`tools/amdgpu/`). "확인 필요" 표시는 실제 설치본에서 확인하기 전에는 사실로 쓰지 않는다는 뜻이다.

## 1. 방침

- AMD가 공개한 런타임(HIP 런타임, ROCm에 포함, MIT 라이선스)을 **호출만** 한다. 그 소스나 헤더를 저장소에 복사하지 않는다.
- 연결 층 규격(`backends/interface/ocu_backend.h`)의 함수 26개를 HIP 런타임 함수에 대응시킨다. 위층(런타임·라이브러리)은 바꾸지 않는다.
- 장치 코드는 공개 컴파일러(Clang)가 HIP 모드로 만든 코드 객체를 쓴다. 이 저장소의 컴파일러 실험에서 gfx1100(RDNA3)·gfx1201(RDNA4)용 코드 객체가 이미 만들어졌다(`docs/compiler-probe.md`).
- 행렬 연산은 AMD의 공개 BLAS 라이브러리(rocBLAS, MIT 라이선스)를 BLAS 제공자 뒤에 둔다(`backends/interface/ocu_blas_provider.h`).

## 2. 함수 대응

| 연결 층 | HIP 런타임 | 메모 |
|---|---|---|
| `dev_count` | `hipGetDeviceCount` | |
| `dev_info` | `hipGetDeviceProperties` | 이름, 전체 메모리, `warpSize`(제품에 따라 32 또는 64), 블록·격자 한도, 블록당 공유 메모리 |
| `dev_open` / `dev_close` | `hipSetDevice` + 내부 대기열 하나 생성 / 해제 | 내부 대기열은 `queue == NULL` 동기 실행용 |
| `mem_alloc` / `mem_free` | `hipMalloc` / `hipDeviceSynchronize` 후 `hipFree` | 규격: 해제는 대기 중인 작업을 먼저 기다린다. `hipFree`의 암묵적 동기화 여부는 확인 필요. 확인 전에는 명시적으로 기다린다 |
| `mem_copy` | `hipMemcpyAsync` + (대기열 없으면) `hipStreamSynchronize` | 방향 상수 대응 |
| `mem_set` | `hipMemsetAsync` | |
| `mem_info` | `hipMemGetInfo` | |
| `host_alloc_pinned` / `host_free_pinned` | `hipHostMalloc` / 동기화 후 `hipHostFree` | |
| `queue_create` / `queue_destroy` | `hipStreamCreateWithFlags`(non-blocking) / 동기화 후 `hipStreamDestroy` | 규격: 해제는 남은 일을 끝낸 뒤 |
| `queue_sync` / `queue_query` | `hipStreamSynchronize` / `hipStreamQuery` | `hipErrorNotReady` → `OCU_ERR_NOT_READY` |
| `module_load` / `module_unload` | `hipModuleLoadData` / `hipModuleUnload` | 이미지 = Clang이 만든 코드 객체(ELF). 여러 칩용을 한 파일에 넣는 형식은 컴파일러 쪽에서 정한다 |
| `module_get_kernel` | `hipModuleGetFunction` | 없으면 `OCU_ERR_NOT_FOUND` |
| `kernel_launch` | `hipModuleLaunchKernel(f, grid, block, shared, stream, args, NULL)` | 우리 `args` 배열(인자 값마다 포인터 하나)이 이 함수의 인자 형식과 같다. 인자 값을 실행 시점에 복사하는지는 확인 필요 |
| `marker_*` | `hipEventCreate`, `hipEventRecord`, `hipEventSynchronize`, `hipEventQuery`, `hipEventElapsedTime`, `hipEventDestroy` | 기록한 적 없는 표식의 조회·대기 결과가 규격(완료로 취급)과 같은지 확인 필요. 다르면 백엔드가 상태를 따로 기억한다 |
| `get_provider("blas")` | rocBLAS 래퍼 | 아래 4장 |

## 3. 규격과 다를 수 있는 곳

CPU 백엔드에서 통과한 시험(`tests/backend/test_cpu_backend.c`, `test_cpu_regressions.c`)을 AMD 백엔드에 그대로 돌려 차이를 찾는다. 미리 예상하는 차이는 다음과 같다.

| 규격 | 예상 | 대응 |
|---|---|---|
| 해제는 대기 작업을 기다린다 | 런타임마다 다름 | 해제 전에 장치 동기화 |
| 대기열 오류는 다음 동기화·조회에서 한 번 알리고 지운다 | HIP의 오류는 "끈끈한" 종류가 있다 | 오류 종류별로 대응표를 만든다 |
| 커널 안에서 백엔드 호출은 "지원 안 함" | GPU에서는 커널이 호스트 함수를 부를 수 없으므로 해당 없음 | 없음 |
| 묶음 단위 크기는 실제 값 | `warpSize`가 제품에 따라 32 또는 64 | 그대로 보고 |
| 인자 합계 32768바이트 이하 | HIP의 상한은 확인 필요 | 작은 쪽을 규격으로 |

## 4. BLAS 제공자

- 대기열마다 rocBLAS 핸들 하나를 만들고 `rocblas_set_stream`으로 묶는다.
- 스칼라 결과(dot, nrm2, asum, iamax)는 장치 메모리로 받아야 하므로 포인터 모드를 장치 쪽으로 둔다.
- 음수 증가폭, `beta == 0`일 때 출력을 읽지 않는 규칙, iamax의 1부터 세기가 BLAS 표준과 같은지는 기준 사례 526건(`conformance_blas_*`)으로 확인한다. 다르면 제공자 안에서 맞춘다.
- 속도 기준선은 같은 rocBLAS를 직접 부른 시간이다. 우리 층을 거친 시간과의 비율이 PLAN의 속도 지표다.

## 5. 장비가 들어온 날 할 일

1. 설치 기록: 운영체제, 커널, 드라이버, ROCm 버전, 지원 목록에 그 GPU가 있는지 (`docs/setup.md`)
2. `hipGetDeviceProperties`로 장치 정보와 `warpSize` 확인
3. 컴파일러 실험의 gfx 코드 객체를 `hipModuleLoadData`로 올려 `saxpy` 실행
4. 2장의 "확인 필요" 항목을 하나씩 작은 프로그램으로 확인하고 이 문서를 고침
5. 백엔드 구현 후 CPU 백엔드 시험 묶음 전체와 BLAS 기준 사례를 AMD에서 실행

## 7. 실측 기록 (Radeon 780M, 2026-11-21)

장비: GMKtec K12, Ryzen 7 H 255, Radeon 780M(gfx1103, RDNA3, 연산 유닛 6개 보고), RAM 16GB(그래픽 공용 2GB), Windows 11 Pro 26300, AMD 드라이버 3686.0(Adrenalin 25.4.1), HIP 런타임 7.1(`amdhip64_7.dll`), 컴파일러 라이브러리 `amd_comgr_3.dll`(API 3.5, AMD Clang 24.0.0git). ROCm은 설치하지 않았다. 두 DLL은 드라이버와 함께 System32에 들어 있고 서명되어 있다.

### 7.1 장치 코드 경로

- **컴파일:** Clang 실행 파일 없이 드라이버의 `amd_comgr`를 Python에서 불러 쓴다(`tools/amdgpu/comgr.py`, `ocu_amd_cc.py`).
  - CUDA 소스를 HIP 모드로, `-nogpuinc -nogpulib`로 컴파일한다. AMD HIP 헤더와 AMD 장치 라이브러리는 쓰지 않는다.
  - OpenCuda 헤더(`cuda_runtime.h` → `ocu_cuda_lang_amdgcn.h`)를 강제로 포함한다.
  - 결과물은 코드 객체(ELF)다. 파일 하나에 0.1~0.15초 걸린다.
- **comgr 3.x 동작 번호:** 2.x 헤더의 번호와 다르다. 실행해서 확인한 값은 다음과 같다.
  - 0x0 전처리, 0x2 소스 → 비트코드, 0x4 비트코드 → 재배치 객체
  - 0x6 재배치 객체 → 재배치 객체(ld.lld -r), 0x7 재배치 객체 → 실행 객체
  - 0xC 소스 → 비트코드(장치 라이브러리 포함), 0xD 소스 → 재배치 객체, 0x14 소스 → SPIR-V
- **comgr에 들어 있는 헤더:** Clang 기본 헤더 중 `stddef.h`와 `stdint.h`만 있다.
  - `limits.h`, `float.h`, `stdarg.h`, `stdbool.h`는 `runtime/include/freestanding/`에 새로 넣었고, 검색 경로 맨 뒤(`-idirafter`)에 둔다.
  - libc++는 일부만 들어 있다(`<cstddef>`는 있지만 동작하지 않고, `<utility>`는 없다). 그래서 `<utility>`가 있을 때만 C++ 표준 라이브러리가 있다고 본다.
- **기본 언어 판:** C++14라서 `-std=c++17`을 넘긴다.
- **미리 정의된 매크로:** `__HIP__`, `__HIP_DEVICE_COMPILE__`, `__AMDGCN__`, `__gfx1103__`, `_MSC_VER`(1933)가 정의된다. `__CUDA__`, `__CUDACC__`, `__CUDA_ARCH__`는 정의되지 않는다. 언어 헤더가 `__CUDACC__`와 `__CUDA_ARCH__`(기본 800)를 정의한다.
- **커널 이름:** Itanium 맹글링이다(`_Z5saxpyifPKfPf`). NVIDIA 경로와 같아서 같은 시험 드라이버를 그대로 쓴다.
- **워프 폭:** gfx11의 기본은 wave32이고, CUDA의 워프와 같은 32다. wave64로 빌드하면 언어 헤더가 오류를 낸다.

### 7.2 시험 결과 (모두 수정하지 않은 .cu)

| 시험 | 내용 | 결과 |
|---|---|---|
| `compiler/probe/probe_kernel.cu` | saxpy 100,000개, 공유 메모리 블록 합 | 3 / 3 |
| `tests/metal/cu2msl_features.cu` | 주소 공간 오버로드, 커널·함수 템플릿, `__constant__`/`__device__` 변수(초깃값, 호스트 복사, atomic), 구조체 인자(32바이트·8바이트), 동적 공유 메모리, 모듈 두 개의 변수 분리, 인자 크기 목록 | 15 / 15 |
| `tests/amd/amd_lang.cu` | 3차원 인덱스, 셔플 11종(폭 8·16·32), 64비트·float 셔플, ballot/any/all/match_any/activemask, atomic 10종(int·unsigned·u64·float·double·Inc·Dec·Min·Max·Or·Xor·And·CAS), `__syncthreads_count/and/or`, 정수·변환 내장 함수 24종(포화·NaN 포함), 수학 12종, 워프 리덕션, 템플릿 커널 + `__device__` 변수 | 29 / 29 |

- **수학 함수 오차** (최대, ulp): expf 1, logf 2, sinf 1, cosf 1, tanhf 0, sqrtf 0, powf 0, atan2f 1, erff 1, exp2f 1, log2f 2, fmodf 0
- **빌드 조사** (`tools/amdgpu/build_survey.py`): 저장소의 .cu 48개 중 **9개**가 이 PC에서 장치 컴파일된다.
  - 나머지는 호스트 코드가 쓰는 C/C++ 표준 헤더(`<cstdio>`, `<cmath>`, `<string.h>` 등)가 이 PC에 없어서 멈춘다.
  - 표준 라이브러리가 있는 환경(Linux, MSVC/MinGW 설치)에서는 이 원인이 없어질 것으로 본다.

### 7.3 2장의 "확인 필요" 항목

| 항목 | 관찰 |
|---|---|
| 장치 속성 번호 | `hipDeviceGetAttribute`를 0~159번까지 읽어 문서의 열거 순서와 맞춰 보았다. 블록 차원 26~28(1024), 격자 29~31(2147483647, 65535, 65535), 블록당 스레드 56(1024), 블록당 공유 메모리 74(65536), 연산 유닛 수 63(6), 워프 87(32), 통합형 16(1), 관리 메모리 24(0). PCI 버스 번호 67(101 = 0x65)도 순서가 맞는다는 근거다. 값은 `amd_hip.h`에 있다 |
| 인자 값을 실행 시점에 복사하는가 | **복사한다.** 비동기 실행 직후 인자 변수를 바꿔도 커널은 원래 값을 썼다 |
| 기록한 적 없는 표식 | `hipEventQuery`와 `hipEventSynchronize` 모두 성공(완료)을 돌려준다. 규격과 같다. 기록 안 된 표식으로 `hipEventElapsedTime`을 부르면 400(잘못된 핸들)이고, 이것도 규격과 같다 |
| 오류가 계속 남는가 | 설정 오류(블록 2048, 공유 메모리 70000바이트)는 1(잘못된 값)이고 남지 않는다. 바로 다음 정상 실행은 성공했다. 실행 중 결함(잘못된 주소)은 원격 장비의 화면 드라이버를 재시작시킬 수 있어 **시험하지 않았다** |
| 오류 번호 | 커널 이름 없음 500, 메모리 부족 2, 장치 번호 잘못 101, 이미지 잘못 200, 바쁜 대기열 조회 600. CUDA와 같은 번호다 |
| `hipFree`가 남은 일을 기다리는가 | 결과가 애매해서 결정하지 못했다. 백엔드는 해제 전에 장치를 동기화한다 |
| 관리 메모리(`hipMallocManaged`) | 이 장비의 속성은 0이다(Windows). `mem_alloc_shared`는 "지원 안 함"을 돌려준다 |
| 인자 크기 목록 | 코드 객체의 AMDGPU 메타데이터(MessagePack)에서 읽는다. 컴파일러가 넣는 `hidden_*` 인자는 뺀다. `with_struct`는 [32, 4, 8]로, C 시험의 기대값과 같다 |
| 인자 합계 상한 | 확인 안 함 |

### 7.4 같은 장비의 OpenCL 드라이버 (참고)

같은 GPU는 AMD OpenCL 드라이버로도 보인다. 플랫폼은 "AMD Accelerated Parallel Processing"이고, 장치는 OpenCL 2.0, 이름 `gfx1103`이다. 다만 OpenCuda OpenCL 백엔드의 조건은 만족하지 못한다.

- **SPIR-V:** 받지 않는다. `cl_khr_il_program`이 없고 IL 버전도 비어 있다.
- **C++ for OpenCL:** 받지 않는다. `-cl-std=CLC++` / `CLC++2021`로는 빈 커널도 컴파일에 실패한다.
- **SVM:** 거친 단위와 고운 단위 모두 있다. 손으로 쓴 OpenCL C 커널을 SVM 포인터로 실행해 결과가 맞았다.
- **바이너리 입력:** 드라이버는 `clCompileProgram`이 내는 "컴파일된 객체"를 다시 받아 링크한다. 이 객체는 `.llvmir` 섹션에 amdgcn LLVM 비트코드를 담은 ELF 재배치 파일이고, e_machine 값은 1022다.
  - 그러니 외부 Clang으로 만든 비트코드를 같은 형식에 담아 넘기는 길이 있을 수 있다.
  - 이 PC에서는 Clang이 실행되지 않아 확인하지 못했다.

그래서 AMD GPU는 OpenCL이 아니라 HIP 경로(이 백엔드)로 지원한다.

### 7.5 남은 일

1. C 백엔드를 실제로 빌드하고 시험한다. Linux(ROCm) 또는 스마트 앱 컨트롤을 끈 Windows에서 `-DOCU_BACKEND_AMD=ON`으로 빌드하고, `amd_lang`, `amd_run_probe_kernel`, `amd_run_cu2msl_features`, CPU 백엔드 시험 묶음을 돌린다.
2. 실행 API(`runtime/cudart`)와 연결한다. `OCU_BACKEND=amd`, `ocu_cc.py --target amd` 호스트 패스.
3. rocBLAS 제공자(4장).
4. 장치 `printf`. 지금은 링크 오류로 멈춘다.
5. wave64 장치(GCN/CDNA)의 워프 함수.

## 6. 미리 알아 둘 위험

- WSL에서 쓸 수 있는 AMD GPU와 ROCm 버전의 범위는 리눅스보다 좁다. 확인 필요. 리눅스 전용 PC가 더 안전하다.
- 소비자용 RDNA 제품의 ROCm 지원은 데이터센터용보다 늦게 들어오는 경우가 있다. 장비 결정(D-01) 전에 공식 지원 목록을 본다.
- 실행 시간 측정은 준비 실행과 반복 측정 규칙(중앙값과 느린 10%)을 따른다. 첫 실행에는 코드 적재와 컴파일 시간이 섞인다.

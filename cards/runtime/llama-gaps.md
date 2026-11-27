---
id: runtime-llama-gaps
kind: component
name: Runtime and header gaps for the llama.cpp CUDA backend
spec_sources:
  - public CUDA Runtime API documentation (device management, peer access, cooperative launch, attribute enumeration)
  - public CUDA Math API reference (bfloat16 intrinsics, SIMD intrinsics, integer intrinsics, type casting intrinsics)
  - public CUDA C++ Programming Guide (Cooperative Groups, synchronization functions, read-only data cache loads)
  - public CUDA library-types documentation (cudaDataType)
  - docs/llama-cpp-requirements.md, sections 1.2, 4.1, 4.3, 6.3, 7 and 8 (items 3 to 7, and the loud-failure part of 9)
backend_map:
  cpu: runtime/ over backends/cpu/
  nvidia: runtime/ over backends/nvidia/
state: impl-written-untested
owner_spec: AI Lab 2026-11-13
owner_tests: (to be assigned: a session other than owner_impl)
owner_impl: AI Lab 2026-11-13
cost: { hours: 2.0, ai_usage: 0 }
---

## 동작

llama.cpp의 CUDA 백엔드가 요구하는 이름 중 우리 헤더와 실행 API에 없던 것을 채운다(조사: `docs/llama-cpp-requirements.md`).
모두 공개 문서만 보고 새로 썼다.

- `cuda_bf16.h`: `__nv_bfloat16`/`nv_bfloat16`, `__nv_bfloat162`/`nv_bfloat162`(정렬 4). float에서 가장 가까운 짝수로, float로는 정확히. `_rz`·`_rd`·`_ru` 변환, double에서는 두 번 반올림하지 않음(홀수 반올림을 거쳐서). 사칙·fma(float에서 계산해 한 번 반올림, 24 ≥ 2·8+2이므로 정확한 반올림과 같음), 순서·비순서 비교, 연산자, 수학 함수, 쌍 연산, `make_bfloat162`, `__float22bfloat162_rn`, `__bfloat1622float2`, `__low2bfloat16`, `__high2bfloat16`.
- `library_types.h`(`cuda_runtime_api.h`가 포함): `cudaDataType_t`/`cudaDataType`과 문서의 `CUDA_R_*`/`CUDA_C_*` 값(`CUDA_R_32F`=0, `CUDA_R_16F`=2, `CUDA_R_16BF`=14, `CUDA_R_8F_E4M3`=28 등).
- 장치 속성 정책(`cudaGetDeviceProperties`): 계산 능력 5.0(환경변수 `OCU_COMPUTE_CAPABILITY=M.m`으로 바꿀 수 있음), `warpSize`는 백엔드 값, `sharedMemPerBlockOptin`은 CPU·엔비디아에서 48 KiB 이상, `cudaFuncGetAttributes().ptxVersion` = 10·major+minor(< 90), PCI 번호는 합성값(아래).
- 속성 열거: `cudaDevAttrCooperativeLaunch`(95)와 `cudaDevAttrCooperativeMultiDeviceLaunch`(96)는 늘 0. `cudaDevAttrPciBusId`(33)·`PciDeviceId`(34)·`PciDomainId`(50)는 속성 구조체와 같은 값.
- SIMD 내장 함수 전부(2026-11-27): 반워드·바이트 단위 abs·absdiff·add·sub(감김·포화)·평균·cmp·set·max·min·neg·sad, 32비트·16x2의 `__vimax3_*`·`__vimin3_*`·`__viaddmax_*`·`__viaddmin_*`·`__vibmax_*`·`__vibmin_*`·`_relu`. 문서가 정하지 않은 곳은 이렇게 읽었다: 포화는 레인의 표현 범위, `__vsads`/`__vsadu`는 레인별 |a-b|의 합 하나, 16x2 술어는 pred_hi가 위쪽 반워드, add-then-max/min의 a+b는 레인 안에서 감김. 시험은 `tests/runtime/rt_simd.cu`.
- 장치 쪽 내장 함수(CPU 경로·NVPTX 경로 모두): `__vsub4`, `__vcmpne4`, `__isnanf`, `__isinff`, `__ldg`(템플릿, 평범한 읽기). `make_half2`는 `cuda_fp16.h`. 공통 정의는 `ocu_cuda_lang_common.h`.
- CPU 경로에 추가(NVPTX에는 이미 있었음): `__byte_perm`, `__umulhi`, `__mulhi`, `__float2int_{rn,rz,rd,ru}`, `__double2int_{rn,rz,rd,ru}`, `__syncthreads_{count,and,or}`, `__trap`.
- `__trap`: CPU 경로에서는 stdout·stderr를 비우고 스레드 위치가 든 메시지를 찍은 뒤 `abort()`. 두 경로 모두 `[[noreturn]]`.
- `cooperative_groups.h`와 `cooperative_groups/reduce.h`: `thread_block`, `this_thread_block()`, `thread_block_tile<N>`(N은 32 이하의 2의 거듭제곱)과 `tiled_partition<N>`, 타일의 `sync`·`shfl`·`shfl_up`·`shfl_down`·`shfl_xor`·`ballot`·`any`·`all`, `grid_group`/`this_grid()`(`is_valid()`는 false, `sync()`는 메시지를 찍고 `__trap`), `cg::reduce`와 `plus`·`less`·`greater`·`bit_and`·`bit_or`·`bit_xor`(타일만). 그 밖의 것(런타임 크기 분할, `coalesced_threads`, `memcpy_async` 등)은 이름이 없어 컴파일 오류.
- 실행 API: `cudaDeviceGetPCIBusId(buf, len, dev)`는 합성 식별자 `"0000:BB:00.0"`(BB = 1 + 장치 번호, 16진수). `len < 13`이면 `cudaErrorInvalidValue`. `cudaDeviceEnablePeerAccess(peer, flags)`: 플래그가 0이 아니면 `cudaErrorInvalidValue`, 없는 장치나 현재 장치면 `cudaErrorInvalidDevice`, 그 밖에는 `cudaErrorPeerAccessUnsupported`(`cudaDeviceCanAccessPeer`가 늘 0이므로). `cudaDeviceDisablePeerAccess`: `cudaErrorPeerAccessNotEnabled`. `cudaLaunchCooperativeKernel`: 늘 `cudaErrorNotSupported`(블록이 모두 동시에 올라 있음을 어느 백엔드도 보장하지 못함). 함수 포인터를 받는 C++ 중복 정의 있음.

## 경계 조건

- NO_DEVICE_CODE: llama.cpp의 매크로는 `__CUDA_ARCH__`가 정의되어 있어야 `no_device_code()`(→ `__trap()`)로 풀린다. NVIDIA 경로는 장치 컴파일에서 `__CUDA_ARCH__`가 정의되므로 잘못 고른 커널이 "unspecified launch failure"로 실패한다. CPU 경로는 `-D__CUDA_ARCH__=500 -D__CUDA_ARCH_LIST__=500`을 주어야 같은 효과(프로세스 중단)가 난다. 드라이버가 이를 기본으로 정의하는 일은 이 카드의 범위 밖(조사 문서 8장 1번).
- 문서에 있는 `__vsub4`·`__vcmpne4` 등 SIMD 함수는 장치 전용이다(호스트에서 부르면 컴파일 오류, CUDA와 같음). `__isnanf`·`__isinff`는 호스트에서는 C 라이브러리의 것을 쓴다.
- `__ldg`는 값만 보장한다(읽기 전용 캐시의 속도 효과 없음).
- `grid_group::sync()`는 어떤 경우에도 성공하지 않는다.
- 피어·협력 실행 항목은 단일 장치 기준. 장치 둘 이상에서 피어를 지원하는 백엔드가 생기면 `cudaDeviceCanAccessPeer`부터 바꾼다.

## 검증할 것

시험 작성자(구현자와 다른 세션)가 확인할 동작. 괄호는 근거 문서.

### cuda_bf16.h (CUDA Math API, Bfloat16 Precision Intrinsics; bfloat16 형식 정의)
1. 모든 유한 bf16 65,536개: `__bfloat162float` 후 `__float2bfloat16`로 왕복하면 같은 비트(NaN 제외). 부호 있는 0과 무한대 보존.
2. 이웃한 두 bf16 사이의 정확한 중간 float(동률)은 짝수 가수 쪽, 중간보다 한 float 위·아래는 각각 위·아래로(`_rn`). `_rz`는 0 쪽, `_rd`는 아래, `_ru`는 위. 최대 유한값 근처에서 `_rn`·`_ru`는 무한대로 넘침, `_rz`·`_rd`(양수)는 최대 유한값에 머무름.
3. NaN 입력은 NaN(조용한 NaN)이 되고 `__hisnan`이 참. 무한대 입력은 무한대.
4. double 변환: 동률 바로 위·아래 double(float로는 구분 안 되는 거리)이 올바른 쪽으로 반올림됨(두 번 반올림 결함이 없음). 큰 double은 무한대, 아주 작은 double은 0.
5. 독립 기준(정수 연산)과 비교한 덧셈·뺄셈·곱셈·나눗셈·fma: 무작위 조합에서 정확한 반올림과 일치.
6. 비교: 순서 비교는 NaN에서 false, `u`로 끝나는 비순서 비교는 NaN에서 true, `operator!=`는 NaN에서 true(=`__hneu`). `__hmax`/`__hmin`은 NaN 하나를 무시.
7. `__nv_bfloat162`: 크기 4·정렬 4, `{a, b}` 초기화, `.x`가 하위 반쪽, `__low2bfloat16`/`__high2bfloat16`/`__halves2bfloat162`/`__lows2bfloat162`/`__highs2bfloat162`/`__lowhigh2highlow`, `__bfloat1622float2`/`__float22bfloat162_rn`, 쌍 연산자와 `__hfma2`.
8. 정수 변환: NaN은 0, 범위 밖은 포화.
9. 같은 함수들이 커널 안(CPU 경로와 NVIDIA 경로)에서 호스트와 같은 비트를 낸다.

### 라이브러리 형 (CUDA library-types documentation)
10. `CUDA_R_32F=0, CUDA_R_64F=1, CUDA_R_16F=2, CUDA_R_8I=3, CUDA_C_32F=4, CUDA_R_16BF=14, CUDA_R_8F_E4M3=28, CUDA_R_8F_E5M2=29` 등 문서의 모든 값. `cudaDataType_t`와 `cudaDataType`이 같은 형. C와 C++ 양쪽에서 포함됨.

### 장치 속성 정책 (조사 문서 6.3)
11. 기본 `major.minor` = 5.0, `warpSize` = 32, `sharedMemPerBlockOptin` ≥ 48 KiB(CPU·NVIDIA), `multiProcessorCount` > 0, `cudaFuncGetAttributes().ptxVersion` < 90. `OCU_COMPUTE_CAPABILITY=8.0`이면 8.0, 잘못된 값("x", "50")은 무시하고 5.0. `cudaDeviceGetAttribute`의 계산 능력 값이 속성 구조체와 같음.
12. `cudaDeviceGetAttribute(cudaDevAttrCooperativeLaunch)`와 `...MultiDeviceLaunch`는 0, 잘못된 장치 번호는 `cudaErrorInvalidDevice`. PCI 속성 3개가 구조체의 `pciBusID`·`pciDeviceID`·`pciDomainID`와 같음.

### 실행 API (CUDA Runtime API, Device Management)
13. `cudaDeviceGetPCIBusId`: 장치 0은 `"0000:01:00.0"`, 장치 1이 있으면 `"0000:02:00.0"`; 호출마다 같은 값; `len` 13 미만과 NULL은 `cudaErrorInvalidValue`; 범위 밖 장치는 `cudaErrorInvalidDevice`; 버퍼가 딱 13바이트여도 끝에 NUL.
14. `cudaDeviceEnablePeerAccess`: 플래그 ≠ 0 → `cudaErrorInvalidValue`; 자기 자신·범위 밖 → `cudaErrorInvalidDevice`; 장치가 둘 이상이면 다른 장치 → `cudaErrorPeerAccessUnsupported`. 모든 실패는 `cudaGetLastError`에 남음. `cudaDeviceDisablePeerAccess`는 `cudaErrorPeerAccessNotEnabled`.
15. `cudaLaunchCooperativeKernel`(`void*`판과 함수 포인터판 모두)은 `cudaErrorNotSupported`를 돌려주고 커널을 실행하지 않음(출력 버퍼가 그대로).

### 장치 쪽 내장 함수 (CUDA Math API, SIMD / Integer / Type casting intrinsics)
16. `__vsub4`: 바이트별 mod 256 뺄셈, 바이트 사이 빌림 없음(예: 0x00000000 - 0x00000001 = 0x000000ff, 0x00000100 - 0x00000001 = 0x000001ff). 임의 입력을 바이트 단위 기준과 비교.
17. `__vcmpne4`: 다른 바이트는 0xff, 같은 바이트는 0x00. 모두 같으면 0, 모두 다르면 0xffffffff.
18. `__byte_perm(x, y, s)`: 8바이트 {y,x}에서 선택자 4니블의 하위 3비트로 고름. 예: `(0x03020100, 0x07060504, 0x5432)` = 0x05040302. 니블의 최상위 비트와 `s`의 상위 16비트는 무시(부호 복제 없음: `0x0000fedc`는 `0x00006543`과 같은 결과). CPU와 NVIDIA 경로가 모든 선택자에서 일치. (2026-11-13 수정, 아래 이의 참고)
19. `__umulhi`: 상위 32비트(`0xffffffff * 0x80000000` → `0x7fffffff`). `__mulhi`는 부호 있는 상위 32비트.
20. `__float2int_rn`(2.5→2, 3.5→4, -2.5→-2), `_rz`(-2.7→-2), `_rd`(-2.1→-3), `_ru`(2.1→3); NaN → 0; ±1e20 → INT_MAX/INT_MIN(포화). double판도 같은 규칙. CPU와 NVIDIA 경로가 같은 값.
21. `__isnanf`/`__isinff`: NaN·무한대에서 1, 그 밖은 0. 커널 안에서 동작.
22. `__ldg`: 스칼라·`float2`·`int4`·`__half2`에서 `*p`와 같은 값.
23. `__syncthreads_count/and/or`(CUDA C++ Programming Guide, Synchronization Functions): 블록 크기 1·32·33·64·128·1024(1차원)와 2차원 블록에서, 모든 스레드가 같은 값을 받음. 술어가 전부 거짓 / 일부 참 / 전부 참 각각에서 count = 참 개수, and = (전부 참), or = (하나라도 참). 한 블록에서 연속으로 여러 번 불러도 앞 호출 값이 섞이지 않음. 블록이 여러 개이고 블록마다 술어가 다를 때 블록별로 독립.
24. `make_half2(x, y)`: `.x`가 하위, 정수·float 인자가 `__half`로 변환됨.
25. `__trap` 시험(별도 프로세스에서 실행하고 종료 상태를 봄): CPU 경로 — 프로세스가 비정상 종료(abort)하고 stderr에 `__trap()`과 블록·스레드 위치가 찍힘, 그 앞의 커널 `printf` 출력이 사라지지 않음. NVIDIA 경로 — 이후 동기화가 오류를 돌려줌(`cudaErrorLaunchFailure`). 커널이 trap 뒤의 쓰기를 하지 않음.
26. NO_DEVICE_CODE 패턴(`__CUDA_ARCH__` 정의 시 `printf` 후 `__trap()`을 부르는 `[[noreturn]]` 함수): NVIDIA 경로에서 `cudaDeviceSynchronize`가 오류, CPU 경로에서 `-D__CUDA_ARCH__=500`을 주면 프로세스 중단. (`__CUDA_ARCH__` 없이 CPU 경로에서 매크로가 빈 것은 알려진 한계이며 시험 대상이 아님.)

### cooperative_groups (CUDA C++ Programming Guide, Cooperative Groups)
27. `this_thread_block()`: `size()`, `thread_rank()`(x + y·dx + z·dx·dy), `group_index()`, `thread_index()`, `sync()`가 `__syncthreads`처럼 동작.
28. `tiled_partition<N>`(N = 1, 2, 4, 8, 16, 32): `size()` = N, `thread_rank()` = 블록 안 선형 번호 % N. `shfl(v, src)`는 타일 안의 순위 기준(타일 둘 이상일 때 다른 타일 값을 읽지 않음), `shfl_up/down`은 타일 경계에서 자기 값 유지, `shfl_xor`, `ballot`은 타일 순위로 번호 붙은 비트, `any`·`all`. `float`·`int`·`double`·`__half2`에서 동작. 블록 크기 32가 아니거나 64일 때도.
29. `cg::reduce(tile, v, cg::plus<int>())` 등: 모든 순위가 같은 값을 받음. `plus`·`less`·`greater`(최솟값·최댓값)·`bit_and`·`bit_or`·`bit_xor`. float 합은 순서 차이를 감안한 오차 안.
30. `this_grid()`: `is_valid()`는 false. `sync()`는 메시지를 찍고 trap(CPU: 프로세스 중단, NVIDIA: 실행 실패). `size()`·`thread_rank()` 값.
31. 컴파일 오류가 나야 하는 것(컴파일 실패 시험): `thread_block_tile<3>`, `thread_block_tile<64>`, `cg::reduce(this_thread_block(), ...)`, `tiled_partition<16>(tile8)`.

### 기존 시험 유지
32. 전체 ctest가 이 변경 전과 같은 수로 통과(`rt_*`의 CPU·GPU 판 포함). `cudaGetDeviceProperties` 정책 변경(8.0 → 5.0)으로 깨지는 기존 시험이 없음.

## 이의

(없음)

- 2026-11-13 (llama.cpp 적용 세션): 18번의 "니블 최상위 비트면 부호 복제"는 PTX `prmt` 기본 모드의 동작이고, 공개 문서의 `__byte_perm`은 니블마다 3비트 색인만 쓴다. llama.cpp `get_int_from_table_16`(IQ4_NL·IQ4_XS·MXFP4·NVFP4·Q1_0·Q2_0의 16칸 표 찾기)은 4비트 색인을 그대로 넘기며 넷째 비트가 무시되기를 기대한다. 부호 복제 구현으로 test-backend-ops의 해당 형식 행렬곱이 두 경로 모두에서 실패해, 두 경로 모두 3비트 색인으로 고쳤다(NVPTX는 `prmt`에 `s & 0x7777`). 18번 시험은 이 수정된 문장대로 써 주기 바란다.

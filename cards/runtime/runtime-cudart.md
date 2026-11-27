---
id: runtime-cudart
kind: component
name: Runtime API (CUDA-named, first subset)
spec_sources:
  - public CUDA Runtime API documentation (function names, parameters, documented behaviour)
  - ARCHITECTURE.md, section 5 (runtime layer)
backend_map:
  cpu: runtime/cudart/ over backends/cpu/
state: cpu-pass
owner_spec: AI Lab 2026-11-12
owner_tests: AI Lab 2026-11-12 (same author as impl: see note)
owner_impl: AI Lab 2026-11-12
cost: { hours: 5.0, ai_usage: 0 }
---

## 동작

보통의 CUDA 프로그램이 이름을 고치지 않고 CPU 경로에서 빌드·실행되게 하는 실행 API 첫 묶음.
헤더(`runtime/include/cuda_runtime_api.h`, `cuda_runtime.h`, `cuda.h`)는 공개 문서만 보고 새로 썼다.

- 장치: 개수·현재 장치·속성·동기화·초기화, 버전(12.3으로 보고).
- 오류: 스레드별 마지막 오류. `cudaGetLastError`는 읽고 지움, `cudaPeekAtLastError`는 남김. "아직 안 끝남"은 기록하지 않음.
- 메모리: 할당·해제·고정 메모리·남은 양, 1차원·2차원 복사와 채우기(동기·비동기), 기호 복사(`__constant__`·`__device__`).
  `cudaMemcpyDefault`는 런타임이 할당한 포인터로 방향을 정한다(시작 주소 기준, 범위 검사는 그대로).
- 포인터 종류 조회(`cudaPointerGetAttributes`: 장치·고정 호스트·통합·일반), 통합 메모리(`cudaMallocManaged`, 호스트와 메모리를 공유하는 장치에서만. 아니면 "지원 안 함"), 장치 속성 개별 조회(`cudaDeviceGetAttribute`, 자주 쓰는 27개. 모르는 값은 오류).
- 스트림: 막는 스트림·안 막는 스트림, 동기화·조회, 사건 기다리기.
  예전 기본 스트림 규칙: 기본 스트림 작업은 막는 스트림을 먼저 끝내고, 막는 스트림 작업은 기본 스트림을 먼저 끝낸다.
- 사건: 만들기(시간 재기 끄기 포함)·기록·동기화·조회·경과 시간.
- 실행: `K<<<g, b, shm, s>>>(...)`. 크기 한도를 넘으면 `cudaErrorInvalidConfiguration`.
- `cudaLaunchKernel(func, grid, block, args, shm, stream)`: CPU 경로에서는 커널의 함수 형에서 매개변수 형을 알아 인자를 제 형으로 복사한다(호출 시점 값). 주소만 있는 `const void *` 커널은 아직 실행 못 함(주소→커널 표 없음, 선언하지 않아 컴파일 오류).
- 장치 쪽: 색인 변수(`threadIdx`·`blockIdx`는 `uint3`, `blockDim`·`gridDim`은 `dim3`, `warpSize`는 `int`. 매크로가 아닌 진짜 변수라 같은 이름의 지역 변수를 써도 된다), 원자 연산 묶음(값 인자는 주소 형으로 변환), 울타리, 셔플·투표·`__syncwarp`, 자주 쓰는 수학 함수(형이 섞인 `min`·`max` 포함).
- 벡터 형: `char1`~`double4`와 `make_*`, 문서에 적힌 정렬.
- 추가: 피치 할당·2차원 채우기, 비동기 기호 복사(돌아오기 전에 끝남), 호스트 할당 플래그, 캐시 설정 힌트(값은 확인·기록, 효과 없음), 공개 문서의 오류 값 목록 전체와 이름·설명.
- 컴파일 경로: `.cu`는 전처리 뒤 바꾸기 때문에 포함한 파일·매크로 안의 실행 구문도 처리. 띄어 쓴 `<< <`·`>> >`, 인자의 리터럴 `0`·`NULL`도 처리.

- 고정 안 된(일반) 호스트 메모리와의 비동기 복사는 돌아오기 전에 끝난다(문서 동작: 원본을 그 자리에서 읽고, 장치→호스트는 돌아올 때 완료). `cudaMallocHost`·`cudaHostAlloc` 메모리만 나중에 읽힌다.
- 커널 실패는 장치에 남는다: 이후의 동기화·제출 호출이 같은 오류를 내고, `cudaDeviceReset`이 지운다.
- 기본 스트림·막는 스트림에 넣는 일은 순서 확인부터 대기열에 넣기까지 잠금을 쥔다(여러 호스트 스레드에서도 순서 규칙 유지).

- 스트림 더하기: 우선순위(한 단계뿐이라 범위 밖 값은 그 단계로 맞춤)·플래그 조회, 캡처 상태 조회(늘 "캡처 안 함"), 캡처 시작은 "지원 안 함"(그래프 미구현), 호스트 함수(앞선 작업이 끝난 뒤 실행하고 돌아옴).
- 호스트 메모리 등록·해제·장치 주소(호스트와 메모리를 공유하는 장치에서만), 장치 플래그(값 확인, 힌트), 피어 접근 조회(지금은 늘 불가).
- 함수 속성: 동적 공유 메모리 상한은 장치의 opt-in 한도까지 받음, 조회는 장치 한도로 답함. 점유율: 블록 크기·동적 공유 메모리 대 장치 한도.

- 반정밀도(`cuda_fp16.h`): `__half`·`__half2`, 변환(가장 가까운 짝수·0쪽·아래·위 반올림, 정수 변환은 넘치면 포화·NaN은 0), 사칙·fma(넓은 형에서 계산해 한 번 반올림 = 정확한 반올림), 순서·비순서 비교, 연산자, 수학 함수(float로 계산 후 반올림), `__half2` 묶음 연산.
  - 16비트 부동소수 원자 덧셈: `atomicAdd(__half *, __half)`, `atomicAdd(__half2 *, __half2)`, `atomicAdd(__nv_bfloat16 *, ...)`, `atomicAdd(__nv_bfloat162 *, ...)`. 옛 값을 돌려주고, 합은 그 형의 `__hadd`(가장 가까운 짝수로 한 번 반올림). 두 원소 형은 원소마다 원자적(32비트 한 번이 아님). 16비트 값은 32비트 워드의 어느 쪽에 있어도 된다. 헤더 순서(`cuda_fp16.h`·`cuda_bf16.h`와 `cuda_runtime.h`)에 상관없이 선언된다(`ocu_half_atomics.h`).

- 작은 추가 (실제 프로그램 조사 뒤, 11/13; 응용 결과는 카드 `apps-app-corpus`):
  - 3차원 메모리: `cudaMalloc3D`, `cudaMemset3D`·`Async`, `cudaMemcpy3D`·`Async`(피치 있는 원본·대상, `srcPos`·`dstPos`, `ysize`; 배열이 얽히면 `cudaErrorNotSupported`), `cudaMemset2DAsync`, 형식 `cudaExtent`·`cudaPos`·`cudaPitchedPtr`·`cudaMemcpy3DParms`와 `make_cudaExtent` 등.
  - `cudaDeviceGetLimit`·`cudaDeviceSetLimit`: 스택 크기·printf FIFO·말록 힙은 기록해 돌려줌(효과 없음, 0은 `cudaErrorInvalidValue`). 장치 런타임 한도 둘과 L2 상주 한도는 `cudaErrorUnsupportedLimit`. L2 가져오기 단위는 힌트라 0.
  - `cuda_profiler_api.h`의 `cudaProfilerStart`·`cudaProfilerStop`: 붙은 도구가 없으므로 효과 없이 `cudaSuccess`.
  - 진단: 환경변수 `OCU_MALLOC_FILL=B`(0~255, 10진수나 0x..)이면 새 장치 할당(`cudaMalloc` 계열)을 모두 바이트 B로 채운다. 아무도 쓰지 않은 메모리를 읽는 곳을 찾는 용도(0xff면 f32·f16·bf16 값이 NaN). 없거나 바이트 값이 아니면 할당을 건드리지 않는다. 값은 프로세스당 한 번 읽는다.
  - `math_constants.h`: `CUDART_*` 단·배정밀도 상수와 `_HI`/`_LO`.
  - 장치 쪽: 캐시 힌트 적재·저장 9개(`__ldcg` 등), 범위 지정 원자(`atomicAdd_block`/`_system` 등; NVIDIA 경로의 `_system`은 컴파일 때 지원 안 함), 무동기 워프 함수(`__shfl` 등)는 컴파일 때 지원 안 함과 대체 함수 안내. Cooperative Groups 타일의 `meta_group_rank`·`meta_group_size`.

## 경계 조건

- 선언하지 않은 API는 컴파일 오류로 드러난다(실행 중에 조용히 실패하지 않음).
- 기호 복사는 통합 메모리를 지원하는 장치에서만 된다. 아니면 "지원 안 함".
- fork한 자식에서 부모의 스트림·사건은 "지원 안 함".

## 시험 목록

`tests/runtime/rt_api.cu`: 보통의 CUDA 프로그램 하나(ocu 이름 없음). 기본 모드와 무작위 순서·스레드 3개 모드로 실행.

- [x] 장치·오류 읽기와 지우기·버전·오류 이름
- [x] 할당·복사(H2D, D2H, D2D, H2H, 2차원)·채우기·범위 넘는 복사 오류·남의 포인터 해제 오류
- [x] 기호 복사, 원자 연산 합, 실행 설정 오류, 동적 공유 메모리
- [x] 스트림 순서·조회, 안 막는 스트림 독립, 사건 시간, 예전 기본 스트림 순서, 사건 기다리기
- [x] `tests/probe/main_saxpy.cu`: 프로브 커널을 실행 API로 돌림
- [x] `tests/runtime/rt_lang.cu`(무작위 순서·스레드 3개·워프 64 모드 포함): 벡터 형 크기·정렬과 연산, 내장 변수 형과 이름 가리기, 리터럴 0·NULL 인자, 형 섞인 min·max, 원자 연산 변환, 포함 파일·매크로·띄어 쓴 실행 구문, 피치 할당·2차원 채우기, 스트림 위 기호 복사, 캐시 힌트, 오류 이름
- [x] `tests/runtime/rt_apps_gaps.cu`(스모크 49개 점검): 캐시 힌트, 범위 지정 원자, 상수, 3D 할당·채우기·복사 왕복, 한도. 완전한 시험은 다른 세션이 쓴다(항목 목록은 카드 `apps-app-corpus`의 시험 목록)
- [x] 독립 시험(다른 세션, 11/13): `rt_apps_consts.cu`(math_constants 정확한 반올림 기준 116), `rt_apps_mem3d.cu`(3D 할당·채우기·복사 무작위 500건과 오류, 한도, 프로파일러), `rt_apps_dev.cu`(캐시 힌트 29형, 범위 지정 원자 순차·병렬, 타일 메타 그룹), `rt_apps_align.cu`, `apps_compile_cases.cu`(무동기 워프 함수 대체 이름 안내, 시스템 범위 원자, 구조체 뒤 정렬). 확인된 결함 2건은 기대 실패로 표시: CPU 경로에서 `typedef struct {..} __align__(8) S;`와 `struct S {..} __align__(8) g;`의 정렬이 빠짐, 3D 복사·채우기가 범위 초과 오류를 돌려주면서 앞 조각을 이미 씀
- [x] 외부: Rodinia 18개 중 17개 빌드, 8개 독립 기준과 일치(`conformance/external/rodinia/run.sh`)

- [x] `tests/runtime/rt_mem.cu`: 포인터 종류(할당 안쪽 주소, 해제 뒤), `cudaMemcpyDefault` 네 방향과 범위 초과 오류, 통합 메모리를 호스트·커널이 함께 씀, 장치 속성과 속성 구조체 일치·잘못된 값
- [x] `tests/runtime/rt_misc.cu`: 우선순위·플래그·캡처 상태, 호스트 함수 순서, 호스트 메모리 등록과 그 메모리에서 커널 실행·중복 등록·해제 오류, 장치 플래그·피어, 함수 속성 상한, 점유율
- [x] `tests/runtime/rt_fp16.cu`: 구현과 독립인 정확한 기준(모든 유한 half를 2^-24 단위 정수로, 합·곱은 128비트 정수로 계산하고 정렬한 전체 표에서 가장 가까운 값). 65,536개 왕복, 이웃 사이 중간값 6만여 개와 그 바로 위아래, 넘침 경계·특수값, 20만 개 무작위 조합의 덧셈·뺄셈·곱셈·fma와 5만 개 나눗셈, 커널 안 `__half2` fma. 결함 두 가지(동률 올림, 자리올림 경계)를 넣으면 실패함을 확인
- [x] `tests/runtime/rt_half_atomics.cu`: 16비트 카운터 4개(32비트 워드 두 개의 양쪽)에 스레드마다 1을 더해 합이 정확하고 돌려받은 옛 값이 0..N-1의 순열(half 512개, bf16 256개씩), `__half2`·`__nv_bfloat162` 두 원소가 따로 쌓임, 한 번 더하기의 가장 가까운 짝수 반올림(동률 4건, 넘침 경계 아래, 0). CPU 경로에서 통과
- [x] `tests/runtime/rt_malloc_fill.cu`: `OCU_MALLOC_FILL` 없음·0xff·7·잘못된 값("x", "300")에서 1바이트·4 KiB·3 MiB 할당이 채워지거나(값 일치) 건드려지지 않음(쓴 값 그대로 읽힘). CPU 경로와 OpenCL(Arc Pro B70)에서 통과
- [x] `tests/runtime/rt_launch_api.cu`: 구조체 인자 포함 실행, 호출 뒤 값을 바꿔도 영향 없음, 동적 공유 메모리, 설정 오류와 인자 없음 오류
- [x] `tests/runtime/rt_review.cu`(다섯째 검토 회귀): 바쁜 스트림에 일반 메모리 비동기 복사, 템플릿 커널의 리터럴 0(형 추론 int), `<<<g, b, 0, NULL>>>`, 동적 공유 메모리 바이트 배열의 형 변환, `operator<< <>`, 커널 실패 유지와 초기화

## 다섯째 검토 (11/12, 다른 세션)

확인된 버그 5개 모두 수정: 일반 메모리 비동기 복사가 늦게 읽힘(조용히 틀린 결과), 템플릿 커널 리터럴 0의 형, 설정의 NULL, 동적 공유 메모리 형 변환, `operator<< <>` 오인. 의심 3개 중 2개 수정(여러 스레드의 기본 스트림 순서, 커널 실패가 한 번 읽히면 사라짐). 남은 1개: 내장 변수를 C++에서 `const`로 선언(값은 각 GPU 스레드 안에서 변하지 않음. 블록의 파이버가 OS 스레드를 옮기지 않는다는 설계에 기대므로 그 보장을 유지해야 함).

## 여섯째 검토 (11/12 밤, 다른 세션)

재현 4건 수정: 호스트 함수 안에서 실행 API를 부르면 멈춤(이제 `cudaErrorNotPermitted`), `h = 0;` 같은 `__half` 정수 대입이 컴파일 안 됨, 등록 범위를 감싸는 `cudaHostRegister`를 받아들임(이제 어떤 겹침도 거부, 주소 넘침도 거부), 크기 0 할당이 기록에 쌓임. 의심 중 2건 수정: `cudaFree`가 기록을 해제 뒤에 지워 생기는 경합(이제 먼저 지우고 실패하면 되돌림), 점유율을 워프 단위로 계산. 남김: `cudaDeviceReset`이 할당을 풀지 않음(프로그램이 쥔 포인터를 지키는 쪽을 택함), NaN 비트 모양(0x7e00), `cudaFuncGetAttributes`가 설정값을 반영하지 않음. 호스트 함수는 호출한 스레드에서 돌고 끝나야 돌아온다(콜백이 호출 뒤의 호스트 신호를 기다리면 멈춤): 문서에 적힌 제약.
- [x] `tests/runtime/rt_review2.cu`: 위 수정의 회귀 시험

## 메모

시험과 구현을 같은 세션이 썼다. 다른 세션이 시험을 다시 검토해야 한다.

## 이의

(없음)

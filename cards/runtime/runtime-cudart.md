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
cost: { hours: 3.5, ai_usage: 0 }
---

## 동작

보통의 CUDA 프로그램이 이름을 고치지 않고 CPU 경로에서 빌드·실행되게 하는 실행 API 첫 묶음.
헤더(`runtime/include/cuda_runtime_api.h`, `cuda_runtime.h`, `cuda.h`)는 공개 문서만 보고 새로 썼다.

- 장치: 개수·현재 장치·속성·동기화·초기화, 버전(12.3으로 보고).
- 오류: 스레드별 마지막 오류. `cudaGetLastError`는 읽고 지움, `cudaPeekAtLastError`는 남김. "아직 안 끝남"은 기록하지 않음.
- 메모리: 할당·해제·고정 메모리·남은 양, 1차원·2차원 복사와 채우기(동기·비동기), 기호 복사(`__constant__`·`__device__`).
  `cudaMemcpyDefault`는 "지원 안 함".
- 스트림: 막는 스트림·안 막는 스트림, 동기화·조회, 사건 기다리기.
  예전 기본 스트림 규칙: 기본 스트림 작업은 막는 스트림을 먼저 끝내고, 막는 스트림 작업은 기본 스트림을 먼저 끝낸다.
- 사건: 만들기(시간 재기 끄기 포함)·기록·동기화·조회·경과 시간.
- 실행: `K<<<g, b, shm, s>>>(...)`. 크기 한도를 넘으면 `cudaErrorInvalidConfiguration`.
- 장치 쪽: 색인 변수(`threadIdx`·`blockIdx`는 `uint3`, `blockDim`·`gridDim`은 `dim3`, `warpSize`는 `int`. 매크로가 아닌 진짜 변수라 같은 이름의 지역 변수를 써도 된다), 원자 연산 묶음(값 인자는 주소 형으로 변환), 울타리, 셔플·투표·`__syncwarp`, 자주 쓰는 수학 함수(형이 섞인 `min`·`max` 포함).
- 벡터 형: `char1`~`double4`와 `make_*`, 문서에 적힌 정렬.
- 추가: 피치 할당·2차원 채우기, 비동기 기호 복사(돌아오기 전에 끝남), 호스트 할당 플래그, 캐시 설정 힌트(값은 확인·기록, 효과 없음), 공개 문서의 오류 값 목록 전체와 이름·설명.
- 컴파일 경로: `.cu`는 전처리 뒤 바꾸기 때문에 포함한 파일·매크로 안의 실행 구문도 처리. 띄어 쓴 `<< <`·`>> >`, 인자의 리터럴 `0`·`NULL`도 처리.

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
- [x] 외부: Rodinia 18개 중 17개 빌드, 8개 독립 기준과 일치(`conformance/external/rodinia/run.sh`)

## 메모

시험과 구현을 같은 세션이 썼다. 다른 세션이 시험을 다시 검토해야 한다.

## 이의

(없음)

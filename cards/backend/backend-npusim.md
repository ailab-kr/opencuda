---
id: backend-npusim
kind: component
name: NPU constraint simulator backend (npusim)
spec_sources:
  - docs/spec/op-table.md section 7 (the rules it enforces) and sections 3-6
  - backends/interface/ocu_backend.h, backends/interface/ocu_ops.h
backend_map:
  npusim: backends/npusim/npusim.c (compute: backends/ops_ref)
state: tests
owner_spec: AI Lab 2026-11-13
owner_tests: AI Lab, separate author 2026-11-13 (spec and headers only)
owner_impl: AI Lab 2026-11-13
cost: { hours: 1.5, ai_usage: 0 }
---

## 동작

SIMT 스레드가 없는 가속기처럼 행동하는 호스트 백엔드다. 실제 NPU 없이 상위 층이 그 제약을 견디는지 확인한다. 속도 흉내는 하지 않는다.

- 커널 없음: `module_load`, `module_get_kernel`, `kernel_launch`는 "지원 안 함". 거부한 실행 수를 센다. `dev_info`는 묶음 1, 블록당 스레드 1, 공유 메모리 = 온칩 메모리 크기, 통합 메모리 없음.
- 따로 떨어진 메모리: 장치 메모리는 접근 금지로 잡아 둔 주소 범위이고, 내용은 이 백엔드만 아는 별도 매핑에 있다. 호스트가 장치 포인터를 읽으면 프로그램이 죽는다. 복사는 방향을 검사한다(호스트 쪽에 장치 주소, 장치 쪽에 호스트 주소나 범위 밖이면 오류). 할당 사이에 256바이트 틈을 둔다.
- 연산 표만 계산: 실행은 계획과 같은 모양(또는 `VAR_ROWS`로 더 적은 행), 장치 메모리, 64바이트 정렬이어야 한다.
- 온칩 메모리 한도(`OCU_NPUSIM_SCRATCH_KB`, 기본 256): GEMM은 128~16 크기의 정육면체 타일로 나눈다(`OCU_NPUSIM_TILING=0`이면 통째로 들어가야 함). RMS_NORM·SOFTMAX는 한 행이 들어가야 한다. 안 되면 "지원 안 함".
- 설정: 입력 자료형 제한(`OCU_NPUSIM_DTYPES`), 계획 지연(`OCU_NPUSIM_PLAN_MS`), 메모리 크기(`OCU_NPUSIM_MEM_MB`). 집계(`stats`): 연산 호출, 계획 성공·거부, 거부한 커널 실행, 방향별 DMA 바이트. `OCU_NPUSIM_REPORT=1`이면 장치를 닫을 때 표준 오류로 출력.
- 모든 일을 잠금 하나 아래 동기로 한다(엔진 하나). 대기열 순서는 저절로 지켜진다.
- 등록 순서가 마지막이라 기본 백엔드가 되지 않는다. `OCU_BACKEND=npusim`으로 고른다.

## 경계 조건

- 크기 0 할당은 1바이트로 잡는다. 할당 해제는 할당 시작 주소만 받는다.
- 메모리 매핑을 못 잡으면 장치 0개로 보고한다.

## 시험 목록

- [x] `npusim_profile`, `_scratch4`, `_scratch1`, `_tiling0`, `_dtypes_f16`, `_env`: 규격 7장 규칙
- [x] `ops_npusim`: 연산 표 적합성(CPU와 같은 시험)
- [x] MinGW-w64 교차 빌드(라이브러리만; Windows 실행은 안 해 봄)
- [ ] 시험 코드의 독립 검토

npusim이 잡아낸 제약 위반의 예(시험 출력에서): 호스트 포인터를 실행에 줌 → `INVALID_VALUE`; 계획보다 큰 행 수로 실행 → `INVALID_VALUE`; 행 10만 개짜리 softmax를 온칩 256KB에 계획 → `NOT_SUPPORTED`(계획 거부 집계 +1); 커널 실행 → `NOT_SUPPORTED`(거부 집계 +1).

## 메모

- 계산은 CPU 기준 구현(`backends/ops_ref`)을 별도 매핑 위에서 돌린다. 타일은 크기 판정과 보고에만 쓰고, 실제 계산 순서는 타일과 무관하다.
- 섞어 쓰기(지원 안 하는 커널을 CPU로 보내기)는 아직 없다. 지금은 "지원 안 함"을 그대로 알린다.
- 실제 NPU의 정렬·자료형·타일 규칙은 칩마다 다르다. 지금 값은 대표적인 모양을 흉내 낸 것이다.

## 이의

(없음)

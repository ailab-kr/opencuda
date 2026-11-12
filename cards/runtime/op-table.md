---
id: runtime-op-table
kind: component
name: Operator table (ocu_ops), version 1
spec_sources:
  - docs/spec/op-table.md (the contract)
  - backends/interface/ocu_ops.h
  - IEEE 754-2019 (binary16, rounding); bfloat16 as the upper half of binary32
backend_map:
  cpu: backends/cpu/cpu_ops.c (computes with backends/ops_ref)
  npusim: backends/npusim/npusim.c
  amd: (none)
state: tests
tolerance: docs/spec/op-table.md section 6 (classes in the spec, not in conformance/tolerances yet)
owner_spec: AI Lab 2026-11-13
owner_tests: AI Lab, separate author 2026-11-13 (spec and headers only)
owner_impl: AI Lab 2026-11-13 (same as spec)
cost: { hours: 3.5, ai_usage: 0 }
---

## 동작

커널을 못 돌리는 칩(NPU)도 쓸 수 있게, 백엔드가 라이브러리 수준의 연산을 내놓는 두 번째 길이다. `get_provider("ops")`로 표를 내놓으므로 `ocu_backend`와 ABI 번호는 그대로다.

- 순서: `support`(연산 × 자료형, 지원표용) → `query`(설명자 하나, 정확한 답) → `plan`(모양 고정, 컴파일 가능, 실패 가능) → `execute`(대기열 순서) → `destroy`. `stats`는 선택.
- 텐서: 장치 포인터 + 자료형 + 모양 + 보폭(원소 단위, 가장 안쪽 보폭 1).
- 연산: GEMM(전치, alpha·beta, 3차원 배치와 배치 1 방송, f16/bf16 입력에 f32 출력), ADD, MUL(방송), RMS_NORM(가중치 선택), SOFTMAX(scale). ROPE는 규격만, QMATMUL과 블록 양자화 자료형은 자리만.
- 행 수 가변(`OCU_OP_FLAG_VAR_ROWS`): 가장 큰 행 수로 한 번 계획하고 실행 때 더 적은 행을 준다. 모델 작업은 가중치 모양 `(K, N)` 가짓수는 적고 행 수(M)만 바뀌므로 계획 수를 줄인다.
- 허용 오차 등급: EXACT(ADD·MUL, 한 번 반올림), ULP(k)(ROPE), ROW(RMS_NORM·SOFTMAX, 행 길이에 비례), DOT(GEMM, `(K+4)·u_acc·(|α|Σ|a||b| + |β||c|)`). 누적 순서와 융합 곱셈-덧셈을 가리지 않고, 백엔드가 밝힌 누적 자료형으로 채점한다.

## 경계 조건

- 크기 0인 차원은 기술하지 않는다(호출하는 쪽이 건너뜀).
- GEMM에서 beta 0이면 C를 읽지 않고, alpha 0이면 A·B를 읽지 않는다.
- 제자리 실행은 원소별·행 연산의 첫 입력과 같은 포인터·같은 설명자일 때만. 그 밖의 겹침은 오류.

## 오류 코드

- 잘못된 설명자·텐서, 계획과 다른 모양, 장치 메모리가 아닌 포인터, 할당을 벗어난 범위: `INVALID_VALUE`.
- 없거나 해제된 계획: `INVALID_HANDLE`.
- 정의되지 않은 자료형, 예약 연산, 칩 한도(npusim의 온칩 메모리 등): `NOT_SUPPORTED`. 아직 안 만든 것(ROPE): `NOT_IMPLEMENTED`.

## 시험 목록

`tests/ops/` (규격과 헤더만 본 별도 세션이 작성, 구현 파일은 열지 않음):

- [x] `ops_cpu`, `ops_npusim`: 수치(각 1만 1천~1만 2천여 확인 항목), 오류 코드, query·plan·support 일치, 대기열 순서
- [x] `npusim_profile*`: npusim 규칙(아래 카드), 환경 6가지
- [x] `op_support_table`: 백엔드별 지원표(마크다운)
- 첫 실행에서 구현 결함 5건 발견 → 고침(e4d27cd). 규격의 모호한 곳 7건 → 규격에 못 박음(b332a75).
- [ ] 시험 코드의 독립 검토 (아직 안 함)

## 메모

- `libs/blas`와 잇는 법(규격 8장): `ocu_ops_of(backend)`로 표를 얻고, `query`가 OK면 연산 길로, 아니면 기존 BLAS 제공자·커널 길로 간다. 열 우선 GEMM은 복사 없이 슬롯을 바꿔(B 버퍼를 A 자리에, A 버퍼를 B 자리에) 행 우선 표에 넣는다. 계획은 설명자 전체를 열쇠로 저장하고, M이 바뀌면 `VAR_ROWS`를 쓴다.
- 같은 세션이 규격과 구현을 썼다. 시험은 다른 세션이 규격만 보고 썼다.
- ROPE 기준 구현, 정수 GEMM, 블록 양자화 행렬곱은 다음 판.

## 이의

- (구현 쪽 의견, 검토자 판단 필요) `test_ops.c`의 "span leaves the allocation" 두 번째 사례는 할당 바로 끝(`dy + 256`)을 넘긴다. 할당이 붙어 있는 백엔드에서는 그 주소가 다음 할당의 시작일 수 있어 거부할 수 없다(규격 5.4 괄호). CPU 백엔드는 스레드 검사 빌드에서 이 사례가 실패했다(할당이 붙어 나옴). 구현 쪽에서 CPU 할당 끝에 추적하지 않는 256바이트를 덧붙여 지금은 두 백엔드 모두 거부하지만, 할당 사이 틈을 보장하지 않는 백엔드에서는 이 사례가 맞는 요구인지 검토가 필요하다.

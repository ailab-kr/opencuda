# 적합성 시험

어떤 칩과 어떤 호환 방식이 CUDA 기능을 얼마나 정확하고 빠르게 처리하는지 재는 시험이다.

현재 상태: 채점 규칙과 결과 기록 검사만 있다. 시험 묶음과 실행기는 아직 없다.

## 폴더

| 폴더 | 내용 | 상태 |
|---|---|---|
| `harness/ocu_conf/` | 채점 규칙, 결과 기록 검사 | 있음 |
| `harness/tests/` | 채점기 자체의 시험 | 있음 |
| `suites/L1_build/` | 소스가 수정 없이 컴파일되는가 | 없음 |
| `suites/L2_runtime/` | 실행 API가 문서대로 동작하는가 | 없음 |
| `suites/L3_language/` | GPU 코드의 의미가 같은가 | 없음 |
| `suites/L4_libs/` | 라이브러리의 수치 결과가 허용 오차 안인가 | 없음 |
| `suites/L5_apps/` | 실제 프로그램이 같은 결과를 내는가 | 없음 |
| `oracle/` | 수학적 기준 답 생성기 | 없음 |
| `tolerances/` | 연산별 허용 오차 | 없음 |
| `mutants/` | 일부러 오류를 넣은 구현(채점기 검증용) | 없음 |
| `results/` | 결과 기록(JSON). 저장소에 올리지 않음 | |

## 기준 답의 우선순위

1. 공개 문서에 적힌 동작
2. 수학적으로 정해지는 답(CPU에서 높은 정밀도로 계산)
3. 엔비디아 GPU에서 관찰한 결과(1, 2로 정해지지 않는 부분만, 사람 승인 뒤)

## 채점 규칙

| 대상 | 규칙 | 함수 |
|---|---|---|
| 정수, 제어 흐름 | 완전 일치 | `score.exact_match` |
| 실수 연산 | 최대 상대 오차가 허용 오차 이내 | `score.within_tolerance` |
| NaN, 무한대 | 양쪽이 같은 자리에 같은 값일 때만 일치 | `score.max_relative_error` |
| 속도 | 중앙값, 느린 10% 구간, 가장 느린 시험 이름 | `score.summarize` |

## 결과 기록 형식

실행 한 번에 파일 하나. 예시는 `harness/tests/data/example_result.json`.

| 항목 | 내용 |
|---|---|
| `schema` | 형식 번호. 지금은 1 |
| `run` | `date`, `suite_version`, `set`(`public` 또는 `private`) |
| `device` | `vendor`, `model`, `driver`, `os` |
| `impl` | `name`, `version`, `backend` |
| `tests[]` | 시험별 결과 |

시험별 결과

| 항목 | 필수 | 내용 |
|---|---|---|
| `id` | 예 | 파일 안에서 겹치지 않는 이름. 예: `L4.blas.saxpy.n1024_incx1` |
| `card` | 예 | 기능 카드 id |
| `level` | 예 | `L1`~`L5` |
| `status` | 예 | `pass`, `fail`, `skip`, `unsupported` |
| `reason` | `skip`일 때 | 건너뛴 이유 |
| `max_rel_err`, `tolerance` | 아니오 | 실수 연산의 오차와 허용 오차 |
| `time_ms_median` | 아니오 | 측정 시간의 중앙값 |
| `baseline_native_ms` | 아니오 | 같은 칩에서 칩 회사 전용 방식의 시간 |
| `speed_ratio` | 아니오 | 전용 방식 대비 속도 비율. 1이면 같음 |

`unsupported`는 그 칩이나 구현이 기능을 지원하지 않는다고 밝힌 경우다. `fail`과 구분한다.

## 실행

```sh
cd harness
python -m unittest discover -s tests
python -m ocu_conf validate <결과 파일> [...]
```

## 시험을 쓸 때 지킬 것

- 사용자와 같은 표면만 쓴다. 저장소 내부 헤더를 쓰지 않는다.
- 시험은 기능 카드만 보고 쓴다. 구현을 보고 쓰지 않는다.
- 입력을 넓힌다: 크기(0, 1, 큰 값, 2의 거듭제곱 근처), 증가폭(양수, 음수), 비정상 값(NaN, 무한대, 아주 작은 수).
- 시험이 구현의 오류를 실제로 잡는지 `mutants/`로 확인한다.

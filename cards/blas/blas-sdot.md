---
id: blas-sdot
kind: library-function
name: sdot (BLAS standard operation `dot`, element type f32)
spec_sources:
  - BLAS standard, netlib reference BLAS definition of SDOT (public)
backend_map:
  cpu: backends/cpu/cpu_blas.c (provider) via libs/blas
  amd: none yet
state: cpu-pass
tolerance: conformance/tolerances/blas.json
owner_spec: AI Lab 2026-11-12
owner_tests: AI Lab 2026-11-12
owner_impl: AI Lab 2026-11-12
cost: { hours: 0, ai_usage: 0 }
---

## 동작

BLAS 표준의 `dot` 연산. 기준 답은 `conformance/oracle/blas_ref.py`의 `dot`이 정확한 수학 값으로 계산한다.

## 시험 목록

- [x] 기준 답과 시험 사례: `conformance/oracle/cases.py` (카드 `blas-sdot`)
- [ ] CUDA 이름의 라이브러리 함수에 대한 사례 대응 (함수 이름·인자 형식은 공개 문서 기준)
- [x] CPU 백엔드에서 통과 (11/12, `conformance_blas_cpu`: 이 카드의 사례 전부, 동기·대기열·무작위 순서)
- [ ] AMD GPU에서 통과

## 메모

시험 사례의 의미는 BLAS 표준만 따른다. CUDA 라이브러리 쪽 차이(색인 기준, 결과를 돌려주는 방식, 오류 코드)는 이 기준 위에 덧붙이는 대응 층에서 다룬다.

---
id: blas-saxpy
kind: library-function
name: saxpy (BLAS standard operation `axpy`, element type f32)
spec_sources:
  - BLAS standard, netlib reference BLAS definition of SAXPY (public)
backend_map:
  cpu: none yet
  amd: none yet
state: tests
tolerance: conformance/tolerances/blas.json
owner_spec: AI Lab 2026-11-12
owner_tests: AI Lab 2026-11-12
owner_impl: (none)
cost: { hours: 0, ai_usage: 0 }
---

## 동작

BLAS 표준의 `axpy` 연산. 기준 답은 `conformance/oracle/blas_ref.py`의 `axpy`이 정확한 수학 값으로 계산한다.

## 시험 목록

- [x] 기준 답과 시험 사례: `conformance/oracle/cases.py` (카드 `blas-saxpy`)
- [ ] CUDA 이름의 라이브러리 함수에 대한 사례 대응 (함수 이름·인자 형식은 공개 문서 기준)
- [ ] CPU 백엔드에서 통과
- [ ] AMD GPU에서 통과

## 메모

시험 사례의 의미는 BLAS 표준만 따른다. CUDA 라이브러리 쪽 차이(색인 기준, 결과를 돌려주는 방식, 오류 코드)는 이 기준 위에 덧붙이는 대응 층에서 다룬다.

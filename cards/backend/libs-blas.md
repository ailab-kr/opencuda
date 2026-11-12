---
id: libs-blas
kind: component
name: BLAS layer (libs/blas) and the CPU provider
spec_sources:
  - BLAS standard, netlib reference BLAS (argument rules, quick returns)
  - ARCHITECTURE.md section 4, "라이브러리 제공자"
  - backends/interface/ocu_blas_provider.h
backend_map:
  cpu: backends/cpu/cpu_blas.c
  amd: (later: vendor BLAS behind the same provider table)
state: cpu-pass
owner_spec: AI Lab 2026-11-12
owner_tests: AI Lab 2026-11-12
owner_impl: AI Lab 2026-11-12
cost: { hours: 1, ai_usage: 0 }
---

## 동작

칩과 무관한 층(`libs/blas`)이 BLAS 표준의 인자 규칙과 빠른 반환을 맡고, 계산은 활성 백엔드의 제공자(`get_provider("blas")`)가 한다. 스칼라 결과(dot, nrm2, asum, iamax)는 층이 호스트로 가져온다. 이름은 내부 이름(`ocu_blas_*`)이며, CUDA 이름의 라이브러리는 법률 검토 뒤에 이 위에 얹는다.

## 시험 목록

- [x] 기준 사례 526건 전부 통과: 동기 실행, 대기열 + 무작위 순서 (`tests/conformance/blas_driver.c`, `conformance/harness/ocu_conf/blas_run.py`)
- [x] 일부러 망가뜨린 C 커널 3종(beta 0일 때 C를 읽음, iamax 0부터, 오른쪽 trsm의 연산 방향) 모두 검출
- [ ] 잘못된 인자(음수 크기, lda 부족, 증가폭 0)에 대한 오류 코드 시험

## 메모

- 증가폭 0은 오류로 다룬다. 참조 BLAS는 1단계 함수에서 0을 허용하지만, 병렬 커널에서 출력 증가폭 0은 경쟁 상태가 되므로 막았다. CUDA 쪽 동작은 공개 문서로 확인한다.
- CPU 제공자의 커널은 정확성 기준이며 속도용이 아니다.

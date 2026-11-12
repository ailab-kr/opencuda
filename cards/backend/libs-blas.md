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

칩과 무관한 층(`libs/blas`)이 BLAS 표준의 인자 규칙과 빠른 반환을 맡고, 계산은 활성 백엔드의 제공자(`get_provider("blas")`)가 한다. 스칼라 결과(dot, nrm2, asum, iamax)는 층이 호스트로 가져온다. 이름은 내부 이름(`ocu_blas_*`)이며, CUDA 이름의 라이브러리(libs/cublas)는 이 위에 얹는다.

## 시험 목록

- [x] 기준 사례 526건 전부 통과: 동기 실행, 대기열 + 무작위 순서 (`tests/conformance/blas_driver.c`, `conformance/harness/ocu_conf/blas_run.py`)
- [x] 두 번째 독립 검토(11/12): alpha 0일 때 A·B를 읽어 NaN, 넓은 행렬(열 1,048,560 초과) 실패, 드라이버의 대기열 오류 무시와 프로토콜 함정을 고침. alpha 0 사례 11건(총 537건), `test_blas_large`. 수정 전 커널은 새 사례 11건을 모두 틀림
- [x] 일부러 망가뜨린 C 커널 3종(beta 0일 때 C를 읽음, iamax 0부터, 오른쪽 trsm의 연산 방향) 모두 검출
- [x] 잘못된 인자(음수 크기, lda 부족, 증가폭 0, 잘못된 열거값)와 빠른 반환 시험 (`tests/backend/test_blas_args.c`, 45개 확인 항목)

## 메모

- isamax는 NaN을 건너뛴다(기준 답과 같음). 참조 BLAS는 첫 원소가 NaN이면 1을 돌려준다. CUDA 쪽 동작은 공개 문서로 확인한다.
- 증가폭 0은 오류로 다룬다. 참조 BLAS는 1단계 함수에서 0을 허용하지만, 병렬 커널에서 출력 증가폭 0은 경쟁 상태가 되므로 막았다. CUDA 쪽 동작은 공개 문서로 확인한다.
- CPU 제공자의 커널은 정확성 기준이며 속도용이 아니다.

---
id: <묶음>-<이름>            # 예: blas-saxpy, runtime-malloc
kind: library-function        # runtime-function | language-feature | library-function | app | component
name: <공개 문서의 이름>
spec_sources:
  - <공개 문서 주소와 절 이름>
backend_map:
  cpu: <자체 구현 | 대응 함수>
  amd: <대응 함수 | 공통 구현>
state: spec                   # spec → tests → cpu-pass → gpu-pass → perf-pass
tolerance: <conformance/tolerances/ 안의 파일, 실수 연산일 때>
owner_spec: <세션 또는 사람>
owner_tests: <세션 또는 사람>
owner_impl: <세션 또는 사람>
cost: { hours: 0, ai_usage: 0 }
---

## 동작

공개 문서의 내용을 자기 말로 요약한다. 문서를 그대로 옮겨 적지 않는다.

## 경계 조건

크기 0, 음수 증가폭, 널 포인터, 겹치는 메모리, 아주 큰 값 등.

## 오류 코드

어떤 입력에 어떤 오류가 나와야 하는지.

## 시험 목록

- [ ] <시험 id>: <무엇을 확인하는지>

## 메모

칩 사이의 차이, 연결 대상과 다른 점, 문서가 모호한 부분과 그때 택한 해석.

## 이의

시험이 틀렸다고 보이면 여기에 적는다. 구현 담당은 시험을 직접 고치지 않는다.

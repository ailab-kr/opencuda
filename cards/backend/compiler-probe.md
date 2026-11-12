---
id: compiler-probe
kind: component
name: Compiler probe (PLAN P0-W1-11)
spec_sources:
  - Clang documentation, "Compiling CUDA with clang" (public)
  - docs/compiler-probe.md
state: cpu-pass
owner_spec: AI Lab 2026-11-12
owner_tests: AI Lab 2026-11-12 (same author: see backend-cpu note)
owner_impl: AI Lab 2026-11-12
cost: { hours: 1.5, ai_usage: 0 }
---

## 동작

배포하지 않는 실현성 시험이다. 고치지 않은 `.cu` 파일이 엔비디아·AMD·CPU 세 경로로 빌드되고, CPU 경로에서는 실제로 정답을 내는지 확인한다.

## 시험 목록

- [x] `compiler/probe/run_probe.sh`: 두 `.cu` 파일을 엔비디아 sm_80, AMD gfx1100, gfx1201로 컴파일
- [x] `tests/probe/test_probe_cpu.cpp`: CPU 경로 전 구간 실행 6묶음, 고정·무작위 순서
- [x] `compiler/probe/test_cpu_rewrite.py`: 변환기 단위 시험 11개

## 메모

CUDA 언어 이름(`threadIdx`, `__syncthreads`, `atomicAdd`, `dim3`)을 쓰는 헤더가 들어 있다. 제품 헤더가 아니라 이 실험 안에만 둔다. 실행 API 이름(`cudaMalloc` 등)은 쓰지 않는다.

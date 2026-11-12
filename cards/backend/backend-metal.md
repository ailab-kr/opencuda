---
id: backend-metal
kind: component
name: Apple GPU (Metal) backend and CUDA-to-MSL translation
spec_sources:
  - backends/interface/ocu_backend.h (ABI 2)
  - Apple's public Metal API and Metal Shading Language documentation
  - backends/metal/README.md (design)
backend_map:
  metal: backends/metal/, compiler/metal/
state: gpu-pass on Apple M3 (macOS 15.5) and M1 Pro (macOS 15.0.1); translation first cut
owner_spec: AI Lab 2026-11-13
owner_tests: AI Lab 2026-11-13 (same author as impl: see note)
owner_impl: AI Lab 2026-11-13
cost: { hours: 1.5, ai_usage: 0 }
---

## 동작

- 백엔드(`backends/metal/mt_backend.m`): 공개 Metal API. 장치 포인터 = 공유 저장 버퍼의 GPU 주소(Metal 3), 복사는 같은 메모리의 CPU 주소로, 커널은 MSL 소스를 실행 중에 컴파일, 인자는 버퍼 0에 값 덩어리 + 버퍼 1에 위치표. 모든 할당을 실행마다 GPU에 알림.
- 변환(`compiler/metal/cu2msl.py`, 1차): Clang 장치 패스의 형 정보 구문 트리를 읽어 원본에서 필요한 곳만 고친다(커널 머리, 포인터 주소 공간, 공유 메모리, 내장 변수 문맥). 장벽·원자·수학·워프 함수는 `msl_prelude.h`. 템플릿·double·전역 장치 변수는 이유를 밝히고 거부.

## 경계 조건

- Apple GPU에는 double이 없다: 변환 단계에서 오류.
- 블록당 공유 메모리 32KB(M3 측정): 48KB를 가정한 프로그램은 실행 설정 오류.
- 공유(통합) 메모리, 호스트 메모리 등록, 모듈 전역 변수는 아직 "지원 안 함".

## 시험 목록

- [x] `tests/backend/test_metal_backend.c`(손으로 쓴 MSL): Apple M3에서 53/53. 장치 Apple M3, 작업 집합 10.7GiB, 공유 메모리 32768바이트, 100만 값 saxpy 정답(2.3ms), 공유 메모리 블록 합 정답
- [x] `tests/backend/test_metal_cu2msl.c`: `probe_kernel.cu`를 cu2msl로 바꿔 실행, 호스트 계산과 대조: M3·M1 Pro 모두 23/23
- [x] M1 Pro: 백엔드 53/53(saxpy 100만 값 1.4ms), 변환 23/23
- [x] 두 Mac 모두 전체 시험 42/42, check.sh ALL PASS(빌드 경고 0, Release)

## 메모

이 세션은 Mac이 없어 Metal 코드를 컴파일하지 못한다. 결과는 M3 시험기 보고로 확인한다. 시험과 구현을 같은 세션이 썼다.

## 다음

1. 변환기 넓히기: 장치 함수 포인터 인자(템플릿화), 템플릿 커널, `__constant__`·`__device__` 변수(버퍼로), 구조체 인자 배치 확인
2. GPU 경로 빌드 연결: `ocu_cc --target metal`(호스트는 Clang CUDA 모드 호스트 패스, 장치 코드는 cu2msl 이미지를 컨테이너에)
3. 런타임 시험 프로그램과 Rodinia를 M3에서

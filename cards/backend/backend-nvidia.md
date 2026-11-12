---
id: backend-nvidia
kind: component
name: NVIDIA GPU backend
spec_sources:
  - backends/interface/ocu_backend.h (the contract, ABI 2)
  - public CUDA Driver API reference (function names, parameters, documented constants)
backend_map:
  nvidia: backends/nvidia/
state: gpu-pass (RTX 5070 Ti, driver 610.88, WSL2)
owner_spec: AI Lab 2026-11-12
owner_tests: AI Lab 2026-11-12 (same author as impl: see note)
owner_impl: AI Lab 2026-11-12
cost: { hours: 1.5, ai_usage: 0 }
---

## 동작

엔비디아 GPU에서 연결 층 규격 26개 함수와 선택 항목 2개(ABI 2)를 구현한다.

- 설치된 드라이버 라이브러리(`libcuda.so.1`)를 실행 중에 불러 공개 API만 부른다. 빌드에는 엔비디아 파일이 하나도 필요 없다. 드라이버가 없으면 장치 0개.
- 선언(`nv_driver.h`)은 공개 문서를 보고 새로 썼다. 엔비디아 헤더를 쓰지 않는다.
- 장치 코드 이미지는 PTX 문자열(Clang의 NVPTX 출력). 모듈을 올릴 때 드라이버가 설치된 GPU용으로 번역한다(sm_80용 PTX가 RTX 5070 Ti(12.0)에서 동작).
- 장치마다 기본 문맥 하나, 모든 호출 앞에서 그 문맥을 현재 스레드에 둔다.
- 할당 표를 직접 관리해 범위를 넘는 복사·채우기·중간 포인터 해제를 오류로 돌려준다(드라이버에 맡기면 메모리 손상 가능).
- 해제·대기열 해제·모듈 해제는 남은 작업을 먼저 끝낸다(규격).
- 기록한 적 없는 표식은 완료로 본다(규격). 시간 측정은 둘 다 기록된 경우만.
- `module_get_global`: 모듈 전역 변수 주소와 크기. 그 범위를 할당 표에 넣어 복사 대상이 되게 한다.
- `kernel_arg_sizes`: PTX의 `.entry` 매개변수 목록에서 인자 수와 크기를 읽는다.

## 경계 조건

- 블록·격자 0, 블록 크기 초과는 `OCU_ERR_INVALID_VALUE`. 실행 중 오류(7xx)는 `OCU_ERR_LAUNCH_FAILED`.
- 엔비디아 장비의 결과는 기준 답으로 쓰지 않는다. 우리 쪽 기준(CPU 백엔드, 정확한 계산)과 대조한다.

## 시험 목록

`tests/backend/test_nvidia_backend.c`(장치 없으면 SKIP): 장치 정보, PTX 올리기·잘못된 이미지, 커널 찾기·없는 이름, 100만 값 saxpy를 호스트 계산과 대조, 표식·시간, 범위 초과 복사·중간 포인터 해제 오류, 채우기·장치 간 복사, 잘못된 실행 설정, 고정 메모리, 메모리 정보.

- [x] RTX 5070 Ti: 50/50
- [x] 주소 검사기 빌드(누수 검사만 끔: 드라이버와 드라이버가 올리는 코드 생성기의 할당)
- [x] 공유(통합) 메모리·호스트 메모리 등록: 런타임 시험 rt_mem·rt_misc·rt_review2를 GPU에서 통과
- [x] 런타임 시험 프로그램 8개 + saxpy를 GPU용으로 빌드해 실행(`gpu_*` 시험, 장치 없으면 건너뜀)
- [x] Rodinia(`OCU_TARGET=nvidia`): 17/18 빌드, 9개 독립 기준과 값 차이 0

## 측정 (RTX 5070 Ti, WSL2)

- 프로그램 시작 비용: 빈 프로그램 약 165ms(드라이버 초기화), 첫 실행 1.6s(PTX 번역, 이후 드라이버가 캐시).
- lavaMD(boxes 10): 커널 43.6ms, 드라이버 초기화 154ms. 짧은 벤치마크는 전체 시간보다 커널 시간으로 비교해야 한다.
- float 수학 함수는 double로 계산 후 반올림하므로, GeForce(FP64 속도가 1/64)에서 float 초월 함수가 많은 커널은 느리다. float 전용 구현이 다음 과제.

## 메모

시험과 구현을 같은 세션이 썼다. 다른 세션이 다시 검토해야 한다.

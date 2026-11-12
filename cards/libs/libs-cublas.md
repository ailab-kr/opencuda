---
id: libs-cublas
kind: component
name: BLAS library API (cublas_v2.h subset), libs/cublas
spec_sources:
  - cuBLAS library documentation (public): helper functions, level-1/2/3 functions, BLAS-like extensions (cublasGemmEx, cublasGemmBatchedEx, cublasGemmStridedBatchedEx), data types (cublasStatus_t, cublasComputeType_t, cublasGemmAlgo_t, cublasMath_t ...)
  - CUDA library types documentation (cudaDataType_t, libraryPropertyType)
  - BLAS standard, netlib reference BLAS (quick returns, alpha == 0 and beta == 0 rules, negative increments)
  - IEEE 754 (binary16, bfloat16 conversion with round to nearest even)
backend_map:
  cpu: libs/cublas/cublas.cu, compiled for the CPU path (ocu_cublas_cpu)
  nvidia: the same source compiled to PTX (ocu_cublas_nvidia); no vendor library is called
  metal: (next) the same source through cu2msl, built with OCU_CUBLAS_NO_FP64
state: gpu-pass
tolerance: tests/libs/cublas_accuracy.cu (bound in the file header); conformance/tolerances/blas.json for the reference cases
owner_spec: AI Lab 2026-11-13
owner_tests: AI Lab 2026-11-13 (same author as impl, as the task asked; separate commits)
owner_impl: AI Lab 2026-11-13
cost: { hours: 1, ai_usage: 0 }
---

## 동작

`cublas_v2.h`(공개 문서의 이름·인자·열거값으로 새로 작성)의 부분집합. 호스트 쪽은 실행 API만 부르는 칩 중립 C++, 계산은 같은 `.cu`의 CUDA C 커널이다. 하나의 소스를 `ocu_cc`가 대상마다 컴파일해 `ocu_cublas_cpu`·`ocu_cublas_nvidia`를 만든다(ARCHITECTURE.md 4장 "CUDA 이름의 라이브러리").

| 묶음 | 함수 |
|---|---|
| 핸들·스트림 | `cublasCreate_v2`, `cublasDestroy_v2`, `cublasSetStream_v2`, `cublasGetStream_v2`, `cublasSetWorkspace_v2`(받아 둠), `cublasSetPointerMode_v2`/`Get`(호스트·장치 스칼라), `cublasSetMathMode`/`Get`, `cublasSetAtomicsMode`/`Get`, `cublasGetStatusName`, `cublasGetStatusString`, `cublasGetVersion_v2`, `cublasGetProperty`, `cublasGetCudartVersion`. 평이한 이름(`cublasCreate` 등)은 `cublas_v2.h`의 매크로 |
| GEMM | `cublasSgemm`, `cublasDgemm`, `cublasHgemm`, `cublasSgemmBatched`, `cublasDgemmBatched`, `cublasHgemmBatched`, `cublasSgemmStridedBatched`, `cublasDgemmStridedBatched`, `cublasHgemmStridedBatched`, `cublasGemmEx`, `cublasGemmBatchedEx`, `cublasGemmStridedBatchedEx` |
| 1·2단계 | `cublasSaxpy`, `cublasSscal`, `cublasSdot`, `cublasSnrm2`, `cublasSgemv` |
| 3단계 그 밖 | `cublasStrsmBatched`(llama.cpp의 solve_tri가 씀, 오른쪽 항마다 스레드 하나인 단순 구현) |

`*Ex`의 형 조합: COMPUTE_32F(`_PEDANTIC` 포함)에서 A=B=32F→C 32F, 16F→16F·32F, 16BF→16BF·32F. COMPUTE_16F에서 16F→16F(스칼라 half). COMPUTE_32F_FAST_16F/16BF/TF32에서 32F→32F. COMPUTE_64F(`_PEDANTIC`)에서 64F→64F. 예전 방식의 `CUDA_R_16F/32F/64F` 계산형 값도 받는다.

GEMM 커널: 256스레드 블록, 공유 메모리 타일 두 가지(128×128×16·스레드당 8×8, 64×64×16·4×4), 다음 타일을 레지스터에 미리 읽음, 전치별로 이웃 스레드가 이웃 주소를 읽는 순서. 큰 타일이 다중처리기 절반 이상을 채우면 큰 타일(`OCU_CUBLAS_TILE=64|128`로 강제 가능). 배치는 grid z(65535 넘으면 나눠 실행).

## 경계 조건

- m·n·배치 0: 아무것도 하지 않고 성공. k 0 또는 alpha 0: A·B를 읽지 않고 C = beta·C. beta 0: C를 읽지 않음(NaN이 있어도 됨). 호스트 모드에서 alpha 0·beta 1이면 바로 돌아옴.
- 장치 포인터 모드: alpha·beta를 커널이 장치 메모리에서 읽음(스트림 순서 유지). dot·nrm2 결과도 장치 메모리로. 호스트 모드의 dot·nrm2는 결과가 올 때까지 기다림.
- 음수 증가폭: 참조 BLAS 방식(먼 끝부터). scal·nrm2는 incx ≤ 0이면 빈 벡터.
- 반정밀도 변환: 읽기는 정확, 쓰기는 가장 가까운 짝수로(전체 2^32 중 1/3 표본과 half 65536개 전부를 기준 변환과 대조, 불일치 0).

## 오류 코드

- 핸들 NULL: `NOT_INITIALIZED`. 크기 음수, ld 부족, 잘못된 연산·측·삼각·대각 값, 배치 음수, 잘못된 algo 값, 스칼라 NULL: `INVALID_VALUE`.
- 계산하지 않는 조합은 `CUBLAS_STATUS_NOT_SUPPORTED`: 복소수와 정수 형(`CUDA_C_*`, `CUDA_R_8I`→`32I`, `COMPUTE_32I`), FP8, `COMPUTE_16F_PEDANTIC`(반정밀도 산술을 요구하는데 여기서는 32F로 누적), A와 B 형이 다른 경우, 문서에 없는 형 조합, axpy의 incy 0(n > 1), grid y 한도(n/타일 > 65535). `OCU_CUBLAS_NO_FP64`로 빌드한 변형에서 64F 전부.
- 실행 실패: `EXECUTION_FAILED`, 메모리 부족: `ALLOC_FAILED`.

## 시험 목록

- [x] `cublas_cpu`, `cublas_cpu_tile64_random`, `cublas_cpu_tile128`, `gpu_cublas_default/64/128` (`tests/libs/cublas_accuracy.cu`): 1395개 확인, CPU·RTX 5070 Ti 모두 통과. 고정밀(long double) 기준, 오차 한계는 파일 머리말. 모양 12가지(타일 배수 아님, k 0, 1×300 등) × 전치 4 × alpha/beta 3, OP_C, 형 조합 12가지, 배치(배열·간격·B 공유·65537개), 장치 포인터 모드(float·half 스칼라), 스트림, 1·2단계, strsm 배치 16조합(잔차 기준), 상태 코드
- [x] `conformance_cublas_cpu`(+스트림·무작위 순서), `gpu_conformance_cublas`: 기존 기준 사례(conformance/oracle) 중 이 라이브러리 함수 251건 전부 통과(오차/한계 최대 0.495 CPU, 0.472 GPU)
- [x] 일부러 망가뜨린 커널: beta 0에서 C를 읽게 함 → 252건 실패로 검출, 출력 열 하나 넘게 씀 → 프로그램 중단으로 검출

## 메모

- 정밀도: 모든 실수 계산형을 그 형 이상의 정밀도로 누적한다(16F도 32F로 누적 후 한 번 반올림). 문서가 허용하는 오차 안이고 더 정확하다. 그래서 16F_PEDANTIC만 "지원 안 함"이다.
- math mode·atomics mode·workspace는 받아 두기만 한다. 줄어든 정밀도를 "허용"하는 설정이라 전체 정밀도로 계산해도 맞다.
- 처리량(RTX 5070 Ti, 정사각, 3회 예열 뒤 10회 중앙값, 90% 구간): 카드 아래 "측정" 절.
- llama.cpp(docs/llama-cpp-requirements.md)의 cuBLAS 함수 12개는 모두 있다. 없는 것: `cublasLt.h`와 cuBLASLt 12개(CUDART 12030 프로필의 FP8 경로에서만 씀, 별도 카드 일), `cublasLoggerConfigure`(llama.cpp에서 주석 처리됨).
- 링크: `ocu-nvcc -lcublas`는 대상에 맞는 `libocu_cublas_<cpu|nvidia>.a`, 툴킷의 `CUDA::cublas`는 `OCU_TARGET`에 맞는 변형.
- Metal: 커널은 템플릿·람다·구조체 인자를 쓴다. cu2msl이 그대로 받는지는 Apple 장비에서 확인할 일이다.

## 측정

RTX 5070 Ti(70 SM), `tuning/bench_cublas`, 정사각 n, 3회 예열 뒤 10회 × 3표본, 표본 사이 퍼짐 5% 이내(모두 유효). 11/13 03:28, GPU는 다른 작업 없음, CPU는 다른 에이전트 작업으로 부하 평균 약 7/24(완전히 조용하지는 않음).

| 연산 | n | 중앙값 ms | 90% ms | GFLOP/s |
|---|---|---|---|---|
| Sgemm NN | 1024 | 0.173 | 0.180 | 12,442 |
| Sgemm NN | 2048 | 1.201 | 1.223 | 14,299 |
| Sgemm NN | 4096 | 9.802 | 9.837 | 14,022 |
| Sgemm TN | 4096 | 9.291 | 9.332 | 14,792 |
| Hgemm(16F) NN | 1024 | 0.157 | 0.170 | 13,721 |
| Hgemm(16F) NN | 2048 | 1.201 | 1.216 | 14,305 |
| Hgemm(16F) NN | 4096 | 7.961 | 8.040 | 17,263 |
| GemmEx f16→f16, 32F NN | 4096 | 7.971 | 7.998 | 17,242 |
| GemmEx f16→f32, 32F TN | 4096 | 7.321 | 7.354 | 18,774 |

개선 기록: 첫 판 700 GFLOP/s(누적 배열이 펼쳐지지 않아 지역 메모리로) → 펼침 8,000 → 다음 타일 레지스터 선읽기 14,000(fp32). fp16은 변환을 읽기 직후에 해 읽기가 멈추던 것을 저장형으로 선읽기하게 바꿔 6,700 → 14,900. 텐서 코어는 쓰지 않는다(칩 중립 CUDA C만).

## 이의

없음.

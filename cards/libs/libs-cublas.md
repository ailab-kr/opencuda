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

연산 표 경로(11/13): GEMM은 커널보다 먼저 활성 백엔드의 연산 표(`docs/spec/op-table.md` 8절)에 묻는다. 조건이 모두 맞으면 열 우선 `C(m×n)`을 행 우선 `C'(n×m) = op(B)'·op(A)'`로 옮긴 설명자를 질의 → 계획(핸들마다 32칸 캐시, 설명자 전체가 열쇠) → 스트림의 대기열에서 실행(`ocu_rt_submit`: 실행과 같은 기본 스트림 규칙). 조건: 호스트 모드 스칼라, alpha ≠ 0, k > 0, 간격 배치(포인터 배열 아님, A·B 간격 0은 공유, C 간격 0은 안 됨), A·B 형이 32F·16F·16BF, 표가 알리는 누적형이 binary32. 하나라도 어긋나거나 표가 거절하면(질의·계획 실패, 실행 전 검사 실패) 위의 커널로 간다. 칩 이름은 보지 않는다: CPU 백엔드는 표가 있어 이 길로, NVIDIA 백엔드는 표가 없어 커널로.

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

### 연산 표 경로(11/13)에서 시험 작성자가 확인할 동작

구현 세션이 쓴 목록이다. 시험은 다른 세션이 쓴다. 기존 시험(`cublas_cpu*` 1395개, `conformance_cublas_cpu*`, `ops_cpu`)은 이 경로로 그대로 통과한다(`OCU_CPU_STATS=1`로 보면 `cublas_accuracy`의 GEMM 421번이 블록 1스레드짜리 표 실행으로 감).

1. 길 고르기: 장치 포인터 모드, 포인터 배열 배치, 64F, alpha 0, k 0, C 간격 0 배치는 커널로 가고 결과가 이전과 같음. 같은 문제를 호스트 모드로 부르면 표 경로.
2. 표 거절 시 되돌아감: 백엔드가 할당하지 않은 메모리(예: `cudaHostAlloc` 고정 메모리를 장치 포인터로 넘김)에서 실행 전 검사가 실패하면 오류 없이 커널로 계산, 마지막 오류도 그대로.
3. 순서: 기본 스트림·막는 스트림·막지 않는 스트림에서 GEMM 앞뒤의 복사·커널과 순서가 맞음(실행과 같은 규칙). 스트림에 GEMM 여러 개를 쌓은 뒤 한 번 동기화.
4. 계획 캐시: 서로 다른 모양 33개 이상을 번갈아 불러도 결과가 맞음(밀려난 계획 해제), 핸들 둘이 같은 모양을 써도 맞음, `cublasDestroy` 뒤 누수 없음(주소 검사기).
5. 정밀도: 16F 입력 → 16F 출력에서 한 번만 반올림(가장 가까운 짝수), beta 0이면 C의 NaN을 읽지 않음, alpha ≠ 0이고 A나 B에 Inf·NaN이 있으면 결과에 퍼짐.
6. `OCU_CPU_GEMM=ref|generic`(백엔드 쪽 선택)에서 결과가 오차 한계 안.

## 메모

- 속도(CPU 경로, 11/13, `tuning/bench_cublas_cpu 256 512`, `OCU_CPU_THREADS=8`, 다른 작업으로 부하 평균 약 7): 512³ Sgemm TN 79 → 529 GFLOP/s, Hgemm TN 76 → 535, GemmEx 16F→32F TN 75 → 538, NN(B를 옮겨 읽음) 75 → 약 300. 256³은 38 → 160~370. `OCU_CPU_GEMM=generic`(SSE2만)은 512³ 140~196.
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

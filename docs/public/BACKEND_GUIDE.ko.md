# 백엔드 안내서

[English](BACKEND_GUIDE.md) · **한국어**

> 검토용 초안. 코드 식별자는 짧은 접두어 `ocu`(OpenCuda)를 쓴다.

이 안내서는 새 칩(GPU나 NPU)에서 수정하지 않은 CUDA C++ 프로그램을 돌리려는 칩 벤더와 커뮤니티 구성원을 위한 것이다.
백엔드는 함수 표 하나를 채우는 C 파일 묶음이다. 표 위쪽에는 칩에 묶인 것이 없으므로,
새 백엔드를 만들어도 실행 환경, 라이브러리, 적합성 시험은 바뀌지 않는다.

## 1. 백엔드의 모양

```
CUDA C++ source and build files (unmodified)
        |
   compiler glue      runtime API (CUDA-named)      libraries (BLAS, ...)
        |                    |                            |
        +--------------------+----------------------------+
                             |
                  backends/interface/ocu_backend.h      <- the only chip boundary
                             |
        cpu | nvidia | metal | your backend
```

경계를 깨끗하게 지키는 규칙:

- 백엔드는 인터페이스 헤더와 자기 칩의 라이브러리나 드라이버를 쓸 수 있다.
  다른 백엔드를 쓰면 안 되고, 인터페이스 위쪽의 어떤 것도 칩 이름으로 분기하면 안 된다.
- 적합성 시험은 사용자 프로그램에 보이는 것만 쓴다. 내부 헤더를 포함하지 않는다.
- 새 백엔드가 인터페이스 위쪽의 변경을 필요로 해 보이면 인터페이스에 빈틈이 있는 것이다.
  RFC 절차([GOVERNANCE.ko.md](GOVERNANCE.ko.md))로 변경을 제안한다. 호환되지 않는 변경은 `OCU_BACKEND_ABI`를 올린다.

## 2. ABI: 헤더 하나

ABI는 `backends/interface/ocu_backend.h`다(독립적인 C11, 현재 `OCU_BACKEND_ABI`는 2).
백엔드는 채워진 `ocu_backend` 표의 포인터를 돌려주고 등록한다
(`ocu_register_backend`, 또는 `runtime/core/registry.c`의 내장 목록에 한 줄 추가).
ABI 번호가 다른 등록은 거부된다.

### 필수 항목

| 묶음 | 항목 | 한 줄 계약 |
|---|---|---|
| 식별 | `abi`, `name` | `name`은 소문자이고 바뀌지 않는다(`"cpu"`, `"metal"`) |
| 장치 | `dev_count`, `dev_info`, `dev_open`, `dev_close` | 실제 워프/웨이브프런트 크기(32 또는 64), 실제 한도, 실제 메모리를 보고한다 |
| 메모리 | `mem_alloc`, `mem_free`, `mem_copy`, `mem_set`, `mem_info`, `host_alloc_pinned`, `host_free_pinned` | 장치 포인터는 평범한 주소다. 할당 안에서는 `ptr + n`이 유효하다. `NULL` 큐는 동기 실행을 뜻한다. 해제는 큐에 쌓인 작업을 기다린다 |
| 큐 | `queue_create`, `queue_destroy`, `queue_sync`, `queue_query` | 한 큐의 작업은 제출 순서대로 실행된다. 큐에 쌓인 작업이 낸 오류는 다음 동기화나 조회에서 보고된 뒤 지워진다 |
| 코드 | `module_load`, `module_unload`, `module_get_kernel` | 이미지 형식은 백엔드 고유다(PTX 텍스트, MSL 소스, 함수 표, 벤더 바이너리) |
| 실행 | `kernel_launch` | 반환하기 전에 인자 값을 복사한다. 인자 합계는 `OCU_MAX_ARG_BYTES`(32768) 이하다 |
| 표식 | `marker_create`, `marker_destroy`, `marker_record`, `marker_wait`, `marker_query`, `marker_elapsed_ms` | 한 번도 기록되지 않은 표식은 완료된 것이다. 대기는 호출 시점의 기록을 기다린다 |

### 선택 항목

선택 항목은 `NULL`일 수 있다. 그러면 실행 환경은 그 항목이 필요한 API에 "지원 안 함"을 돌려주거나 이식 가능한 경로로 대체한다.

| 항목 | 용도 | 없으면 |
|---|---|---|
| `get_provider` | 조정된 라이브러리 표를 돌려준다(예: `ocu_blas_provider.h`의 BLAS 제공자) | 라이브러리 계층이 이식 가능한 자체 구현을 쓴다 |
| `module_get_global` | 장치 코드 변수(`__device__`, `__constant__`)의 주소와 크기 | 심볼 복사가 "지원 안 함"을 보고한다 |
| `kernel_arg_sizes` | 매개변수의 개수와 크기. 인자 포인터 배열만 가진 실행에 필요하다 | 그런 실행은 "지원 안 함"을 보고한다 |
| `mem_alloc_shared`, `host_register`, `host_unregister` | 호스트와 장치가 같은 주소로 보는 메모리 | 관리 메모리와 호스트 등록은 `supports_unified_memory`가 설정된 곳에서만 제공한다 |

### 상태 코드

칩이 할 수 없는 일에는 `OCU_ERR_NOT_SUPPORTED`를, 백엔드가 아직 하지 않는 일에는 `OCU_ERR_NOT_IMPLEMENTED`를 돌려준다.
점수판은 둘을 다르게 보여 준다. 하지 않은 일에 성공을 보고해서는 안 된다.

### 동시성과 수명

백엔드 함수는 여러 스레드에서 동시에 불릴 수 있다. 실행 중인 커널 안에서 `kernel_launch`, `queue_sync`,
`queue_destroy`, `marker_wait`, 해제를 부르면 `OCU_ERR_NOT_SUPPORTED`를 돌려준다.
상태 규칙은 [ARCHITECTURE.ko.md](../../ARCHITECTURE.ko.md)의 6절에 있다.

## 3. 장치 코드

인터페이스는 장치 코드를 어떻게 만드는지는 정하지 않고, 어떻게 불러오는지만 정한다.
각 백엔드는 자기 이미지 형식을 문서화한다. 기존 예시:

| 백엔드 | 이미지 | 만드는 방법 |
|---|---|---|
| cpu | 호스트 함수 표 | CPU 경로가 실행 구문을 다시 쓰고 C++로 컴파일한다 |
| nvidia | PTX 텍스트 | CUDA 모드의 오픈소스 Clang이 NVPTX를 목표로 만든다. 수학 함수는 프로젝트 자체의 장치 수학 헤더에서 온다 |
| metal | Metal Shading Language 소스 | Clang 장치 단계의 구문 트리를 읽고 MSL에 필요한 부분(커널 머리, 주소 공간, 공유 메모리, 내장 변수)만 다시 쓰는 변환기 |

새 GPU 백엔드는 보통 오픈소스 컴파일러 프런트엔드를 다시 쓰고, 자기 ISA나 셰이딩 언어를 위한
장치 코드 생성기나 변환기를 더한다.

## 4. 적합성 수준

백엔드는 자기가 통과한 수준을 선언한다. 수준은 누적된다.

| 수준 | 필요한 증거 |
|---|---|
| B0 인터페이스 | 백엔드 시험(`tests/backend`)을 통과한다: 메모리, 큐, 표식, 실행, 오류 경우, 플랫폼에 있는 곳에서는 새니타이저 빌드 |
| B1 실행 환경 | 실행 시험 프로그램(`tests/runtime`, 평범한 CUDA C++ 프로그램)이 수정 없이 빌드되고 백엔드에서 통과한다 |
| B2 언어 | 장치 코드 의미: 장벽, 공유 메모리, 원자 연산, 워프 셔플과 투표, 허용 오차(L3) 안의 수학 함수 |
| B3 라이브러리 | 허용 오차 안에서 L4 시험(현재는 BLAS). 제공자는 선택이다 |
| B4 응용 프로그램 | L5: 공개 프로그램(현재는 Rodinia)이 독립 기준과 같은 결과를 낸다 |

결과는 점수판 도구로 보고한다.

```sh
ctest --test-dir build --output-on-failure > ctest.txt
python3 tools/scoreboard.py build/tests/blas_<backend>_result.json \
    --ctest <backend>=ctest.txt --rodinia <backend>=rodinia.txt --out-dir scoreboard/
```

점수판은 건너뛴 시험과 지원 안 하는 시험을 따로 나열하며, 통과로 세지 않는다.
현재 실행 가능한 증거가 있는 수준은 B0, B1, B3(BLAS), B4(Rodinia)다. L1~L3 시험 모음은 작성 중이며,
그것이 생길 때까지 백엔드의 B2 주장은 실행 시험 프로그램과 언어 시험 프로그램이 뒷받침한다.

속도는 중앙값과 느린 쪽 10 % 값으로 보고한다. 백엔드의 순위를 속도로 매기지 않고, 속도로 설명한다.

## 5. 백엔드를 추가하는 순서

1. `backends/<이름>/`을 만들고 표를 구현한다. 구조는 CPU 백엔드에서 시작한다.
   빠진 부분은 `OCU_ERR_NOT_IMPLEMENTED`로 처리한다.
2. `backends/CMakeLists.txt`에 빌드 옵션을, `runtime/core/registry.c`에 등록 한 줄을 추가한다.
3. 장치 코드 이미지 형식과 만드는 방법을 문서화한다.
4. `tests/backend` 아래에 백엔드 시험을 추가한다. 장치가 없으면 깔끔하게 건너뛰어야 한다.
5. 실제 하드웨어에서 시험 모음을 돌리고 결과 기록을 공개한다.
6. 인터페이스에 새 항목이 필요하면 RFC를 연다.

승인 기준은 [GOVERNANCE.ko.md](GOVERNANCE.ko.md)에 있다.

## 6. NPU용 라이브러리·연산 수준 백엔드 (계획)

상태: 계획. 이 절의 어떤 것도 아직 트리에 없다.

일부 가속기에는 SIMT 코어가 없어 임의의 CUDA C++ 커널을 실행할 수 없다. 그래도 시간의 대부분을
몇 개의 라이브러리 호출(행렬 곱, 합성곱, 리덕션, 어텐션)에 쓰는 많은 CUDA 프로그램은 돌릴 수 있다.
계획한 방식:

- 백엔드는 메모리, 큐, 표식 항목을 평소처럼 구현하고, 임의의 장치 코드에 대해서는
  `module_load`와 `kernel_launch`에서 `OCU_ERR_NOT_SUPPORTED`를 돌려준다.
- 연산은 `get_provider(<라이브러리>)`로 내놓는다. 제공자 표는 라이브러리별로 인터페이스 옆에 정의한다
  (BLAS는 있고, 합성곱·정규화 같은 다른 표는 RFC로 추가한다).
- 적합성은 라이브러리 수준으로만 보고한다(B2 없이 B3). 자기 커널을 실행하는 프로그램은 L1과 L3에서
  숨기지 않고 "지원 안 함"으로 보고한다.
- 그래프나 연산 융합 항목이 필요할 것으로 예상한다. 이것은 인터페이스 확장이므로 코드를 쓰기 전에 RFC를 거친다.

이런 백엔드를 계획하는 벤더는 제공자 표가 자기 연산 집합에 맞도록 RFC를 일찍 열기를 권한다.

## 7. 백엔드에만 적용되는 기여 규칙

- 공개 문서만 쓴다. 다른 벤더의 헤더, 예제, 소스를 복사하거나 옮겨 적지 않고,
  벤더 바이너리를 역어셈블하거나 추적하지 않는다. 변경의 근거가 된 공개 문서를 커밋 메시지에 적는다.
- 자기 칩에 코드를 기여하는 벤더 엔지니어는 풀 리퀘스트에 그 사실을 밝힌다. 검토와 시험 규칙은 모두에게 같다.
- 비공개 시험 세트에서 만든 결과 기록은 증거로 받지 않는다. 공개 시험 모음을 쓴다.

---
id: apps-app-corpus
kind: app
name: Real-program corpus on the CPU path (PolyBench-ACC, Parboil, llm.c fp32 trainer), built unmodified with ocu-nvcc
spec_sources:
  - public nvcc documentation (command line)
  - public CUDA C++ Programming Guide (atomic functions with scopes, load/store functions using cache hints, warp shuffle and vote functions, Cooperative Groups tiles and meta groups)
  - public CUDA Runtime API reference (3D memory management, device limits, profiler control)
  - public CUDA Math API reference (constants)
  - the programs' own sources and checks (listed below)
backend_map:
  cpu: ocu-nvcc --ocu-target=cpu over backends/cpu
state: impl-written-untested
owner_spec: AI Lab 2026-11-13
owner_tests: (to be assigned: a session other than owner_impl)
owner_impl: AI Lab 2026-11-13
cost: { hours: 5, ai_usage: 0 }
---

## 동작

llama.cpp 말고도 소스를 고치지 않은 실제 프로그램이 CPU 경로에서 빌드·실행되는지, 그 프로그램이 스스로 갖춘 점검(또는 같은 묶음의 CPU 판)과 결과가 맞는지를 싸게 잰다. 빌드·실행 스크립트는 `conformance/external/<묶음>/`에 있고, 소스는 저장소 밖(기본 `~/.cache/ocu-<묶음>`)에 받는다.

| 묶음 | 스크립트 | 점검 방법 |
|---|---|---|
| PolyBench-ACC CUDA 21개 (`70ea4ca`) | `conformance/external/polybench/run.sh <빌드> [작업 폴더] [MINI/SMALL/STANDARD]` | 프로그램이 같은 입력으로 자기 직렬 CPU 코드를 돌려 두 결과의 차이 개수를 출력한다(그 줄이 점검) |
| Parboil CUDA 11개 (`ccf3d31`, 비공식 미러) | `conformance/external/parboil/run.sh <빌드> [작업 폴더]` | 데이터셋이 없어 `gen.py`가 작은 입력을 만든다. 같은 묶음의 CPU 판(`src/cpu`·`src/base`)을 같은 입력으로 돌려 `cmp.py`로 비교 |
| llm.c `train_gpt2_fp32.cu` (`f1e2ace`) | `conformance/external/llmc/run.sh <빌드> [작업 폴더]` | `gen.py`가 작은 GPT-2(2층·4헤드·128채널, 가중치 약 690만 개)와 토큰 파일을 만든다. 같은 저장소의 `train_gpt2.c`(gcc)를 같은 가중치·데이터로 돌려 8걸음 손실 비교 |

내려받기는 소스뿐이다(llm.c의 124M 가중치 497 MB와 텍스트 데이터는 쓰지 않는다).

## 결과 (2026-11-13, CPU 경로, WSL2)

PolyBench-ACC (SMALL 데이터셋과 기본 STANDARD 데이터셋 둘 다):

| 항목 | SMALL | STANDARD |
|---|---|---|
| 빌드 | 21/21 | 21/21 |
| 실행 | 21/21 | 21/21 |
| 프로그램 자체 점검 일치 | 20/21 | 20/21 |

- 어긋난 하나는 `mvt`: 프로그램 결함이다. 블록을 32×8로 띄우면서 색인은 `threadIdx.x`만 써서 8개 스레드가 같은 `x1[i]`·`x2[i]`에 더한다. 워프가 박자를 맞춰 도는 GPU에서는 가려지지만(같은 값을 읽고 같은 값을 쓴다), 스레드를 차례로 도는 CPU 백엔드에서는 값이 직렬 결과의 8배가 된다(SMALL 4094개, STANDARD 8190개 차이). 작업 폴더의 복사본에서 `DIM_THREAD_BLOCK_Y`를 1로 바꾸면 차이 0.
- 점검이 CPU 판과 GPU 판을 같은 컴파일러·같은 부동소수점 규칙으로 돌려 비교하는 것이라 칼같이 일치한다(나머지 20개 모두 차이 0). 같은 연산 순서로 짜인 프로그램이라는 뜻이다.

Parboil (11개 모두. bfs·sad·spmv는 텍스처 참조를 쓴다: 카드 `runtime-texture`, 2026-11-13 추가):

| 프로그램 | 빌드 | 실행 | CPU 판과 비교 |
|---|---|---|---|
| bfs | 예 (텍스처 추가 후) | 예 | 다름: 4,096개 노드 중 782개. 원인은 프로그램의 GPU 알고리즘(텍스처를 보통 읽기로 바꾼 복사본도 같은 782개, RTX 5070 Ti에서도 CPU 판과 615개 다름; 카드 `runtime-texture`) |
| cutcp | 예 | 예 (원자 300개) | 기준 없음: 출력만 확인 |
| histo | 예 | 예 | 일치 (비트맵 147,510바이트 동일) |
| lbm | 예 | 예 (2걸음) | 일치 (6,480,000개 값 모두 허용 오차 안, 실제로는 동일) |
| mri-gridding | 예 | 아니오 | 입력 형식(궤적 파일) 생성기 없음: 빌드만 |
| mri-q | 예 | 예 | 일치 (2048개 값) |
| sad | 예 (텍스처 추가 후) | 예 | 일치 (SAD 4,286,312바이트 동일, 128×96 두 프레임) |
| sgemm | 예 | 예 (128×64 · 64×64) | 일치 (8192개 값) |
| spmv | 예 (텍스처 추가 후) | 예 | 일치 (fidapm05.mtx, 42개 값; 패턴 행렬은 묶음의 변환기가 두 판 모두에서 죽음) |
| stencil | 예 | 예 | 일치 (4096개 값) |
| tpacf | 예 | 아니오 | 입력(카탈로그 파일) 생성기 없음: 빌드만 |

합계: 11개 모두 빌드, 9개 실행, 7개가 CPU 판과 일치(cutcp는 기준 없음이라 일치 칸에서 뺌). 텍스처 전에는 8개 빌드·5개 실행·5개 일치였다(빌드 실패 3개는 모두 CUDA 12에서 사라진 `texture` 참조 API가 원인).

Parboil 빌드에 쓴 컴파일러 옵션(소스는 그대로): `-Xcompiler -Wno-c++11-narrowing`(histo의 좁히기 초기화, nvcc의 호스트 컴파일러는 경고만 냄), `-Xcompiler -fno-strict-return`(sgemm의 `readColMajorMatrixFile`은 `bool`을 돌려주는 함수인데 `return` 없이 끝난다. C++에서 정의되지 않은 동작이고 Clang은 최적화에서 그 길을 없앤다). 기준(CPU 판)도 같은 이유로 gcc가 아니라 clang으로 빌드한다(gcc는 그 지점에 `ud2`를 넣어 실행이 죽는다).
입력 크기는 프로그램의 가정에 맞췄다: sgemm은 행이 128의 배수·열이 16의 배수, histo는 히스토그램 너비 = 영상 너비이고 24576칸 단위.

llm.c `train_gpt2_fp32.cu` (CPU 경로, 소스 그대로, `cublas`는 `ocu_blas` 경유):

- 빌드·링크 성공, 실행 성공. 8걸음에 걸음당 약 0.4~0.5초(약 600 토큰/초).
- 손실 비교(걸음 1~8, `train_gpt2.c` 대 우리 경로, 같은 가중치·데이터): 소스 그대로는 최대 차이 0.0648 (걸음 1에서 0.0012, 8에서 0.0648). 아래 경쟁(race) 한 줄을 막은 복사본은 최대 차이 0.000002(6자리 일치): 10.830529/10.830528, 10.755331/10.755333, …, 10.372774/10.372773.
- 소스 그대로가 어긋나는 이유: `fused_classifier_kernel3`에서 0번 스레드가 손실용으로 `logits[idx*P+ix]`를 읽는 동안, 다른 스레드들이 같은 배열을 로짓 기울기로 덮어쓴다. 둘 사이에 순서를 잡는 것이 없다. GPU는 0번 스레드의 읽기가 거의 늘 먼저라 드러나지 않지만, CPU 백엔드에서는 워프 연산 뒤의 실행 순서에 따라 256행 중 46행(고정 순서), 131행(`OCU_CPU_ORDER=random`)에서 이미 덮인 값을 읽는다. 기울기와 학습은 영향이 없고 보고되는 손실만 틀린다(그 행의 손실이 대략 ln(어휘 크기)로 나옴). `__syncthreads()` 한 줄을 읽기 바로 뒤에 넣은 복사본(`run.sh`가 작업 폴더에서 만든다)으로 확인했다.
- 이 프로그램을 돌리려고 `cooperative_groups.h`에 `thread_block_tile::meta_group_rank()`·`meta_group_size()`가 필요했다(아래).

## 이 작업에서 더한 것

런타임·헤더(카드 `runtime-cudart`의 "작은 추가" 절도 본다):

- 캐시 힌트 적재·저장 9개: `__ldcg`·`__ldca`·`__ldcs`·`__ldlu`·`__ldcv`, `__stcg`·`__stcs`·`__stwb`·`__stwt`. 값을 바꾸지 않는 힌트라 일반 접근으로 구현(`__ldcv`는 컴파일러가 앞선 읽기를 재사용하지 못하게 함). 두 경로 공통(`ocu_cuda_lang_common.h`).
- 범위 지정 원자 연산: `atomic{Add,Sub,Exch,Min,Max,And,Or,Xor,Inc,Dec,CAS}_block`과 `_system`(`ocu_cuda_lang_scoped.h`). 블록 범위는 장치 전체 원자가 만족하므로 같은 구현. 시스템 범위는 CPU 경로(모든 스레드가 호스트 스레드)에서만 제공하고 NVIDIA 경로에서는 컴파일할 때 "지원 안 함" 메시지로 멈춘다(장치 범위 원자가 호스트와의 동시 접근을 보장하지 않으므로 조용히 약하게 하지 않음).
- 사라진 무동기 워프 함수 `__shfl`·`__shfl_up`·`__shfl_down`·`__shfl_xor`·`__ballot`·`__any`·`__all`: 계산 능력 7.0 이상에서 지원하지 않는다고 문서에 적혀 있고 그 이상의 의미가 정의돼 있지 않으므로, 부르면(템플릿 인스턴스화 시) 대체할 `_sync` 함수를 알려 주는 메시지와 함께 컴파일이 멈춘다.
- `math_constants.h`: `CUDART_PI_F`, `CUDART_INF_F`, `CUDART_SQRT_HALF_HI` 등 단·배정밀도 상수와 `_HI`/`_LO` 쪼갠 값. 값은 60자리로 계산해 한 번 반올림했다.
- `cuda_profiler_api.h`: `cudaProfilerStart`·`cudaProfilerStop`(붙은 도구가 없으면 효과 없음, 성공을 돌려줌).
- 런타임 호출: `cudaMalloc3D`, `cudaMemset2DAsync`, `cudaMemset3D`·`Async`, `cudaMemcpy3D`·`Async`(배열이 얽힌 복사는 "지원 안 함"), `cudaDeviceGetLimit`·`cudaDeviceSetLimit`, 형식 `cudaExtent`·`cudaPos`·`cudaPitchedPtr`·`cudaMemcpy3DParms`·`cudaLimit`과 `make_cuda*`.
- Cooperative Groups: `thread_block_tile`이 부모 형을 두 번째 템플릿 인자로 가지며(`thread_block_tile<N>`은 부모 없음 형) `meta_group_rank()`·`meta_group_size()`를 제공. 부모가 있는 타일에서 부모 없는 타일로의 변환은 그 값을 가져간다.
- CPU 경로 번역기: `struct S { ... } __align__(8);`(닫는 중괄호 뒤 정렬 지정)을 `__attribute__((aligned(8)))`로 바꿈(Parboil sad).

## 경계 조건

- 텍스처: 카드 `runtime-texture`(객체·배열·CPU 경로의 참조, 소프트웨어 표본 추출).
- `cudaMemcpy3D`의 배열 인자: 텍스처 작업에서 구현(살아 있는 배열이 아닌 손잡이는 `cudaErrorInvalidValue`).
- `cudaDeviceSetLimit`: 스택·printf FIFO·말록 힙은 값을 기록해 돌려주기만 하며 효과가 없다. 0은 `cudaErrorInvalidValue`. 장치 런타임 한도(동기화 깊이, 대기 실행 수)와 L2 상주 한도는 `cudaErrorUnsupportedLimit`, `cudaLimitMaxL2FetchGranularity`는 힌트라 0을 돌려준다.
- 무동기 워프 함수와 NVIDIA 경로의 시스템 범위 원자는 컴파일 오류(위).
- PolyBench `mvt`와 llm.c 손실 값은 프로그램의 경쟁 상태 때문에 CPU 백엔드에서 GPU와 다르게 보인다(위). 백엔드의 스레드 실행 순서가 GPU와 닮을수록 덜 드러나지만 보장은 아니다.

## 시험 목록

시험 작성자(구현자와 다른 세션)가 확인할 동작은 각 API마다 아래 "문서가 요구하는 것"과 같다. 구현자가 둔 스모크 시험은 `tests/runtime/rt_apps_gaps.cu`(49개 점검) 하나뿐이다.

1. 캐시 힌트: 모든 문서화된 자료형(정수 1·2·4·8바이트, 부호 있음·없음, float, double, 벡터형 2·4개)에서 읽기는 `*p`와 같고 쓰기는 `*p = v`와 같다. `__ldcv`는 다른 스레드가 쓴 값을 다시 읽는다. 값 인자는 가리키는 형으로 변환된다(`__stcs(float *, 0)`).
2. 범위 지정 원자: 각 함수가 같은 이름의 원자(범위 없음)와 같은 반환값(이전 값)과 같은 결과를 낸다. 자료형은 `int`·`unsigned`·`unsigned long long`(`Add`·`Exch`·`CAS`는 `float`·`double`도 문서에 따라), 블록 안 모든 스레드가 부름. NVIDIA 경로의 `_system`은 컴파일되지 않는다(메시지에 "system-scope"가 있음). `atomicInc`/`Dec`의 감김.
3. 무동기 워프 함수: 부르는 코드는 컴파일되지 않고 메시지에 대체 함수 이름이 있다. 부르지 않으면 영향이 없다.
4. `math_constants.h`: 단정밀도 값은 해당 수학 상수를 float로 한 번 반올림한 값과 같고 배정밀도도 같다. `_HI + _LO`가 상수에 더 가깝다. `CUDART_INF_F`는 무한대, `CUDART_NAN_F`는 NaN, `CUDART_NEG_ZERO_F`는 부호 비트가 선 0.
5. `cuda_profiler_api.h`: Start/Stop이 `cudaSuccess`, 여러 번 불러도 같음.
6. `cudaMalloc3D`: 피치 ≥ 너비, `xsize`=너비, `ysize`=높이, 할당 크기 = 피치 × 높이 × 깊이. 너비 0 등 경계. `cudaMemset3D`: 지정한 범위(너비×높이×깊이)만 채운다(피치 패딩과 범위 밖은 그대로). `cudaMemcpy3D`: 네 방향, `srcPos`·`dstPos` 오프셋, 피치와 `ysize`가 다른 원본·대상, 범위 초과는 `cudaErrorInvalidValue`, 배열 인자는 카드 `runtime-texture`의 시험 목록 3. `Async` 쪽은 스트림 순서를 따른다. 너비·높이·깊이 중 하나가 0이면 성공하고 아무것도 안 함.
7. `cudaDeviceGetLimit`/`SetLimit`: 설정한 값을 그대로 돌려준다(스택·printf FIFO·말록 힙). 모르는 열거값은 `cudaErrorInvalidValue`, 장치 런타임·L2 상주 한도는 `cudaErrorUnsupportedLimit`.
8. Cooperative Groups: `tiled_partition<N>(this_thread_block())`로 얻은 타일의 `meta_group_size()` = 블록 스레드 수 / N, `meta_group_rank()` = 블록 내 선형 스레드 번호 / N. `tiled_partition<M>(타일<N>)`에서는 N/M과 (타일 내 순번)/M. `thread_block_tile<N> t = tiled_partition<N>(block);`로 부모 없는 형에 담아도 같은 값. `cg::reduce`가 부모 있는 타일에서 동작.
9. 번역기: `struct S {int a; int b;} __align__(8);`가 CPU 경로에서 컴파일되고 `alignof(S) == 8`.
10. 응용 시험(자동화 후보): 위 세 스크립트를 그대로 돌려 각 묶음의 숫자(PolyBench 21/21/21/20, Parboil 11 빌드·9 실행·7 일치, llm.c 8걸음 손실 차이 그대로 0.0648·막은 복사본 0.000002)가 줄지 않을 것.

## 이의

(없음)

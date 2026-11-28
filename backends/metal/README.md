# Apple GPU (Metal) 백엔드 설계

상태: 백엔드 1차 구현이 Apple M3(macOS 15.5)에서 처음 빌드 그대로 통과(시험 53/53, 2026-11-13). 변환기(4장)는 2차 구현(주소 공간별 장치 함수, 템플릿, 모듈 변수, 구조체 인자 — Mac 확인 필요). Mac에서만 빌드된다. "확인 필요"는 실제 Mac에서 확인하기 전에는 사실로 쓰지 않는다.

## 1. 방침

- Apple이 공개한 Metal API(공개 문서)만 호출한다. Apple의 헤더는 macOS SDK에 있고 우리 저장소에 복사하지 않는다(엔비디아와 달리 SDK 헤더를 쓰는 것이 Apple의 공개 사용 방식이다).
- Apple은 GPU 기계어를 만드는 공개 컴파일러를 내놓지 않는다. 공개된 장치 코드 입력은 **Metal 셰이더 언어(MSL) 소스**뿐이고, 실행 중에 드라이버가 컴파일한다(`newLibraryWithSource`).
  그래서 이 백엔드의 장치 코드 이미지는 MSL 소스 문자열이다. CUDA 커널을 MSL로 바꾸는 일은 컴파일러 층(4장, M3)이 맡는다.
- 연결 층 규격(`backends/interface/ocu_backend.h`, ABI 2)을 그대로 구현한다. 위층은 바꾸지 않는다.

## 2. 메모리와 주소

Apple 칩은 CPU와 GPU가 같은 메모리를 쓴다. 그러나 같은 할당이라도 CPU 주소(`contents`)와 GPU 주소(`gpuAddress`, Metal 3)는 다르다.

- `mem_alloc`: `newBufferWithLength:options:MTLResourceStorageModeShared`. 돌려주는 "장치 포인터"는 **GPU 주소**다. CUDA와 같이 호스트는 이 값을 역참조하지 않고, 커널 인자로 그대로 넘긴다. 포인터 산술도 그대로 된다.
- 백엔드는 할당 표(GPU 주소 범위 → 버퍼, CPU 주소)를 가진다. 복사는 이 표로 CPU 주소를 찾아 `memcpy`한다(같은 메모리이므로 복사 명령이 필요 없다). 범위를 넘는 복사는 오류.
- 커널 실행 때 표의 모든 버퍼를 `useResources`로 GPU에 알린다(커널이 어떤 포인터를 쓸지 모르므로).
- `supports_unified_memory = 0`: 주소가 달라 호스트가 장치 포인터를 직접 쓸 수 없기 때문이다. 공유 메모리(`mem_alloc_shared`)는 두 주소가 같아야 하므로 1차 구현에서는 "지원 안 함".

## 3. 함수 대응

| 연결 층 | Metal | 메모 |
|---|---|---|
| `dev_count` / `dev_info` | `MTLCopyAllDevices` | 이름, `recommendedMaxWorkingSetSize`, `maxThreadsPerThreadgroup`, `maxThreadgroupMemoryLength`(Apple GPU는 32KB: 48KB를 가정한 CUDA 프로그램은 오류로 드러난다), 묶음 폭 32 |
| `queue_*` | `MTLCommandQueue` | 실행마다 명령 버퍼 하나를 만들어 넣는다. 동기화 = 마지막 명령 버퍼 `waitUntilCompleted` |
| `mem_copy` / `mem_set` | 대기열을 먼저 끝내고 CPU에서 `memcpy` / `memset` | 같은 메모리라 순서만 지키면 된다. 겹쳐 돌리기는 다음 단계 |
| `module_load` | `newLibraryWithSource:options:error:` | 이미지 = MSL 소스 |
| `module_get_kernel` | `newFunctionWithName:` + `newComputePipelineStateWithFunction:error:` | |
| `kernel_launch` | 계산 명령 + `dispatchThreadgroups:threadsPerThreadgroup:` | 인자 값들은 버퍼 0에 한 덩어리로, 각 인자의 위치표는 버퍼 1로 넘긴다(5장). 동적 공유 메모리는 `setThreadgroupMemoryLength:atIndex:0` |
| `marker_*` | 명령 버퍼 완료 시점 | 기록 = 그 대기열의 다음 완료 시각. 시간은 CPU 시계 기준(확인 필요: GPU 시각 `GPUEndTime` 사용 가능 여부) |
| `module_get_global` | 모듈마다 버퍼 하나(`mem_alloc`으로 할당, 할당 표에 들어감) | 이미지의 `global`/`init` 줄로 위치·크기·초기값. 커널 실행 때 버퍼 2로 묶는다 |
| `kernel_arg_sizes` | MSL 생성기가 이미지에 적은 인자 크기 | |

## 4. CUDA 커널 → MSL (M3, 컴파일러 층)

MSL은 C++14 기반이라 문장·식은 대부분 그대로 옮겨진다. 바꿔야 하는 것:

| CUDA | MSL |
|---|---|
| 포인터 `T *` (주소 공간 없음) | `device T *` / `threadgroup T *` / `constant T *`: 출처에 따라 정해야 한다 |
| `threadIdx` 등 내장 변수 | 커널 인자 `[[thread_position_in_threadgroup]]` 등. 장치 함수에는 문맥으로 넘긴다 |
| `__shared__` 변수 | 커널 함수 안의 `threadgroup` 변수(MSL은 장치 함수 안에 둘 수 없다) |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup \| mem_flags::mem_device)` |
| 원자 연산 | `atomic_fetch_add_explicit` 등(대상을 `device atomic_int *`로) |
| 워프 함수 | `simd_shuffle`, `simd_ballot` 등(묶음 폭 32) |
| `double` | Apple GPU에 없다. "지원 안 함"으로 컴파일 오류(조용히 float로 바꾸지 않는다) |

포인터 주소 공간은 글자만 보고는 정할 수 없다. 그래서 변환은 Clang이 만든 **형 정보가 붙은 구문 트리**(공개 컴파일러의 출력)를 읽어 한다.

### 4.1 변환기가 하는 일 (2차 구현)

- 입력: 번역 단위 전체의 장치 쪽 구문 트리(한 번). 포함된 사용자 헤더(시스템·런타임 폴더 밖)의 커널·장치 함수도 옮긴다.
- 장치 함수: 포인터·참조 매개변수가 불리는 주소 공간 조합마다 하나씩(오버로드). 포인터를 돌려주는 함수는 반환문이 가리키는 공간으로 반환형을 정한다.
- 지역 포인터·참조 선언과 포인터 형변환: 가리키는 대상에서 공간을 정한다(device, threadgroup, thread, constant). 한 선언에 공간이 다른 포인터가 섞이면 "나눠 쓰라"는 오류.
- 템플릿: 커널 템플릿의 인스턴스마다 MSL 커널 하나(이름은 구문 트리의 맹글 이름), 장치 함수 템플릿도 인스턴스마다 하나. 템플릿 인자는 함수 안의 typedef·상수로 넣는다.
- `__device__`/`__constant__` 변수: 모듈 버퍼 하나 안의 위치로 바꾼다(`ocu_ctx.glob + 위치`). 이미지 머리에 `global <이름> <위치> <크기>`, `init <위치> <16진 바이트>`. 크기·정렬은 CUDA 컴파일러(Clang 장치 단계)에게 물어서 정한다.
- 구조체·typedef·enum·상수: 장치 코드가 이름을 쓰는 것만 옮긴다(포인터 필드에는 device). 인자 구조체마다 `static_assert(sizeof == 호스트 크기)`.
- 매크로: 장치 단계 전처리기가 실제로 본 정의(-E -dD)만 옮긴다. 매크로 본문 안에서 고쳐야 하는 구문(장치 함수 호출, 변수 참조)은 오류, 매크로 인자 안은 고친다.
- 이름공간: 이름공간 안의 장치 함수는 파일 범위에 고유한 이름(`<이름>_ocu<번호>`)으로 내보내고 호출을 그 이름으로 고친다. 이름공간 안의 형·상수는 아직 옮기지 않는다(이름으로만 찾은 다른 항목처럼 경고 후 빼고, 그 이름을 쓰는 코드는 Metal 컴파일에서 멈춘다). 이름공간 안의 `__device__`/`__constant__` 변수는 오류.
- 함수형 템플릿 인자(`template <float (*op)(float, float)>`): 인스턴스마다 `op(...)` 호출을 인자로 준 장치 함수의 직접 호출로 고친다. 본문이 없는 함수가 인자이면 오류.
- `_ocu<번호>`로 끝나는 이름은 변환기가 쓴다. 프로그램의 파일·이름공간 범위 선언이 변환기가 만든 이름과 같으면 오류로 멈춘다(지역 이름은 검사하지 않는다).
- 장치 코드를 읽는 아키텍처(`__CUDA_ARCH__`)는 기본 sm_80이고 환경 변수 `OCU_METAL_ARCH`로 바꾼다. 아키텍처로 커널을 고르는 프로그램은 빌드에 쓴 값과 같게 준다.
- 옮기지 못하는 것(오류로 멈춤): double, 이름공간 안의 형·상수·장치 변수, 멤버 함수·클래스 템플릿, 장치 함수 안의 정적 `__shared__`, 포인터의 포인터, 함수 포인터, 재귀, 인라인 어셈블리, 텍스처. `--keep-going`은 그런 커널만 빼고 나머지를 옮긴다(빠진 커널은 stderr에 적힌다).
- 문법 검사: Metal 컴파일러가 없는 곳에서는 `tests/metal/msl_check.py`가 생성된 MSL을 C++ for OpenCL로 컴파일해 본다(주소 공간 규칙이 MSL과 같다). Rodinia CUDA 79개 커널 중 35개가 변환·검사를 통과(전에는 17개 변환, 검사 없음). 나머지 44개: double 6개, 이름공간 4개, 장치 단계 컴파일 자체가 안 되는 파일 34개(엔비디아 예제 헤더·텍스처·빠진 헤더).

## 5. 인자 전달

CUDA 커널 인자는 값의 나열이다(포인터는 GPU 주소 값). 호스트는 인자 값을 크기 순서대로, 각 값을 자기 크기의 2의 거듭제곱 약수(최대 16) 경계에 맞춰 한 덩어리에 쌓고, 각 인자의 시작 위치를 따로 넘긴다. 생성된 MSL 커널은 그 위치에서 제 형으로 읽는다. 구조체 인자는 C 배치 그대로 쌓이고, 생성된 MSL은 같은 구조체 정의로 읽는다. 스칼라·포인터(8바이트)·2·4성분 벡터만 든 구조체는 C와 MSL의 배치가 같다. 3성분 벡터(float3 등)는 MSL에서 16바이트라 배치가 달라지므로 구조체 필드에서는 오류, 커널 인자로는 `packed_float3`으로 읽는다. 크기는 `static_assert`로 Metal 컴파일 때 다시 확인한다.

## 6. 확인할 것 (Mac에서)

1. `gpuAddress`(macOS 13 이상, Apple7 이상 GPU)를 버퍼에서 읽은 포인터로 커널이 쓸 수 있는지 — **M3에서 확인됨**(saxpy, 할당 안쪽 주소에서 시작하는 포인터 산술). M1 Pro는 확인 필요
2. `useResources` 수천 개의 비용
3. 위 1·2가 안 되면 대안: 커널 인자를 버퍼 색인 + 오프셋으로 바꾸는 변환(포인터 산술은 오프셋 산술로)

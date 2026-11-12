# Apple GPU (Metal) 백엔드 설계

상태: 설계와 1차 구현(M2). Mac에서만 빌드된다. "확인 필요"는 실제 Mac에서 확인하기 전에는 사실로 쓰지 않는다.

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
| `module_get_global` | 1차 구현 없음 | `__constant__`·`__device__` 변수는 M3에서 버퍼로 바꾼다 |
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

## 5. 인자 전달

CUDA 커널 인자는 값의 나열이다(포인터는 GPU 주소 값). 호스트는 인자 값을 크기 순서대로, 각 값을 자기 크기의 2의 거듭제곱 약수(최대 16) 경계에 맞춰 한 덩어리에 쌓고, 각 인자의 시작 위치를 따로 넘긴다. 생성된 MSL 커널은 그 위치에서 제 형으로 읽는다. 구조체 배치 규칙이 C와 MSL에서 같은지에 기대지 않는다.

## 6. 확인할 것 (Mac에서)

1. `gpuAddress`(macOS 13 이상, Apple7 이상 GPU)를 버퍼에서 읽은 포인터로 커널이 쓸 수 있는지 — M1·M3 둘 다
2. `useResources` 수천 개의 비용
3. 위 1·2가 안 되면 대안: 커널 인자를 버퍼 색인 + 오프셋으로 바꾸는 변환(포인터 산술은 오프셋 산술로)

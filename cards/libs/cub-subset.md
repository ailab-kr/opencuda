---
id: libs-cub-subset
kind: component
name: CUB parallel primitives subset (cub/cub.cuh) and programmatic dependent launch declarations
spec_sources:
  - CUB API documentation (nvidia.github.io/cccl, cub API pages): DeviceReduce, DeviceSegmentedReduce, DeviceScan, DeviceRadixSort, DeviceSegmentedRadixSort, DeviceSegmentedSort, BlockScan, BlockLoad, BlockStore (signatures, two-phase temporary storage, stability, floating-point key order, bit ranges, data arrangements)
  - CUDA Runtime API reference (docs.nvidia.com/cuda/cuda-runtime-api): cudaLaunchKernelEx / cudaLaunchKernelExC, cudaLaunchConfig_t, cudaLaunchAttribute, cudaLaunchAttributeValue, enum cudaLaunchAttributeID
  - CUDA C++ Programming Guide (public): programmatic dependent launch (cudaGridDependencySynchronize, cudaTriggerProgrammaticLaunchCompletion), __syncthreads memory visibility
backend_map:
  cpu: header-only CUDA C++ (runtime/include/cub/cub.cuh), compiled with the program by the CPU path
  nvidia: the same header compiled with the program (Clang CUDA, PTX); no vendor library
state: impl-written-untested
owner_spec: AI Lab 2026-11-13
owner_tests: (to be assigned: a session other than owner_impl)
owner_impl: AI Lab 2026-11-13
cost: { hours: 1.5, ai_usage: 0 }
---

## 동작

llama.cpp(`0fb8f15`)의 CUDA 백엔드가 기본 프로필(`CUDART_VERSION` 12030)에서 쓰는 이름만 갖춘다.

PDL(프로그램 종속 실행):
- 형: `cudaLaunchConfig_t`(gridDim, blockDim, dynamicSmemBytes, stream, attrs, numAttrs), `cudaLaunchAttribute`(id, val), `cudaLaunchAttributeValue`(공용체), `cudaLaunchAttributeID`(0~14, 문서의 값), 보조 형(`cudaAccessPolicyWindow`, `cudaSynchronizationPolicy`, `cudaClusterSchedulingPolicy`, `cudaLaunchMemSyncDomain(Map)`).
- 호스트: `cudaLaunchKernelExC(config, func, args)`(C), `cudaLaunchKernelEx(config, kernel, args...)`(C++ 템플릿, 인자를 커널 매개변수 형으로 변환). 격자·블록·동적 공유 메모리·스트림을 config에서 가져와 `cudaLaunchKernel`과 같은 길로 실행한다.
- 속성: 성능 힌트(접근 정책, 동기화 정책, 우선순위, 메모리 동기 영역, 클러스터 스케줄 선호, 선호 클러스터 크기, 공유 메모리 비율)는 받아들이고 효과 없음. `ProgrammaticStreamSerialization`은 받아들이되 스트림 순서를 그대로 지킨다(속성은 앞 커널과 겹쳐 실행되는 것을 "허용"할 뿐이라, 완전한 순서도 허용된 실행 중 하나다). `ProgrammaticEvent`·`LaunchCompletionEvent`는 커널 뒤에 그 이벤트를 기록한다(문서가 말하는 시점보다 늦은 시점; 기다리는 쪽은 늦게 온 신호와 구별할 수 없다).
- 장치: `cudaGridDependencySynchronize()`, `cudaTriggerProgrammaticLaunchCompletion()`은 즉시 반환. 커널이 시작될 때 앞선 스트림 작업이 이미 끝나 있고 메모리가 보이므로(스트림 순서), 기다릴 것도 풀어 줄 것도 없다.

CUB(`namespace cub`, 헤더 하나):
- 연산자 `Sum`, `Max`, `Min`, `NullType`; 알고리즘 열거형 `BlockScanAlgorithm`, `BlockLoadAlgorithm`, `BlockStoreAlgorithm`(속도만 바꾸고 결과는 같다).
- `BlockScan<T, X, ALG, Y, Z>`: `TempStorage`, `InclusiveSum/ExclusiveSum/InclusiveScan/ExclusiveScan`(항목 하나, 블록 합계 출력 있는/없는 형, 스레드당 배열 형). 배열 형은 블록 배치(스레드 i가 i*N..i*N+N-1).
- `BlockLoad/BlockStore<T, X, N, ALG>`: `STRIPED`는 줄무늬 배치(i*블록스레드+t), 나머지는 블록 배치. 부분 적재(`valid_items`, 범위 밖 기본값), 부분 저장. 전치 알고리즘은 공유 메모리를 거쳐 읽고 쓴다.
- `DeviceSegmentedReduce::Sum/Max/Min/Reduce`(구간 i는 [begin[i], end[i]), end <= begin이면 빈 구간이라 초깃값. 구간마다 블록 하나라 아주 긴 구간 하나가 전체 시간을 정한다. ArgMin/ArgMax는 없음).
- `DeviceReduce::Sum/Max/Min/Reduce`, `DeviceScan::InclusiveSum/ExclusiveSum/InclusiveScan/ExclusiveScan`(제자리 가능).
  - 누적형(2026-11-28, 문서대로): `Sum`은 출력의 값 형에서 시작한다(출력에 값 형이 없으면 입력의 형). 누적기는 (초깃값, 입력)에 연산자를 적용한 형이다. 그래서 bool 반복자를 int64_t로 더하면 개수가 되고(전에는 입력 형 bool로 누적해 1이 됐다), float을 double로 더하면 double로 누적한다. `Max`·`Min`은 그대로 입력 형.
- `KeyValuePair<Key, Value>`(사소한 기본 생성자라 `__shared__`에 둘 수 있음, `operator!=`), `ArgMax`·`ArgMin`(큰·작은 값, 같으면 낮은 키): `BlockReduce<KeyValuePair, ...>`와 함께(HeCBench moe-cuda).
- 입력 반복자 `CountingInputIterator<T, OffsetT>`, `TransformInputIterator<V, Op, InIt, OffsetT>`(무작위 접근, 호스트와 장치 코드, `cub/iterator/*.cuh` 경로로도 포함 가능).
- `DeviceRadixSort::SortPairs(Descending)/SortKeys(Descending)`: 안정 정렬, 비트 범위 [begin_bit, end_bit).
- `DeviceSegmentedRadixSort::SortPairs(Descending)/SortKeys(Descending)`, `DeviceSegmentedSort::(Stable)SortPairs(Descending)/(Stable)SortKeys(Descending)`: 구간마다 블록 하나, 안정 정렬.
- 장치 전체 호출의 공통 규약: `d_temp_storage == nullptr`이면 `temp_storage_bytes`만 쓰고 아무것도 실행하지 않는다(값은 항상 1 이상). 그 밖에는 `stream`에 작업을 넣고 실행 상태를 돌려준다. 동기화하지 않는다.

## 경계 조건

- 항목 0개: Reduce는 초기값(Sum은 0, Max는 `numeric_limits::lowest()`, Min은 `max()`)을 쓴다. Scan·Sort는 아무것도 쓰지 않는다.
- 실수 키: 음수는 모든 비트 반전, 양수는 부호 비트를 켠 정수 순서. -0.0과 +0.0은 같은 키(입력 순서 유지). NaN은 변환된 비트 순서대로. 내림차순은 변환된 키를 반전해 오름차순 안정 정렬(같은 키는 입력 순서 유지). 비트 범위는 반전까지 한 뒤의 비트에 적용.
- 구간 정렬: `begin >= end`인 구간은 빈 구간. 어느 구간에도 들지 않는 항목은 읽지도 쓰지도 않는다.
- 입력과 출력 범위가 겹치면 안 됨(정렬). `begin_bit == end_bit`면 그대로 복사.
- 요소 형은 사소하게 복사 가능해야 한다. BlockScan의 임시 저장소는 값 2*블록스레드 개.
- CCCL 판 매크로(`CCCL_MAJOR_VERSION` 등)는 정의하지 않는다: llama.cpp의 `DeviceTopK`, `cuda::make_strided_iterator` 경로는 꺼진 채로 남는다(대체 경로가 있다).

## 오류 코드

- `cudaLaunchKernelEx(C)`: config가 NULL, 또는 numAttrs > 0인데 attrs가 NULL → `cudaErrorInvalidValue`. 모르는 속성 id, `DeviceUpdatableKernelNode`(그래프 전용) → `cudaErrorInvalidValue`. `Cooperative` ≠ 0 → `cudaErrorNotSupported`(`cudaLaunchCooperativeKernel`과 같음). `ClusterDimension` x*y*z > 1 → `cudaErrorNotSupported`. 이벤트 속성의 이벤트가 NULL → `cudaErrorInvalidResourceHandle`. 실행 오류는 `cudaLaunchKernel`과 같다.
- CUB: 주어진 `temp_storage_bytes`가 질의 값보다 작으면 `cudaErrorInvalidValue`. 비트 범위가 키 폭 밖이거나 begin > end → `cudaErrorInvalidValue`. 커널 실행 실패는 그 오류(`cudaPeekAtLastError`).

## 시험 목록

시험 작성자(구현자와 다른 세션)가 확인할 동작. 근거: CUB API 문서의 각 클래스 쪽, CUDA Runtime API의 cudaLaunchKernelEx·cudaLaunchAttributeID 항목.

1. PDL 헤더: C와 C++에서 `cudaLaunchConfig_t cfg = {}`, `cudaLaunchAttribute a; a.id = cudaLaunchAttributeProgrammaticStreamSerialization; a.val.programmaticStreamSerializationAllowed = 1;`이 컴파일. 열거값이 문서와 같음(Ignore 0 … ClusterDimension 4, ProgrammaticStreamSerialization 6, ProgrammaticEvent 7, Priority 8, LaunchCompletionEvent 12). `val.clusterDim.x/y/z`가 있음.
2. `cudaLaunchKernelEx(&cfg, kernel, args...)`: 격자·블록·동적 공유 메모리·스트림이 반영됨(결과와 `blockDim`/`gridDim`/동적 공유 메모리 크기로 확인), 인자 형 변환(int → float 등), 인자 없는 커널, CPU·NVIDIA 두 경로.
3. 같은 스트림에 PDL 속성으로 생산자 → 소비자 커널(소비자가 `cudaGridDependencySynchronize()` 뒤 생산자 결과를 읽음)을 여러 번 쌓아도 결과가 항상 맞음.
4. 오류: NULL config, attrs NULL + numAttrs 1, 모르는 id(예: 999), Cooperative=1, ClusterDimension(2,1,1) → 위 오류 코드. Ignore·Priority·AccessPolicyWindow 속성은 성공. `cudaLaunchKernelExC`(void** 형)도 NVIDIA 경로에서 같은 결과.
5. ProgrammaticEvent: 실행 뒤 그 이벤트를 `cudaEventSynchronize`하면 커널 결과가 보임.
6. `cudaGridDependencySynchronize()`·`cudaTriggerProgrammaticLaunchCompletion()`이 커널 안에서 컴파일·즉시 반환.
7. 장치 전체 호출의 질의: `d_temp_storage == nullptr`이면 출력이 바뀌지 않고(미리 채운 값 그대로) `temp_storage_bytes > 0`; 같은 인자로 두 번 질의하면 같은 값; 질의 값보다 작은 크기 → `cudaErrorInvalidValue`.
8. `DeviceReduce::Sum`: n = 0(출력 0), 1, 255, 256, 257, 2047, 2048, 2049, 10^5, 10^6; 정수와 실수(정확히 표현되는 값으로 순서 무관 비교); `Max`/`Min`/`Reduce(op, init)`(init이 결과에 한 번 들어감, 빈 입력이면 init).
9. `DeviceScan::InclusiveSum/ExclusiveSum`: 같은 크기들, 제자리(`d_in == d_out`), 정수 값 정확 일치; `ExclusiveScan(op, init)` 첫 출력 = init; 비가환 연산(예: 2x2 행렬 곱 구조체)으로 순서 확인.
10. `DeviceRadixSort::SortPairs`/`SortPairsDescending`/`SortKeys`: n = 1, 1023, 1024, 1025, 5000, 2*10^5; 같은 키가 많은 입력에서 값(원래 위치)이 입력 순서대로(안정성); -0.0/+0.0이 같은 키로 입력 순서 유지; 음수·양수·무한대·NaN 섞인 순서; int32·uint32·int64·double 키; `begin_bit`/`end_bit` 일부 범위(예: 0~8)에서는 그 비트만으로 안정 정렬.
11. `DeviceSegmentedRadixSort`·`DeviceSegmentedSort`: 연속 구간(offsets, offsets+1), 빈 구간, 구간 사이 틈(틈의 출력은 바뀌지 않음), 구간 수 1과 많음(예: 1000), 오름·내림.
12. `BlockScan<int, 128>`(및 `<float, 256>`, 2차원 블록 `X=16, Y=8`): 포함·배타 합과 블록 합계, 연산자 형, 스레드당 배열 형(블록 배치). 같은 임시 저장소를 `__syncthreads()` 뒤 다시 씀.
13. `BlockLoad/BlockStore<float, 128, 16, *_WARP_TRANSPOSE>`(llama.cpp ssm-scan 형태): 결과가 블록 배치, 두 TempStorage를 공용체로 묶어 씀; `BLOCK_LOAD_STRIPED`는 줄무늬 배치; 부분 적재(valid_items)와 범위 밖 기본값.
14. 응용: llama.cpp 기본 프로필 빌드에서 test-backend-ops의 ARGSORT, TOP_K, CUMSUM, SUM, MEAN, SSM_SCAN이 CPU 기준과 일치(아래 메모의 값보다 줄지 않음).

## 메모

- 모든 알고리즘은 헤더 안의 평범한 CUDA C++ 커널이다. 프로그램과 함께 대상마다 컴파일되므로 `libs/`의 정적 라이브러리가 필요 없다(CUB 자체도 헤더 라이브러리다). 칩 이름으로 갈리는 코드는 없다.
- 구현 방식: 블록 스캔은 공유 메모리 두 버퍼 Hillis-Steele; 장치 전체 합은 8192항목까지 한 블록 한 번 실행, 그 위로는 블록별 부분 합(최대 256블록) + 한 블록 마무리; 스캔은 4타일(8192항목)까지 한 블록이 타일을 차례로 돌고, 그 위로는 타일(2048항목) 합계 → 합계 스캔(한 블록) → 타일 스캔; 기수 정렬은 4비트씩 LSD, 타일(1024항목) 안의 안정 순위를 16개 16비트 계수기 묶음의 블록 스캔으로 구함; 구간 정렬은 구간마다 블록 하나가 모든 자릿수 단계를 돈다.
- 성능은 정확성 다음이다. 큰 단일 구간 정렬은 단계마다 커널 3번(히스토그램, 스캔, 흩뿌리기), 32비트 키에 8단계.
- 구현자의 자체 점검(저장소 밖, 시험 작성자 몫과 별개): 위 시험 목록 2·4·8~13의 일부를 한 프로그램으로 CPU·NVIDIA 경로에서 돌려 모두 일치. llama.cpp 결과와 속도는 `cards/apps/llama-cpp.md` 메모.
- 알려진 느린 곳: 32768항목 정도의 스캔을 행마다 부르면(llama.cpp CUMSUM 32768×4) 커널 3번 실행 비용이 지배해 대체 경로보다 약 4배 느리다. 한 번에 끝내는 스캔이 개선 후보.

## 이의

(없음)

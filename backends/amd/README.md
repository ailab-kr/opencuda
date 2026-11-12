# AMD GPU 백엔드 설계

상태: 설계만 있다. 장비(PLAN의 D-01)가 들어오면 이 문서대로 구현한다. "확인 필요" 표시는 실제 설치본에서 확인하기 전에는 사실로 쓰지 않는다는 뜻이다.

## 1. 방침

- AMD가 공개한 런타임(HIP 런타임, ROCm에 포함, MIT 라이선스)을 **호출만** 한다. 그 소스나 헤더를 저장소에 복사하지 않는다.
- 연결 층 규격(`backends/interface/ocu_backend.h`)의 함수 26개를 HIP 런타임 함수에 대응시킨다. 위층(런타임·라이브러리)은 바꾸지 않는다.
- 장치 코드는 공개 컴파일러(Clang)가 HIP 모드로 만든 코드 객체를 쓴다. 이 저장소의 컴파일러 실험에서 gfx1100(RDNA3)·gfx1201(RDNA4)용 코드 객체가 이미 만들어졌다(`docs/compiler-probe.md`).
- 행렬 연산은 AMD의 공개 BLAS 라이브러리(rocBLAS, MIT 라이선스)를 BLAS 제공자 뒤에 둔다(`backends/interface/ocu_blas_provider.h`).

## 2. 함수 대응

| 연결 층 | HIP 런타임 | 메모 |
|---|---|---|
| `dev_count` | `hipGetDeviceCount` | |
| `dev_info` | `hipGetDeviceProperties` | 이름, 전체 메모리, `warpSize`(제품에 따라 32 또는 64), 블록·격자 한도, 블록당 공유 메모리 |
| `dev_open` / `dev_close` | `hipSetDevice` + 내부 대기열 하나 생성 / 해제 | 내부 대기열은 `queue == NULL` 동기 실행용 |
| `mem_alloc` / `mem_free` | `hipMalloc` / `hipDeviceSynchronize` 후 `hipFree` | 규격: 해제는 대기 중인 작업을 먼저 기다린다. `hipFree`의 암묵적 동기화 여부는 확인 필요. 확인 전에는 명시적으로 기다린다 |
| `mem_copy` | `hipMemcpyAsync` + (대기열 없으면) `hipStreamSynchronize` | 방향 상수 대응 |
| `mem_set` | `hipMemsetAsync` | |
| `mem_info` | `hipMemGetInfo` | |
| `host_alloc_pinned` / `host_free_pinned` | `hipHostMalloc` / 동기화 후 `hipHostFree` | |
| `queue_create` / `queue_destroy` | `hipStreamCreateWithFlags`(non-blocking) / 동기화 후 `hipStreamDestroy` | 규격: 해제는 남은 일을 끝낸 뒤 |
| `queue_sync` / `queue_query` | `hipStreamSynchronize` / `hipStreamQuery` | `hipErrorNotReady` → `OCU_ERR_NOT_READY` |
| `module_load` / `module_unload` | `hipModuleLoadData` / `hipModuleUnload` | 이미지 = Clang이 만든 코드 객체(ELF). 여러 칩용을 한 파일에 넣는 형식은 컴파일러 쪽에서 정한다 |
| `module_get_kernel` | `hipModuleGetFunction` | 없으면 `OCU_ERR_NOT_FOUND` |
| `kernel_launch` | `hipModuleLaunchKernel(f, grid, block, shared, stream, args, NULL)` | 우리 `args` 배열(인자 값마다 포인터 하나)이 이 함수의 인자 형식과 같다. 인자 값을 실행 시점에 복사하는지는 확인 필요 |
| `marker_*` | `hipEventCreate`, `hipEventRecord`, `hipEventSynchronize`, `hipEventQuery`, `hipEventElapsedTime`, `hipEventDestroy` | 기록한 적 없는 표식의 조회·대기 결과가 규격(완료로 취급)과 같은지 확인 필요. 다르면 백엔드가 상태를 따로 기억한다 |
| `get_provider("blas")` | rocBLAS 래퍼 | 아래 4장 |

## 3. 규격과 다를 수 있는 곳

CPU 백엔드에서 통과한 시험(`tests/backend/test_cpu_backend.c`, `test_cpu_regressions.c`)을 AMD 백엔드에 그대로 돌려 차이를 찾는다. 미리 예상하는 차이는 다음과 같다.

| 규격 | 예상 | 대응 |
|---|---|---|
| 해제는 대기 작업을 기다린다 | 런타임마다 다름 | 해제 전에 장치 동기화 |
| 대기열 오류는 다음 동기화·조회에서 한 번 알리고 지운다 | HIP의 오류는 "끈끈한" 종류가 있다 | 오류 종류별로 대응표를 만든다 |
| 커널 안에서 백엔드 호출은 "지원 안 함" | GPU에서는 커널이 호스트 함수를 부를 수 없으므로 해당 없음 | 없음 |
| 묶음 단위 크기는 실제 값 | `warpSize`가 제품에 따라 32 또는 64 | 그대로 보고 |
| 인자 합계 32768바이트 이하 | HIP의 상한은 확인 필요 | 작은 쪽을 규격으로 |

## 4. BLAS 제공자

- 대기열마다 rocBLAS 핸들 하나를 만들고 `rocblas_set_stream`으로 묶는다.
- 스칼라 결과(dot, nrm2, asum, iamax)는 장치 메모리로 받아야 하므로 포인터 모드를 장치 쪽으로 둔다.
- 음수 증가폭, `beta == 0`일 때 출력을 읽지 않는 규칙, iamax의 1부터 세기가 BLAS 표준과 같은지는 기준 사례 526건(`conformance_blas_*`)으로 확인한다. 다르면 제공자 안에서 맞춘다.
- 속도 기준선은 같은 rocBLAS를 직접 부른 시간이다. 우리 층을 거친 시간과의 비율이 PLAN의 속도 지표다.

## 5. 장비가 들어온 날 할 일

1. 설치 기록: 운영체제, 커널, 드라이버, ROCm 버전, 지원 목록에 그 GPU가 있는지 (`docs/setup.md`)
2. `hipGetDeviceProperties`로 장치 정보와 `warpSize` 확인
3. 컴파일러 실험의 gfx 코드 객체를 `hipModuleLoadData`로 올려 `saxpy` 실행
4. 2장의 "확인 필요" 항목을 하나씩 작은 프로그램으로 확인하고 이 문서를 고침
5. 백엔드 구현 후 CPU 백엔드 시험 묶음 전체와 BLAS 기준 사례를 AMD에서 실행

## 6. 미리 알아 둘 위험

- WSL에서 쓸 수 있는 AMD GPU와 ROCm 버전의 범위는 리눅스보다 좁다. 확인 필요. 리눅스 전용 PC가 더 안전하다.
- 소비자용 RDNA 제품의 ROCm 지원은 데이터센터용보다 늦게 들어오는 경우가 있다. 장비 결정(D-01) 전에 공식 지원 목록을 본다.
- 실행 시간 측정은 준비 실행과 반복 측정 규칙(중앙값과 느린 10%)을 따른다. 첫 실행에는 코드 적재와 컴파일 시간이 섞인다.

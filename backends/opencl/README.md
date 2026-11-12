# OpenCL 백엔드 설계

상태: 1차 구현(2026-11-13). CPU용 OpenCL 구현(PoCL 6.0, SPIR-V 입력)에서 시험 통과. GPU 드라이버에서는 아직 확인하지 않았다.

## 1. 방침

- Khronos가 공개한 OpenCL API(3.0 규격)만 부른다. 라이브러리는 실행 중에 찾는다(`dlopen("libOpenCL.so.1")`, 다른 파일은 `OCU_OPENCL_LIBRARY`). 빌드에는 OpenCL SDK가 필요 없다: 쓰는 함수와 상수만 `ocl_api.h`에 규격을 보고 적었다.
- OpenCL은 인텔 GPU, AMD(ROCm·rusticl), Arm Mali·Adreno, OpenCL 드라이버가 있는 가속기에 닿는다. 칩마다 다른 것은 드라이버가 맡고, 이 백엔드는 OpenCL 2.0 이상의 공통 기능만 쓴다.
- 연결 층 규격(`backends/interface/ocu_backend.h`, ABI 2)을 그대로 구현한다.

## 2. 포인터 모델

CUDA 커널은 장치 포인터를 값으로 받아 더하고, 메모리에 저장하고, 구조체에 넣어 넘긴다. OpenCL의 보통 버퍼(`cl_mem`)는 커널 밖에서 주소가 없으므로 이걸 할 수 없다.

| 방법 | 장점 | 단점 | 선택 |
|---|---|---|---|
| 거친 공유 가상 메모리(SVM, OpenCL 2.0 핵심 기능) | 주소가 커널과 호스트 호출에서 같다. 덧셈·저장·구조체 모두 그대로 | OpenCL 3.0에서는 선택 기능. SVM 없는 드라이버(일부 모바일)에서는 못 쓴다 | **사용** |
| `cl_ext_buffer_device_address`, 인텔 USM | 장치 주소를 직접 얻는다 | 확장이라 드라이버마다 다르다 | 다음 단계(SVM 없는 장치용) |
| 큰 버퍼 하나 + 오프셋 | OpenCL 1.2에서도 된다 | 포인터를 정수 오프셋으로 바꾸는 변환이 커널 전체에 필요하고, 할당 크기가 미리 정해진다 | 하지 않음 |

- `mem_alloc`: `clSVMAlloc`(거친 단위). 드라이버가 자기 최대 자료형보다 큰 정렬을 거부할 수 있어서, 256바이트 더 크게 잡고 안에서 256바이트 경계로 올린 주소를 돌려준다(CUDA 할당의 정렬).
- 백엔드는 할당 표를 가진다. 복사·채우기는 범위를 표로 확인하고 `clEnqueueSVMMemcpy`/`clEnqueueSVMMemFill`. 범위를 넘으면 오류.
- 커널 실행마다 표의 모든 할당을 `CL_KERNEL_EXEC_INFO_SVM_PTRS`로 알린다(인자 덩어리나 메모리에서 읽은 포인터도 쓸 수 있게). Metal의 `useResources`와 같은 이유.
- `mem_alloc_shared`: 고운 단위 SVM 버퍼가 있으면 그것(호스트와 장치가 같은 주소를 직접 쓴다), 없으면 "지원 안 함".
- `supports_unified_memory = 0`: 거친 단위 SVM은 호스트가 직접 읽고 쓰면 안 된다.
- 장치 목록에는 OpenCL 2.0 이상이고 거친 단위 SVM이 있으며 SPIR-V나 C++ for OpenCL 소스를 받는 장치만 넣는다. 빠진 장치와 이유는 `OCU_OPENCL_VERBOSE=1`로 본다.

## 3. 함수 대응

| 연결 층 | OpenCL | 메모 |
|---|---|---|
| `dev_count` / `dev_info` | `clGetPlatformIDs`, `clGetDeviceIDs`, `clGetDeviceInfo` | 묶음 크기는 작은 시험 커널의 `get_max_sub_group_size()`(하위 묶음이 없으면 1). PoCL CPU는 블록 전체(256개 블록에서 256)를 하위 묶음으로 알린다 |
| `queue_*` | 순서대로 도는 명령 대기열(측정 켬) | 마지막 명령의 사건을 기억해 조회·동기화에 쓴다 |
| `mem_free` | 장치의 모든 대기열 `clFinish` 뒤 `clSVMFree` | 규격의 "해제는 남은 일을 기다린다" |
| `module_load` | SPIR-V면 `clCreateProgramWithIL`, 아니면 C++ for OpenCL 소스(`-cl-std=CLC++2021`, `cl_ext_cxx_for_opencl` 장치), 아니면 OpenCL C 소스 | 장치가 받는 형태가 이미지에 없으면 "지원 안 함". 빌드 실패 기록은 `OCU_OPENCL_VERBOSE=1` |
| `module_get_kernel` | `clCreateKernel` | 이름이 없으면 `NOT_FOUND`. 인자 4개면 변수 블록을 받는 커널 |
| `kernel_launch` | 인자 덩어리·위치표를 읽기 전용 버퍼로, 동적 공유 메모리는 `__local` 인자, `clEnqueueNDRangeKernel`(3차원) | 블록 크기·공유 메모리 한도를 먼저 확인 |
| `marker_*` | `clEnqueueMarkerWithWaitList`, 사건 측정 정보(끝 시각) | 드라이버가 표식 시각을 주지 않으면 `elapsed_ms`는 "지원 안 함" |
| `module_get_global` | 모듈 변수 블록(SVM 하나) + 위치 | 이미지의 `global`·`init` 줄로 채운다 |
| `kernel_arg_sizes` | 이미지의 `param` 줄 | |

## 4. 인자 전달과 이미지

인자 전달은 Metal 백엔드와 같다: 값들을 크기 순서대로, 각 값을 자기 크기의 2의 거듭제곱 약수(최대 16) 경계에 맞춰 한 덩어리에 쌓고 위치를 따로 넘긴다. 커널 머리는

```
kernel(constant uchar *ocu_args, constant uint *ocu_offs, [device uchar *ocu_glob,] threadgroup uchar *ocu_dyn)
```

이미지(`compiler/opencl/cu2ocl.py`가 만든다): `OCUOCL1`, 커널마다 `param`, 변수마다 `global`·`init`, 길이 줄 `source`·`spirv`·`clc`, `---`, 그 순서의 내용.

## 5. 장치 코드 경로 (`compiler/opencl/cu2ocl.py`)

- 고른 길: **cu2msl의 변환을 그대로 쓰고 C++ for OpenCL로 낸 뒤 SPIR-V로**. 이 PC의 Clang 18(우분투)은 SPIR-V 백엔드가 없고, CUDA 소스를 SPIR-V로 바로 내는 경로도 없다(`clang -print-targets`에 spirv 없음). 반면 OpenCL의 주소 공간(global·local·constant·private)은 MSL과 같은 모양이라 cu2msl의 주소 공간 분석을 그대로 쓸 수 있다. 다른 것은 커널 머리, 인덱스(`get_local_id` 등), 3성분 벡터 인자(CUDA 12바이트, OpenCL 16바이트), double 허용, 머리말(`ocl_prelude.h`)뿐이라 cu2msl에 걸이(hook)를 두고 하위 클래스로 만들었다.
- 생성한 소스는 변환 때 Clang으로 C++ for OpenCL 2021(spir64)로 컴파일해 본다(오류면 변환 실패). `llvm-spirv`(Khronos 공개 번역기)가 있으면 SPIR-V로 바꿔 이미지에 넣는다.
- 워프 함수: 하위 묶음 폭이 장치마다 다르다(PoCL CPU는 블록 전체). 폭 32를 보장할 수 없어 머리말에 넣지 않았다: 쓰는 코드는 변환 단계에서 "컴파일 안 됨"으로 거부된다(틀린 레인 수로 돌지 않게).
- 전체 프로그램 빌드: `ocu_cc.py --target opencl`이 cu2ocl 이미지를 컨테이너에 넣고 Clang CUDA 호스트 패스로 호스트 객체를 만든다. 실행 API(`<<<>>>`, `cudaMalloc` …)가 그대로 OpenCL 백엔드로 간다(`OCU_BACKEND=opencl`).

## 6. 확인할 것

1. GPU 드라이버(인텔 NEO, Mesa rusticl, ROCm)에서 SPIR-V 입력과 SVM
2. SVM 없는 장치: `cl_ext_buffer_device_address` 경로
3. 하위 묶음 폭을 고정할 수 있는 장치(`cl_intel_required_subgroup_size` 등)에서 워프 함수

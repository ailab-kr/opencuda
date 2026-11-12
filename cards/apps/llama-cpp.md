---
id: apps-llama-cpp
kind: app
name: llama.cpp CUDA backend (ggml-cuda), built unmodified with ocu-nvcc
spec_sources:
  - public nvcc documentation (command line, predefined macros __CUDA_ARCH__ / __CUDA_ARCH_LIST__)
  - public CUDA C++ Programming Guide (execution configuration, warp shuffle functions, alignment specifiers)
  - public CUDA Math API reference (FP8 intrinsics, single precision intrinsics)
  - public cuBLAS / cuBLASLt documentation (cublasLt objects and cublasLtMatmul)
  - docs/llama-cpp-requirements.md (survey), docs/compiler-driver.md
backend_map:
  cpu: ocu-nvcc --ocu-target=cpu (CPU rewrite) over backends/cpu
  nvidia: ocu-nvcc --ocu-target=nvidia (Clang CUDA, sm_50 PTX) over backends/nvidia
state: impl-written-untested
owner_spec: AI Lab 2026-11-13
owner_tests: (to be assigned: a session other than owner_impl)
owner_impl: AI Lab 2026-11-13
cost: { hours: 4.5, ai_usage: 0 }
---

## 동작

llama.cpp(`0fb8f15`, MIT)의 CUDA 백엔드를 소스 수정 없이 우리 도구 사슬로 빌드해 실행한다.
llama.cpp 저장소 밖 빌드 폴더에서 CMake 도구 사슬 파일 하나로 구성한다.

```
cmake -S llama.cpp -B build -DCMAKE_TOOLCHAIN_FILE=<toolkit>/cmake/OcuCUDA.cmake -DOCU_TARGET=cpu|nvidia \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=50 \
  -DGGML_CUDA_FA=OFF -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_NO_PEER_COPY=ON -DGGML_CUDA_NCCL=OFF \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_CCACHE=OFF \
  "-DCMAKE_CUDA_FLAGS=-DOCU_CUDART_VERSION=11060"
```

- `OCU_CUDART_VERSION=11060`: llama.cpp가 `CUDART_VERSION`으로 켜는 PDL(`cudaLaunchKernelEx`), FP8, CUB, cublasLt 경로를 끈다(조사 문서 6.1). 헤더의 함수는 그대로다. 12030(기본)으로는 아직 시도하지 않았다(CUB 부분집합, PDL 선언이 필요).
- 아키텍처 50: 장치가 보고하는 계산 능력 5.0과 `__CUDA_ARCH_LIST__=500`이 맞물려, 호스트가 고르는 커널과 장치 코드에 든 본문이 일치한다. 텐서 코어·cp.async PTX는 컴파일되지 않는다. 남는 인라인 PTX `prefetch.global.L2`도 이 llama.cpp 판에서는 부르는 곳(`mmvq.cu`, `mmq-few.cu`)이 특정 아키텍처(`__CUDA_ARCH__ ==` DGX Spark) 조건 안에 있어, 아키텍처 50에서는 어느 경로에도 들어가지 않는다.
- 이 작업에서 고친 것: CPU 경로 실행 람다가 복사 캡처(`[=]`)라 함수 포인터 변수·템플릿 인자로 실행 가능, `__restrict` 포인터 인자 복사, `__shared__ alignas(N)`·`const alignas(N)` 순서, CPU 경로에서 `__CUDA_ARCH__=500`·`__CUDA_ARCH_LIST__=500`(NO_DEVICE_CODE가 `__trap`으로 멈춤), NVIDIA 경로의 `__CUDA_ARCH_LIST__`, NVPTX 1·2바이트 셔플, `cuda_fp8.h`, `__half2_raw`, `<limits.h>`, `__frcp_rn` 등, `cublasLt.h`(객체만, 행렬곱은 지원 안 함), 툴킷 include의 `cooperative_groups/` 폴더.

## 경계 조건

- 계산 능력 6.1 미만 등급에서 llama.cpp는 비연속(뷰·치환) K-양자·IQ-양자 행렬의 곱을 cuBLAS로 보내는데, 그 형식의 비연속 역양자화 함수가 없어 `GGML_ASSERT(convert_func != nullptr)`로 멈춘다(llama.cpp 쪽 한계, 우리 두 경로에서 같음). 모델 추론(연속 가중치)에서는 생기지 않는다. test-backend-ops 측정에서는 이 경우를 `-p` 정규식으로 뺐다.
- NVIDIA 경로의 첫 실행은 sm_50 PTX를 드라이버가 GPU용으로 번역하느라 느리다(첫 프롬프트 약 25초). 두 번째부터는 드라이버의 번역 캐시를 쓴다.
- FA(flash attention), 그래프, VMM, 피어 복사, NCCL은 꺼 둔 상태로만 확인했다.

## 오류 코드

- 지원하지 않는 cublasLt 행렬곱: `CUBLAS_STATUS_NOT_SUPPORTED`(알고리즘 0개).
- 잘못 고른 커널(장치 코드 없음): CPU 경로는 `__trap()` 메시지 후 프로세스 중단, NVIDIA 경로는 실행 실패.

## 시험 목록

시험 작성자(구현자와 다른 세션)가 확인할 동작:

1. 실행 문법: 템플릿 함수 안에서 함수 포인터 매개변수로 `kernel<<<g, b, s, st>>>(args...)`(인자 전달은 `std::forward`), 지역 변수 `auto k = cond ? k1 : k2; k<<<...>>>()`, 이름 있는 커널과 같은 결과. 캡처된 포인터가 비동기 실행에서도 맞는 커널을 부름(스트림에 여러 실행을 쌓은 뒤 동기화).
2. `T *__restrict p`를 받아 다시 실행 인자로 넘기는 함수(포인터·구조체 포인터)가 CPU 경로에서 컴파일·실행됨.
3. `__shared__ __align__(16) T a[n]`, `__shared__ alignas(16) T a[n][m]`, `const alignas(16) T z[n] = {...}`(지역), `extern __shared__ __align__(128) unsigned char b[]`: CPU 경로에서 컴파일되고 주소가 정렬 조건을 만족.
4. CPU 경로 `.cu`에서 `__CUDA_ARCH__ == 500`, `__CUDA_ARCH_LIST__ == 500`; `-D__CUDA_ARCH__=700`을 주면 그 값(경고 없이), `-U__CUDA_ARCH__`면 정의 안 됨. `.cpp`·`.c`(호스트 전용)에서는 정의 안 됨. NVIDIA 경로 `-arch=sm_50`이면 두 단계 모두 `__CUDA_ARCH_LIST__ == 500`.
5. NO_DEVICE_CODE 패턴(`#ifdef __CUDA_ARCH__`면 printf 후 `__trap()`): CPU 경로에서 그 커널을 실행하면 프로세스가 비정상 종료(별도 프로세스로 시험).
6. NVPTX 셔플: `unsigned char`, `signed char`, `short`, `__half`를 `__shfl_sync`/`_up`/`_down`/`_xor`로 옮긴 값이 CPU 경로와 같음.
7. `cuda_fp8.h`: E4M3·E5M2 각 256개 비트 → double → 비트 왕복(NaN 제외), 0x38 = 1.0, 448 → 0x7e, 464 → 0x7e(짝수 쪽), 480 → NOSAT 0x7f / SATFINITE 0x7e, E5M2 61440 → NOSAT 0x7c / SATFINITE 0x7b, 부호 있는 0, 무한대 입력(SATFINITE는 최대 유한값), NaN → 0x7f, 가장 작은 비정규수 근처의 반올림(0.75·2^-9 → 0x01), double 입력이 float를 거치지 않음(동률 바로 옆 double). 클래스의 생성자(정수 포함)·변환 연산자, x2·x4 형의 바이트 순서(.x가 하위). 커널 안에서 같은 비트.
8. `OCU_CUDART_VERSION=11060`이면 `CUDART_VERSION == 11060`, 없으면 12030. `<cuda_runtime.h>`만으로 `INT_MAX` 사용 가능.
9. `cublasLt.h`: C와 C++에서 포함됨, Create/Destroy/SetAttribute 성공, NULL 인자는 `CUBLAS_STATUS_INVALID_VALUE`, `cublasLtMatmulAlgoGetHeuristic`은 결과 0개와 NOT_SUPPORTED, `cublasLtMatmul`은 NOT_SUPPORTED.
10. 툴킷 폴더에 `include/cooperative_groups/reduce.h`가 있고 `-I<toolkit>/include`만으로 포함됨.
11. 응용 시험(자동화 후보): 위 구성으로 llama.cpp를 CPU·NVIDIA 대상 각각 빌드(모든 `.cu` 컴파일·링크), Qwen2.5-0.5B-Instruct Q8_0로 탐욕 생성 64토큰이 의미 있는 문장, CPU 경로와 NVIDIA 경로의 토큰 일치율, test-backend-ops 통과 수(아래 메모의 값보다 줄지 않음).

## 메모

측정(2026-11-13, WSL2, RTX 5070 Ti, Qwen2.5-0.5B-Instruct Q8_0, 프롬프트 27토큰 + 64토큰 탐욕 생성):

- 생성: CPU 경로·NVIDIA 경로 모두 문장이 자연스럽다. 기준(llama.cpp 자체 CPU 백엔드)과는 7번째 토큰에서 갈라진다("light and history" ↔ "light and culture"); 같은 위치 일치 47/64(CPU 경로), 48/64(NVIDIA 경로). 우리 CPU 경로와 NVIDIA 경로는 63/64 일치(첫 34토큰 같음, "1st century AD" ↔ "BC" 한 토큰). 백엔드 사이 차이(MMVQ가 활성값을 q8_1로 양자화하는 등)로 보이며, 같은 CUDA 소스를 두 칩에서 돌린 결과는 거의 같다.
- 둘째 모델(TinyLlama-1.1B-Chat Q4_K_M, 64토큰): 세 결과 모두 자연스러운 문장, 첫 갈림은 기준↔NVIDIA 경로 26토큰, NVIDIA↔CPU 경로 28토큰 뒤(근소한 차이의 동률 근처).
- test-backend-ops(CUDA0 대 CPU, FA 끔, `-p`로 비연속 K·IQ 양자 행렬곱 76건 제외): NVIDIA 경로 15866/15866, CPU 경로 15866/15866 통과, 지원 안 함 11418(그중 FLASH_ATTN_EXT 5721, OUT_PROD 1472, SET_ROWS 588, MUL_MAT 469). 제외한 76건을 NVIDIA 경로에서 하나씩 돌리면 66 통과, 10건은 llama.cpp의 `GGML_ASSERT(convert_func)`로 중단. 고치기 전(`__byte_perm` 부호 복제)에는 NVIDIA 경로 1010건 실패(IQ4_NL·IQ4_XS·MXFP4·NVFP4·Q1_0·Q2_0 행렬곱).
- `CUDART_VERSION` 12030(기본)으로 빌드하면 CPU 경로에서 150개 모두 실패, 오류 1335개: 근본 원인은 PDL 선언 없음(`cudaLaunchConfig_t`, `cudaLaunchAttribute`, `cudaLaunchAttributeProgrammaticStreamSerialization`, `cudaLaunchKernelEx`; 나머지 1000여 개는 그 연쇄)과 `cub/cub.cuh` 없음(argsort, cumsum, mean, ssm-scan, sum, top-k). `cuda_fp8.h`와 `cublasLt.h` 경로는 오류 없이 컴파일됐다. 그래서 11060 프로필을 쓴다.
- 속도(llama-bench, 프롬프트 32토큰·생성 32토큰(CPU 경로 16), 중앙값 / 느린 10%): 기준 CPU 백엔드 8스레드 554 / 534 t/s 처리, 62.6 / 61.3 t/s 생성. NVIDIA 경로 1373 / 1350, 95.5 / 93.3 t/s(드라이버 번역 캐시가 찬 뒤; 첫 실행은 첫 프롬프트에 약 25초). CPU 경로 8스레드 4.7 / 2.9, 0.37 / 0.35 t/s; 24스레드 11.9 / 11.4, 0.79 / 0.71 t/s. 측정 동안 같은 PC에서 다른 작업이 돌아 부하 평균이 18~24였다(기준 24스레드는 흔들려 8스레드 값을 씀).
- CPU 경로가 기준보다 생성에서 약 170배 느린 주된 이유(추정): 행렬-벡터 커널이 행마다 워프 셔플 축소를 하는데, CPU 백엔드는 셔플 한 번마다 32개 파이버를 모두 갈아탄다. 토큰당 수천만 번의 전환으로 어림된다. 백엔드의 워프 실행 방식(묶음 실행 등) 문제이고 이 카드의 범위 밖이다.

## 이의

(없음)

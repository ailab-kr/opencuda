---
id: compiler-driver
kind: component
name: ocu-nvcc, nvcc-compatible compiler driver and toolkit folder
spec_sources:
  - public nvcc documentation (command-line options: phases, file and path options, -arch/-code/-gencode, -Xcompiler/-Xlinker, -rdc, -cudart)
  - CMake documentation and its open-source modules (CMakeDetermineCUDACompiler, Clang-CUDA, FindCUDAToolkit, CMAKE_USER_MAKE_RULES_OVERRIDE_<LANG>, CMAKE_CUDA_RUNTIME_LIBRARY)
  - Clang documentation, "Compiling CUDA with clang"
backend_map:
  cpu: compiler/probe/ocu_cc.py CPU path (preprocess, rewrite, C++ compile)
  nvidia: compiler/probe/ocu_cc.py whole-program path (Clang CUDA mode, PTX in our container)
state: cpu-pass
owner_spec: AI Lab 2026-11-13
owner_tests: AI Lab 2026-11-13 (same author as impl: see note)
owner_impl: AI Lab 2026-11-13
cost: { hours: 3.0, ai_usage: 0 }
---

## 동작

빌드 시스템이 nvcc를 부르던 자리에서 부르는 명령 `compiler/driver/ocu-nvcc`(파이썬).
`.cu`는 `ocu_cc.py`의 컴파일 단계를 그대로 쓰고(모듈로 가져와 부름, 복사 없음), `.c`·`.cpp`는 호스트 컴파일러로, 링크는 호스트 컴파일러에 우리 실행 라이브러리를 붙여 한다.

- 대상: `--ocu-target=cpu|nvidia`, 없으면 불린 이름(`ocu-nvcc-cpu`, `ocu-nvcc-nvidia`), 없으면 환경 변수 `OCU_TARGET`, 없으면 `cpu`.
- 호스트·장치 컴파일러: LLVM Clang(기본 `/usr/lib/llvm-N/bin/clang++`). `-ccbin`으로 바꿀 수 있다.
- `--version`: 우리 이름, Clang의 버전 줄(그대로 옮김), 헤더가 선언한 CUDA API 수준(12.3). 엔비디아 이름은 쓰지 않는다.
- 단계: `-c`, `-E`, `-M`/`-MM`, `-MD`/`-MMD`/`-MF`/`-MT`/`-MP`, 링크(기본), `-shared`, `-lib`, `-ptx`(nvidia), `-dc`/`-rdc`/`-dlink`(cpu만: 장치 코드가 이미 호스트 코드), `-dryrun`, `-v`, `-keep`.
- 경로·매크로: `-I`, `-isystem`, `-include`, `-D`, `-U`, `-L`, `-l`, 응답 파일(`@file`, `-optf`).
- 코드 생성: `-O0..3`, `-g`, `-G`, `-lineinfo`, `-std=`, `--use_fast_math`(nvidia 장치 코드에 근사 함수·서브노멀 0), `--expt-relaxed-constexpr`, `--extended-lambda`(Clang은 원래 허용).
- 아키텍처: `-arch`, `-code`, `-gencode`, `--cuda-gpu-arch`. 기록만 하고, nvidia 대상은 Clang이 아는 가장 낮은 것으로 PTX를 만든다(sm_90보다 새 번호는 sm_90 PTX, 드라이버가 GPU용으로 번역).
- 그대로 넘김: `-Xcompiler`(쉼표와 공백으로 나눔), `-Xlinker`, 호스트 옵션(`-f*`, `-W*`, `-m*`, `-Wl,`, `-Xclang` 등).
- 받고 무시(`-v`에서 알림): `-Xptxas`, `-Xnvlink`, `-Xfatbin`, `-Xcudafe`, `-compress-mode`, `--threads` 등.
- 라이브러리 이름: `-lcudart`·`-lcudart_static`·`-lcudadevrt`·`-lcuda` → 우리 실행 라이브러리, `-lcublas` → `ocu_blas`, `-lcublasLt` → 구현 없음(알림, 함수는 링크에서 미해결).
- 거절: 공급사 SDK 안의 `-I`/`-isystem`/`-L`, 모르는 옵션, `-cubin`/`-fatbin`/`-cuda`, nvidia 대상의 장치 링크.

### 툴킷 폴더

`<build>/toolkit`(빌드 트리, `ocu_toolkit` 대상), 설치 시 `<prefix>/<libdir>/ocu/toolkit`:
`bin/ocu-nvcc`(+ `ocu-nvcc-cpu`, `ocu-nvcc-nvidia`, `nvcc` 링크), `include/`(우리 헤더), `lib/`(정적 라이브러리), 빈 `nvvm/libdevice/`, `cmake/OcuCUDA.cmake`(도구 사슬 파일), `cmake/OcuCUDARules.cmake`, `cmake/modules/FindCUDAToolkit.cmake`.

CMake 사용: `-DCMAKE_TOOLCHAIN_FILE=<toolkit>/cmake/OcuCUDA.cmake [-DOCU_TARGET=nvidia]`. CMake는 이 컴파일러를 Clang 기반 CUDA 컴파일러로 식별하고(실제로 그렇다) Clang 규칙을 쓴다. `find_package(CUDAToolkit)`은 우리 모듈이 받아 `CUDA::cudart` 등을 우리 라이브러리로 잇는다.

## 경계 조건

- 입력이 여럿이고 `-o`가 있으면 `-c`/`-E`/`-M`에서 오류(nvcc와 같음).
- `-o` 없는 기본 출력: `-c`는 `<이름>.o`, 링크는 `a.out`, `-E`·`-M`은 표준 출력.
- `-O`가 없으면 `.cu`는 `-O3`(nvcc의 장치 코드 기본 최적화와 맞춤). `-G`면 기본 최적화 없음.
- `-x cu`는 그 앞뒤 모든 입력에 적용(nvcc처럼).

## 오류 코드

드라이버 오류는 종료 코드 1과 `ocu-nvcc: error: ...`. 컴파일러가 실패하면 그 종료 코드를 그대로 돌려준다.

## 시험 목록

- [x] `ocu_nvcc_options` (`tests/driver/test_ocu_nvcc.py`, 20개): 철자, `-D` 값의 쉼표, 단계, 의존 파일, 전처리, dryrun, 아키텍처 기록, `.c`/`.cpp`·정적·공유 라이브러리, 응답 파일, rdc·dlink, 지원 안 함 오류, 공급사 경로 거절, 이름으로 대상 고르기, PTX
- [x] `ocu_nvcc_cpu_<prog>`, `ocu_nvcc_nvidia_<prog>`: `tests/runtime/*.cu` 8개와 `main_saxpy`를 파일별 `-c` 후 링크해 실행(nvidia는 장치가 있을 때만)
- [x] `ocu_nvcc_cmake_project_{cpu,nvidia,installed}`: `enable_language(CUDA)`와 `find_package(CUDAToolkit)`을 쓰는 보통 CMake 프로젝트(공유 라이브러리·C++ 호스트 프로그램·.cu 단독 프로그램)를 빌드 트리 툴킷과 설치한 툴킷으로 빌드·실행

## 메모

- `bin/nvcc`라는 이름은 CMake가 툴킷을 찾을 때 그 이름의 파일을 찾기 때문에 둔 링크다. 내용은 우리 드라이버이고 엔비디아 이름이나 문구를 출력하지 않는다. 이 이름도 피하려면 CMake의 식별 단계를 통째로 대신해야 해서(강제 컴파일러 설정) CMake 판마다 깨지기 쉽다.
- CMake 3.28 기준으로 확인. CMake가 CUDA 컴파일러를 Clang으로 보면 Clang식 옵션(`-x cuda`, `--cuda-gpu-arch=`, `--cuda-path=`)을 붙이고, 프로젝트는 nvcc식 옵션을 붙인다. 드라이버는 둘 다 받는다.
- llama.cpp(ggml CUDA) 첫 시도: 구성은 통과, 빌드는 없는 헤더(`cublas_v2.h` 등)에서 멈춘다. 다음 오류 목록과 CPU 경로 변환의 한계 2가지는 `docs/compiler-driver.md`.
- 11/13 이후: 실행 람다 복사 캡처, `__restrict` 인자, 정렬 지정자 순서, CPU 경로 `__CUDA_ARCH__=500`·`__CUDA_ARCH_LIST__=500`, NVIDIA 경로 `__CUDA_ARCH_LIST__`로 llama.cpp 150개 `.cu`가 두 대상 모두 컴파일·링크된다(`cards/apps/llama-cpp.md`).
- 시험과 구현을 같은 세션이 썼다(카드 backend-cpu와 같은 사정). 다른 세션의 검토가 필요하다.

## 이의

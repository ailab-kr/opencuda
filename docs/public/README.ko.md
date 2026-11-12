# PROJECT_NAME

> 검토용 초안. `PROJECT_NAME`은 자리표시자이며 공개 이름은 아직 정해지지 않았다.

PROJECT_NAME은 고치지 않은 CUDA C++ 소스를 엔비디아가 아닌 칩에서 빌드해 실행하게 하는 호환 층이다.
함께, 각 칩과 구현이 CUDA 런타임 API와 CUDA C++ 언어의 공개 문서 동작을 얼마나 따르는지 재는 공개 적합성 시험을 제공한다.

- **호환 층**: CUDA 호환 이름의 실행 API와 헤더, 라이브러리 층(BLAS부터), `.cu` 파일을 그대로 받는 컴파일러 경로.
  칩마다 다른 부분은 작은 C 인터페이스(`backends/interface/ocu_backend.h`) 아래에만 있다.
  새 칩을 붙이는 일은 이 인터페이스를 구현하는 일이 전부다.
- **측정 도구**: 정확한 기준 답, 오차 한계, 결과 기록 형식, 정적 성적표를 갖춘 적합성 시험.
  시험은 사용자 프로그램이 보는 표면만 쓰므로 다른 호환 방식도 같은 시험으로 잴 수 있다.

> CUDA는 NVIDIA Corporation의 상표입니다. 이 프로젝트는 NVIDIA와 제휴 관계가 아니며 NVIDIA의 보증을 받지 않았습니다.
> "CUDA C++ 소스", "CUDA 런타임 API"는 이 소프트웨어가 받아들이는 대상을 설명하기 위해서만 쓴다.
> 헤더와 구현은 공개된 문서를 보고 작성했다.

## 백엔드

| 백엔드 | 상태 | 내용 |
|---|---|---|
| CPU (x86-64, ARM64) | 동작, 기준 구현 | 인터페이스 전체. GPU 스레드를 파이버로 실행, 고정·무작위 순서, 워프 폭 32·64 |
| NVIDIA GPU (기준) | 동작 | 설치된 드라이버의 공개 API를 실행 중에 불러 씀. 빌드에 엔비디아 파일 불필요. 장치 코드는 오픈소스 컴파일러가 만든 PTX |
| Apple GPU (Metal) | 동작, 1차 | 백엔드와 CUDA→MSL 변환기. Apple M3, M1 Pro에서 시험 |
| OpenCL | 진행 중 | 아직 저장소에 없음 |
| AMD, Intel | 장치 코드 컴파일만 확인, 실행 백엔드 없음 | gfx1100, gfx1201, sm_80 컴파일 확인 |

## 지금 되는 것

Linux(WSL2, x86-64)와 NVIDIA RTX 5070 Ti에서 잰 값이다. Apple 값은 M3와 M1 Pro에서 쟀다. 최신 값은 성적표를 본다. 아래 표는 한 시점의 기록이다.

| 영역 | 결과 |
|---|---|
| 시험 묶음(CTest) | 66개 중 66개 통과 (CPU, NVIDIA GPU, 변환기 호스트 검사) |
| BLAS 층, 20개 함수 | CPU 백엔드에서 기준 사례 537건 중 537건 통과. 일부러 망가뜨린 구현 15종은 모두 검출 |
| 실행 API | 80여 개 함수(장치, 오류, 통합 메모리 포함 메모리, 스트림, 사건, 실행, 점유율), 벡터 형, 반정밀도, 장치 내장 변수, 원자 연산, 워프 함수 |
| 실제 프로그램의 API 포괄 | llama.cpp가 쓰는 실행 API 호출 이름의 87%, Rodinia의 90% (이름 기준) |
| 고치지 않은 프로그램 | Rodinia CUDA 프로그램 18개 중 17개가 소스 수정 없이 빌드. 최근 실행에서 CPU 경로 13개, NVIDIA 백엔드 14개가 독립 기준(같은 프로그램의 OpenMP판, 직렬 기준 구현, 프로그램 자체 검사)과 일치 |
| NVIDIA 백엔드 (RTX 5070 Ti) | 백엔드 시험 50/50, 실행 API 시험 프로그램 9개 통과 |
| Apple 백엔드 | M3, M1 Pro 모두 백엔드 시험 53/53, 변환 커널 23/23. 100만 값 saxpy 2.3ms(M3), 1.4ms(M1 Pro) |
| 컴파일러 경로 | 고치지 않은 `.cu`가 sm_80, gfx1100, gfx1201용으로 컴파일됨. CPU 경로는 전처리 뒤 실행 구문을 바꿔 C++로 컴파일 |
| CPU 백엔드 속도 | 400만 스레드 saxpy 19ms, 64x64x64 sgemm 0.17ms (24코어) |

아직 없는 것: 텍스처 참조와 텍스처 API, 그래프, 피어 접근, `nvcc` 같은 명령줄 드라이버, 칩 회사 런타임과의 커널 시간 대조.
지원하지 않는 기능은 "지원 안 함" 오류를 돌려준다. 하지 않은 일을 성공했다고 알리지 않는다.

## 빠른 시작

필요한 것: CMake 3.20 이상, C11/C++17 컴파일러, Python 3.9 이상. 컴파일러 경로에는 해당 대상을 지원하는 LLVM Clang.

### Linux

```sh
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

NVIDIA 백엔드는 기본으로 빌드되고 실행 중에 드라이버를 찾는다. 드라이버가 없으면 그 시험은 건너뛴다.
실행 중 백엔드 선택은 `OCU_BACKEND=cpu`(또는 `nvidia`). 컴파일러와 검사기까지 전부 검증: `sh tools/check.sh`.

### macOS (Apple silicon)

```sh
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Apple 플랫폼에서는 Metal 백엔드가 기본으로 켜진다. GPU가 없는 환경(예: GPU 없는 CI 실행기)에서는 Metal 시험을 건너뛴다.

### Windows

CPU 경로는 Linux에서 MinGW-w64(POSIX 스레드)로 교차 빌드해 Wine 또는 Windows에서 실행한다.

```sh
cmake -S . -B build-win -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/x86_64-w64-mingw32.cmake
cmake --build build-win -j4
ctest --test-dir build-win
python3 tools/make_test_bundle.py build-win tests.zip   # 도구 없는 Windows PC에서 실행 가능
```

MSVC 직접 빌드는 아직 지원하지 않는다.

### CPU 경로로 CUDA C++ 파일 빌드

`cmake --install build --prefix /opt/project` 뒤 다른 CMake 프로젝트가 `find_package(Ocu)`, `ocu::cudart` 링크, `.cu` 파일에 `ocu_add_cpu_cu_executable()`을 쓴다.
(패키지와 함수 이름의 접두어는 가칭이며 공개 이름이 정해지면 바뀐다.)

## 적합성 측정 방법

시험은 "컴파일되는가"에서 "실제 프로그램이 같은 답을 내는가"까지 다섯 단계다.

| 단계 | 질문 |
|---|---|
| L1 빌드 | 소스가 수정 없이 컴파일되는가 |
| L2 실행 API | 각 호출이 문서대로 동작하는가 |
| L3 언어 | 장치 코드의 의미가 같은가 |
| L4 라이브러리 | 수치 결과가 정확한 기준의 허용 오차 안인가 |
| L5 응용 | 실제 프로그램이 독립 기준과 같은 결과를 내는가 |

기준 답의 순서: 공개 문서에 적힌 동작, 수학적으로 정해지는 답(CPU에서 높은 정밀도로 계산), 앞의 둘로 정해지지 않는 부분에 한해 검토를 거친 관찰값.
정수와 제어 흐름은 완전 일치, 실수는 연산별 허용 오차 이내, NaN과 무한대는 같은 자리에 같은 값이어야 한다.
속도는 중앙값과 느린 10% 구간, 가장 느린 시험의 이름으로 보고하고 평균만 내지 않는다.

실행 한 번이 JSON 결과 기록 하나를 만든다(형식은 `conformance/README.md`).
`tools/scoreboard.py`가 결과 기록, CTest 출력, Rodinia 출력을 Markdown 파일과 외부 자원 없는 HTML 한 장으로 만든다.

```sh
python3 tools/scoreboard.py build/tests/blas_cpu_result.json \
    --ctest cpu=ctest.txt --rodinia cpu=rodinia.txt --out-dir scoreboard/
```

시험 묶음 자체도 검증한다. 정상 구현은 통과하고 일부러 망가뜨린 구현은 모두 잡혀야 한다.

## 새 칩 붙이기

[BACKEND_GUIDE.md](BACKEND_GUIDE.md)를 본다(영어). 운영 방식은 [GOVERNANCE.md](GOVERNANCE.md), 계획은 [ROADMAP.md](ROADMAP.md), 보안 정책은 [SECURITY.md](SECURITY.md).

## 라이선스

아직 정하지 않았다. [LICENSING-OPTIONS.md](LICENSING-OPTIONS.md)를 본다.

[English](README.md)

---
id: backend-metal
kind: component
name: Apple GPU (Metal) backend and CUDA-to-MSL translation
spec_sources:
  - backends/interface/ocu_backend.h (ABI 2)
  - Apple's public Metal API and Metal Shading Language documentation
  - backends/metal/README.md (design)
backend_map:
  metal: backends/metal/, compiler/metal/
state: gpu-pass on Apple M3 (macOS 15.5) and M1 Pro (macOS 15.0.1); translation first cut
owner_spec: AI Lab 2026-11-13
owner_tests: AI Lab 2026-11-13 (same author as impl: see note)
owner_impl: AI Lab 2026-11-13
cost: { hours: 1.5, ai_usage: 0 }
---

## 동작

- 백엔드(`backends/metal/mt_backend.m`): 공개 Metal API. 장치 포인터 = 공유 저장 버퍼의 GPU 주소(Metal 3), 복사는 같은 메모리의 CPU 주소로, 커널은 MSL 소스를 실행 중에 컴파일, 인자는 버퍼 0에 값 덩어리 + 버퍼 1에 위치표. 모든 할당을 실행마다 GPU에 알림.
- 변환(`compiler/metal/cu2msl.py`, 1차): Clang 장치 패스의 형 정보 구문 트리를 읽어 원본에서 필요한 곳만 고친다(커널 머리, 포인터 주소 공간, 공유 메모리, 내장 변수 문맥). 장벽·원자·수학·워프 함수는 `msl_prelude.h`. 템플릿·double·전역 장치 변수는 이유를 밝히고 거부.

## 경계 조건

- Apple GPU에는 double이 없다: 변환 단계에서 오류.
- 블록당 공유 메모리 32KB(M3 측정): 48KB를 가정한 프로그램은 실행 설정 오류.
- 공유(통합) 메모리, 호스트 메모리 등록, 모듈 전역 변수는 아직 "지원 안 함".

## 시험 목록

- [x] `tests/backend/test_metal_backend.c`(손으로 쓴 MSL): Apple M3에서 53/53. 장치 Apple M3, 작업 집합 10.7GiB, 공유 메모리 32768바이트, 100만 값 saxpy 정답(2.3ms), 공유 메모리 블록 합 정답
- [x] `tests/backend/test_metal_cu2msl.c`: `probe_kernel.cu`를 cu2msl로 바꿔 실행, 호스트 계산과 대조: M3·M1 Pro 모두 23/23
- [x] M1 Pro: 백엔드 53/53(saxpy 100만 값 1.4ms), 변환 23/23
- [x] 두 Mac 모두 전체 시험 42/42, check.sh ALL PASS(빌드 경고 0, Release)

- [x] Rodinia translation count (host side, `cu2msl.py --keep-going` over the pinned Rodinia tree, template instances counted once): 35/79 before, 49/79 after; `msl_check` passes on every image. Not yet run on a GPU.
  - legacy texture declarations (`texture<>`, channel descriptors, bind calls) parse; a kernel that uses a texture reference is skipped with "texture reference '<name>' is not supported yet", the other kernels of the file translate (hybridsort 4, kmeans 1)
  - own minimal helper headers in `compiler/metal/compat/` (`helper_cuda.h`, `helper_timer.h`, empty GL/OpenMP stubs, `vector_types.h`), searched after the program's own `-I` directories (cfd 7)
  - `__constant__ float3` globals are `packed_float3` in the module buffer (12-byte host layout) (cfd 2)
  - struct-typed `__constant__` globals without an initializer, and pointer fields spelled through a macro (heartwall)
- [x] Namespaces and function-valued template arguments (2026-11-28, `tests/metal/test_cu2msl.py` 25/25 on Apple M3; reviewed on x86-64 by another session: full ctest 334/334 before and after, 8 extra checks of its own):
  - a device function inside a namespace (nested, anonymous, a template inside one) is emitted at file scope as `<name>_ocu<N>` and its calls are rewritten; two namespaces with the same function name get two names
  - types and constants inside a namespace are left out with a warning, like any declaration found by name only; a `__device__`/`__constant__` variable inside a namespace is refused
  - `template <float (*op)(float, float)>`: each instance calls the argument function directly; an argument without a body is refused
  - a program declaration spelled like a generated name (`f_ocu0`) is refused ("reserved")
  - `OCU_CU_DEVICE_ARCH` selects the architecture device code is read for (default sm_80)
- [ ] llama.cpp `ggml-cuda` at `0fb8f15` (host side, `OCU_CU_DEVICE_ARCH=sm_50`, 74 files, `--keep-going`; kernel template instances counted each): 0 files before (every file failed in Clang at sm_80), 53 files / 307 instances with `OCU_CU_DEVICE_ARCH`, 56 files / 589 instances with the two items above, 1,342 instances skipped. `msl_check` only; not compiled by Metal, not run. Main reasons left: member function calls 451, a constexpr function that calls itself (`calc_nwarps`, stops `mul_mat_vec_q`) 271, template parameter packs (`k_bin_bcast`) 228, string literals 226, address of a function chosen by a constexpr function 83, conditional pointers into two address spaces (`soft_max_f32`) 20.
- [ ] Still failing: double (policy open), texture use (hybridsort 3, kmeans 1, leukocyte 2, mummergpu 2), mummergpu (driver-API names, old literal syntax), pointer-to-pointer arguments, class templates and types inside namespaces (dwt2d 4, not re-counted after the namespace change)

## 메모

이 세션은 Mac이 없어 Metal 코드를 컴파일하지 못한다. 결과는 M3 시험기 보고로 확인한다. 시험과 구현을 같은 세션이 썼다.

## 다음

1. 변환기 넓히기: 장치 함수 포인터 인자(템플릿화), 템플릿 커널, `__constant__`·`__device__` 변수(버퍼로), 구조체 인자 배치 확인
2. GPU 경로 빌드 연결: `ocu_cc --target metal`(호스트는 Clang CUDA 모드 호스트 패스, 장치 코드는 cu2msl 이미지를 컨테이너에)
3. 런타임 시험 프로그램과 Rodinia를 M3에서

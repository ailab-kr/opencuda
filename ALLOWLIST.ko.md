# 자료 허용 목록

[English](ALLOWLIST.md) · **한국어**

이 저장소의 코드는 공개된 자료만 보고 독자적으로 만든다. 모든 기여자가 이 목록을 따른다.

상태: 초안.

## 볼 수 있는 것

- 엔비디아가 공개한 문서: 함수 설명, 프로그래밍 안내서, 공개된 규격 문서
- 허용적 라이선스로 공개된 소스: 기반 컴파일러, 칩 회사의 공개 런타임과 라이브러리
- 표준 문서, 논문, 교과서
- 우리가 짠 프로그램을 엔비디아 장비에서 실행해 얻은 입력·출력·시간

새 자료가 필요하면 이 목록에 먼저 추가하고 쓴다.

## 하지 않는 것

- 엔비디아가 배포한 프로그램과 라이브러리를 뜯어보는 것: 역어셈블, 역컴파일, 내부 동작 추적
- 엔비디아의 헤더, 예제, 소스를 복사하거나 옆에 두고 옮겨 적는 것
- 엔비디아 도구가 만든 결과물을 분석해 다른 칩용으로 바꾸는 것
- 복제 제한이 강한 라이선스의 코드를 복사하는 것
- 출처를 알 수 없는 코드를 넣는 것

## 기록

- 구현 변경에는 근거가 된 공개 문서를 남긴다. 커밋 메시지의 `Spec-Source:` 줄에 적는다.
- 가져다 쓰는 외부 부품과 그 라이선스는 `docs/third-party.md`에 적는다.

## 허용한 자료

| 자료 | 주소 | 추가한 날 |
|---|---|---|
| Khronos OpenCL 3.0 API·OpenCL C 명세, SPIR-V 명세 | https://registry.khronos.org/OpenCL/ , https://registry.khronos.org/SPIR-V/ | 2026-11-26 |
| Khronos OpenCL 확장 명세: cl_intel_subgroups, cl_intel_subgroup_2d_block_io, cl_intel_subgroup_matrix_multiply_accumulate, cl_intel_unified_shared_memory, cl_khr_integer_dot_product | https://registry.khronos.org/OpenCL/extensions/ | 2026-11-26 |
| LLVM / Clang 문서(HIP 언어 모드, SPIR-V 백엔드, InferAddressSpaces, BypassSlowDivision) | https://llvm.org/docs/ , https://clang.llvm.org/docs/ | 2026-11-26 |
| CUB 공개 API 문서(BlockLoad, BlockStore, BlockReduce, WarpReduce) | https://nvidia.github.io/cccl/cub/ | 2026-11-26 |
| cuBLAS / cuBLASLt API 설명서(함수 설명만) | https://docs.nvidia.com/cuda/cublas/ | 2026-11-26 |
| bitsandbytes 소스(MIT): 저장 형식을 맞추려고 읽은 NF4 코드 표와 블록 배치(블록 64, f32 absmax, 앞 값이 상위 4비트). 시험 프로그램으로 고치지 않고 빌드 | https://github.com/bitsandbytes-foundation/bitsandbytes | 2026-11-26 |
| QLoRA 논문(Dettmers 외 2023): NF4 자료형 | https://arxiv.org/abs/2305.14314 | 2026-11-26 |
| GNU ld 매뉴얼(-rpath, --disable-new-dtags)과 ld.so(8) 매뉴얼 페이지(DT_RPATH·DT_RUNPATH 검색 순서) | https://sourceware.org/binutils/docs/ld/ , https://man7.org/linux/man-pages/man8/ld.so.8.html | 2026-11-27 |
| Intel oneAPI DPC++/C++ 컴파일러 문서(OpenMP 옵션, libiomp5) | https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/ | 2026-11-27 |

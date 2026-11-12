---
id: compiler-ptx-shift
kind: language-feature
name: PTX shift semantics in device code (ocu_cc.py --ptx-shift, opt-in)
spec_sources:
  - PTX ISA 9.5, 9.7.9.8 "Logic and Shift Instructions: shl" (docs.nvidia.com/cuda/parallel-thread-execution, public)
  - PTX ISA 9.5, 9.7.9.9 "Logic and Shift Instructions: shr" (same document)
  - PTX ISA 9.5, 9.4 "Operand Type Information" (the shift amount operand is always .u32)
  - ISO C++17 [expr.shift] (amount negative or >= width of the promoted left operand: undefined)
backend_map:
  cpu: source rewrite (compiler/probe/ptx_shift.py) + helpers (runtime/include/ocu_ptx_shift.h)
  nvidia: same rewrite (Clang's NVPTX output does not keep shl's clamping, see 메모)
  metal: not available (see 메모)
state: spec
owner_spec: AI Lab 2026-11-13
owner_tests: (another session)
owner_impl: AI Lab 2026-11-13
cost: { hours: 3, ai_usage: 0 }
---

## 동작

기본은 표준 C++이다. `ocu_cc.py --ptx-shift`(CMake는 `ocu_add_cpu_cu_executable(<대상> PTX_SHIFT ...)` 또는 변수 `OCU_PTX_SHIFT`, Rodinia 스크립트는 `OCU_PTX_SHIFT=1`)를 주면 장치 함수 안의 정수 자리 옮김이 PTX의 `shl`/`shr` 정의를 따른다.

- 옮김 양은 부호 없는 수로 본다. 너비(정수 승격 뒤 왼쪽 피연산자의 비트 수, 32 또는 64) 이상이면 너비로 줄인다.
- `<<`: 너비 이상이면 0. 부호 있는 값도 비트를 옮긴다(음수 포함).
- `>>`, 부호 없음: 너비 이상이면 0.
- `>>`, 부호 있음: 산술 옮김. 너비 이상이면 부호 비트로 채운다(0 또는 -1).
- 음수 양은 큰 부호 없는 수이므로 너비로 줄어든다.
- 결과의 형과 값(너비 안의 양)은 내장 연산자와 같다. `<<=`, `>>=`도 같고, 왼쪽은 한 번만 평가한다.

바꾸는 범위

- `__global__`, `__device__`(그리고 `__host__ __device__`) 함수의 본문. 그 안의 람다와 템플릿 인스턴스 포함. 호스트 함수는 그대로 둔다.
- 정수 리터럴로 32 미만만큼 옮기는 것은 그대로 둔다(C++에서도 정의됨, 비용 없음). 나머지(변수, 식, 32 이상 리터럴, 음수)는 도우미 호출로 바꾼다.
- 원본 파일과 그 폴더 아래에서 따옴표로 포함한 파일. 작업 폴더에 사본을 만들어 바꾸고 `#line`으로 원래 이름과 줄 번호를 지킨다. 원본은 고치지 않는다.

그대로 두고 알리는 것(표준 C++ 유지, `ocu_cc: --ptx-shift: <파일>:<줄>: ...` 경고)

- 식 일부만 매크로 안에 있는 것(매크로 인자로 들어온 피연산자, 피연산자 자체가 매크로). 식 전체가 한 매크로 본문에 있으면 본문을 바꾼다. 그러면 그 매크로를 쓰는 모든 곳(호스트 포함)이 같은 의미가 된다.
- 원본 폴더 밖 파일의 장치 코드.
- 비트 필드의 복합 대입(`s.f <<= n`): 참조로 묶을 수 없다.

## 경계 조건

- 양 = 0, 31, 32, 33, 63, 64, 큰 값, 음수(-1, INT_MIN).
- 8·16비트 왼쪽 피연산자: 승격 뒤 int(32비트)로 옮긴다.
- 64비트 왼쪽, 32비트 양 / 32비트 왼쪽, 64비트 양. 2^32 이상인 64비트 양도 값으로 비교해 줄인다(PTX의 양은 .u32라 이 경우는 PTX가 정하지 않음, 우리 선택).
- 상수식 안의 자리 옮김(배열 크기, `static_assert`, 템플릿 인자): 도우미가 constexpr이라 그대로 상수식이다.

## 오류 코드

- `--target amd`나 `.ptx` 출력에 옵션을 주면 실행하지 않고 2를 돌려준다(지원 안 함).
- 구문 트리를 만드는 Clang이 실패하면 1(옵션을 조용히 건너뛰지 않는다).

## 시험 목록

- [ ] (시험 담당 세션) 위 동작·경계 조건을 CPU와 엔비디아에서 같은 값으로
- [ ] 옵션 없이 빌드한 결과가 바뀌지 않음(호스트 함수 포함)
- [ ] 경고가 나는 세 경우의 위치와, 그 경우 원래 C++ 그대로임
- [ ] Rodinia huffman: CPU·GPU에서 옵션 켜면 PASS

## 메모

- 엔비디아 경로: Clang은 C++의 자리 옮김을 LLVM `shl`/`lshr`/`ashr`로 내리고, 양이 너비 이상이면 결과가 poison이다. PTX `shl`이 줄여 주는 것은 최적화가 그 값을 건드리지 않았을 때뿐이다. 확인(clang 18, sm_80, -O2): `b == 32 ? a << b : 5`는 상수 5가 되고, `if (bit == 0) o[0] = dw << (32 - bit);`는 저장이 통째로 지워지고, `(unsigned long long)((1 << k) - 1)`은 k=32에서 `ffffffff`(부호 확장이 사라짐)가 된다. 그래서 옵션은 엔비디아 경로에도 같은 변환을 한다.
- 장치 패스(`--cuda-device-only`)의 구문 트리로 정한다. CPU는 호스트 패스로 본다(`__CUDA_ARCH__` 없음, CPU 경로와 같음). 어느 쪽에서도 보이지 않는 `#if` 분기 안의 자리 옮김은 바뀌지 않는다.
- Metal(`compiler/metal/cu2msl.py`): 아직 없다. 같은 사본을 파일 겹침(vfs overlay)으로 주고 `msl_prelude.h`에 같은 도우미를 두면 되지만, 이 PC에서 MSL을 컴파일해 확인할 수 없어 넣지 않았다. 옵션을 주면 돌려주는 오류가 없으므로 Metal 쪽 드라이버가 생길 때 함께 정한다.
- 비용과 Rodinia 결과는 `conformance/external/rodinia/README.md`.

## 이의

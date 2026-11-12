---
id: compiler-ptx-shift
kind: language-feature
name: PTX shift semantics in device code (ocu_cc.py --ptx-shift, opt-in)
spec_sources:
  - PTX ISA 8.5, 9.7.7.8 "Logic and Shift Instructions: shl" (docs.nvidia.com/cuda/parallel-thread-execution, public; 9.7.9.8 in ISA 9.5)
  - PTX ISA 8.5, 9.7.7.9 "Logic and Shift Instructions: shr" (same document; 9.7.9.9 in ISA 9.5)
  - PTX ISA, 9.4 "Operand Type Information" (the shift amount operand is always .u32)
  - ISO C++17 [expr.shift] (amount negative or >= width of the promoted left operand: undefined; /4: E1 sequenced before E2)
  - ISO C++17 [expr.ass]/1 (in E1 op= E2, E2 sequenced before E1), [over.match.oper]/2 (an overloaded operator in operator notation keeps the built-in order)
backend_map:
  cpu: source rewrite (compiler/probe/ptx_shift.py) + helpers (runtime/include/ocu_ptx_shift.h)
  nvidia: same rewrite (Clang's NVPTX output does not keep shl's clamping, see 메모)
  metal: not available (see 메모)
  opencl: not available, refused (exit 2)
state: spec
owner_spec: AI Lab 2026-11-13
owner_tests: (another session)
owner_impl: AI Lab 2026-11-13
cost: { hours: 3, ai_usage: 0 }
---

## 동작

기본은 표준 C++이다. `ocu_cc.py --ptx-shift`(CMake는 `ocu_add_cpu_cu_executable`·`ocu_add_nvidia_cu_executable`·`ocu_add_cu_library`에 `PTX_SHIFT`, 또는 변수 `OCU_PTX_SHIFT`; Rodinia 스크립트는 `OCU_PTX_SHIFT=1`; `ocu_cc.build_commands`를 쓰는 드라이버는 `args.ptx_shift`)를 주면 장치 함수 안의 정수 자리 옮김이 PTX의 `shl`/`shr` 정의를 따른다.

- 옮김 양은 부호 없는 수로 본다. 너비(정수 승격 뒤 왼쪽 피연산자의 비트 수, 32 또는 64) 이상이면 너비로 줄인다.
- `<<`: 너비 이상이면 0. 부호 있는 값도 비트를 옮긴다(음수 포함).
- `>>`, 부호 없음: 너비 이상이면 0.
- `>>`, 부호 있음: 산술 옮김. 너비 이상이면 부호 비트로 채운다(0 또는 -1).
- 음수 양은 큰 부호 없는 수이므로 너비로 줄어든다.
- 결과의 형과 값(너비 안의 양)은 내장 연산자와 같다. `<<=`, `>>=`도 같고, 왼쪽은 한 번만 평가한다.
- 평가 순서는 C++17 그대로다: `a << b`, `a >> b`는 a 다음 b, `a <<= b`, `a >>= b`는 b 다음 a. 그래서 바꾼 꼴은 함수 호출이 아니라 왼쪽만 감싸고 연산자를 남긴다: `__ocu_shift_lhs(a) << b`, `__ocu_shift_ref(a) <<= b`. 연산자 표기로 쓴 다중 정의 연산자는 내장 연산자의 순서를 따른다([over.match.oper]/2). `f(a, b)` 꼴이면 순서가 정해지지 않는다(처음 구현의 결함, GCC는 b 먼저, Clang `<<=`는 a 먼저였다).
- volatile 정수의 복합 자리 옮김(`v <<= n`)은 결과로 volatile 참조 대신 값을 쓸 때 한 번 읽는 작은 객체를 돌려준다. 문장으로 쓸 때(흔한 경우) GCC가 "implicit dereference will not access object" 경고를 내지 않게 하려는 것이다. 값으로 쓰면 내장 연산자와 같다(volatile 읽기 한 번). 참조로 묶을 수는 없다(`volatile int &r = (v <<= n);`은 컴파일 오류, 조용히 다르지 않음). `auto x = (v <<= n);`이면 x는 그 객체가 되어 쓸 때 읽는다.

바꾸는 범위

- `__global__`, `__device__`(그리고 `__host__ __device__`) 함수의 본문. 그 안의 람다와 템플릿 인스턴스 포함. 호스트 함수는 그대로 둔다.
- 정수 리터럴로 32 미만만큼 옮기는 것은 그대로 둔다(C++에서도 정의됨, 비용 없음). 나머지(변수, 식, 32 이상 리터럴, 음수)는 위 꼴로 바꾼다.
- 매크로: 두 피연산자의 시작과 끝, 그리고 연산자가 모두 한 매크로 본문의 글자이면 그 본문을 바꾼다. 예: `#define SHL(a, b) ((a) << (b))`(괄호가 본문의 것), `#define W (v << n)`. 본문의 글자만 바뀌므로 어떤 인자로 펼쳐도 맞다. 그 매크로를 쓰는 모든 곳(호스트 포함)이 같은 의미가 된다.
- 원본 파일과 그 폴더 아래에서 따옴표로 포함한 파일. 작업 폴더에 사본을 만들어 바꾸고 `#line`으로 원래 이름과 줄 번호를 지킨다. 원본은 고치지 않는다.

그대로 두고 알리는 것(표준 C++ 유지, `ocu_cc: --ptx-shift: <파일>:<줄>: ...` 경고)

- 피연산자의 시작이나 끝이 매크로 인자이거나 연산자와 다른 곳에 있는 것. 예: `#define SH(x) x << k`(왼쪽 피연산자가 인자 글자로 시작), `x << AMOUNT`(피연산자 자체가 매크로), 매크로와 지역 헤더에 걸친 식.
- 위 규칙으로 바꿀 매크로가 `#if`·`#elif` 줄에도 쓰이는 것(전처리기는 감싼 꼴을 계산하지 못한다).
- 원본 폴더 밖 파일의 장치 코드.
- 비트 필드의 복합 대입(`s.f <<= n`): 참조로 묶을 수 없다.

## 경계 조건

- 양 = 0, 31, 32, 33, 63, 64, 큰 값, 음수(-1, INT_MIN).
- 8·16비트 왼쪽 피연산자: 승격 뒤 int(32비트)로 옮긴다.
- 64비트 왼쪽, 32비트 양 / 32비트 왼쪽, 64비트 양. 2^32 이상인 64비트 양도 값으로 비교해 줄인다(PTX의 양은 .u32라 이 경우는 PTX가 정하지 않음, 우리 선택).
- 상수식 안의 자리 옮김(배열 크기, `static_assert`, 템플릿 인자): 도우미가 constexpr이라 그대로 상수식이다.

## 오류 코드

- `--target amd`, `--target opencl`, `.ptx` 출력에 옵션을 주면 실행하지 않고 2를 돌려준다(지원 안 함). OpenCL C는 양을 너비로 나눈 나머지만큼 옮기므로 `cu2ocl.py`가 따로 바꿔야 한다.
- 구문 트리를 만드는 Clang이 실패하면 1(옵션을 조용히 건너뛰지 않는다).

## 시험 목록

- [ ] (시험 담당 세션) 위 동작·경계 조건을 CPU와 엔비디아에서 같은 값으로
- [ ] 옵션 없이 빌드한 결과가 바뀌지 않음(호스트 함수 포함)
- [ ] 경고가 나는 세 경우의 위치와, 그 경우 원래 C++ 그대로임
- [ ] Rodinia huffman: CPU·GPU에서 옵션 켜면 PASS

## 메모

- 엔비디아 경로: Clang은 C++의 자리 옮김을 LLVM `shl`/`lshr`/`ashr`로 내리고, 양이 너비 이상이면 결과가 poison이다. PTX `shl`이 줄여 주는 것은 최적화가 그 값을 건드리지 않았을 때뿐이다. 확인(clang 18, sm_80, -O2): `b == 32 ? a << b : 5`는 상수 5가 되고, `if (bit == 0) o[0] = dw << (32 - bit);`는 저장이 통째로 지워지고, `(unsigned long long)((1 << k) - 1)`은 k=32에서 `ffffffff`(부호 확장이 사라짐)가 된다. 그래서 옵션은 엔비디아 경로에도 같은 변환을 한다.
- 장치 패스(`--cuda-device-only`)의 구문 트리로 정한다. CPU는 호스트 패스로 본다(`__CUDA_ARCH__` 없음, CPU 경로와 같음). 어느 쪽에서도 보이지 않는 `#if` 분기 안의 자리 옮김은 바뀌지 않는다.
- 엔비디아 장치 패스에서는 컴파일러가 접을 수 없는 양(32비트 이하)이면 도우미가 PTX `shl`/`shr`를 그대로 쓴다(인라인 PTX, 명령 자체가 양을 줄임). 상수 양은 이식 가능한 형태로 두어 접힌다. CPU 경로는 비교와 선택이 더해진다.
- 구문 트리는 수백 MB(huffman 374MB)라 Clang 출력을 최상위 선언 단위로 흘려 읽고, 사용자 파일의 장치 함수가 든 선언만 해석한다(최대 메모리 1.3GB → 약 390MB, 그중 Clang 124MB).
- Metal(`compiler/metal/cu2msl.py`): 아직 없다(`cu2msl.py`에 옵션이 없다). 같은 사본을 파일 겹침(vfs overlay)으로 주고 `msl_prelude.h`에 같은 도우미를 두면 되지만, 이 PC에서 MSL을 컴파일해 확인할 수 없어 넣지 않았다.
- `ocu-nvcc`(compiler/driver)는 아직 이 옵션을 내놓지 않는다. 내놓을 때 의존 파일(-MD)이 사본 경로를 가리키지 않게 해야 한다.
- 비용과 Rodinia 결과는 `conformance/external/rodinia/README.md`("compat").

## 이의

- (구현 담당, 2026-11-13) `ptx_shift_cmd_case1_warn_*`, `ptx_shift_cmd_case8_warn_*`: `#define SHL_ARGS(a, b) ((a) << (b))`를 경고하고 그대로 두기를 기대한다. 처음 카드의 "매크로 인자로 들어온 피연산자"라는 말이 이 꼴까지 포함하는 것처럼 읽혀서 생긴 차이다. 구현은 이 꼴을 바꾸는 쪽을 택했다(위 "매크로"). 이유: 바꾸는 것은 본문의 괄호와 연산자 둘레의 글자뿐이라 인자가 무엇이든 맞다(`W` 같은 본문 전체 규칙과 같은 근거). 그대로 두면 장치 코드에서 흔한 비트 도우미 매크로가 표준 C++로 남아 이 옵션의 목적(PTX 결과)에 어긋난다. 위험한 곳(`#if` 안의 사용)은 따로 경고한다. 진짜로 걸친 꼴(`#define SH(x) x << k`)은 지금도 경고한다. 시험은 카드에 맞춰 고쳐 주기를 청한다: case 1·8은 경고 없음, 값은 옵션 켤 때 0(40만큼 옮김).

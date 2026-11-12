# ocu (가칭)

CUDA용으로 짠 소스를 고치지 않고 다른 GPU에서 빌드해 실행하게 하는 호환 층과, 그 호환 수준을 재는 적합성 시험.

## 지금 되는 것

- **CPU 기준 백엔드**: 연결 층 규격의 함수 26개 전부. 메모리, 대기열, 표식, 커널 실행(블록 동기화, 워프 셔플·투표 포함), 작업 스레드 묶음. x86-64와 ARM64.
- **BLAS 20개 함수**(내부 이름 `ocu_blas_*`): CPU 백엔드에서 기준 사례 537건을 모두 통과한다.
- **적합성 시험의 바탕**: BLAS의 정확한 기준 답, 오차 한계, 결과 기록 형식, 시험 묶음 자체의 검증(정상 구현은 통과, 일부러 망가뜨린 구현 15종은 모두 검출).
- **컴파일러 실험**: 고치지 않은 `.cu` 파일 네 개가 엔비디아(sm_80)·AMD(gfx1100, gfx1201)용으로 컴파일되고, CPU에서 실행되어 정답을 낸다. 배포하지 않는 실험이다(`compiler/probe/`).

아직 없는 것: CUDA 이름의 실행 API와 헤더, AMD·인텔 백엔드. 그래서 일반 CUDA 프로그램은 아직 빌드할 수 없다.

## 읽을 순서

1. `ARCHITECTURE.md`: 부품, 경계, 규칙, 측정값
2. `ALLOWLIST.md`, `CONTRIBUTING.md`: 볼 수 있는 자료, 커밋 규칙
3. `backends/interface/ocu_backend.h`: 장치 연결 층 규격
4. `docs/compiler-probe.md`: 컴파일러 실험 결과
5. `backends/amd/README.md`: AMD 백엔드 설계

## 빌드와 시험

리눅스(또는 WSL)에서:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

한 번에 전부 검증(gcc·clang, 주소·스레드 검사기, ARM64 에뮬레이터, 채점기, 컴파일러 실험, 클린룸 점검):

```sh
sh tools/check.sh
```

BLAS 기준 사례를 CPU 백엔드에서 실행하고 채점:

```sh
cd conformance/harness
OCU_BACKEND=cpu python3 -m ocu_conf.blas_run ../../build/tests/blas_driver
```

## 폴더

| 폴더 | 내용 |
|---|---|
| `backends/` | 장치 연결 층 규격(`interface/`), CPU 백엔드(`cpu/`), AMD 설계(`amd/`) |
| `runtime/` | 백엔드 등록(`core/`). 실행 API는 법률 검토 뒤 |
| `libs/` | BLAS 층(`blas/`) |
| `compiler/` | 컴파일러 실험과 드라이버(`probe/`) |
| `conformance/` | 기준 답, 오차 한계, 채점기, 실행기, 일부러 망가뜨린 구현 |
| `tuning/` | 측정 규칙을 지키는 측정기, BLAS 측정 |
| `cards/` | 기능 카드 |
| `docs/` | 실험 기록, 외부 부품 목록 |
| `tests/` | 백엔드·BLAS·컴파일러 경로 시험 |
| `tools/` | 전체 검사, git 훅 |

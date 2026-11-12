# ocu (가칭)

CUDA용으로 짠 소스를 고치지 않고 다른 GPU에서 빌드해 실행하게 하는 호환 층과, 그 호환 수준을 재는 적합성 시험.

## 지금 되는 것

- **CPU 기준 백엔드**: 연결 층 규격의 함수 26개 전부. 메모리, 대기열, 표식, 커널 실행(블록 동기화, 워프 셔플·투표 포함), 작업 스레드 묶음. x86-64와 ARM64.
- **BLAS 20개 함수**(내부 이름 `ocu_blas_*`): CPU 백엔드에서 기준 사례 537건을 모두 통과한다.
- **적합성 시험의 바탕**: BLAS의 정확한 기준 답, 오차 한계, 결과 기록 형식, 시험 묶음 자체의 검증(정상 구현은 통과, 일부러 망가뜨린 구현 15종은 모두 검출).
- **실행 API 첫 묶음과 헤더**(`runtime/`): 장치·오류·메모리·기호 복사·스트림·사건·실행 등 50여 개 함수, 벡터 형, 장치 쪽 내장 변수·원자 연산·워프 함수. 공개 문서만 보고 새로 썼다.
- **보통의 CUDA 프로그램을 CPU에서**: 공개 벤치마크 Rodinia의 CUDA 프로그램 18개 중 17개가 소스 수정 없이 빌드되고, 8개는 독립 기준(같은 프로그램의 OpenMP판, 계산식을 옮긴 직렬 구현, 프로그램 자체 검사)과 결과가 같다(`conformance/external/rodinia/`).
- **컴파일러 실험**: 고치지 않은 `.cu` 파일이 엔비디아(sm_80)·AMD(gfx1100, gfx1201)용으로 컴파일된다. CPU 경로는 전처리 뒤 실행 구문을 바꿔 C++로 컴파일한다(`compiler/probe/`, 배포 전 실험 단계).

아직 없는 것: AMD·인텔 백엔드, nvcc와 같은 명령줄 드라이버, 텍스처·그래프·통합 메모리 같은 API.

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

보통의 CUDA 프로그램(Rodinia)을 CPU 경로로 빌드해 결과 대조(소스는 저장소 밖에 내려받는다):

```sh
conformance/external/rodinia/run.sh build
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
| `runtime/` | 백엔드 등록(`core/`), 실행 API(`cudart/`), 헤더(`include/`) |
| `libs/` | BLAS 층(`blas/`) |
| `compiler/` | 컴파일러 실험과 드라이버(`probe/`) |
| `conformance/` | 기준 답, 오차 한계, 채점기, 실행기, 일부러 망가뜨린 구현, 공개 프로그램 대조(`external/`) |
| `tuning/` | 측정 규칙을 지키는 측정기, BLAS 측정 |
| `cards/` | 기능 카드 |
| `docs/` | 실험 기록, 외부 부품 목록 |
| `tests/` | 백엔드·BLAS·컴파일러 경로 시험 |
| `tools/` | 전체 검사, git 훅 |

# ocu (가칭)

CUDA용으로 짠 소스를 고치지 않고 다른 GPU에서 빌드해 실행하게 하는 호환 층과, 그 호환 수준을 재는 적합성 시험.

현재 상태: 뼈대만 있다. 사용자 프로그램을 빌드하거나 실행할 수는 없다.

## 읽을 순서

1. `ARCHITECTURE.md`: 부품, 경계, 규칙
2. `ALLOWLIST.md`: 개발할 때 볼 수 있는 자료와 금지 자료
3. `backends/interface/ocu_backend.h`: 장치 연결 층 규격
4. `cards/_template.md`: 작업 단위(기능 카드) 양식

## 빌드와 시험

리눅스(또는 WSL)에서:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

채점기(파이썬 3.10 이상, 추가 설치 없음):

```sh
cd conformance/harness
python -m unittest discover -s tests
python -m ocu_conf validate tests/data/example_result.json
```

## 폴더

| 폴더 | 내용 |
|---|---|
| `backends/` | 장치 연결 층 규격과 칩별 구현 |
| `runtime/` | 백엔드 등록(`core/`), 실행 API(예정) |
| `libs/` | 라이브러리별 호환 층(예정) |
| `compiler/` | 명령 호환과 빌드 도구 연동(예정) |
| `conformance/` | 적합성 시험과 채점기 |
| `tuning/` | 속도 측정과 자동 튜닝(예정) |
| `cards/` | 기능 카드 |
| `docs/` | 설치 기록, 문법 차이 목록, 외부 부품 목록 |
| `tests/` | 뼈대의 기본 동작 시험 |

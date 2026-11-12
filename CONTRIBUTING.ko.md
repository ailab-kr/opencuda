# 기여 안내

[English](CONTRIBUTING.md) · **한국어**

이 저장소는 클린룸 방식으로 개발한다. 공개 문서만 보고 작성한다.
코드를 쓰기 전에 `ALLOWLIST.ko.md`를 읽고, 경계를 바꾸기 전에 `ARCHITECTURE.ko.md`를 읽는다.

## 클론마다 한 번 하는 설정

```sh
git config core.hooksPath tools/hooks
```

훅은 커밋 형식을 검사하고, 시험과 구현이 섞이지 않게 한다.

## 커밋 형식

```
<area>: <one-line summary>

<what changed and why>

Role: <spec | tests | impl | review | perf | infra | docs>
Card: <card id>
Spec-Source: <public document the change is based on>
```

| 트레일러 | 필요한 경우 |
|---|---|
| `Role` | 항상 |
| `Card` | `Role`이 `spec`, `tests`, `impl`, `perf`일 때 |
| `Spec-Source` | `Role`이 `impl`일 때 |

`<area>`는 최상위 폴더 또는 구성 요소다: `backends/cpu`, `runtime`,
`conformance`, `docs`, `build`, ...

언어: 코드, 주석, 커밋 메시지는 영어로 쓴다.

## 시험과 구현은 분리한다

- 한 커밋은 시험 경로(`conformance/suites`, `conformance/tolerances`,
  `conformance/oracle`)나 구현 경로(`backends`, `runtime`, `libs`,
  `compiler`, `tuning`) 중 한쪽만 바꿀 수 있고, 둘을 함께 바꾸면 안 된다.
  `pre-commit` 훅이 섞인 커밋을 거부한다.
- 같은 카드의 시험과 구현은 서로 다른 사람이 쓴다.
- 시험이 잘못돼 보여도 구현 쪽에서 고치지 않는다. 그 카드의 "이의" 절에
  반대 의견을 적는다.
- `OCU_ALLOW_MIXED=1`은 섞임 검사를 건너뛴다. 저장소 전체에 걸친 기계적 변경
  (이름 바꾸기, 서식 정리)에만 쓰고, 커밋 메시지에 그렇게 밝힌다.

## 넣지 않는 것

- `ALLOWLIST.ko.md`의 "하지 않는 것"에 나열된 모든 것.
- 빌드 산출물, 결과 기록(`conformance/results/*.json`), 비공개 평가 데이터.
- 계획과 전략 메모. 이런 것은 이 저장소 밖에 둔다.

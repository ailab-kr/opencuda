# 연산 표(`ocu_ops`), 버전 1

[English](op-table.md) · **한국어**

상태: 초안, 버전 1 (2026-11-13). 헤더: `backends/interface/ocu_ops.h`.

## 1. 목적과 위치

장치 백엔드 인터페이스(`ocu_backend.h`)는 CUDA 커널을 실행한다. 일부 가속기에는
SIMT 스레드가 없다. 이런 칩의 공개 런타임은 모양이 고정된 미리 컴파일된 연산을
칩 안의 스크래치패드 메모리와 DMA로 실행한다. 이런 칩은 고치지 않은 커널을 돌릴
수 없지만, 라이브러리 수준의 연산(GEMM, 정규화, softmax, 원소별 산술)은 돌릴 수 있다.

연산 표는 백엔드가 일을 하는 두 번째이자 선택적인 방법이다.

```
 libs/blas, libs/dnn, an out-of-tree ggml backend, ...
        │                         │
  kernel path                operator path
  (kernel_launch)            (op table: support → query → plan → execute)
        │                         │
  backends with threads       any backend that fills the table
```

- 백엔드는 기존의 선택적 진입점 `get_provider`로 표를 내놓는다.
  `get_provider("ops")`(`OCU_OPS_PROVIDER`)는 `const ocu_op_table *`을 돌려주고,
  백엔드에 표가 없으면 `NULL`을 돌려준다. 백엔드 표(`ocu_backend`)와
  `OCU_BACKEND_ABI`는 바뀌지 않는다.
- 연산 표는 자체 버전(`OCU_OPS_VERSION`, 현재 1)과 자체 크기 필드(`struct_size`)를
  가진다. 호출자는 항목을 쓰기 전에 둘 다 확인한다. 항목은 뒤에 덧붙이기만 한다.
  더 오래된 헤더로 컴파일한 호출자는 자신이 아는 표의 앞부분만 본다.
- `backends/` 위의 호출자는 백엔드 이름이 아니라 표에 물어서 경로를 고른다.
  인라인 도우미 `ocu_ops_of(backend)`는 표를 돌려주고, 표가 없거나 버전 또는 크기가
  버전 1에 못 미치면 `NULL`을 돌려준다.

## 2. 데이터 형식

| `ocu_dtype` | 값 | 원소 | 바이트 |
|---|---|---|---|
| `OCU_DT_NONE` | 0 | 텐서 없음(서술자에서는 유효하지 않음) | 0 |
| `OCU_DT_F32` | 1 | IEEE 754 binary32 | 4 |
| `OCU_DT_F16` | 2 | IEEE 754 binary16 | 2 |
| `OCU_DT_BF16` | 3 | bfloat16(binary32의 상위 16비트) | 2 |
| `OCU_DT_I8` | 4 | 부호 있는 8비트 정수 | 1 |
| `OCU_DT_I32` | 5 | 부호 있는 32비트 정수 | 4 |
| `OCU_DT_Q8_B32` | 0x101 | 블록 양자화: 값 32개 단위 블록, binary16 스케일 하나 다음에 `int8` 32개 | 블록당 34 |
| `OCU_DT_Q4_B32` | 0x102 | 블록 양자화: 값 32개 단위 블록, binary16 스케일 하나 다음에 4비트 값 16바이트 | 블록당 18 |

`ocu_dtype_is_blocked(dt)`는 값이 `>= 0x100`(블록 계열)이면 참이다. 버전 1의
어떤 연산도 블록 양자화 텐서를 받지 않는다. 서술자와 지원 표가 이 형식을 이름으로
가리킬 수 있도록 값만 예약해 둔 것이다. 블록 형식에서 `shape`는 여전히 값의 개수를
센다. 가장 안쪽 차원은 블록 길이의 배수여야 하고, 바깥 차원의 보폭은 값 단위로
세며 블록 길이의 배수여야 한다. 블록 텐서가 둘 중 하나라도 어기면 잘못된 것이다
(`OCU_ERR_INVALID_VALUE`). 블록 안의 바이트 배치는 그 형식을 쓰는 연산을 처음
내놓는 버전에서 정한다.

binary16과 bfloat16으로의 변환은 가장 가까운 값으로 반올림하고, 가운데 값은
짝수 쪽으로 보낸다. 구현은 비정규 결과를 0으로 비워도 된다(6.2 참고).

## 3. 텐서

```c
typedef struct ocu_tensor_desc {
    ocu_dtype dtype;
    uint32_t  rank;                       /* 1 .. OCU_OPS_MAX_RANK (4) */
    int64_t   shape[OCU_OPS_MAX_RANK];    /* shape[0] is the outermost dim */
    int64_t   stride[OCU_OPS_MAX_RANK];   /* in elements, not bytes */
} ocu_tensor_desc;

typedef struct ocu_tensor {
    void           *data;   /* device pointer to element [0,0,..] */
    ocu_tensor_desc desc;
} ocu_tensor;
```

원소 `(i0, .., i_{r-1})`는 `data + sum(i_k * stride[k]) * elem_size`에 있다.
`rank` 이후의 항목은 무시한다.

서술자가 다음을 모두 만족하면 **올바른 형태**다. 그렇지 않으면 `query`, `plan`,
`execute`는 `OCU_ERR_INVALID_VALUE`를 돌려준다.

1. `dtype`이 2절의 값 중 하나이고 `1 <= rank <= 4`이다.
2. 모든 `shape[k] >= 1`이다. (빈 텐서는 기술하지 않는다. BLAS의 빠른 반환처럼
   호출자가 호출을 건너뛴다.)
3. 가장 안쪽 차원이 연속이다: `stride[rank-1] == 1`.
4. 바깥 보폭이 안쪽 범위와 겹치지 않는다:
   `k < rank-1`에 대해 `stride[k] >= stride[k+1] * shape[k+1]`.
5. 원소 개수 `prod(shape)`와 범위 `1 + sum((shape[k]-1) * stride[k])`가
   `int64_t`에 들어간다.

지원 질의에서 쓰는 배치 이름:

- **dense**: 모든 `k`에 대해 `stride[k] == stride[k+1] * shape[k+1]`
  (빈틈없는 행 우선).
- **row-strided**: 올바른 형태이지만 dense가 아닌 것(행 또는 행렬 사이에 패딩이
  있는 경우, 예를 들어 선행 차원이 행 길이보다 큰 경우).

## 4. 연산

```c
typedef struct ocu_op_desc {
    uint32_t        struct_size;   /* sizeof(ocu_op_desc) of the caller */
    ocu_op          op;
    uint32_t        num_tensors;   /* inputs, then the output (last) */
    uint32_t        flags;         /* OCU_OP_FLAG_* bits (4.7) */
    ocu_tensor_desc tensors[OCU_OPS_MAX_TENSORS];  /* 4 */
    union { ... } p;               /* per-operator parameters */
} ocu_op_desc;
```

출력은 항상 마지막 슬롯이다. `flags`에는 4.7절의 비트만 둘 수 있고,
`struct_size`는 적어도 버전 1의 크기여야 한다. 아니면 `OCU_ERR_INVALID_VALUE`이다.
모든 텐서 슬롯은 올바른 형태여야 한다(3절).

아래의 모양은 가장 바깥 차원부터 적는다. 행 연산의 "행"은 가장 안쪽 차원을 뺀
모든 차원이고, `N = shape[rank-1]`은 행 길이다.

| `ocu_op` | 값 | 슬롯 | 결과 |
|---|---|---|---|
| `OCU_OP_GEMM` | 1 | A, B, C | `C = alpha * op(A) · op(B) + beta * C` |
| `OCU_OP_ADD` | 2 | a, b, y | `y = a + b` (b는 브로드캐스트) |
| `OCU_OP_MUL` | 3 | a, b, y | `y = a * b` (b는 브로드캐스트) |
| `OCU_OP_RMS_NORM` | 4 | x, y 또는 x, w, y | 행마다 `y = x / sqrt(mean(x²) + eps) [* w]` |
| `OCU_OP_SOFTMAX` | 5 | x, y | 행마다 `y = softmax(scale * x)` |
| `OCU_OP_ROPE` | 6 | x, pos, y | 회전 위치 임베딩(4.6) |
| `OCU_OP_QMATMUL` | 7 | 예약 | 블록 양자화 행렬곱, 매개변수는 이후 버전에서 정함 |

### 4.1 GEMM

매개변수 `p.gemm`: `uint32_t trans_a, trans_b`(0 또는 1), `float alpha, beta`.

- 랭크 2: A는 `[M,K]`로 저장하고(`trans_a`이면 `[K,M]`), B는 `[K,N]`로 저장하며
  (`trans_b`이면 `[N,K]`), C는 `[M,N]`이다. `op(X)`는 X 또는 X의 전치다.
- 랭크 3: 맨 앞에 배치 차원이 있다. C는 `[Bt,M,N]`이고, A와 B의 배치는 `Bt` 또는
  1이다(1은 모든 배치 항목에 같은 행렬을 쓴다는 뜻이다). 세 텐서의 랭크는 같다.
  그 밖의 랭크는 `OCU_ERR_INVALID_VALUE`이다.
- 형식: A와 B는 dtype이 같고 f32, f16, bf16 중 하나다. C는 f32이거나 A, B와 같은
  dtype이다. (f16 또는 bf16 입력에 f32 C를 쓰는 경우를 "혼합"이라 한다.)
  int8 입력은 이후 버전을 위해 예약되어 있다(`NOT_SUPPORTED`).
- 곱은 백엔드의 누적 형식(`caps.accum_dtype`, 5.2 참고)으로 누적한다. 그다음
  `alpha * acc + beta * C_old`를 적어도 그 정밀도로 계산하고 C의 dtype으로 한 번만
  반올림한다.
- `beta == 0`: C를 읽지 않는다(C의 NaN이나 Inf가 전파되지 않는다).
  `alpha == 0`: A와 B를 읽지 않는다. `C = beta * C`.
- `trans_a`/`trans_b`가 0 또는 1이 아니거나, M/N/K가 맞지 않거나, 배치 규칙을
  어기면 `OCU_ERR_INVALID_VALUE`이다.
- C는 A 또는 B와 겹치면 안 된다.

### 4.2 ADD와 MUL

매개변수는 없다. `a`, `b`, `y`는 랭크가 같고 dtype이 같다(f32, f16, bf16).
`y`의 모양은 `a`의 모양이다. 각 `k`에 대해 `b.shape[k]`는 `a.shape[k]`와 같거나
1이다(그 차원을 따라 브로드캐스트). `a + b`(또는 `a * b`)의 정확한 결과를 dtype으로
한 번만 반올림한다. `y`는 `a`와 같은 텐서(같은 포인터와 같은 서술자)일 수 있다.
`y`와 입력이 그 밖의 방식으로 겹치는 것은 허용하지 않는다.

### 4.3 RMS_NORM

매개변수 `p.rms_norm.eps`(float, `>= 0`, 아니면 `OCU_ERR_INVALID_VALUE`).
`x`와 `y`는 모양과 dtype이 같다(f32, f16, bf16). 선택 입력인 가중치 `w`는
랭크 1, 모양 `[N]`, 같은 dtype이다. 각 행에 대해
`y_i = x_i * (1 / sqrt(sum_j x_j² / N + eps)) * w_i`이다(`w_i` 인수는 `w`가 있을
때만 곱한다). `y`는 `x`와 같은 텐서일 수 있다.

### 4.4 SOFTMAX

매개변수 `p.softmax.scale`(float, 유한한 값, 아니면 `OCU_ERR_INVALID_VALUE`).
`x`와 `y`는 모양과 dtype이 같다(f32, f16, bf16). 각 행에서
`t_j = scale * x_j`, `m = max_j t_j`라 할 때
`y_i = exp(t_i - m) / sum_j exp(t_j - m)`이다. `y`는 `x`와 같은 텐서일 수 있다.
NaN 또는 `+Inf`가 든 행은 그 행의 값이 지정되지 않으며, 다른 행은 영향을 받지 않는다.

### 4.5 공통 규칙

- 장치 포인터는 백엔드의 `mem_alloc`이 돌려준 포인터에 오프셋을 더한 것이다.
  모든 텐서의 전체 범위는 하나의 할당 안에 있어야 한다.
- `num_tensors`가 그 연산에 나열된 값과 다르면 `OCU_ERR_INVALID_VALUE`이다.
- 헤더에 나열되지 않은 연산 값은 `OCU_ERR_INVALID_VALUE`이다.
- 슬롯의 dtype이 그 연산의 일치 규칙을 어기면(예를 들어 A는 f32이고 B는 f16)
  `OCU_ERR_INVALID_VALUE`이다.
- 이 버전에서 그 연산에 정의되지 않은 dtype(int8 GEMM, i32 위의 ADD, 모든 블록
  양자화 텐서)은 `OCU_ERR_NOT_SUPPORTED`이다. 어느 슬롯이든 그런 dtype이면 dtype
  일치 규칙은 적용하지 않는다(f32 A, B에 i32 C를 쓴 GEMM은 `NOT_SUPPORTED`이다).
  ROPE도 마찬가지다. ROPE를 구현하지 않은 백엔드에서도 블록 양자화 텐서는
  `NOT_SUPPORTED`이다. 예약된 연산 `OCU_OP_QMATMUL`은 `OCU_ERR_NOT_SUPPORTED`이다.
- 검사 순서: 잘못된 형태의 텐서나 서술자(`INVALID_VALUE`)를
  `NOT_SUPPORTED` / `NOT_IMPLEMENTED`보다 먼저 보고한다.

### 4.6 ROPE (정의하되 버전 1에서는 구현이 선택)

매개변수 `p.rope`: `uint32_t mode`(0 = 인접 쌍 `(2i, 2i+1)`,
1 = 반으로 나눈 쌍 `(i, i + d/2)`), `uint32_t n_dims`(`d`, 짝수, `<= N`),
`float theta_base`, `float freq_scale`.
`x`는 `[T, H, N]`(토큰, 헤드, 헤드 크기) 또는 `[T, N]`이고, `pos`는
`OCU_DT_I32`인 `[T]`이며, `y`는 `x`와 모양과 dtype이 같다. 토큰 `t`에 대해
`p = pos[t]`이고 쌍 인덱스 `i < d/2`일 때 각도는
`a = p * freq_scale * theta_base^(-2i/d)`이고, 쌍 `(u, v)`는
`(u cos a - v sin a, u sin a + v cos a)`가 된다. 각 행에서 위치 `>= d`인 원소는
그대로 복사한다. 허용 오차 등급은 f32에서 ULP(8)이며(6.1), binary64로 계산한
기준 값과 비교해 잰다. 아직 구현하지 않은 백엔드는 `support`, `query`, `plan`에서
`OCU_ERR_NOT_IMPLEMENTED`를 돌려준다.

### 4.7 가변 행 수(`OCU_OP_FLAG_VAR_ROWS`, 값 1)

가속기가 컴파일하는 것은 모양이 고정된 계획이다. 그러나 나머지는 그대로인데
호출마다 행 수만 바뀌는 경우가 흔하다(언어 모델의 프롬프트 길이, 행이 하나인
마지막 토큰 단계). 모델 작업 부하에는 서로 다른 가중치 모양 `(K, N)`이 적고 행
수의 종류도 적다. 그래서 `(K, N, dtype)`마다 가변 행 수 계획 하나만 두면 계획
수가 작게 유지된다. 이 플래그를 쓰면 계획한 서술자에는 **가장 큰** 행 수를 적고,
`execute`는 `1 <= m' <=` 계획한 값인 임의의 행 수 `m'`을 넘길 수 있다.

- GEMM: 변하는 차원은 M이다(A에서는 `shape[rank-2]`, `trans_a`이면
  `shape[rank-1]`이고, C에서는 `shape[rank-2]`이다). B는 변하지 않는다.
- ADD, MUL, RMS_NORM, SOFTMAX: 변하는 차원은 출력의 `shape[0]`이며, 출력과 랭크가
  같고 `shape[0]`이 출력의 계획된 `shape[0]`과 같은 모든 입력의 `shape[0]`도
  변한다(그 자리가 1인 브로드캐스트 입력과 랭크 1인 RMS 가중치는 변하지 않는다).
  출력의 랭크는 2 이상이어야 하며, 아니면 `OCU_ERR_INVALID_VALUE`이다.
- 한 호출에서 변하는 모든 차원은 같은 `m'`을 갖는다. 보폭과 나머지 모든 차원은
  계획한 값과 같아야 한다.
- `query`와 `plan`은 계획한(가장 큰) 모양을 판정하고, `caps`도 그 모양을 기술한다.
- 더 큰 행 수로 만든 계획으로 더 작은 행 수를 실행할 수 없는 백엔드는 이 플래그가
  붙은 서술자에 `OCU_ERR_NOT_SUPPORTED`를 돌려주고, `support`에 `var_rows = 0`을
  보고한다. 그러면 호출자는 행 수마다 계획하거나, 계획된 구간까지 행을 채운다.

단계적으로 커지는 모양(블록의 배수로 채운 어텐션 길이)은 호출자가 구간까지 채워
구간마다 한 번 계획해야 한다. 이후 버전에서 마스크가 있는 동적 안쪽 길이를 추가할
수 있다.

## 5. 표 항목

```c
typedef struct ocu_op_table {
    uint32_t version;      /* OCU_OPS_VERSION the backend implements */
    uint32_t struct_size;  /* sizeof(ocu_op_table) the backend was built with */
    ocu_status (*support)(ocu_device *dev, ocu_op op, ocu_dtype dtype,
                          ocu_op_support *out);
    ocu_status (*query)(ocu_device *dev, const ocu_op_desc *desc,
                        ocu_op_caps *caps);
    ocu_status (*plan)(ocu_device *dev, const ocu_op_desc *desc,
                       ocu_op_plan **plan);
    ocu_status (*execute)(ocu_op_plan *plan, ocu_queue *queue,
                          const ocu_tensor *tensors, uint32_t num_tensors);
    ocu_status (*destroy)(ocu_op_plan *plan);
    ocu_status (*stats)(ocu_device *dev, ocu_op_stats *out);   /* may be NULL */
} ocu_op_table;
```

상태 값의 의미는 `ocu_backend.h`와 같다. 특히 `OCU_ERR_NOT_SUPPORTED`는 칩이 이
경우를 할 수 없다는 뜻이고, `OCU_ERR_NOT_IMPLEMENTED`는 백엔드가 아직 하지 않는다는
뜻이다. 모든 항목은 장치가 `NULL`이면 `OCU_ERR_INVALID_DEVICE`를, 출력 포인터가
`NULL`이면 `OCU_ERR_INVALID_VALUE`를 돌려준다.

### 5.1 `support`: 연산 × dtype별 대략적인 능력

지원 표를 위한 항목이다. `dtype`은 입력 dtype이다(GEMM에서는 A와 B의 dtype).
그 연산과 dtype의 서술자 중 일부라도 계획할 수 있으면 `OCU_OK`를 돌려주고 `*out`을
채운다. 그렇지 않으면 `OCU_ERR_NOT_SUPPORTED` 또는 `OCU_ERR_NOT_IMPLEMENTED`를
돌려준다(이때 `*out`은 0으로 채운다).

```c
typedef struct ocu_op_support {
    uint32_t  layouts;       /* OCU_LAYOUT_DENSE | OCU_LAYOUT_ROW_STRIDED */
    uint32_t  max_rank;      /* largest rank accepted */
    int64_t   max_row;       /* largest innermost dim, 0 = no fixed limit */
    uint32_t  align_bytes;   /* required alignment of every data pointer */
    ocu_dtype accum_dtype;   /* accumulation type (GEMM, reductions) */
    uint32_t  out_dtypes;    /* bit (1u << dtype) per accepted output dtype
                                (plain dtypes only, values below 32) */
    uint32_t  needs_plan_ms; /* typical plan time in ms (0 = negligible) */
    uint32_t  var_rows;      /* 1 when OCU_OP_FLAG_VAR_ROWS plans are accepted */
} ocu_op_support;
```

`support`는 요약이다. 서술자가 `query`에서 거절될 수도 있다(예를 들어 행이
스크래치패드보다 긴 경우). `support`가 `NOT_SUPPORTED`를 돌려주면 그 쌍의 모든
서술자가 거절된다는 뜻이다.

### 5.2 `query`: 서술자 하나에 대한 정확한 답

같은 서술자로 `plan`을 하면 성공할 때(메모리 부족은 제외) `OCU_OK`를 돌려주고
`*caps`를 채운다. 그렇지 않으면 `plan`이 돌려줄 상태를 돌려준다.

```c
typedef struct ocu_op_caps {
    ocu_tol_class tol;          /* the class of the operator (section 6) */
    ocu_dtype     accum_dtype;  /* accumulation type used for this plan */
    uint64_t      scratch_bytes;/* on-chip working set of one tile, 0 if n/a */
    uint64_t      tiles;        /* number of tiles the plan runs, >= 1 */
} ocu_op_caps;
```

`accum_dtype`은 f32 입력에서는 `OCU_DT_F32`이다. f16 또는 bf16 입력에서는 백엔드가
실제로 누적하는 대로 f32, f16, bf16 중 하나다. 적합성 시험은 선언한 형식으로
검사하므로(6.1) 잘못 선언하면 실패한다.

### 5.3 `plan`

바로 이 서술자에 대한 계획을 만든다. 계획은 컴파일하느라 오래 걸릴 수 있으며,
텐서 주소를 갖지 않는다. 서술자는 복사되므로 호출자가 다시 써도 된다. 실패하면
`*plan`을 `NULL`로 둔다. 계획은 다른 계획을 만들거나 없애더라도 `destroy`까지
유효하다. 계획은 여러 스레드에서 쓸 수 있지만, 한 계획을 자신의 `destroy`와
동시에 실행해서는 안 된다.

### 5.4 `execute`

계획을 `queue`에서 실행한다(`NULL`이면 실행하고 기다린다). `tensors`는
`num_tensors`개의 항목을 슬롯 순서로 담는다. 그 서술자는 계획한 것과 같아야 하며
(dtype, 랭크, `rank`까지의 `shape`와 `stride`. `OCU_OP_FLAG_VAR_ROWS`가 있으면
행 수가 더 작을 수 있다, 4.7), `num_tensors`도 맞아야 한다. 아니면
`OCU_ERR_INVALID_VALUE`이다: **모양은 계획 시점에 고정된다**. 겹침 규칙은 넘겨받은
그대로의 텐서에 적용한다. 이 백엔드의 장치 메모리가 아니거나, 텐서 범위가 할당을
벗어나거나, `align_bytes`를 어기는 데이터 포인터는 `OCU_ERR_INVALID_VALUE`이다.
(할당 끝 바로 다음을 가리키는 포인터가 다른 할당의 첫 바이트일 수 있다. 백엔드는
할당이 인접하지 않을 때만 이를 거절할 수 있다. npusim은 간격을 둔다, 7절.)
`NULL`이거나 이미 없앤 계획은 `OCU_ERR_INVALID_HANDLE`이다. 이 검사는 `execute`가
돌아오기 전에 이루어지며, 실패하면 아무것도 큐에 들어가지 않는다.

작업은 같은 큐의 다른 작업(복사, 채우기, 커널, 표식, 다른 `execute` 호출)과
순서가 정해진다. 작업이 도는 동안 생긴 오류는 커널과 마찬가지로 다음
`queue_sync` 또는 `queue_query`가 보고한다.

### 5.5 `destroy`

계획을 해제한다. 그 계획을 쓰는 큐의 작업이 끝나기를 기다린다. `NULL`은
`OCU_ERR_INVALID_HANDLE`이다.

### 5.6 `stats` (선택 항목, `NULL`일 수 있음)

장치를 연 뒤로 센 계수기이며 보고서에 쓴다.

```c
typedef struct ocu_op_stats {
    uint64_t op_calls[OCU_OP_COUNT];  /* successful execute calls per op */
    uint64_t plans_created;
    uint64_t plans_refused;           /* plan calls that failed */
    uint64_t bytes_to_device;         /* mem_copy host → device */
    uint64_t bytes_to_host;           /* mem_copy device → host */
    uint64_t bytes_on_device;         /* mem_copy device → device */
    uint64_t kernel_launches_refused; /* kernel_launch → NOT_SUPPORTED */
} ocu_op_stats;
```

## 6. 수치 허용 오차 등급

적합성 시험은 출력 원소 `y`를 기준 값 `r`과 비교한다. `r`은 같은(이미 반올림된)
입력 값으로 binary64에서 계산한다. 표기는 다음과 같다.
`u(dt)`는 단위 반올림 오차로, `2^-24`(f32), `2^-11`(f16), `2^-8`(bf16)이다.
`u_out = u(출력 dtype)`, `u_acc = u(accum_dtype)`.
`tiny(dt)`는 출력 dtype에서 가장 작은 양의 정규수로,
`2^-126`(f32, bf16), `2^-14`(f16)이다.

`r`이 NaN이면 `y`도 NaN이어야 한다. `r`이 ±Inf이면 `y`는 같은 부호의 무한대여야
한다. `OCU_TOL_EXACT`에서 "출력 dtype으로 반올림"에는 오버플로가 포함된다. dtype의
범위를 넘는 유한한 정확한 결과는 그 부호의 무한대이고, `y`는 그 무한대여야 한다.
그 밖의 경우 `y`는 유한해야 하며 등급의 한계를 만족해야 한다.

### 6.1 등급

| 등급 | `|y - r|`의 한계 | 연산 |
|---|---|---|
| `OCU_TOL_EXACT` | `y`가 출력 dtype으로 반올림한 `r`과 같다(비트 단위. `r`이 0이면 ±0 모두 허용. `tiny`보다 작은 결과는 0이어도 된다) | ADD, MUL |
| `OCU_TOL_ULP` (k) | `k · ulp_out(r)`. `ulp_out(r)`은 `|r|`에서 출력 dtype의 간격(`|r| < tiny`이면 `tiny`에서의 간격) | ROPE (k = 8) |
| `OCU_TOL_ROW` | `rtol · |r| + tiny_out`, `rtol`은 아래 연산별로 정한다 | RMS_NORM, SOFTMAX |
| `OCU_TOL_DOT` | `(K + 4) · u_acc · (|alpha| · S_ij + |beta| · |C_old_ij|) + u_out · |r| + tiny_out`, 여기서 `S_ij = sum_k |op(A)_ik| · |op(B)_kj|` | GEMM |

행 연산(`N` = 행 길이):

- RMS_NORM: `rtol = (N + 8) · u_acc + 2 · u_out`.
- SOFTMAX: `rtol = (2N + 2·T_i + 16) · u_acc + 2 · u_out`, 여기서
  `T_i = |scale · x_i| + max_j |scale · x_j|`이다(지수 인수의 오차는 그 크기에 따라
  커진다). 최댓값은 유한한 `x_j`에 대해 취하며, `scale · x_i = -Inf`인 원소는
  `r = 0`이므로 `|y| <= tiny_out`이어야 한다.

행 연산에서 `accum_dtype`은 행의 합을 담는 dtype이다.

이 한계는 어떤 누적 순서(트리, 블록, 순차, 타일로 나눔)와 곱셈-덧셈 융합 여부도
허용한다. 선언한 `accum_dtype`보다 좁은 형식으로 누적하는 것은 허용하지 않는다.

### 6.2 참고

- 어느 등급에서든 비정규 출력은 0으로 비워도 된다(`tiny_out` 항, 또는 `tiny`보다
  작은 결과에 대한 EXACT 규칙).
- GEMM의 한계는 내적에 대한 교과서적 분석이 주는 종류의 원소별 전방 오차 한계
  (`|fl(x·y) - x·y| <= γ_K |x|·|y|`)이며, alpha 곱, beta 항, 마지막 반올림에 대한
  여유를 두었다.

## 7. NPU 시뮬레이터 프로파일(`npusim`)

`backends/npusim`은 호스트에서 돌지만 비SIMT 가속기처럼 동작하는 백엔드다. 하드웨어
없이도 위 계층을 그런 제약에 맞서 시험할 수 있다. 이 백엔드가 강제하는 규칙은
다음과 같다.

1. **커널 없음.** `module_load`, `module_get_kernel`, `kernel_launch`는
   `OCU_ERR_NOT_SUPPORTED`를 돌려준다. `dev_info`는 `warp_size = 1`,
   `max_threads_per_block = 1`, 그리드 한계 1, `supports_unified_memory = 0`을
   보고하고, `max_shared_memory_per_block`은 스크래치패드 크기다.
2. **연산만.** 일은 연산 표와, 큐에서 하는 `mem_copy` / `mem_set`으로 한다.
3. **분리된 메모리 공간.** 장치 메모리는 호스트가 읽거나 쓸 수 없는 주소 범위다
   (호스트에서 장치 포인터를 역참조하면 오류가 난다). 데이터가 드나드는 길은
   `mem_copy`와 `mem_set`뿐이며, 이들은 방향을 검사한다. 호스트 쪽 포인터는 장치
   주소여서는 안 되고, 장치 쪽 포인터는 살아 있는 할당 하나 안에 있어야 한다(아니면
   `OCU_ERR_INVALID_VALUE`). `execute`에 호스트 포인터를 주면
   `OCU_ERR_INVALID_VALUE`이다.
4. **실행 전에 계획.** `execute`는 텐서를 계획과 대조해 검사한다(5.4).
5. **스크래치패드 한계.** 설정한 스크래치패드 크기 `S`바이트
   (`OCU_NPUSIM_SCRATCH_KB`, 기본값 256)가 타일 하나의 작업 집합을 제한한다.
   - GEMM은 타일로 나눈다(`tm × tn` 출력 타일, `tk` 깊이 조각. 각각 최대 128이고
     2의 거듭제곱이며, 128부터 16까지 내려가며 고른다): 작업 집합
     `= (tm·tk + tk·tn) · in_size + tm·tn · 4`. `16 × 16 × 16`조차 들어가지
     않으면 `NOT_SUPPORTED`이다. 타일 나누기를 끄면(`OCU_NPUSIM_TILING=0`) 문제
     전체 `M, N, K`가 타일 하나에 들어가야 하고, 아니면 `NOT_SUPPORTED`이다.
   - RMS_NORM과 SOFTMAX는 한 행을 칩 안에 둔다:
     `N · (in_size + out_size) + 64`바이트(RMS 가중치가 있으면 `N · in_size`를
     더한다)가 `S`에 들어가야 하며, 아니면 `NOT_SUPPORTED`이다. 행은 나누지 않는다.
   - ADD와 MUL은 스트리밍한다: 항상 들어간다.
   `query`는 고른 `scratch_bytes`와 `tiles`를 보고한다.
6. **정렬.** 모든 텐서 데이터 포인터는 64바이트 정렬이다(`align_bytes = 64`).
   아니면 `execute`가 `OCU_ERR_INVALID_VALUE`를 돌려준다.
7. **dtype 집합.** `OCU_NPUSIM_DTYPES`(`f32,f16,bf16`의 쉼표 목록, 기본값은 셋
   모두)가 입력 dtype을 제한한다. 그 밖의 dtype은 `support`, `query`, `plan`에서
   `NOT_SUPPORTED`이다.
8. **계획 지연.** `OCU_NPUSIM_PLAN_MS`(기본값 0)는 성공하는 모든 `plan`에서 그만큼
   잠들어 컴파일을 대신한다. `support`는 같은 값을 `needs_plan_ms`에 보고한다.
9. **계수기.** `stats`는 연산 호출, 계획, 거절된 커널 실행, 방향별 DMA 바이트를
   센다. `OCU_NPUSIM_REPORT=1`이면 장치를 닫을 때 이를 표준 오류로 출력한다.
10. **메모리 크기.** `OCU_NPUSIM_MEM_MB`(기본값 256)는 장치 메모리 공간의 크기다.
    할당은 256바이트 정렬이고 256바이트 이상의 간격을 두고 떨어져 있어서, 한
    할당 바로 다음을 가리키는 포인터가 다른 할당 안에 있는 일은 없다.
11. 모든 연산에서 누적은 f32다(`accum_dtype = OCU_DT_F32`).
12. 가변 행 수(4.7)를 받는다(`var_rows = 1`). 스크래치패드와 타일 수치는 계획한
    (가장 큰) 모양의 것이다.

## 8. 라이브러리 계층에서 표 쓰기

`libs/blas` 같은 라이브러리는 연산을 다음과 같이 표를 거쳐 보낸다(칩 이름은 쓰지
않는다).

1. `t = backend->get_provider ? backend->get_provider("ops") : NULL`.
   `t->version >= 1`이고 `t->struct_size`가 쓰려는 항목을 덮을 때만 쓴다.
2. 서술자를 만들고 `query`를 호출한다. `OCU_OK`이면 연산 경로를 쓴다.
   `NOT_SUPPORTED` / `NOT_IMPLEMENTED`이면 백엔드에 커널이 있을 때 커널 경로(또는
   BLAS 공급자)를 쓰고, 없으면 사용자에게 `NOT_SUPPORTED`를 보고한다.
3. 서술자 전체(모양, 보폭, dtype, alpha와 beta를 포함한 매개변수)를 키로 계획을
   캐시한다. 행 수가 변하는 곳(GEMM의 M)에서는 `support`가 `var_rows`를 보고하면
   `OCU_OP_FLAG_VAR_ROWS`와 예상되는 가장 큰 M으로 한 번만 계획한다. 그렇지 않으면
   M도 캐시 키에 넣고 몇 개의 행 구간을 쓴다.
4. 스트림의 큐에서 `execute`한다.

열 우선 GEMM(BLAS 관례)은 복사 없이 이 행 우선 표에 대응된다. 선행 차원이 `ld`인
열 우선 `m × n` 행렬은 행 보폭이 `ld`인 행 우선 `[n, m]` 행렬과 같다. 따라서
(열 우선, `m, n, k`의) `C = alpha · op(A) · op(B) + beta · C`는 표의 GEMM에서
B 버퍼를 슬롯 A에, A 버퍼를 슬롯 B에, C를 슬롯 C에 넣은 것이다.

대응의 자세한 내용: 열 우선 `C(m×n) = op(A)(m×k) · op(B)(k×n)`은 행 우선
`C'(n×m) = op(B)'(n×k) · op(A)'(k×m)`이다. 슬롯 A'은 B 버퍼다. 그 행 우선 보기는
`opb = N`이면 보폭 `ldb`인 `[n, k]`(`trans_a' = 0`)이고, `opb = T`이면 보폭
`ldb`인 `[k, n]`(`trans_a' = 1`)이다. 슬롯 B'은 A 버퍼다: `opa = N`이면 보폭
`lda`인 `[k, m]`(`trans_b' = 0`)이고, `opa = T`이면 보폭 `lda`인 `[m, k]`
(`trans_b' = 1`)이다. 슬롯 C'은 보폭 `ldc`인 `[n, m]`이다.

## 9. 버전 이력

| 버전 | 변경 |
|---|---|
| 1 | 첫 버전: GEMM, ADD, MUL, RMS_NORM, SOFTMAX, 가변 행 수. ROPE는 정의하되 선택. QMATMUL과 블록 형식은 예약. |

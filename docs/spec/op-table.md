# Operator table (`ocu_ops`), version 1

Status: draft, version 1 (2026-11-13). Header: `backends/interface/ocu_ops.h`.

## 1. Purpose and position

The device backend interface (`ocu_backend.h`) runs CUDA kernels. Some
accelerators have no SIMT threads: their public runtimes run precompiled
operators with fixed shapes, on-chip scratchpad memory and DMA. Such a chip
cannot run an unmodified kernel, but it can run library-level operators
(GEMM, normalisation, softmax, element-wise arithmetic).

The operator table is a second, optional way for a backend to do work:

```
 libs/blas, libs/dnn, an out-of-tree ggml backend, ...
        │                         │
  kernel path                operator path
  (kernel_launch)            (op table: support → query → plan → execute)
        │                         │
  backends with threads       any backend that fills the table
```

- A backend offers the table through the existing optional entry
  `get_provider`: `get_provider("ops")` (`OCU_OPS_PROVIDER`) returns a
  `const ocu_op_table *`, or `NULL` when the backend has none. The backend
  table (`ocu_backend`) and `OCU_BACKEND_ABI` do not change.
- The op table has its own version (`OCU_OPS_VERSION`, now 1) and its own
  size field (`struct_size`). Callers check both before using an entry.
  Entries are only ever appended; a caller compiled against an older header
  sees a prefix of the table it knows.
- Callers above `backends/` choose between paths by asking the table, never
  by backend name. The inline helper `ocu_ops_of(backend)` returns the
  table, or `NULL` when there is none or its version or size is too small
  for version 1.

## 2. Data types

| `ocu_dtype` | Value | Element | Bytes |
|---|---|---|---|
| `OCU_DT_NONE` | 0 | no tensor (never valid in a descriptor) | 0 |
| `OCU_DT_F32` | 1 | IEEE 754 binary32 | 4 |
| `OCU_DT_F16` | 2 | IEEE 754 binary16 | 2 |
| `OCU_DT_BF16` | 3 | bfloat16 (upper 16 bits of binary32) | 2 |
| `OCU_DT_I8` | 4 | signed 8-bit integer | 1 |
| `OCU_DT_I32` | 5 | signed 32-bit integer | 4 |
| `OCU_DT_Q8_B32` | 0x101 | block-quantised: blocks of 32 values, one binary16 scale then 32 `int8` | 34 per block |
| `OCU_DT_Q4_B32` | 0x102 | block-quantised: blocks of 32 values, one binary16 scale then 16 bytes of 4-bit values | 18 per block |

`ocu_dtype_is_blocked(dt)` is true for values `>= 0x100` (the block
family). No version 1 operator accepts a block-quantised tensor; the values
are reserved so that descriptors and support tables can name them. For a
block type, `shape` still counts values, the innermost dimension must be a
multiple of the block length, and strides of the outer dimensions count
values (multiples of the block length). The layout of the bytes inside a
block is fixed in the version that first gives an operator using it.

Conversions to binary16 and bfloat16 round to nearest, ties to even.
Implementations may flush subnormal results to zero (see 6.2).

## 3. Tensors

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

Element `(i0, .., i_{r-1})` is at `data + sum(i_k * stride[k]) * elem_size`.
Entries at and after `rank` are ignored.

A descriptor is **well formed** when all of the following hold; otherwise
`query`, `plan` and `execute` return `OCU_ERR_INVALID_VALUE`:

1. `dtype` is one of the values in section 2 and `1 <= rank <= 4`.
2. Every `shape[k] >= 1`. (Empty tensors are not described; callers skip
   the call, as BLAS quick returns do.)
3. The innermost dimension is contiguous: `stride[rank-1] == 1`.
4. Outer strides do not overlap inner extents:
   `stride[k] >= stride[k+1] * shape[k+1]` for `k < rank-1`.
5. The element count `prod(shape)` and the span
   `1 + sum((shape[k]-1) * stride[k])` fit in `int64_t`.

Layout names used by the support query:

- **dense**: `stride[k] == stride[k+1] * shape[k+1]` for every `k` (packed
  row-major).
- **row-strided**: well formed but not dense (padding between rows or
  matrices, such as a leading dimension larger than the row length).

## 4. Operators

```c
typedef struct ocu_op_desc {
    uint32_t        struct_size;   /* sizeof(ocu_op_desc) of the caller */
    ocu_op          op;
    uint32_t        num_tensors;   /* inputs, then the output (last) */
    uint32_t        flags;         /* 0 in version 1 */
    ocu_tensor_desc tensors[OCU_OPS_MAX_TENSORS];  /* 4 */
    union { ... } p;               /* per-operator parameters */
} ocu_op_desc;
```

The output is always the last slot. `flags` must be 0 and `struct_size`
must be at least the version 1 size, else `OCU_ERR_INVALID_VALUE`. Every
tensor slot must be well formed (section 3).

Shapes below are written outermost first. "Rows" of a row operator are all
dimensions but the innermost; `N = shape[rank-1]` is the row length.

| `ocu_op` | Value | Slots | Result |
|---|---|---|---|
| `OCU_OP_GEMM` | 1 | A, B, C | `C = alpha * op(A) · op(B) + beta * C` |
| `OCU_OP_ADD` | 2 | a, b, y | `y = a + b` (b broadcasts) |
| `OCU_OP_MUL` | 3 | a, b, y | `y = a * b` (b broadcasts) |
| `OCU_OP_RMS_NORM` | 4 | x, y or x, w, y | `y = x / sqrt(mean(x²) + eps) [* w]` per row |
| `OCU_OP_SOFTMAX` | 5 | x, y | `y = softmax(scale * x)` per row |
| `OCU_OP_ROPE` | 6 | x, pos, y | rotary position embedding (4.6) |
| `OCU_OP_QMATMUL` | 7 | reserved | block-quantised matrix product, parameters in a later version |

### 4.1 GEMM

Parameters `p.gemm`: `uint32_t trans_a, trans_b` (0 or 1), `float alpha, beta`.

- Rank 2: A stored `[M,K]` (or `[K,M]` when `trans_a`), B stored `[K,N]`
  (or `[N,K]` when `trans_b`), C `[M,N]`. `op(X)` is X or its transpose.
- Rank 3: a leading batch dimension. C is `[Bt,M,N]`; A and B have batch
  `Bt` or 1 (1 means the same matrix for every batch entry). All three
  tensors have the same rank.
- Types: A and B have the same dtype, one of f32, f16, bf16. C is f32, or
  the same dtype as A and B. (f16 or bf16 inputs with an f32 C is "mixed".)
  int8 inputs are reserved for a later version (`NOT_SUPPORTED` or
  `NOT_IMPLEMENTED`).
- The products are accumulated in the backend's accumulation type
  (`caps.accum_dtype`, see 5.2), then `alpha * acc + beta * C_old` is
  formed in at least that precision and rounded once to C's dtype.
- `beta == 0`: C is not read (NaN or Inf in C does not propagate).
  `alpha == 0`: A and B are not read; `C = beta * C`.
- `trans_a`/`trans_b` other than 0 or 1, mismatched M/N/K, or a batch rule
  violation: `OCU_ERR_INVALID_VALUE`.
- C must not overlap A or B.

### 4.2 ADD and MUL

No parameters. `a`, `b`, `y` have the same rank and the same dtype (f32,
f16, bf16). `y` has the shape of `a`. For each `k`, `b.shape[k]` equals
`a.shape[k]` or is 1 (broadcast along that dimension). The exact result of
`a + b` (or `a * b`) is rounded once to the dtype. `y` may be the same
tensor as `a` (same pointer and descriptor); any other overlap between `y`
and an input is not allowed.

### 4.3 RMS_NORM

Parameter `p.rms_norm.eps` (float, `>= 0`, else `OCU_ERR_INVALID_VALUE`).
`x` and `y` have the same shape and dtype (f32, f16, bf16). The optional
weight `w` is rank 1, shape `[N]`, same dtype. For each row:
`y_i = x_i * (1 / sqrt(sum_j x_j² / N + eps)) * w_i` (the `w_i` factor only
when present). `y` may be the same tensor as `x`.

### 4.4 SOFTMAX

Parameter `p.softmax.scale` (float, finite, else `OCU_ERR_INVALID_VALUE`).
`x` and `y` have the same shape and dtype (f32, f16, bf16). For each row,
with `t_j = scale * x_j` and `m = max_j t_j`:
`y_i = exp(t_i - m) / sum_j exp(t_j - m)`. `y` may be the same tensor as `x`.
Rows containing NaN or `+Inf` give unspecified values in that row; other
rows are unaffected.

### 4.5 Shared rules

- Device pointers are pointers the backend's `mem_alloc` returned, plus an
  offset; the whole span of every tensor lies inside one allocation.
- `num_tensors` other than listed for the operator: `OCU_ERR_INVALID_VALUE`.
- An operator value the header does not list: `OCU_ERR_INVALID_VALUE`.

### 4.6 ROPE (defined, optional to implement in version 1)

Parameters `p.rope`: `uint32_t mode` (0 = adjacent pairs `(2i, 2i+1)`,
1 = split halves `(i, i + d/2)`), `uint32_t n_dims` (`d`, even, `<= N`),
`float theta_base`, `float freq_scale`.
`x` is `[T, H, N]` (tokens, heads, head size) or `[T, N]`; `pos` is `[T]`
of `OCU_DT_I32`; `y` has the shape and dtype of `x`. For token `t` with
`p = pos[t]` and pair index `i < d/2`, the angle is
`a = p * freq_scale * theta_base^(-2i/d)`, and the pair `(u, v)` becomes
`(u cos a - v sin a, u sin a + v cos a)`. Elements at positions `>= d` in
each row are copied. Tolerance class ULP(8) on f32 (6.1), measured against
the reference evaluated in binary64. Backends that do not implement it yet
return `OCU_ERR_NOT_IMPLEMENTED` from `support`, `query` and `plan`.

## 5. Table entries

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

Status values have the meaning they have in `ocu_backend.h`. In particular
`OCU_ERR_NOT_SUPPORTED` means the chip cannot do this case, and
`OCU_ERR_NOT_IMPLEMENTED` means the backend does not do it yet. Every entry
returns `OCU_ERR_INVALID_DEVICE` for a `NULL` device and
`OCU_ERR_INVALID_VALUE` for a `NULL` output pointer.

### 5.1 `support`: coarse capability per operator × dtype

For the support table. `dtype` is the input dtype (for GEMM, the dtype of A
and B). Returns `OCU_OK` and fills `*out` when some descriptors with that
operator and dtype can be planned; `OCU_ERR_NOT_SUPPORTED` or
`OCU_ERR_NOT_IMPLEMENTED` otherwise (`*out` is then zeroed).

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
} ocu_op_support;
```

`support` is a summary. A descriptor can still be refused by `query`
(for example a row longer than the scratchpad). `support` returning
`NOT_SUPPORTED` means every descriptor with that pair is refused.

### 5.2 `query`: exact answer for one descriptor

Returns `OCU_OK` when `plan` with the same descriptor would succeed (apart
from running out of memory), and fills `*caps`. Otherwise returns the
status `plan` would return.

```c
typedef struct ocu_op_caps {
    ocu_tol_class tol;          /* the class of the operator (section 6) */
    ocu_dtype     accum_dtype;  /* accumulation type used for this plan */
    uint64_t      scratch_bytes;/* on-chip working set of one tile, 0 if n/a */
    uint64_t      tiles;        /* number of tiles the plan runs, >= 1 */
} ocu_op_caps;
```

`accum_dtype` is `OCU_DT_F32` for f32 inputs. For f16 or bf16 inputs it is
f32, f16 or bf16 as the backend actually accumulates; conformance checks
with the declared type (6.1), so declaring it wrongly fails.

### 5.3 `plan`

Creates a plan for exactly this descriptor. A plan may compile and take
long; it holds no tensor addresses. The descriptor is copied; the caller
may reuse it. On failure `*plan` is set to `NULL`. A plan stays valid until
`destroy`, even if other plans are created or destroyed. Plans may be used
from several threads, but one plan is not executed concurrently with its
own `destroy`.

### 5.4 `execute`

Runs a plan on `queue` (`NULL`: run and wait). `tensors` has
`num_tensors` entries in slot order; their descriptors must be identical to
the planned ones (dtype, rank, and `shape` and `stride` up to `rank`) and
`num_tensors` must match, else `OCU_ERR_INVALID_VALUE`: **shapes are fixed
at plan time**. A data pointer that is not device memory of this backend,
whose tensor span leaves its allocation, or that breaks `align_bytes`,
gives `OCU_ERR_INVALID_VALUE`. A `NULL` or destroyed plan gives
`OCU_ERR_INVALID_HANDLE`. These checks happen before `execute` returns;
nothing is queued when they fail.

Work is ordered with other work on the same queue (copies, fills, kernels,
markers, other `execute` calls). Errors raised while the work runs are
reported by the next `queue_sync` or `queue_query`, as for kernels.

### 5.5 `destroy`

Releases a plan. Waits for queued work that uses it. `NULL` gives
`OCU_ERR_INVALID_HANDLE`.

### 5.6 `stats` (optional entry, may be `NULL`)

Counters since the device was opened, for reports.

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

## 6. Numeric tolerance classes

Conformance compares an output element `y` with a reference `r` computed in
binary64 from the same (already rounded) input values. Notation:
`u(dt)` is the unit roundoff: `2^-24` (f32), `2^-11` (f16), `2^-8` (bf16).
`u_out = u(output dtype)`, `u_acc = u(accum_dtype)`.
`tiny(dt)` is the smallest positive normal number of the output dtype:
`2^-126` (f32, bf16), `2^-14` (f16).

If `r` is NaN, `y` must be NaN. If `r` is ±Inf, `y` must be the same
infinity. Otherwise `y` must be finite and satisfy the class bound.

### 6.1 Classes

| Class | Bound on `|y - r|` | Operators |
|---|---|---|
| `OCU_TOL_EXACT` | `y` equals `r` rounded to the output dtype (bitwise; ±0 both accepted for a zero `r`; a result below `tiny` may also be 0) | ADD, MUL |
| `OCU_TOL_ULP` (k) | `k · ulp_out(r)`, `ulp_out(r)` = spacing of the output dtype at `|r|` (for `|r| < tiny`, the spacing at `tiny`) | ROPE (k = 8) |
| `OCU_TOL_ROW` | `rtol · |r| + tiny_out`, `rtol` per operator below | RMS_NORM, SOFTMAX |
| `OCU_TOL_DOT` | `(K + 4) · u_acc · (|alpha| · S_ij + |beta| · |C_old_ij|) + u_out · |r| + tiny_out`, where `S_ij = sum_k |op(A)_ik| · |op(B)_kj|` | GEMM |

Row operators (`N` = row length):

- RMS_NORM: `rtol = (N + 8) · u_acc + 2 · u_out`.
- SOFTMAX: `rtol = (2N + 2·T_i + 16) · u_acc + 2 · u_out`, where
  `T_i = |scale · x_i| + max_j |scale · x_j|` (the error of the exponent
  argument grows with its size).

For row operators `accum_dtype` is the dtype the row sum is kept in.

These bounds allow any order of accumulation (tree, blocked, sequential,
split over tiles) and fused or unfused multiply-add. They do not allow
accumulating in a narrower type than the declared `accum_dtype`.

### 6.2 Notes

- Subnormal outputs may be flushed to zero in every class (`tiny_out`
  term, or the EXACT rule for results below `tiny`).
- For GEMM, the bound is an element-wise forward error bound of the kind
  textbook analyses of inner products give (`|fl(x·y) - x·y| <= γ_K |x|·|y|`),
  with slack for the scaling by alpha, the beta term and the final
  rounding.

## 7. NPU simulator profile (`npusim`)

`backends/npusim` is a backend that runs on the host but behaves the way a
non-SIMT accelerator does, so the layers above can be tested against those
limits without the hardware. The rules it enforces:

1. **No kernels.** `module_load`, `module_get_kernel` and `kernel_launch`
   return `OCU_ERR_NOT_SUPPORTED`. `dev_info` reports `warp_size = 1`,
   `max_threads_per_block = 1`, grid limits 1, `supports_unified_memory = 0`,
   and `max_shared_memory_per_block` = the scratchpad size.
2. **Only operators.** Work is done by the op table and by `mem_copy` /
   `mem_set` on a queue.
3. **Separate memory space.** Device memory is an address range the host
   cannot read or write (dereferencing a device pointer on the host faults).
   `mem_copy` and `mem_set` are the only way data enters or leaves it, and
   they check direction: host-side pointers must not be device addresses,
   device-side pointers must lie inside one live allocation (else
   `OCU_ERR_INVALID_VALUE`). A host pointer given to `execute` is
   `OCU_ERR_INVALID_VALUE`.
4. **Plan before execute.** `execute` checks the tensors against the plan
   (5.4).
5. **Scratchpad limit.** The configured scratchpad size `S` bytes
   (`OCU_NPUSIM_SCRATCH_KB`, default 256) bounds the working set of one
   tile:
   - GEMM is tiled (`tm × tn` output tile, `tk` depth slice, each at most
     128 and a power of two, chosen from 128 down to 16):
     working set `= (tm·tk + tk·tn) · in_size + tm·tn · 4`. If even
     `16 × 16 × 16` does not fit, `NOT_SUPPORTED`. With tiling disabled
     (`OCU_NPUSIM_TILING=0`) the whole problem `M, N, K` must fit as one
     tile, else `NOT_SUPPORTED`.
   - RMS_NORM and SOFTMAX keep one row on chip:
     `N · (in_size + out_size) + 64` bytes (plus `N · in_size` for the RMS
     weight) must fit in `S`, else `NOT_SUPPORTED`. Rows are never split.
   - ADD and MUL are streamed: always fit.
   `query` reports the chosen `scratch_bytes` and `tiles`.
6. **Alignment.** Every tensor data pointer is 64-byte aligned
   (`align_bytes = 64`), else `OCU_ERR_INVALID_VALUE` from `execute`.
7. **Dtype set.** `OCU_NPUSIM_DTYPES` (comma list of `f32,f16,bf16`,
   default all three) limits the input dtypes; others are `NOT_SUPPORTED`
   in `support`, `query` and `plan`.
8. **Plan latency.** `OCU_NPUSIM_PLAN_MS` (default 0) sleeps in every
   successful `plan`, standing in for compilation.
9. **Counters.** `stats` counts op calls, plans, refused kernel launches
   and DMA bytes by direction. `OCU_NPUSIM_REPORT=1` prints them to
   standard error when the device is closed.
10. **Memory size.** `OCU_NPUSIM_MEM_MB` (default 256) is the size of the
    device memory space.
11. Accumulation is f32 for every operator (`accum_dtype = OCU_DT_F32`).

## 8. Using the table from a library layer

A library such as `libs/blas` routes an operation through the table like
this (no chip names):

1. `t = backend->get_provider ? backend->get_provider("ops") : NULL`; use it
   only when `t->version >= 1` and `t->struct_size` covers the entries used.
2. Build the descriptor; call `query`. `OCU_OK` → use the operator path.
   `NOT_SUPPORTED` / `NOT_IMPLEMENTED` → use the kernel path (or the BLAS
   provider) when the backend has kernels; otherwise report
   `NOT_SUPPORTED` to the user.
3. Cache plans keyed by the whole descriptor (shapes, strides, dtypes,
   parameters, including alpha and beta).
4. `execute` on the stream's queue.

Column-major GEMM (the BLAS convention) maps to this row-major table
without copies: a column-major `m × n` matrix with leading dimension `ld`
is the row-major `[n, m]` matrix with row stride `ld`. So
`C = alpha · op(A) · op(B) + beta · C` (column-major, `m, n, k`) is the
table's GEMM with the B buffer in slot A, the A buffer in slot B, and C in
slot C.

Mapping in detail: column-major `C(m×n) = op(A)(m×k) · op(B)(k×n)` is
row-major `C'(n×m) = op(B)'(n×k) · op(A)'(k×m)`. Slot A' is the B buffer:
its row-major view is `[n, k]` with stride `ldb` when `opb = N`
(`trans_a' = 0`), and `[k, n]` with stride `ldb` when `opb = T`
(`trans_a' = 1`). Slot B' is the A buffer: `[k, m]` with stride `lda`
when `opa = N` (`trans_b' = 0`), and `[m, k]` with stride `lda` when
`opa = T` (`trans_b' = 1`). Slot C' is `[n, m]` with stride `ldc`.

## 9. Version history

| Version | Change |
|---|---|
| 1 | First version: GEMM, ADD, MUL, RMS_NORM, SOFTMAX; ROPE defined, optional; QMATMUL and block types reserved. |

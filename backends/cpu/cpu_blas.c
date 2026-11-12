/*
 * CPU backend: BLAS provider (see backends/interface/ocu_blas_provider.h).
 *
 * Every operation is a kernel run through cpu_kernel_launch, so it obeys
 * queue order and runs on the same fibers as user kernels. The kernels are
 * written the way simple GPU kernels are: one thread per output element,
 * block-level tree reductions in shared memory, plain accumulation in the
 * element type. They are the reference for correctness, not for speed.
 */
#include "cpu_internal.h"
#include "ocu_blas_provider.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define RED_THREADS 256u /* threads of the single block used by reductions */
#define TILE 16u

static inline int64_t vidx(int64_t i, int n, int inc)
{
    return inc > 0 ? i * inc : ((int64_t)n - 1 - i) * (int64_t)(-inc);
}

static uint32_t blocks_for(int64_t count, uint32_t per)
{
    return (uint32_t)((count + per - 1) / per);
}

static ocu_status launch(ocu_queue *q, ocu_cpu_kernel_fn fn, uint32_t gx,
                         uint32_t gy, uint32_t bx, uint32_t by, size_t shared,
                         void *argstruct, size_t argsize)
{
    struct ocu_kernel k = {NULL, fn}; /* fn is copied by the launch */
    void *args[1] = {argstruct};
    size_t sizes[1] = {argsize};
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = gx; p.grid[1] = gy; p.grid[2] = 1;
    p.block[0] = bx; p.block[1] = by; p.block[2] = 1;
    p.shared_memory_bytes = shared;
    p.args = args;
    p.arg_sizes = sizes;
    p.num_args = 1;
    return cpu_kernel_launch(q, &k, &p);
}

static inline uint32_t gtid_x(void)
{
    return ocu_cpu_block_idx().x * ocu_cpu_block_dim().x + ocu_cpu_thread_idx().x;
}

static inline uint32_t gtid_y(void)
{
    return ocu_cpu_block_idx().y * ocu_cpu_block_dim().y + ocu_cpu_thread_idx().y;
}

/* ---- argument blocks ---------------------------------------------------- */

typedef struct vec_args {
    int n, incx, incy;
    double alpha;      /* converted to the element type in the kernel */
    const void *x;
    void *y;
    void *result;
} vec_args;

typedef struct mat_args {
    int m, n, k, lda, ldb, ldc, incx, incy;
    int opa, opb, uplo, side, diag;
    double alpha, beta;
    const void *a, *b, *x;
    void *c, *y;
} mat_args;

/*
 * Kernels for one element type. T is float or double; S is the suffix.
 * Alpha and beta travel as double and are rounded to T once.
 */
#define DEFINE_KERNELS(T, S, SQRT, FABS)                                       \
                                                                               \
static void k_axpy_##S(void **a)                                               \
{                                                                              \
    const vec_args *v = a[0];                                                  \
    uint32_t i = gtid_x();                                                     \
    if ((int)i >= v->n) return;                                                \
    const T *x = v->x;                                                         \
    T *y = v->y;                                                               \
    T al = (T)v->alpha;                                                        \
    int64_t yi = vidx(i, v->n, v->incy);                                       \
    y[yi] = al * x[vidx(i, v->n, v->incx)] + y[yi];                            \
}                                                                              \
                                                                               \
static void k_scal_##S(void **a)                                               \
{                                                                              \
    const vec_args *v = a[0];                                                  \
    uint32_t i = gtid_x();                                                     \
    if ((int)i >= v->n) return;                                                \
    T *x = v->y;                                                               \
    x[(int64_t)i * v->incx] = (T)v->alpha * x[(int64_t)i * v->incx];           \
}                                                                              \
                                                                               \
static void k_copy_##S(void **a)                                               \
{                                                                              \
    const vec_args *v = a[0];                                                  \
    uint32_t i = gtid_x();                                                     \
    if ((int)i >= v->n) return;                                                \
    const T *x = v->x;                                                         \
    T *y = v->y;                                                               \
    y[vidx(i, v->n, v->incy)] = x[vidx(i, v->n, v->incx)];                     \
}                                                                              \
                                                                               \
static void k_swap_##S(void **a)                                               \
{                                                                              \
    const vec_args *v = a[0];                                                  \
    uint32_t i = gtid_x();                                                     \
    if ((int)i >= v->n) return;                                                \
    T *x = (T *)(uintptr_t)v->x;                                               \
    T *y = v->y;                                                               \
    int64_t xi = vidx(i, v->n, v->incx), yi = vidx(i, v->n, v->incy);          \
    T t = x[xi];                                                               \
    x[xi] = y[yi];                                                             \
    y[yi] = t;                                                                 \
}                                                                              \
                                                                               \
/* One block; thread t handles elements t, t+256, ...; then a tree. */        \
static T block_sum_##S(T *tile, T mine)                                        \
{                                                                              \
    uint32_t t = ocu_cpu_thread_idx().x;                                       \
    tile[t] = mine;                                                            \
    ocu_cpu_barrier();                                                         \
    for (uint32_t s = RED_THREADS / 2; s > 0; s >>= 1) {                       \
        if (t < s) tile[t] += tile[t + s];                                     \
        ocu_cpu_barrier();                                                     \
    }                                                                          \
    T r = tile[0];                                                             \
    ocu_cpu_barrier();                                                         \
    return r;                                                                  \
}                                                                              \
                                                                               \
static T block_max_##S(T *tile, T mine)                                        \
{                                                                              \
    uint32_t t = ocu_cpu_thread_idx().x;                                       \
    tile[t] = mine;                                                            \
    ocu_cpu_barrier();                                                         \
    for (uint32_t s = RED_THREADS / 2; s > 0; s >>= 1) {                       \
        if (t < s && tile[t + s] > tile[t]) tile[t] = tile[t + s];             \
        ocu_cpu_barrier();                                                     \
    }                                                                          \
    T r = tile[0];                                                             \
    ocu_cpu_barrier();                                                         \
    return r;                                                                  \
}                                                                              \
                                                                               \
static void k_dot_##S(void **a)                                                \
{                                                                              \
    const vec_args *v = a[0];                                                  \
    const T *x = v->x, *y = v->y;                                              \
    T s = 0;                                                                   \
    for (int64_t i = ocu_cpu_thread_idx().x; i < v->n; i += RED_THREADS)      \
        s += x[vidx(i, v->n, v->incx)] * y[vidx(i, v->n, v->incy)];            \
    s = block_sum_##S(ocu_cpu_dynamic_shared(), s);                            \
    if (ocu_cpu_thread_idx().x == 0) *(T *)v->result = s;                      \
}                                                                              \
                                                                               \
static void k_asum_##S(void **a)                                               \
{                                                                              \
    const vec_args *v = a[0];                                                  \
    const T *x = v->x;                                                         \
    T s = 0;                                                                   \
    for (int64_t i = ocu_cpu_thread_idx().x; i < v->n; i += RED_THREADS)      \
        s += FABS(x[i * v->incx]);                                             \
    s = block_sum_##S(ocu_cpu_dynamic_shared(), s);                            \
    if (ocu_cpu_thread_idx().x == 0) *(T *)v->result = s;                      \
}                                                                              \
                                                                               \
/* Scaled: sqrt(sum (x/scale)^2) * scale never overflows or underflows. */    \
static void k_nrm2_##S(void **a)                                               \
{                                                                              \
    const vec_args *v = a[0];                                                  \
    const T *x = v->x;                                                         \
    T *tile = ocu_cpu_dynamic_shared();                                        \
    T mx = 0;                                                                  \
    int nan_seen = 0;                                                          \
    for (int64_t i = ocu_cpu_thread_idx().x; i < v->n; i += RED_THREADS) {    \
        T e = FABS(x[i * v->incx]);                                            \
        if (isnan(e)) nan_seen = 1;                                            \
        else if (e > mx) mx = e;                                               \
    }                                                                          \
    T any_nan = block_sum_##S(tile, (T)nan_seen);                              \
    T scale = block_max_##S(tile, mx);                                         \
    T s = 0;                                                                   \
    if (any_nan == 0 && scale > 0 && !isinf(scale)) {                          \
        for (int64_t i = ocu_cpu_thread_idx().x; i < v->n; i += RED_THREADS) { \
            T r = x[i * v->incx] / scale;                                      \
            s += r * r;                                                        \
        }                                                                      \
    }                                                                          \
    s = block_sum_##S(tile, s);                                                \
    if (ocu_cpu_thread_idx().x == 0) {                                         \
        T out;                                                                 \
        if (any_nan != 0) out = (T)NAN;                                        \
        else if (scale == 0 || isinf(scale)) out = scale;                      \
        else out = scale * SQRT(s);                                            \
        *(T *)v->result = out;                                                 \
    }                                                                          \
}                                                                              \
                                                                               \
/* First index of the largest |x|; NaN elements are skipped. */               \
static void k_iamax_##S(void **a)                                              \
{                                                                              \
    const vec_args *v = a[0];                                                  \
    const T *x = v->x;                                                         \
    uint32_t t = ocu_cpu_thread_idx().x;                                       \
    T best = -1;                                                               \
    int64_t best_i = 0;                                                        \
    for (int64_t i = t; i < v->n; i += RED_THREADS) {                          \
        T e = FABS(x[i * v->incx]);                                            \
        if (!isnan(e) && e > best) { best = e; best_i = i; }                   \
    }                                                                          \
    T *val = ocu_cpu_dynamic_shared();                                         \
    int64_t *idx = (int64_t *)(void *)(val + RED_THREADS);                     \
    val[t] = best;                                                             \
    idx[t] = best_i;                                                           \
    ocu_cpu_barrier();                                                         \
    for (uint32_t s = RED_THREADS / 2; s > 0; s >>= 1) {                       \
        if (t < s) {                                                           \
            if (val[t + s] > val[t] ||                                         \
                (val[t + s] == val[t] && idx[t + s] < idx[t])) {               \
                val[t] = val[t + s];                                           \
                idx[t] = idx[t + s];                                           \
            }                                                                  \
        }                                                                      \
        ocu_cpu_barrier();                                                     \
    }                                                                          \
    if (t == 0) *(int *)v->result = (int)idx[0] + 1;                           \
}                                                                              \
                                                                               \
static inline T op_get_##S(const T *m, int ld, int op, int i, int j)           \
{                                                                              \
    return op == OCU_BLAS_OP_N ? m[i + (int64_t)j * ld] : m[j + (int64_t)i * ld]; \
}                                                                              \
                                                                               \
static void k_gemv_##S(void **a)                                               \
{                                                                              \
    const mat_args *g = a[0];                                                  \
    int rows = g->opa == OCU_BLAS_OP_N ? g->m : g->n;                          \
    int cols = g->opa == OCU_BLAS_OP_N ? g->n : g->m;                          \
    uint32_t i = gtid_x();                                                     \
    if ((int)i >= rows) return;                                                \
    const T *A = g->a, *x = g->x;                                              \
    T *y = g->y;                                                               \
    T s = 0;                                                                   \
    for (int j = 0; j < cols; ++j)                                             \
        s += op_get_##S(A, g->lda, g->opa, (int)i, j) * x[vidx(j, cols, g->incx)]; \
    int64_t yi = vidx(i, rows, g->incy);                                       \
    T r = (T)g->alpha * s;                                                     \
    if (g->beta != 0) r += (T)g->beta * y[yi];                                 \
    y[yi] = r;                                                                 \
}                                                                              \
                                                                               \
static void k_ger_##S(void **a)                                                \
{                                                                              \
    const mat_args *g = a[0];                                                  \
    uint32_t i = gtid_x(), j = gtid_y();                                       \
    if ((int)i >= g->m || (int)j >= g->n) return;                              \
    const T *x = g->x, *y = g->b;                                              \
    T *A = g->c;                                                               \
    int64_t q = i + (int64_t)j * g->lda;                                       \
    A[q] = (T)g->alpha * x[vidx(i, g->m, g->incx)] * y[vidx(j, g->n, g->incy)] + A[q]; \
}                                                                              \
                                                                               \
static void k_gemm_##S(void **a)                                               \
{                                                                              \
    const mat_args *g = a[0];                                                  \
    uint32_t i = gtid_x(), j = gtid_y();                                       \
    if ((int)i >= g->m || (int)j >= g->n) return;                              \
    const T *A = g->a, *B = g->b;                                              \
    T *C = g->c;                                                               \
    T s = 0;                                                                   \
    for (int p = 0; p < g->k; ++p)                                             \
        s += op_get_##S(A, g->lda, g->opa, (int)i, p) *                        \
             op_get_##S(B, g->ldb, g->opb, p, (int)j);                         \
    int64_t q = i + (int64_t)j * g->ldc;                                       \
    T r = (T)g->alpha * s;                                                     \
    if (g->beta != 0) r += (T)g->beta * C[q]; /* beta == 0: C not read */      \
    C[q] = r;                                                                  \
}                                                                              \
                                                                               \
static void k_syrk_##S(void **a)                                               \
{                                                                              \
    const mat_args *g = a[0];                                                  \
    uint32_t i = gtid_x(), j = gtid_y();                                       \
    if ((int)i >= g->n || (int)j >= g->n) return;                              \
    if (g->uplo == OCU_BLAS_UPPER ? i > j : i < j) return;                     \
    const T *A = g->a;                                                         \
    T *C = g->c;                                                               \
    T s = 0;                                                                   \
    for (int p = 0; p < g->k; ++p)                                             \
        s += op_get_##S(A, g->lda, g->opa, (int)i, p) *                        \
             op_get_##S(A, g->lda, g->opa, (int)j, p);                         \
    int64_t q = i + (int64_t)j * g->ldc;                                       \
    T r = (T)g->alpha * s;                                                     \
    if (g->beta != 0) r += (T)g->beta * C[q];                                  \
    C[q] = r;                                                                  \
}                                                                              \
                                                                               \
/* Solve op(A) x = b in place for one right-hand side (strided). */           \
static void tri_solve_##S(const T *A, int lda, int n, int uplo, int op,        \
                          int unit, T *b, int64_t stride_b, int64_t off_b,      \
                          int nb, int incb)                                     \
{                                                                              \
    /* op(A) is lower when (A lower, N) or (A upper, T) */                     \
    int lower = (uplo == OCU_BLAS_LOWER) == (op == OCU_BLAS_OP_N);             \
    for (int s = 0; s < n; ++s) {                                              \
        int i = lower ? s : n - 1 - s;                                         \
        int64_t bi = off_b + (incb ? vidx(i, nb, incb) : (int64_t)i * stride_b); \
        T v = b[bi];                                                           \
        int j0 = lower ? 0 : i + 1, j1 = lower ? i : n;                        \
        for (int j = j0; j < j1; ++j) {                                        \
            int64_t bj = off_b + (incb ? vidx(j, nb, incb) : (int64_t)j * stride_b); \
            v -= op_get_##S(A, lda, op, i, j) * b[bj];                         \
        }                                                                      \
        if (!unit) v /= op_get_##S(A, lda, op, i, i);                          \
        b[bi] = v;                                                             \
    }                                                                          \
}                                                                              \
                                                                               \
static void k_trsv_##S(void **a)                                               \
{                                                                              \
    const mat_args *g = a[0];                                                  \
    tri_solve_##S(g->a, g->lda, g->n, g->uplo, g->opa,                         \
                  g->diag == OCU_BLAS_UNIT, g->y, 0, 0, g->n, g->incx);        \
}                                                                              \
                                                                               \
/* side L: thread j solves column j; side R: thread i solves row i. */        \
static void k_trsm_##S(void **a)                                               \
{                                                                              \
    const mat_args *g = a[0];                                                  \
    uint32_t r = gtid_x();                                                     \
    T *B = g->c;                                                               \
    T al = (T)g->alpha;                                                        \
    if (g->side == OCU_BLAS_LEFT) {                                            \
        if ((int)r >= g->n) return;                                            \
        int64_t col = (int64_t)r * g->ldb;                                     \
        for (int i = 0; i < g->m; ++i) B[col + i] *= al;                       \
        tri_solve_##S(g->a, g->lda, g->m, g->uplo, g->opa,                     \
                      g->diag == OCU_BLAS_UNIT, B, 1, col, 0, 0);              \
    } else {                                                                   \
        /* X op(A) = alpha B  <=>  op(A)^T X^T = alpha B^T */                  \
        if ((int)r >= g->m) return;                                            \
        for (int j = 0; j < g->n; ++j) B[r + (int64_t)j * g->ldb] *= al;       \
        int op_t = g->opa == OCU_BLAS_OP_N ? OCU_BLAS_OP_T : OCU_BLAS_OP_N;    \
        tri_solve_##S(g->a, g->lda, g->n, g->uplo, op_t,                       \
                      g->diag == OCU_BLAS_UNIT, B, g->ldb, r, 0, 0);           \
    }                                                                          \
}

DEFINE_KERNELS(float, f32, sqrtf, fabsf)
DEFINE_KERNELS(double, f64, sqrt, fabs)

#define PICK(t, name) ((t) == OCU_BLAS_F32 ? k_##name##_f32 : k_##name##_f64)

static double scalar(ocu_blas_type t, const void *p)
{
    return t == OCU_BLAS_F32 ? (double)*(const float *)p : *(const double *)p;
}

static size_t elem(ocu_blas_type t) { return t == OCU_BLAS_F32 ? 4u : 8u; }

/* ---- provider entries --------------------------------------------------- */

static ocu_status p_axpy(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                         const void *alpha, const void *x, int incx, void *y, int incy)
{
    (void)d;
    vec_args v = {n, incx, incy, scalar(t, alpha), x, y, NULL};
    return launch(q, PICK(t, axpy), blocks_for(n, 256), 1, 256, 1, 0, &v, sizeof(v));
}

static ocu_status reduce(ocu_queue *q, ocu_cpu_kernel_fn fn, ocu_blas_type t,
                         int n, const void *x, int incx, const void *y, int incy,
                         void *result, size_t extra)
{
    vec_args v = {n, incx, incy, 0, x, (void *)(uintptr_t)y, result};
    return launch(q, fn, 1, 1, RED_THREADS, 1, RED_THREADS * (elem(t) + extra),
                  &v, sizeof(v));
}

static ocu_status p_dot(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                        const void *x, int incx, const void *y, int incy, void *result)
{
    (void)d;
    return reduce(q, PICK(t, dot), t, n, x, incx, y, incy, result, 0);
}

static ocu_status p_scal(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                         const void *alpha, void *x, int incx)
{
    (void)d;
    vec_args v = {n, incx, 0, scalar(t, alpha), NULL, x, NULL};
    return launch(q, PICK(t, scal), blocks_for(n, 256), 1, 256, 1, 0, &v, sizeof(v));
}

static ocu_status p_nrm2(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                         const void *x, int incx, void *result)
{
    (void)d;
    return reduce(q, PICK(t, nrm2), t, n, x, incx, NULL, 0, result, 0);
}

static ocu_status p_asum(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                         const void *x, int incx, void *result)
{
    (void)d;
    return reduce(q, PICK(t, asum), t, n, x, incx, NULL, 0, result, 0);
}

static ocu_status p_iamax(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                          const void *x, int incx, int *result)
{
    (void)d;
    return reduce(q, PICK(t, iamax), t, n, x, incx, NULL, 0, result, sizeof(int64_t));
}

static ocu_status p_copy(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                         const void *x, int incx, void *y, int incy)
{
    (void)d;
    vec_args v = {n, incx, incy, 0, x, y, NULL};
    return launch(q, PICK(t, copy), blocks_for(n, 256), 1, 256, 1, 0, &v, sizeof(v));
}

static ocu_status p_swap(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                         void *x, int incx, void *y, int incy)
{
    (void)d;
    vec_args v = {n, incx, incy, 0, x, y, NULL};
    return launch(q, PICK(t, swap), blocks_for(n, 256), 1, 256, 1, 0, &v, sizeof(v));
}

static ocu_status p_gemv(ocu_device *d, ocu_queue *q, ocu_blas_type t, ocu_blas_op op,
                         int m, int n, const void *alpha, const void *a, int lda,
                         const void *x, int incx, const void *beta, void *y, int incy)
{
    (void)d;
    mat_args g;
    memset(&g, 0, sizeof(g));
    g.m = m; g.n = n; g.lda = lda; g.incx = incx; g.incy = incy; g.opa = op;
    g.alpha = scalar(t, alpha); g.beta = scalar(t, beta);
    g.a = a; g.x = x; g.y = y;
    int rows = op == OCU_BLAS_OP_N ? m : n;
    return launch(q, PICK(t, gemv), blocks_for(rows, 128), 1, 128, 1, 0, &g, sizeof(g));
}

static ocu_status p_ger(ocu_device *d, ocu_queue *q, ocu_blas_type t, int m, int n,
                        const void *alpha, const void *x, int incx, const void *y,
                        int incy, void *a, int lda)
{
    (void)d;
    mat_args g;
    memset(&g, 0, sizeof(g));
    g.m = m; g.n = n; g.lda = lda; g.incx = incx; g.incy = incy;
    g.alpha = scalar(t, alpha); g.x = x; g.b = y; g.c = a;
    return launch(q, PICK(t, ger), blocks_for(m, TILE), blocks_for(n, TILE), TILE,
                  TILE, 0, &g, sizeof(g));
}

static ocu_status p_trsv(ocu_device *d, ocu_queue *q, ocu_blas_type t, ocu_blas_fill uplo,
                         ocu_blas_op op, ocu_blas_diag diag, int n, const void *a,
                         int lda, void *x, int incx)
{
    (void)d;
    mat_args g;
    memset(&g, 0, sizeof(g));
    g.n = n; g.lda = lda; g.incx = incx; g.uplo = uplo; g.opa = op; g.diag = diag;
    g.a = a; g.y = x;
    return launch(q, PICK(t, trsv), 1, 1, 1, 1, 0, &g, sizeof(g));
}

static ocu_status p_gemm(ocu_device *d, ocu_queue *q, ocu_blas_type t, ocu_blas_op opa,
                         ocu_blas_op opb, int m, int n, int k, const void *alpha,
                         const void *a, int lda, const void *b, int ldb,
                         const void *beta, void *c, int ldc)
{
    (void)d;
    mat_args g;
    memset(&g, 0, sizeof(g));
    g.m = m; g.n = n; g.k = k; g.lda = lda; g.ldb = ldb; g.ldc = ldc;
    g.opa = opa; g.opb = opb; g.alpha = scalar(t, alpha); g.beta = scalar(t, beta);
    g.a = a; g.b = b; g.c = c;
    return launch(q, PICK(t, gemm), blocks_for(m, TILE), blocks_for(n, TILE), TILE,
                  TILE, 0, &g, sizeof(g));
}

static ocu_status p_syrk(ocu_device *d, ocu_queue *q, ocu_blas_type t, ocu_blas_fill uplo,
                         ocu_blas_op op, int n, int k, const void *alpha, const void *a,
                         int lda, const void *beta, void *c, int ldc)
{
    (void)d;
    mat_args g;
    memset(&g, 0, sizeof(g));
    g.n = n; g.k = k; g.lda = lda; g.ldc = ldc; g.uplo = uplo; g.opa = op;
    g.alpha = scalar(t, alpha); g.beta = scalar(t, beta); g.a = a; g.c = c;
    return launch(q, PICK(t, syrk), blocks_for(n, TILE), blocks_for(n, TILE), TILE,
                  TILE, 0, &g, sizeof(g));
}

static ocu_status p_trsm(ocu_device *d, ocu_queue *q, ocu_blas_type t, ocu_blas_side side,
                         ocu_blas_fill uplo, ocu_blas_op op, ocu_blas_diag diag, int m,
                         int n, const void *alpha, const void *a, int lda, void *b, int ldb)
{
    (void)d;
    mat_args g;
    memset(&g, 0, sizeof(g));
    g.m = m; g.n = n; g.lda = lda; g.ldb = ldb; g.side = side; g.uplo = uplo;
    g.opa = op; g.diag = diag; g.alpha = scalar(t, alpha); g.a = a; g.c = b;
    int count = side == OCU_BLAS_LEFT ? n : m;
    return launch(q, PICK(t, trsm), blocks_for(count, 64), 1, 64, 1, 0, &g, sizeof(g));
}

static const ocu_blas_provider cpu_blas = {
    .version = OCU_BLAS_PROVIDER_VERSION,
    .axpy = p_axpy, .dot = p_dot, .scal = p_scal, .nrm2 = p_nrm2,
    .asum = p_asum, .iamax = p_iamax, .copy = p_copy, .swap = p_swap,
    .gemv = p_gemv, .ger = p_ger, .trsv = p_trsv,
    .gemm = p_gemm, .syrk = p_syrk, .trsm = p_trsm,
};

const void *cpu_get_provider(const char *library)
{
    if (library != NULL && strcmp(library, "blas") == 0) {
        return &cpu_blas;
    }
    return NULL;
}

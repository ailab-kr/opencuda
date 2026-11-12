/*
 * Chip-neutral BLAS layer: argument rules, quick returns, scalar results.
 * The arithmetic is done by the active backend's provider.
 */
#include "ocu_blas.h"
#include "ocu_core.h"

#include <stddef.h>

typedef struct ctx {
    const ocu_backend       *b;
    ocu_device              *d;
    const ocu_blas_provider *p;
} ctx;

static ocu_status get_ctx(ctx *c)
{
    ocu_status st = ocu_core_device(&c->b, &c->d);
    if (st != OCU_OK) {
        return st;
    }
    c->p = c->b->get_provider ? c->b->get_provider("blas") : NULL;
    if (c->p == NULL || c->p->version != OCU_BLAS_PROVIDER_VERSION) {
        return OCU_ERR_NOT_SUPPORTED; /* portable fallback: later */
    }
    return OCU_OK;
}

static int max1(int v) { return v > 1 ? v : 1; }

static int valid_op(ocu_blas_op o) { return o == OCU_BLAS_OP_N || o == OCU_BLAS_OP_T; }
static int valid_fill(ocu_blas_fill f) { return f == OCU_BLAS_UPPER || f == OCU_BLAS_LOWER; }
static int valid_diag(ocu_blas_diag d) { return d == OCU_BLAS_NON_UNIT || d == OCU_BLAS_UNIT; }
static int valid_side(ocu_blas_side s) { return s == OCU_BLAS_LEFT || s == OCU_BLAS_RIGHT; }

/* Run a scalar-producing provider call and bring the result to the host. */
typedef ocu_status (*scalar_fn)(const ctx *c, ocu_queue *q, void *dev_result);

static ocu_status scalar_result(const ctx *c, ocu_queue *q, scalar_fn fn, void *host,
                                size_t bytes)
{
    void *dev = NULL;
    ocu_status st = c->b->mem_alloc(c->d, 16, &dev);
    if (st != OCU_OK) {
        return st;
    }
    st = fn(c, q, dev);
    if (st == OCU_OK && q != NULL) {
        st = c->b->queue_sync(q);
    }
    if (st == OCU_OK) {
        st = c->b->mem_copy(c->d, NULL, host, dev, bytes, OCU_COPY_DEVICE_TO_HOST);
    }
    ocu_status fst = c->b->mem_free(c->d, dev);
    return st != OCU_OK ? st : fst;
}

/* ---- level 1 ------------------------------------------------------------ */

static ocu_status axpy(ocu_queue *q, ocu_blas_type t, int n, const void *alpha,
                       const void *x, int incx, void *y, int incy)
{
    if (n < 0 || incx == 0 || incy == 0) return OCU_ERR_INVALID_VALUE;
    if (n == 0) return OCU_OK;
    ctx c;
    ocu_status st = get_ctx(&c);
    if (st != OCU_OK) return st;
    return c.p->axpy(c.d, q, t, n, alpha, x, incx, y, incy);
}

ocu_status ocu_blas_saxpy(ocu_queue *q, int n, float alpha, const float *x, int incx,
                          float *y, int incy)
{
    return axpy(q, OCU_BLAS_F32, n, &alpha, x, incx, y, incy);
}

ocu_status ocu_blas_daxpy(ocu_queue *q, int n, double alpha, const double *x, int incx,
                          double *y, int incy)
{
    return axpy(q, OCU_BLAS_F64, n, &alpha, x, incx, y, incy);
}

typedef struct red_args {
    ocu_blas_type t;
    int n, incx, incy;
    const void *x, *y;
    int which; /* 0 dot, 1 nrm2, 2 asum, 3 iamax */
} red_args;

static _Thread_local red_args tl_red;

static ocu_status run_red(const ctx *c, ocu_queue *q, void *dev)
{
    const red_args *r = &tl_red;
    switch (r->which) {
    case 0: return c->p->dot(c->d, q, r->t, r->n, r->x, r->incx, r->y, r->incy, dev);
    case 1: return c->p->nrm2(c->d, q, r->t, r->n, r->x, r->incx, dev);
    case 2: return c->p->asum(c->d, q, r->t, r->n, r->x, r->incx, dev);
    default: return c->p->iamax(c->d, q, r->t, r->n, r->x, r->incx, dev);
    }
}

static ocu_status reduce(ocu_queue *q, ocu_blas_type t, int which, int n, const void *x,
                         int incx, const void *y, int incy, void *host, size_t bytes)
{
    if (host == NULL) return OCU_ERR_INVALID_VALUE;
    ctx c;
    ocu_status st = get_ctx(&c);
    if (st != OCU_OK) return st;
    tl_red = (red_args){t, n, incx, incy, x, y, which};
    return scalar_result(&c, q, run_red, host, bytes);
}

ocu_status ocu_blas_sdot(ocu_queue *q, int n, const float *x, int incx, const float *y,
                         int incy, float *result)
{
    if (n < 0 || incx == 0 || incy == 0 || result == NULL) return OCU_ERR_INVALID_VALUE;
    if (n == 0) { *result = 0; return OCU_OK; }
    return reduce(q, OCU_BLAS_F32, 0, n, x, incx, y, incy, result, sizeof(*result));
}

ocu_status ocu_blas_ddot(ocu_queue *q, int n, const double *x, int incx, const double *y,
                         int incy, double *result)
{
    if (n < 0 || incx == 0 || incy == 0 || result == NULL) return OCU_ERR_INVALID_VALUE;
    if (n == 0) { *result = 0; return OCU_OK; }
    return reduce(q, OCU_BLAS_F64, 0, n, x, incx, y, incy, result, sizeof(*result));
}

static ocu_status scal(ocu_queue *q, ocu_blas_type t, int n, const void *alpha, void *x,
                       int incx)
{
    if (n < 0) return OCU_ERR_INVALID_VALUE;
    if (n == 0 || incx <= 0) return OCU_OK;
    ctx c;
    ocu_status st = get_ctx(&c);
    if (st != OCU_OK) return st;
    return c.p->scal(c.d, q, t, n, alpha, x, incx);
}

ocu_status ocu_blas_sscal(ocu_queue *q, int n, float alpha, float *x, int incx)
{
    return scal(q, OCU_BLAS_F32, n, &alpha, x, incx);
}

ocu_status ocu_blas_dscal(ocu_queue *q, int n, double alpha, double *x, int incx)
{
    return scal(q, OCU_BLAS_F64, n, &alpha, x, incx);
}

#define EMPTY_REDUCTION(result, zero)                                         \
    do {                                                                      \
        if (n < 0 || (result) == NULL) return OCU_ERR_INVALID_VALUE;          \
        if (n == 0 || incx <= 0) { *(result) = (zero); return OCU_OK; }       \
    } while (0)

ocu_status ocu_blas_snrm2(ocu_queue *q, int n, const float *x, int incx, float *result)
{
    EMPTY_REDUCTION(result, 0.0f);
    return reduce(q, OCU_BLAS_F32, 1, n, x, incx, NULL, 0, result, sizeof(*result));
}

ocu_status ocu_blas_dnrm2(ocu_queue *q, int n, const double *x, int incx, double *result)
{
    EMPTY_REDUCTION(result, 0.0);
    return reduce(q, OCU_BLAS_F64, 1, n, x, incx, NULL, 0, result, sizeof(*result));
}

ocu_status ocu_blas_sasum(ocu_queue *q, int n, const float *x, int incx, float *result)
{
    EMPTY_REDUCTION(result, 0.0f);
    return reduce(q, OCU_BLAS_F32, 2, n, x, incx, NULL, 0, result, sizeof(*result));
}

ocu_status ocu_blas_isamax(ocu_queue *q, int n, const float *x, int incx, int *result)
{
    EMPTY_REDUCTION(result, 0);
    return reduce(q, OCU_BLAS_F32, 3, n, x, incx, NULL, 0, result, sizeof(*result));
}

ocu_status ocu_blas_scopy(ocu_queue *q, int n, const float *x, int incx, float *y, int incy)
{
    if (n < 0 || incx == 0 || incy == 0) return OCU_ERR_INVALID_VALUE;
    if (n == 0) return OCU_OK;
    ctx c;
    ocu_status st = get_ctx(&c);
    if (st != OCU_OK) return st;
    return c.p->copy(c.d, q, OCU_BLAS_F32, n, x, incx, y, incy);
}

ocu_status ocu_blas_sswap(ocu_queue *q, int n, float *x, int incx, float *y, int incy)
{
    if (n < 0 || incx == 0 || incy == 0) return OCU_ERR_INVALID_VALUE;
    if (n == 0) return OCU_OK;
    ctx c;
    ocu_status st = get_ctx(&c);
    if (st != OCU_OK) return st;
    return c.p->swap(c.d, q, OCU_BLAS_F32, n, x, incx, y, incy);
}

/* ---- level 2 ------------------------------------------------------------ */

static ocu_status gemv(ocu_queue *q, ocu_blas_type t, ocu_blas_op op, int m, int n,
                       const void *alpha, double alpha_d, const void *a, int lda,
                       const void *x, int incx, const void *beta, double beta_d, void *y,
                       int incy)
{
    if (!valid_op(op) || m < 0 || n < 0 || lda < max1(m) || incx == 0 || incy == 0)
        return OCU_ERR_INVALID_VALUE;
    if (m == 0 || n == 0 || (alpha_d == 0 && beta_d == 1)) return OCU_OK;
    ctx c;
    ocu_status st = get_ctx(&c);
    if (st != OCU_OK) return st;
    return c.p->gemv(c.d, q, t, op, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

ocu_status ocu_blas_sgemv(ocu_queue *q, ocu_blas_op trans, int m, int n, float alpha,
                          const float *a, int lda, const float *x, int incx, float beta,
                          float *y, int incy)
{
    return gemv(q, OCU_BLAS_F32, trans, m, n, &alpha, alpha, a, lda, x, incx, &beta, beta,
                y, incy);
}

ocu_status ocu_blas_dgemv(ocu_queue *q, ocu_blas_op trans, int m, int n, double alpha,
                          const double *a, int lda, const double *x, int incx,
                          double beta, double *y, int incy)
{
    return gemv(q, OCU_BLAS_F64, trans, m, n, &alpha, alpha, a, lda, x, incx, &beta, beta,
                y, incy);
}

ocu_status ocu_blas_sger(ocu_queue *q, int m, int n, float alpha, const float *x, int incx,
                         const float *y, int incy, float *a, int lda)
{
    if (m < 0 || n < 0 || incx == 0 || incy == 0 || lda < max1(m))
        return OCU_ERR_INVALID_VALUE;
    if (m == 0 || n == 0 || alpha == 0) return OCU_OK;
    ctx c;
    ocu_status st = get_ctx(&c);
    if (st != OCU_OK) return st;
    return c.p->ger(c.d, q, OCU_BLAS_F32, m, n, &alpha, x, incx, y, incy, a, lda);
}

ocu_status ocu_blas_strsv(ocu_queue *q, ocu_blas_fill uplo, ocu_blas_op trans,
                          ocu_blas_diag diag, int n, const float *a, int lda, float *x,
                          int incx)
{
    if (!valid_fill(uplo) || !valid_op(trans) || !valid_diag(diag) || n < 0 ||
        lda < max1(n) || incx == 0)
        return OCU_ERR_INVALID_VALUE;
    if (n == 0) return OCU_OK;
    ctx c;
    ocu_status st = get_ctx(&c);
    if (st != OCU_OK) return st;
    return c.p->trsv(c.d, q, OCU_BLAS_F32, uplo, trans, diag, n, a, lda, x, incx);
}

/* ---- level 3 ------------------------------------------------------------ */

static ocu_status gemm(ocu_queue *q, ocu_blas_type t, ocu_blas_op ta, ocu_blas_op tb,
                       int m, int n, int k, const void *alpha, double alpha_d,
                       const void *a, int lda, const void *b, int ldb, const void *beta,
                       double beta_d, void *c, int ldc)
{
    if (!valid_op(ta) || !valid_op(tb) || m < 0 || n < 0 || k < 0 ||
        lda < max1(ta == OCU_BLAS_OP_N ? m : k) ||
        ldb < max1(tb == OCU_BLAS_OP_N ? k : n) || ldc < max1(m))
        return OCU_ERR_INVALID_VALUE;
    if (m == 0 || n == 0 || ((alpha_d == 0 || k == 0) && beta_d == 1)) return OCU_OK;
    ctx cx;
    ocu_status st = get_ctx(&cx);
    if (st != OCU_OK) return st;
    return cx.p->gemm(cx.d, q, t, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

ocu_status ocu_blas_sgemm(ocu_queue *q, ocu_blas_op transa, ocu_blas_op transb, int m,
                          int n, int k, float alpha, const float *a, int lda,
                          const float *b, int ldb, float beta, float *c, int ldc)
{
    return gemm(q, OCU_BLAS_F32, transa, transb, m, n, k, &alpha, alpha, a, lda, b, ldb,
                &beta, beta, c, ldc);
}

ocu_status ocu_blas_dgemm(ocu_queue *q, ocu_blas_op transa, ocu_blas_op transb, int m,
                          int n, int k, double alpha, const double *a, int lda,
                          const double *b, int ldb, double beta, double *c, int ldc)
{
    return gemm(q, OCU_BLAS_F64, transa, transb, m, n, k, &alpha, alpha, a, lda, b, ldb,
                &beta, beta, c, ldc);
}

ocu_status ocu_blas_ssyrk(ocu_queue *q, ocu_blas_fill uplo, ocu_blas_op trans, int n, int k,
                          float alpha, const float *a, int lda, float beta, float *c,
                          int ldc)
{
    if (!valid_fill(uplo) || !valid_op(trans) || n < 0 || k < 0 ||
        lda < max1(trans == OCU_BLAS_OP_N ? n : k) || ldc < max1(n))
        return OCU_ERR_INVALID_VALUE;
    if (n == 0 || ((alpha == 0 || k == 0) && beta == 1)) return OCU_OK;
    ctx cx;
    ocu_status st = get_ctx(&cx);
    if (st != OCU_OK) return st;
    return cx.p->syrk(cx.d, q, OCU_BLAS_F32, uplo, trans, n, k, &alpha, a, lda, &beta, c,
                      ldc);
}

ocu_status ocu_blas_strsm(ocu_queue *q, ocu_blas_side side, ocu_blas_fill uplo,
                          ocu_blas_op trans, ocu_blas_diag diag, int m, int n, float alpha,
                          const float *a, int lda, float *b, int ldb)
{
    if (!valid_side(side) || !valid_fill(uplo) || !valid_op(trans) || !valid_diag(diag) ||
        m < 0 || n < 0 || lda < max1(side == OCU_BLAS_LEFT ? m : n) || ldb < max1(m))
        return OCU_ERR_INVALID_VALUE;
    if (m == 0 || n == 0) return OCU_OK;
    ctx cx;
    ocu_status st = get_ctx(&cx);
    if (st != OCU_OK) return st;
    return cx.p->trsm(cx.d, q, OCU_BLAS_F32, side, uplo, trans, diag, m, n, &alpha, a, lda,
                      b, ldb);
}

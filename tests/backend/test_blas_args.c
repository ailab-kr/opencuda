/*
 * libs/blas argument rules and quick returns (BLAS standard, as documented
 * in libs/blas/ocu_blas.h). Runs on the CPU backend.
 */
#include "ocu_blas.h"
#include "ocu_core.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_failures, g_checks;

#define CHECK_ST(expr, want)                                                 \
    do {                                                                     \
        ocu_status got_ = (expr);                                            \
        ++g_checks;                                                          \
        if (got_ != (want)) {                                                \
            fprintf(stderr, "FAIL %s:%d: %s -> %s, want %s\n", __FILE__,     \
                    __LINE__, #expr, ocu_status_name(got_),                  \
                    ocu_status_name(want));                                  \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static const ocu_backend *B;
static ocu_device *D;

static float *dev(size_t n, float fill)
{
    float *p = NULL;
    if (B->mem_alloc(D, n * sizeof(float), (void **)&p) != OCU_OK) return NULL;
    for (size_t i = 0; i < n; ++i) p[i] = fill; /* CPU backend: host-visible */
    return p;
}

int main(void)
{
    if (ocu_core_device(&B, &D) != OCU_OK) return 1;
    const ocu_status INV = OCU_ERR_INVALID_VALUE;
    float *x = dev(64, 1.0f), *y = dev(64, 2.0f), *a = dev(64 * 64, 1.0f);
    float r = -1;
    int ir = -1;

    /* level 1: negative sizes and zero increments are errors */
    CHECK_ST(ocu_blas_saxpy(NULL, -1, 1, x, 1, y, 1), INV);
    CHECK_ST(ocu_blas_saxpy(NULL, 4, 1, x, 0, y, 1), INV);
    CHECK_ST(ocu_blas_saxpy(NULL, 4, 1, x, 1, y, 0), INV);
    CHECK_ST(ocu_blas_sdot(NULL, 4, x, 1, y, 1, NULL), INV);
    CHECK_ST(ocu_blas_sdot(NULL, -2, x, 1, y, 1, &r), INV);
    CHECK_ST(ocu_blas_scopy(NULL, 4, x, 0, y, 1), INV);
    CHECK_ST(ocu_blas_sswap(NULL, 4, x, 1, y, 0), INV);
    CHECK_ST(ocu_blas_sscal(NULL, -1, 2, x, 1), INV);
    CHECK_ST(ocu_blas_snrm2(NULL, -1, x, 1, &r), INV);
    CHECK_ST(ocu_blas_isamax(NULL, 4, x, 1, NULL), INV);

    /* quick returns: n == 0, incx <= 0 for scal/nrm2/asum/iamax */
    CHECK_ST(ocu_blas_saxpy(NULL, 0, 1, NULL, 1, NULL, 1), OCU_OK);
    r = -1;
    CHECK_ST(ocu_blas_sdot(NULL, 0, NULL, 1, NULL, 1, &r), OCU_OK);
    CHECK(r == 0);
    CHECK_ST(ocu_blas_sscal(NULL, 4, 9, x, -1), OCU_OK);
    CHECK(x[0] == 1.0f); /* untouched */
    r = -1;
    CHECK_ST(ocu_blas_snrm2(NULL, 4, x, 0, &r), OCU_OK);
    CHECK(r == 0);
    r = -1;
    CHECK_ST(ocu_blas_sasum(NULL, 4, x, -2, &r), OCU_OK);
    CHECK(r == 0);
    ir = -1;
    CHECK_ST(ocu_blas_isamax(NULL, 0, x, 1, &ir), OCU_OK);
    CHECK(ir == 0);

    /* level 2 */
    CHECK_ST(ocu_blas_sgemv(NULL, OCU_BLAS_OP_N, 4, 4, 1, a, 3, x, 1, 0, y, 1), INV); /* lda < m */
    CHECK_ST(ocu_blas_sgemv(NULL, (ocu_blas_op)7, 4, 4, 1, a, 4, x, 1, 0, y, 1), INV);
    CHECK_ST(ocu_blas_sgemv(NULL, OCU_BLAS_OP_N, -1, 4, 1, a, 4, x, 1, 0, y, 1), INV);
    CHECK_ST(ocu_blas_sgemv(NULL, OCU_BLAS_OP_N, 4, 4, 1, a, 4, x, 0, 0, y, 1), INV);
    CHECK_ST(ocu_blas_sgemv(NULL, OCU_BLAS_OP_N, 0, 4, 1, a, 1, x, 1, 0, y, 1), OCU_OK); /* lda >= max(1, 0) */
    y[0] = 5;
    CHECK_ST(ocu_blas_sgemv(NULL, OCU_BLAS_OP_N, 4, 4, 0, a, 4, x, 1, 1, y, 1), OCU_OK);
    CHECK(y[0] == 5); /* alpha 0, beta 1: unchanged */
    CHECK_ST(ocu_blas_sger(NULL, 4, 4, 1, x, 1, y, 1, a, 2), INV);
    CHECK_ST(ocu_blas_strsv(NULL, (ocu_blas_fill)9, OCU_BLAS_OP_N, OCU_BLAS_NON_UNIT, 4, a, 4, x, 1), INV);
    CHECK_ST(ocu_blas_strsv(NULL, OCU_BLAS_UPPER, OCU_BLAS_OP_N, (ocu_blas_diag)3, 4, a, 4, x, 1), INV);
    CHECK_ST(ocu_blas_strsv(NULL, OCU_BLAS_UPPER, OCU_BLAS_OP_N, OCU_BLAS_NON_UNIT, 4, a, 3, x, 1), INV);

    /* level 3: leading dimensions follow op() */
    CHECK_ST(ocu_blas_sgemm(NULL, OCU_BLAS_OP_N, OCU_BLAS_OP_N, 8, 4, 2, 1, a, 7, a, 2, 0, a, 8), INV);
    CHECK_ST(ocu_blas_sgemm(NULL, OCU_BLAS_OP_T, OCU_BLAS_OP_N, 8, 4, 2, 1, a, 2, a, 2, 0, a, 8), OCU_OK);
    CHECK_ST(ocu_blas_sgemm(NULL, OCU_BLAS_OP_N, OCU_BLAS_OP_T, 8, 4, 2, 1, a, 8, a, 3, 0, a, 8), INV);
    CHECK_ST(ocu_blas_sgemm(NULL, OCU_BLAS_OP_N, OCU_BLAS_OP_N, 8, 4, 2, 1, a, 8, a, 2, 0, a, 7), INV);
    CHECK_ST(ocu_blas_sgemm(NULL, OCU_BLAS_OP_N, OCU_BLAS_OP_N, 8, 4, -1, 1, a, 8, a, 1, 0, a, 8), INV);
    float *c = dev(16, 3.0f);
    CHECK_ST(ocu_blas_sgemm(NULL, OCU_BLAS_OP_N, OCU_BLAS_OP_N, 4, 4, 0, 1, a, 4, a, 1, 1, c, 4), OCU_OK);
    CHECK(c[0] == 3.0f); /* k == 0, beta 1: unchanged */
    CHECK_ST(ocu_blas_sgemm(NULL, OCU_BLAS_OP_N, OCU_BLAS_OP_N, 4, 4, 0, 1, a, 4, a, 1, 0.5f, c, 4), OCU_OK);
    CHECK(c[0] == 1.5f); /* k == 0: C = beta * C */
    CHECK_ST(ocu_blas_ssyrk(NULL, OCU_BLAS_UPPER, OCU_BLAS_OP_T, 4, 8, 1, a, 4, 0, c, 4), INV); /* lda < k */
    CHECK_ST(ocu_blas_strsm(NULL, OCU_BLAS_RIGHT, OCU_BLAS_UPPER, OCU_BLAS_OP_N, OCU_BLAS_NON_UNIT,
                            8, 4, 1, a, 3, a, 8), INV); /* side R: lda >= n */
    CHECK_ST(ocu_blas_strsm(NULL, OCU_BLAS_LEFT, OCU_BLAS_UPPER, OCU_BLAS_OP_N, OCU_BLAS_NON_UNIT,
                            8, 4, 1, a, 8, a, 7), INV); /* ldb >= m */
    CHECK_ST(ocu_blas_strsm(NULL, (ocu_blas_side)5, OCU_BLAS_UPPER, OCU_BLAS_OP_N, OCU_BLAS_NON_UNIT,
                            8, 4, 1, a, 8, a, 8), INV);

    B->mem_free(D, x);
    B->mem_free(D, y);
    B->mem_free(D, a);
    B->mem_free(D, c);
    if (g_failures == 0) {
        printf("test_blas_args: ok (%d checks)\n", g_checks);
        return 0;
    }
    fprintf(stderr, "test_blas_args: %d of %d checks failed\n", g_failures, g_checks);
    return 1;
}

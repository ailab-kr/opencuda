/*
 * BLAS shapes past the old grid limits (review finding 2): one row by more
 * than 65535 * 16 columns. Runs on the CPU backend.
 */
#include "ocu_blas.h"
#include "ocu_core.h"

#include <stdio.h>

int main(void)
{
    const ocu_backend *B;
    ocu_device *D;
    if (ocu_core_device(&B, &D) != OCU_OK) return 1;
    const int n = 1100000;
    float *a = NULL, *b = NULL, *c = NULL, *x = NULL;
    if (B->mem_alloc(D, sizeof(float), (void **)&a) != OCU_OK ||
        B->mem_alloc(D, (size_t)n * sizeof(float), (void **)&b) != OCU_OK ||
        B->mem_alloc(D, (size_t)n * sizeof(float), (void **)&c) != OCU_OK ||
        B->mem_alloc(D, sizeof(float), (void **)&x) != OCU_OK) {
        return 1;
    }
    a[0] = 2.0f;
    x[0] = 3.0f;
    for (int j = 0; j < n; ++j) { b[j] = (float)(j % 7); c[j] = 1.0f; }
    int bad = 0;
    /* C(1 x n) = 2 * A(1x1) * B(1 x n) + 1 * C */
    ocu_status st = ocu_blas_sgemm(NULL, OCU_BLAS_OP_N, OCU_BLAS_OP_N, 1, n, 1, 2.0f,
                                   a, 1, b, 1, 1.0f, c, 1);
    bad += st != OCU_OK;
    for (int j = 0; j < n && !bad; ++j) bad += c[j] != 4.0f * (float)(j % 7) + 1.0f;
    /* A(1 x n) += 0.5 * x * y^T, with A = c, y = b */
    st = ocu_blas_sger(NULL, 1, n, 0.5f, x, 1, b, 1, c, 1);
    bad += st != OCU_OK;
    for (int j = 0; j < n && !bad; ++j) bad += c[j] != 5.5f * (float)(j % 7) + 1.0f;
    B->mem_free(D, a);
    B->mem_free(D, b);
    B->mem_free(D, c);
    B->mem_free(D, x);
    printf("test_blas_large: %s (n = %d, last status %s)\n", bad ? "FAIL" : "ok", n,
           ocu_status_name(st));
    return bad ? 1 : 0;
}

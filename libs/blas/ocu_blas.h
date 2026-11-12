/*
 * ocu_blas.h -- chip-neutral BLAS layer (internal names).
 *
 * Semantics follow the public BLAS standard (netlib reference BLAS):
 * column-major matrices, strided vectors where a negative increment starts
 * at the far end, 1-based iamax. The CUDA-named library on top of this
 * (handles, pointer modes, status codes) is provided by the BLAS-compatible library (libs/cublas).
 *
 * Every function takes the queue to run on (NULL: run synchronously) and
 * device pointers. Scalars are host values. Functions that produce a
 * scalar (dot, nrm2, asum, iamax) write it to host memory and return when
 * it is there.
 *
 * Argument rules (OCU_ERR_INVALID_VALUE when broken): sizes >= 0; leading
 * dimensions >= max(1, rows); vector increments != 0. Quick returns as in
 * the reference BLAS: nothing happens for empty problems, and gemm/syrk/
 * gemv with alpha == 0 (or k == 0) and beta == 1 leave the output alone.
 * scal, nrm2, asum and iamax treat incx <= 0 as an empty vector.
 *
 * Works on any backend whose get_provider("blas") returns a table; other
 * backends get OCU_ERR_NOT_SUPPORTED for now.
 */
#ifndef OCU_BLAS_H
#define OCU_BLAS_H

#include "ocu_backend.h"
#include "ocu_blas_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

/* level 1 */
ocu_status ocu_blas_saxpy(ocu_queue *q, int n, float alpha, const float *x, int incx,
                          float *y, int incy);
ocu_status ocu_blas_daxpy(ocu_queue *q, int n, double alpha, const double *x, int incx,
                          double *y, int incy);
ocu_status ocu_blas_sdot(ocu_queue *q, int n, const float *x, int incx, const float *y,
                         int incy, float *result);
ocu_status ocu_blas_ddot(ocu_queue *q, int n, const double *x, int incx, const double *y,
                         int incy, double *result);
ocu_status ocu_blas_sscal(ocu_queue *q, int n, float alpha, float *x, int incx);
ocu_status ocu_blas_dscal(ocu_queue *q, int n, double alpha, double *x, int incx);
ocu_status ocu_blas_snrm2(ocu_queue *q, int n, const float *x, int incx, float *result);
ocu_status ocu_blas_dnrm2(ocu_queue *q, int n, const double *x, int incx, double *result);
ocu_status ocu_blas_sasum(ocu_queue *q, int n, const float *x, int incx, float *result);
ocu_status ocu_blas_isamax(ocu_queue *q, int n, const float *x, int incx, int *result);
ocu_status ocu_blas_scopy(ocu_queue *q, int n, const float *x, int incx, float *y, int incy);
ocu_status ocu_blas_sswap(ocu_queue *q, int n, float *x, int incx, float *y, int incy);

/* level 2 */
ocu_status ocu_blas_sgemv(ocu_queue *q, ocu_blas_op trans, int m, int n, float alpha,
                          const float *a, int lda, const float *x, int incx, float beta,
                          float *y, int incy);
ocu_status ocu_blas_dgemv(ocu_queue *q, ocu_blas_op trans, int m, int n, double alpha,
                          const double *a, int lda, const double *x, int incx,
                          double beta, double *y, int incy);
ocu_status ocu_blas_sger(ocu_queue *q, int m, int n, float alpha, const float *x, int incx,
                         const float *y, int incy, float *a, int lda);
ocu_status ocu_blas_strsv(ocu_queue *q, ocu_blas_fill uplo, ocu_blas_op trans,
                          ocu_blas_diag diag, int n, const float *a, int lda, float *x,
                          int incx);

/* level 3 */
ocu_status ocu_blas_sgemm(ocu_queue *q, ocu_blas_op transa, ocu_blas_op transb, int m,
                          int n, int k, float alpha, const float *a, int lda,
                          const float *b, int ldb, float beta, float *c, int ldc);
ocu_status ocu_blas_dgemm(ocu_queue *q, ocu_blas_op transa, ocu_blas_op transb, int m,
                          int n, int k, double alpha, const double *a, int lda,
                          const double *b, int ldb, double beta, double *c, int ldc);
ocu_status ocu_blas_ssyrk(ocu_queue *q, ocu_blas_fill uplo, ocu_blas_op trans, int n, int k,
                          float alpha, const float *a, int lda, float beta, float *c,
                          int ldc);
ocu_status ocu_blas_strsm(ocu_queue *q, ocu_blas_side side, ocu_blas_fill uplo,
                          ocu_blas_op trans, ocu_blas_diag diag, int m, int n, float alpha,
                          const float *a, int lda, float *b, int ldb);

#ifdef __cplusplus
}
#endif

#endif /* OCU_BLAS_H */

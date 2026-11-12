/*
 * ocu_blas_provider.h -- what a backend offers the BLAS layer (libs/blas).
 *
 * A backend returns a pointer to this table from get_provider("blas"). The
 * table's shape is defined by libs/blas; it lives here, next to the backend
 * interface, because both sides depend on it.
 *
 * Division of work:
 *   libs/blas  checks arguments (BLAS standard rules), handles quick
 *              returns (n == 0, ...), moves scalar results to the host.
 *   provider   computes, on device memory, in queue order. Arguments it
 *              receives are already valid and non-trivial.
 *
 * Conventions: column-major matrices; strided vectors where a negative
 * increment starts from the far end (element i at (n-1-i)*|inc|); scalars
 * by pointer to host memory; scalar results written to device memory.
 */
#ifndef OCU_BLAS_PROVIDER_H
#define OCU_BLAS_PROVIDER_H

#include "ocu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OCU_BLAS_PROVIDER_VERSION 1u

typedef enum ocu_blas_type { OCU_BLAS_F32 = 0, OCU_BLAS_F64 = 1 } ocu_blas_type;
typedef enum ocu_blas_op   { OCU_BLAS_OP_N = 0, OCU_BLAS_OP_T = 1 } ocu_blas_op;
typedef enum ocu_blas_fill { OCU_BLAS_UPPER = 0, OCU_BLAS_LOWER = 1 } ocu_blas_fill;
typedef enum ocu_blas_side { OCU_BLAS_LEFT = 0, OCU_BLAS_RIGHT = 1 } ocu_blas_side;
typedef enum ocu_blas_diag { OCU_BLAS_NON_UNIT = 0, OCU_BLAS_UNIT = 1 } ocu_blas_diag;

typedef struct ocu_blas_provider {
    uint32_t version; /* OCU_BLAS_PROVIDER_VERSION */

    /* level 1 */
    ocu_status (*axpy)(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                       const void *alpha, const void *x, int incx, void *y, int incy);
    ocu_status (*dot)(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                      const void *x, int incx, const void *y, int incy, void *result);
    ocu_status (*scal)(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                       const void *alpha, void *x, int incx);
    ocu_status (*nrm2)(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                       const void *x, int incx, void *result);
    ocu_status (*asum)(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                       const void *x, int incx, void *result);
    /* 1-based index of the first largest |x|, as an int in device memory */
    ocu_status (*iamax)(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                        const void *x, int incx, int *result);
    ocu_status (*copy)(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                       const void *x, int incx, void *y, int incy);
    ocu_status (*swap)(ocu_device *d, ocu_queue *q, ocu_blas_type t, int n,
                       void *x, int incx, void *y, int incy);

    /* level 2 */
    ocu_status (*gemv)(ocu_device *d, ocu_queue *q, ocu_blas_type t, ocu_blas_op op,
                       int m, int n, const void *alpha, const void *a, int lda,
                       const void *x, int incx, const void *beta, void *y, int incy);
    ocu_status (*ger)(ocu_device *d, ocu_queue *q, ocu_blas_type t, int m, int n,
                      const void *alpha, const void *x, int incx, const void *y,
                      int incy, void *a, int lda);
    ocu_status (*trsv)(ocu_device *d, ocu_queue *q, ocu_blas_type t, ocu_blas_fill uplo,
                       ocu_blas_op op, ocu_blas_diag diag, int n, const void *a,
                       int lda, void *x, int incx);

    /* level 3 */
    ocu_status (*gemm)(ocu_device *d, ocu_queue *q, ocu_blas_type t, ocu_blas_op opa,
                       ocu_blas_op opb, int m, int n, int k, const void *alpha,
                       const void *a, int lda, const void *b, int ldb,
                       const void *beta, void *c, int ldc);
    ocu_status (*syrk)(ocu_device *d, ocu_queue *q, ocu_blas_type t, ocu_blas_fill uplo,
                       ocu_blas_op op, int n, int k, const void *alpha, const void *a,
                       int lda, const void *beta, void *c, int ldc);
    ocu_status (*trsm)(ocu_device *d, ocu_queue *q, ocu_blas_type t, ocu_blas_side side,
                       ocu_blas_fill uplo, ocu_blas_op op, ocu_blas_diag diag, int m,
                       int n, const void *alpha, const void *a, int lda, void *b, int ldb);
} ocu_blas_provider;

#ifdef __cplusplus
}
#endif

#endif /* OCU_BLAS_PROVIDER_H */

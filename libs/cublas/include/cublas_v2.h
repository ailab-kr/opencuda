/*
 * cublas_v2.h -- the handle-based BLAS API under its plain names.
 *
 * Written for this project from the public cuBLAS library documentation.
 * Not derived from any vendor header. As documented, including this
 * header makes the plain names (cublasCreate, cublasSgemm, ...) refer to
 * the _v2 entry points declared in cublas_api.h.
 */
#ifndef OCU_CUBLAS_V2_H
#define OCU_CUBLAS_V2_H

#include "cublas_api.h"

#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasGetVersion cublasGetVersion_v2
#define cublasSetStream cublasSetStream_v2
#define cublasGetStream cublasGetStream_v2
#define cublasSetWorkspace cublasSetWorkspace_v2
#define cublasGetPointerMode cublasGetPointerMode_v2
#define cublasSetPointerMode cublasSetPointerMode_v2

#define cublasSaxpy cublasSaxpy_v2
#define cublasSscal cublasSscal_v2
#define cublasSdot cublasSdot_v2
#define cublasSnrm2 cublasSnrm2_v2
#define cublasSgemv cublasSgemv_v2
#define cublasSgemm cublasSgemm_v2
#define cublasDgemm cublasDgemm_v2

#endif /* OCU_CUBLAS_V2_H */

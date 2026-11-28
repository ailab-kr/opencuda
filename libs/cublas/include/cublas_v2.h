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
#define cublasDaxpy cublasDaxpy_v2
#define cublasDscal cublasDscal_v2
#define cublasDdot cublasDdot_v2
#define cublasDnrm2 cublasDnrm2_v2
#define cublasScopy cublasScopy_v2
#define cublasDcopy cublasDcopy_v2
#define cublasSswap cublasSswap_v2
#define cublasDswap cublasDswap_v2
#define cublasSasum cublasSasum_v2
#define cublasDasum cublasDasum_v2
#define cublasIsamax cublasIsamax_v2
#define cublasIdamax cublasIdamax_v2
#define cublasIsamin cublasIsamin_v2
#define cublasIdamin cublasIdamin_v2
#define cublasDgemv cublasDgemv_v2
#define cublasSgemm cublasSgemm_v2
#define cublasDgemm cublasDgemm_v2

#endif /* OCU_CUBLAS_V2_H */

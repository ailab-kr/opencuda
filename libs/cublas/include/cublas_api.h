/*
 * cublas_api.h -- the BLAS library API (handle-based, version 2) as user
 * programs see it.
 *
 * Written for this project from the public cuBLAS library documentation
 * (function names, parameters, enum names and their documented values).
 * Not derived from any vendor header. Only the subset this project
 * implements is declared: anything else fails to compile instead of
 * failing at run time. Combinations of types the implementation does not
 * compute return CUBLAS_STATUS_NOT_SUPPORTED.
 *
 * Include <cublas_v2.h>, which maps the plain names (cublasCreate,
 * cublasSgemm, ...) to the _v2 entry points declared here.
 *
 * C and C++. cublasHgemm and friends, which take __half, are declared in
 * C++ only (cuda_fp16.h is C++).
 */
#ifndef OCU_CUBLAS_API_H
#define OCU_CUBLAS_API_H

#include <stddef.h>
#include <stdint.h>

#include "cuda_runtime_api.h"
#include "library_types.h"

#ifdef __cplusplus
#include "cuda_fp16.h"
extern "C" {
#endif

#define CUBLAS_VER_MAJOR 12
#define CUBLAS_VER_MINOR 3
#define CUBLAS_VER_PATCH 0
#define CUBLAS_VER_BUILD 0
#define CUBLAS_VERSION (CUBLAS_VER_MAJOR * 10000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH)

typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

typedef enum {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL = 2
} cublasFillMode_t;

typedef enum {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1
} cublasDiagType_t;

typedef enum {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1
} cublasSideMode_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_HERMITAN = 2,
    CUBLAS_OP_CONJG = 3
} cublasOperation_t;

typedef enum {
    CUBLAS_POINTER_MODE_HOST = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef enum {
    CUBLAS_ATOMICS_NOT_ALLOWED = 0,
    CUBLAS_ATOMICS_ALLOWED = 1
} cublasAtomicsMode_t;

typedef enum {
    CUBLAS_GEMM_DFALT = -1,
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_ALGO1 = 1,
    CUBLAS_GEMM_ALGO2 = 2,
    CUBLAS_GEMM_ALGO3 = 3,
    CUBLAS_GEMM_ALGO4 = 4,
    CUBLAS_GEMM_ALGO5 = 5,
    CUBLAS_GEMM_ALGO6 = 6,
    CUBLAS_GEMM_ALGO7 = 7,
    CUBLAS_GEMM_ALGO8 = 8,
    CUBLAS_GEMM_ALGO9 = 9,
    CUBLAS_GEMM_ALGO10 = 10,
    CUBLAS_GEMM_ALGO11 = 11,
    CUBLAS_GEMM_ALGO12 = 12,
    CUBLAS_GEMM_ALGO13 = 13,
    CUBLAS_GEMM_ALGO14 = 14,
    CUBLAS_GEMM_ALGO15 = 15,
    CUBLAS_GEMM_ALGO16 = 16,
    CUBLAS_GEMM_ALGO17 = 17,
    CUBLAS_GEMM_ALGO18 = 18,
    CUBLAS_GEMM_ALGO19 = 19,
    CUBLAS_GEMM_ALGO20 = 20,
    CUBLAS_GEMM_ALGO21 = 21,
    CUBLAS_GEMM_ALGO22 = 22,
    CUBLAS_GEMM_ALGO23 = 23,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_DFALT_TENSOR_OP = 99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
    CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,
    CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
    CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,
    CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
    CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,
    CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
    CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,
    CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
    CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,
    CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
    CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,
    CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
    CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,
    CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
    CUBLAS_GEMM_ALGO15_TENSOR_OP = 115
} cublasGemmAlgo_t;

typedef enum {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
} cublasMath_t;

typedef enum {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73
} cublasComputeType_t;

typedef struct ocu_cublas_context *cublasHandle_t;

/* ---- helper functions --------------------------------------------------- */

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version);
cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value);
size_t cublasGetCudartVersion(void);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId);
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace,
                                     size_t workspaceSizeInBytes);
cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode);
cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);
cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode);
cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
const char *cublasGetStatusName(cublasStatus_t status);
const char *cublasGetStatusString(cublasStatus_t status);

/* ---- level 1 ------------------------------------------------------------ */

cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha,
                              const float *x, int incx, float *y, int incy);
cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha, float *x,
                              int incx);
cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx,
                             const float *y, int incy, float *result);
cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx,
                              float *result);

/* ---- level 2 ------------------------------------------------------------ */

cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                              const float *alpha, const float *A, int lda, const float *x,
                              int incx, const float *beta, float *y, int incy);

/* ---- level 3 ------------------------------------------------------------ */

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda, const float *B,
                              int ldb, const float *beta, float *C, int ldc);
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const double *alpha, const double *A, int lda,
                              const double *B, int ldb, const double *beta, double *C,
                              int ldc);
cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const float *alpha, const float *const Aarray[], int lda,
                                  const float *const Barray[], int ldb, const float *beta,
                                  float *const Carray[], int ldc, int batchCount);
cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const double *alpha, const double *const Aarray[],
                                  int lda, const double *const Barray[], int ldb,
                                  const double *beta, double *const Carray[], int ldc,
                                  int batchCount);
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const float *alpha, const float *A, int lda,
                                         long long int strideA, const float *B, int ldb,
                                         long long int strideB, const float *beta, float *C,
                                         int ldc, long long int strideC, int batchCount);
cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const double *alpha, const double *A, int lda,
                                         long long int strideA, const double *B, int ldb,
                                         long long int strideB, const double *beta,
                                         double *C, int ldc, long long int strideC,
                                         int batchCount);

/* Correct but simple (one thread per right-hand side): for small systems. */
cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo, cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n, const float *alpha,
                                  const float *const A[], int lda, float *const B[], int ldb,
                                  int batchCount);

#ifdef __cplusplus
cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const __half *alpha, const __half *A, int lda, const __half *B,
                           int ldb, const __half *beta, __half *C, int ldc);
cublasStatus_t cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const __half *alpha, const __half *const Aarray[], int lda,
                                  const __half *const Barray[], int ldb, const __half *beta,
                                  __half *const Carray[], int ldc, int batchCount);
cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const __half *alpha, const __half *A, int lda,
                                         long long int strideA, const __half *B, int ldb,
                                         long long int strideB, const __half *beta,
                                         __half *C, int ldc, long long int strideC,
                                         int batchCount);
#endif

/*
 * Mixed-precision GEMM. computeType also accepts the older cudaDataType_t
 * values CUDA_R_16F, CUDA_R_32F and CUDA_R_64F (as the documentation
 * allows for backward compatibility); they mean CUBLAS_COMPUTE_16F, _32F
 * and _64F.
 */
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void *alpha, const void *A, cudaDataType_t Atype, int lda,
                            const void *B, cudaDataType_t Btype, int ldb, const void *beta,
                            void *C, cudaDataType_t Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                   cublasOperation_t transb, int m, int n, int k,
                                   const void *alpha, const void *const Aarray[],
                                   cudaDataType_t Atype, int lda, const void *const Barray[],
                                   cudaDataType_t Btype, int ldb, const void *beta,
                                   void *const Carray[], cudaDataType_t Ctype, int ldc,
                                   int batchCount, cublasComputeType_t computeType,
                                   cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                          cublasOperation_t transb, int m, int n, int k,
                                          const void *alpha, const void *A,
                                          cudaDataType_t Atype, int lda,
                                          long long int strideA, const void *B,
                                          cudaDataType_t Btype, int ldb,
                                          long long int strideB, const void *beta, void *C,
                                          cudaDataType_t Ctype, int ldc,
                                          long long int strideC, int batchCount,
                                          cublasComputeType_t computeType,
                                          cublasGemmAlgo_t algo);

#ifdef __cplusplus
}

/* C++: the older calls that pass a cudaDataType_t as the compute type. */
static inline cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                          int n, int k, const void *alpha, const void *A, cudaDataType_t Atype, int lda,
                                          const void *B, cudaDataType_t Btype, int ldb, const void *beta, void *C,
                                          cudaDataType_t Ctype, int ldc, cudaDataType_t computeType, cublasGemmAlgo_t algo)
{
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc,
                        (cublasComputeType_t)computeType, algo);
}
static inline cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                                 int m, int n, int k, const void *alpha, const void *const Aarray[],
                                                 cudaDataType_t Atype, int lda, const void *const Barray[], cudaDataType_t Btype,
                                                 int ldb, const void *beta, void *const Carray[], cudaDataType_t Ctype, int ldc,
                                                 int batchCount, cudaDataType_t computeType, cublasGemmAlgo_t algo)
{
    return cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray,
                               Ctype, ldc, batchCount, (cublasComputeType_t)computeType, algo);
}
static inline cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                                        cublasOperation_t transb, int m, int n, int k, const void *alpha,
                                                        const void *A, cudaDataType_t Atype, int lda, long long int strideA,
                                                        const void *B, cudaDataType_t Btype, int ldb, long long int strideB,
                                                        const void *beta, void *C, cudaDataType_t Ctype, int ldc,
                                                        long long int strideC, int batchCount, cudaDataType_t computeType,
                                                        cublasGemmAlgo_t algo)
{
    return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB,
                                      beta, C, Ctype, ldc, strideC, batchCount, (cublasComputeType_t)computeType, algo);
}
#endif

#endif /* OCU_CUBLAS_API_H */

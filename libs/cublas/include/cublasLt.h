/*
 * cublasLt.h -- the lightweight matrix-multiply API (cuBLASLt): handle,
 * descriptor and layout objects, and cublasLtMatmul.
 *
 * Written for this project from the public cuBLASLt documentation
 * (function names, parameters, the descriptor, layout and preference
 * attribute enumerations). Not derived from any vendor header.
 *
 * NOT IMPLEMENTED beyond its objects: handles, descriptors, layouts and
 * preferences can be created, configured and destroyed (so a program that
 * sets up a cuBLASLt handle at start-up keeps running), but
 * cublasLtMatmulAlgoGetHeuristic finds no algorithm and cublasLtMatmul
 * returns CUBLAS_STATUS_NOT_SUPPORTED. Attribute values are accepted and
 * not kept. The functions are inline: there is no cuBLASLt library to link.
 *
 * C and C++.
 */
#ifndef OCU_CUBLASLT_H
#define OCU_CUBLASLT_H

#include <stddef.h>
#include <stdint.h>

#include "cublas_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ocu_cublaslt_context *cublasLtHandle_t;
typedef struct ocu_cublaslt_matmul_desc *cublasLtMatmulDesc_t;
typedef struct ocu_cublaslt_matrix_layout *cublasLtMatrixLayout_t;
typedef struct ocu_cublaslt_matmul_preference *cublasLtMatmulPreference_t;

typedef struct {
    uint64_t data[8];
} cublasLtMatmulAlgo_t;

typedef struct {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

typedef enum {
    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,
    CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
    CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,
    CUBLASLT_MATMUL_DESC_TRANSA = 3,
    CUBLASLT_MATMUL_DESC_TRANSB = 4,
    CUBLASLT_MATMUL_DESC_TRANSC = 5,
    CUBLASLT_MATMUL_DESC_FILL_MODE = 6,
    CUBLASLT_MATMUL_DESC_EPILOGUE = 7,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
    CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,
    CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21
} cublasLtMatmulDescAttributes_t;

typedef enum {
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
    CUBLASLT_MATRIX_LAYOUT_COLS = 3,
    CUBLASLT_MATRIX_LAYOUT_LD = 4,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,
    CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7
} cublasLtMatrixLayoutAttribute_t;

typedef enum {
    CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
} cublasLtMatmulPreferenceAttributes_t;

/* Objects are tokens: a distinct non-null address per kind, no storage. */
static char ocu_cublaslt_token_[4];

static inline cublasStatus_t cublasLtCreate(cublasLtHandle_t *handle)
{
    if (handle == NULL) return CUBLAS_STATUS_INVALID_VALUE;
    *handle = (cublasLtHandle_t)(void *)&ocu_cublaslt_token_[0];
    return CUBLAS_STATUS_SUCCESS;
}
static inline cublasStatus_t cublasLtDestroy(cublasLtHandle_t handle)
{
    return handle == NULL ? CUBLAS_STATUS_NOT_INITIALIZED : CUBLAS_STATUS_SUCCESS;
}

static inline cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *desc,
                                                      cublasComputeType_t computeType,
                                                      cudaDataType_t scaleType)
{
    (void)computeType;
    (void)scaleType;
    if (desc == NULL) return CUBLAS_STATUS_INVALID_VALUE;
    *desc = (cublasLtMatmulDesc_t)(void *)&ocu_cublaslt_token_[1];
    return CUBLAS_STATUS_SUCCESS;
}
static inline cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t desc)
{
    (void)desc;
    return CUBLAS_STATUS_SUCCESS;
}
static inline cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t desc,
                                                            cublasLtMatmulDescAttributes_t attr,
                                                            const void *buf, size_t sizeInBytes)
{
    (void)attr;
    (void)sizeInBytes;
    return desc == NULL || buf == NULL ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
}

static inline cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *layout,
                                                        cudaDataType type, uint64_t rows,
                                                        uint64_t cols, int64_t ld)
{
    (void)type;
    (void)rows;
    (void)cols;
    (void)ld;
    if (layout == NULL) return CUBLAS_STATUS_INVALID_VALUE;
    *layout = (cublasLtMatrixLayout_t)(void *)&ocu_cublaslt_token_[2];
    return CUBLAS_STATUS_SUCCESS;
}
static inline cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t layout)
{
    (void)layout;
    return CUBLAS_STATUS_SUCCESS;
}
static inline cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t layout,
                                                              cublasLtMatrixLayoutAttribute_t attr,
                                                              const void *buf, size_t sizeInBytes)
{
    (void)attr;
    (void)sizeInBytes;
    return layout == NULL || buf == NULL ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
}

static inline cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)
{
    if (pref == NULL) return CUBLAS_STATUS_INVALID_VALUE;
    *pref = (cublasLtMatmulPreference_t)(void *)&ocu_cublaslt_token_[3];
    return CUBLAS_STATUS_SUCCESS;
}
static inline cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)
{
    (void)pref;
    return CUBLAS_STATUS_SUCCESS;
}
static inline cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void *buf,
    size_t sizeInBytes)
{
    (void)attr;
    (void)sizeInBytes;
    return pref == NULL || buf == NULL ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
}

/* No algorithm is available: zero results, CUBLAS_STATUS_NOT_SUPPORTED. */
static inline cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int *returnAlgoCount)
{
    (void)handle;
    (void)operationDesc;
    (void)Adesc;
    (void)Bdesc;
    (void)Cdesc;
    (void)Ddesc;
    (void)preference;
    (void)requestedAlgoCount;
    (void)heuristicResultsArray;
    if (returnAlgoCount != NULL) *returnAlgoCount = 0;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

static inline cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t computeDesc, const void *alpha, const void *A,
    cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta,
    const void *C, cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspaceSizeInBytes,
    cudaStream_t stream)
{
    (void)handle;
    (void)computeDesc;
    (void)alpha;
    (void)A;
    (void)Adesc;
    (void)B;
    (void)Bdesc;
    (void)beta;
    (void)C;
    (void)Cdesc;
    (void)D;
    (void)Ddesc;
    (void)algo;
    (void)workspace;
    (void)workspaceSizeInBytes;
    (void)stream;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif

#endif /* OCU_CUBLASLT_H */

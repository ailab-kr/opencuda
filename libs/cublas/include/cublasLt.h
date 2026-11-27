/*
 * cublasLt.h -- the lightweight matrix-multiply API (cuBLASLt): handle,
 * descriptor and layout objects, and cublasLtMatmul.
 *
 * Written for this project from the public cuBLASLt documentation
 * (function names, parameters, the descriptor, layout, preference and
 * epilogue enumerations). Not derived from any vendor header.
 *
 * cublasLtMatmul computes D = alpha op(A) op(B) + beta C through the
 * library's cublasGemmStridedBatchedEx (libs/cublas/cublas.cu), so it takes
 * the type and compute combinations that function takes (FP32, FP64, FP16,
 * BF16, int8 into int32). Supported on top: column- or row-major layouts
 * (all four the same order), strided batches, C and D at different
 * addresses (C is copied to D first), host or device pointer mode, and the
 * epilogues DEFAULT, RELU, BIAS, RELU_BIAS, GELU, GELU_BIAS (bias: one value
 * per row of D, of D's type unless BIAS_DATA_TYPE says otherwise). Anything
 * else (FP8, the AUX and gradient epilogues, scale pointers, mixed orders)
 * returns CUBLAS_STATUS_NOT_SUPPORTED. cublasLtMatmulAlgoGetHeuristic
 * returns one algorithm, needing no workspace. Link with -lcublasLt or
 * -lcublas (one library).
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
/* Descriptor, layout and preference objects live in storage of these sizes:
   allocated by the Create functions, or provided by the caller and set up
   by the Init functions (which need no Destroy). */
typedef struct {
    uint64_t data[32];
} cublasLtMatmulDescOpaque_t;
typedef struct {
    uint64_t data[8];
} cublasLtMatrixLayoutOpaque_t;
typedef struct {
    uint64_t data[8];
} cublasLtMatmulPreferenceOpaque_t;
typedef cublasLtMatmulDescOpaque_t *cublasLtMatmulDesc_t;
typedef cublasLtMatrixLayoutOpaque_t *cublasLtMatrixLayout_t;
typedef cublasLtMatmulPreferenceOpaque_t *cublasLtMatmulPreference_t;

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
    CUBLASLT_POINTER_MODE_HOST = 0,
    CUBLASLT_POINTER_MODE_DEVICE = 1,
    CUBLASLT_POINTER_MODE_DEVICE_VECTOR = 2,
    CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO = 3,
    CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST = 4
} cublasLtPointerMode_t;

typedef enum {
    CUBLASLT_ORDER_COL = 0,
    CUBLASLT_ORDER_ROW = 1,
    CUBLASLT_ORDER_COL32 = 2,
    CUBLASLT_ORDER_COL4_4R2_8C = 3,
    CUBLASLT_ORDER_COL32_2R_4R4 = 4
} cublasLtOrder_t;

typedef enum {
    CUBLASLT_EPILOGUE_DEFAULT = 1,
    CUBLASLT_EPILOGUE_RELU = 2,
    CUBLASLT_EPILOGUE_RELU_AUX = 130,
    CUBLASLT_EPILOGUE_BIAS = 4,
    CUBLASLT_EPILOGUE_RELU_BIAS = 6,
    CUBLASLT_EPILOGUE_RELU_AUX_BIAS = 134,
    CUBLASLT_EPILOGUE_DRELU = 136,
    CUBLASLT_EPILOGUE_DRELU_BGRAD = 152,
    CUBLASLT_EPILOGUE_GELU = 32,
    CUBLASLT_EPILOGUE_GELU_AUX = 160,
    CUBLASLT_EPILOGUE_GELU_BIAS = 36,
    CUBLASLT_EPILOGUE_GELU_AUX_BIAS = 164,
    CUBLASLT_EPILOGUE_DGELU = 192,
    CUBLASLT_EPILOGUE_DGELU_BGRAD = 208,
    CUBLASLT_EPILOGUE_BGRADA = 256,
    CUBLASLT_EPILOGUE_BGRADB = 512
} cublasLtEpilogue_t;

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
    CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13,
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = 14,
    CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 15,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
    CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,
    CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 23,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = 24,
    CUBLASLT_MATMUL_DESC_FAST_ACCUM = 25,
    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26
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

cublasStatus_t cublasLtCreate(cublasLtHandle_t *handle);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t handle);
size_t cublasLtGetVersion(void);

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *desc, cublasComputeType_t computeType,
                                        cudaDataType_t scaleType);
cublasStatus_t cublasLtMatmulDescInit(cublasLtMatmulDesc_t desc, cublasComputeType_t computeType,
                                      cudaDataType_t scaleType);
cublasStatus_t cublasLtMatrixLayoutInit(cublasLtMatrixLayout_t layout, cudaDataType type, uint64_t rows,
                                        uint64_t cols, int64_t ld);
cublasStatus_t cublasLtMatmulPreferenceInit(cublasLtMatmulPreference_t pref);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t desc);
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr,
                                              const void *buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr,
                                              void *buf, size_t sizeInBytes, size_t *sizeWritten);

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *layout, cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t layout);
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t layout, cublasLtMatrixLayoutAttribute_t attr,
                                                const void *buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t layout, cublasLtMatrixLayoutAttribute_t attr,
                                                void *buf, size_t sizeInBytes, size_t *sizeWritten);

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref);
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr, const void *buf,
                                                    size_t sizeInBytes);

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t handle, cublasLtMatmulDesc_t operationDesc,
                                              cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
                                              cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
                                              cublasLtMatmulPreference_t preference, int requestedAlgoCount,
                                              cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                              int *returnAlgoCount);

cublasStatus_t cublasLtMatmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t computeDesc, const void *alpha,
                              const void *A, cublasLtMatrixLayout_t Adesc, const void *B,
                              cublasLtMatrixLayout_t Bdesc, const void *beta, const void *C,
                              cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc,
                              const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspaceSizeInBytes,
                              cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* OCU_CUBLASLT_H */

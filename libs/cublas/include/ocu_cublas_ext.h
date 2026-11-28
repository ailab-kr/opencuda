/*
 * ocu_cublas_ext.h -- OpenCuda extensions to the cuBLAS API (not part of CUDA).
 *
 * Programs test OCU_CUBLAS_HAVE_NF4_GEMM and keep their portable path otherwise
 * (for example bitsandbytes' dequantisation followed by cublasGemmEx).
 */
#ifndef OCU_CUBLAS_EXT_H
#define OCU_CUBLAS_EXT_H
#include "cublas_v2.h"
#define OCU_CUBLAS_HAVE_NF4_GEMM 1
#ifdef __cplusplus
extern "C" {
#endif
/*
 * C[M,N] (f32, row stride ldc) = A[M,K] (bf16, row stride lda) * dequant(W)[N,K]^T + beta * C, row-major.
 * W: bitsandbytes NF4 with blocksize 64 (two values per byte, first in the high nibble; absmax f32 per
 * 64 consecutive values of a row). K % 64 == 0. The scale is applied to the f32 partial sums of each
 * 64-k block, so results can differ from dequantise-to-bf16-then-GEMM in the last bits.
 * CUBLAS_STATUS_NOT_SUPPORTED when the device or the arguments are not supported (use the portable path).
 */
cublasStatus_t ocublasGemmNF4(cublasHandle_t handle, int M, int N, int K, const void *A, int lda,
                              const unsigned char *W, const float *absmax, float beta, float *C, int ldc);
#ifdef __cplusplus
}
#endif
#endif

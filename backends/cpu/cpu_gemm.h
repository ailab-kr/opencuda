/*
 * CPU backend: the operator table's GEMM as a blocked, vectorised routine
 * (cpu_gemm.c). Internal to backends/cpu.
 */
#ifndef OCU_CPU_GEMM_H
#define OCU_CPU_GEMM_H

#include "ocu_ops.h"

#include <stdint.h>

#define GEMM_MR 4 /* rows of A per inner step */
#define GEMM_NR 3 /* rows of B (columns of C) per inner step */

/* 1 when this build has the routine (GCC or Clang vector extension). */
int cpu_gemm_available(void);

/*
 * Queue C = alpha op(A) op(B) + beta C for a descriptor that passed
 * ocu_ref_check_tensors (the actual shapes of this call), data[i] being
 * the pointer of slot i. Same contract as the reference: binary32
 * accumulation, beta == 0 never reads C, alpha == 0 never reads A and B.
 */
ocu_status cpu_gemm_launch(ocu_queue *queue, const ocu_op_desc *d, void *const *data);

/* Per-instruction-set inner routines (cpu_gemm_kern.inc). */
#define CPU_GEMM_DECLARE(sfx)                                                          \
    void cpu_gemm_panel_##sfx(float *dst, int64_t kcp, const void *src, ocu_dtype dt,   \
                              int64_t ld, int kmajor, int64_t r0, int64_t nrows,        \
                              int64_t rows_pad, int64_t k0, int64_t kc);                \
    void cpu_gemm_dots_##sfx(float *acc, int64_t ldacc, const float *ap,                \
                             const float *bp, int64_t rows_pad, int64_t cols_pad,       \
                             int64_t kcp);
CPU_GEMM_DECLARE(generic)
CPU_GEMM_DECLARE(avx2)

#endif /* OCU_CPU_GEMM_H */

/*
 * mmvq_rl.h -- shared declarations of the region-loop stage-0 benchmark
 * (docs/cpu-region-loops.md, stage 0).
 *
 * The kernel is llama.cpp's quantised matrix-vector product
 * (ggml-cuda/mmvq.cu, mul_mat_vec_q) for one case: weights Q8_0, activations
 * Q8_1, one output column (generation), no fusion, the GENERIC parameter
 * table that the CPU path selects (__CUDA_ARCH__ 500): 4 warps of 32
 * threads per block, 1 row per block, vdr 2. llama.cpp is MIT licensed;
 * the code here is written for this benchmark following that kernel's
 * structure and arithmetic order, it is not a copy of the file.
 */
#ifndef MMVQ_RL_H
#define MMVQ_RL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MMVQ_QK      32 /* values per Q8_0 / Q8_1 block */
#define MMVQ_QI      8  /* 32-bit ints of quants per block */
#define MMVQ_VDR     2  /* ints per thread and block */
#define MMVQ_WARP    32
#define MMVQ_NWARPS  4
#define MMVQ_THREADS (MMVQ_WARP * MMVQ_NWARPS)

/* Layouts as in ggml: f16 scale (+ f16 sum for Q8_1) then 32 int8 quants. */
typedef struct mmvq_q8_0 { uint16_t d; int8_t qs[MMVQ_QK]; } mmvq_q8_0;
typedef struct mmvq_q8_1 { uint16_t d, s; int8_t qs[MMVQ_QK]; } mmvq_q8_1;

typedef struct mmvq_args {
    const mmvq_q8_0 *vx; /* nrows x (ncols / 32) blocks */
    const mmvq_q8_1 *vy; /* ncols / 32 blocks */
    float *dst;          /* nrows */
    int ncols_x;         /* K */
    int stride_row_x;    /* blocks per row */
    int nrows;           /* N, also the grid size */
} mmvq_args;

/* Loop form, one build per compiler flag set (mmvq_loop.cpp compiled
   three times with MMVQ_LOOP_NAME set): runs the whole grid on `nthreads`
   OS threads of its own pool. `form` 0: uniform fast loop with the
   next-pc check; 1: pc dispatcher only (the fallback schedule).
   Returns the number of blocks that fell back to the dispatcher. */
long mmvq_loop_scalar(const mmvq_args *a, int nthreads, int form);
long mmvq_loop_vec(const mmvq_args *a, int nthreads, int form);
long mmvq_loop_avx2(const mmvq_args *a, int nthreads, int form);
/* The OS-thread pool the loop forms share (mmvq_pool.cpp): runs fn(arg)
   on `nthreads` threads, the caller being one of them, and returns when
   all are done. Workers spin briefly for the next launch, then sleep. */
void mmvq_pool_run(int nthreads, void (*fn)(void *arg), void *arg);
/* An empty launch through the pool (fixed launch cost). */
void mmvq_pool_empty(int nthreads);

#ifdef __cplusplus
}
#endif

#endif

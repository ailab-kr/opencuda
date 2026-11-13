/*
 * GEMM inner routines for x86-64 with AVX2, FMA and F16C. CMake compiles
 * this file with those instruction sets (only on x86-64 with GCC or Clang);
 * cpu_gemm.c calls it only when the processor reports all three.
 */
#if defined(OCU_CPU_GEMM_AVX2) && (defined(__GNUC__) || defined(__clang__))
#define GEMM_SUFFIX avx2
#include "cpu_gemm_kern.inc"
#else
typedef int cpu_gemm_avx2_unused;
#endif

/* GEMM inner routines for the build's baseline instruction set. */
#if defined(__GNUC__) || defined(__clang__)
#define GEMM_SUFFIX generic
#include "cpu_gemm_kern.inc"
#else
typedef int cpu_gemm_generic_unused;
#endif

/*
 * libm_fma.c -- correctly rounded fma and fmaf in place of the C
 * library's, for hosts where those are not (MinGW-w64: up to one ulp off,
 * e.g. fmaf(0x1.02fbdap+0, 0x1.fffffep-1, -0x1.02fbd8p+0)).
 *
 * CUDA documents fma/fmaf as one operation with a single rounding (IEEE
 * 754-2019 5.4.1 fusedMultiplyAdd), and kernels built for the CPU path
 * call the C library's fma and fmaf (directly, through std::fma, or
 * through the compiler's builtin, which becomes a library call on a target
 * without the instruction). A program links ocu_cudart ahead of the C
 * library, so its calls resolve to these definitions.
 *
 * Built into ocu_cudart only when the configure check (check_libm_fma.c)
 * finds the C library's results differ from the exact ones; elsewhere the
 * C library's functions are used unchanged.
 */
#include <math.h>

#include "ocu_fma_exact.h"

#if !defined(OCU_FMA_HAVE_SOFT_DOUBLE)
#error "libm_fma.c needs the software fma of ocu_fma_exact.h (128-bit integers, double evaluation)"
#endif

double fma(double a, double b, double c) { return ocu_fma_soft(a, b, c, OCU_FMA_RN); }

float fmaf(float a, float b, float c) { return ocu_fmaf_soft(a, b, c, OCU_FMA_RN); }

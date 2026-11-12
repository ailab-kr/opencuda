/*
 * fma_libm_probe <0|1>: does the C library this program is linked with (not
 * ocu_cudart) round fmaf once?  The argument is what the configure check
 * (runtime/CMakeLists.txt, OCU_LIBM_FMA_EXACT) decided.  The sample is the
 * one the header runtime/include/ocu_fma_exact.h documents:
 *   fmaf(0x1.02fbdap+0, 0x1.fffffep-1, -0x1.02fbd8p+0)
 * exact (IEEE 754-2019 5.4.1, one rounding):  0x1.fa084cp-25
 * MinGW-w64's library:                          0x1.fa085p-25
 * Exit 0 when the configure result agrees with what the library does: exact
 * library <=> 1.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    int configured_exact = atoi(argv[1]);
    volatile float a = 0x1.02fbdap+0f, b = 0x1.fffffep-1f, c = -0x1.02fbd8p+0f;
    float r = fmaf(a, b, c);
    int exact = r == 0x1.fa084cp-25f;
    printf("C library fmaf sample = %a (%s); configure check says exact=%d\n", (double)r, exact ? "exact" : "not exact",
           configured_exact);
    return exact == (configured_exact != 0) ? 0 : 1;
}

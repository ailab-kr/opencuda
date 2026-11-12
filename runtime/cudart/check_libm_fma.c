/*
 * check_libm_fma.c -- configure check (runtime/CMakeLists.txt): are the C
 * library's fma and fmaf correctly rounded? Compares them with the exact
 * software fma of ocu_fma_exact.h on known hard cases and on pseudo-random
 * operands with c close to -a*b (cancellation and ties), and with random
 * bit patterns. Exit status 0: all equal; 1: the library differs (then
 * ocu_cudart carries replacements, cudart/libm_fma.c).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ocu_fma_exact.h"

#if !defined(OCU_FMA_HAVE_SOFT_DOUBLE)
int main(void) { return 0; } /* nothing to replace the library with */
#else

static uint64_t g_state = 88172645463325252ull;
static uint64_t next(void)
{
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return g_state;
}
static float f_of(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static uint32_t bits_f(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static double d_of(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
static uint64_t bits_d(double d) { uint64_t b; memcpy(&b, &d, 8); return b; }

static int bad;

static void check_f(float a, float b, float c)
{
    volatile float va = a, vb = b, vc = c; /* no constant folding */
    float lib = fmaf(va, vb, vc), want = ocu_fmaf_soft(a, b, c, OCU_FMA_RN);
    if (bits_f(lib) != bits_f(want) && !(isnan(lib) && isnan(want))) {
        if (bad++ < 3) printf("fmaf(%a, %a, %a) = %a, exact %a\n", a, b, c, lib, want);
    }
}

static void check_d(double a, double b, double c)
{
    volatile double va = a, vb = b, vc = c;
    double lib = fma(va, vb, vc), want = ocu_fma_soft(a, b, c, OCU_FMA_RN);
    if (bits_d(lib) != bits_d(want) && !(isnan(lib) && isnan(want))) {
        if (bad++ < 3) printf("fma(%a, %a, %a) = %a, exact %a\n", a, b, c, lib, want);
    }
}

int main(void)
{
    check_f(0x1.02fbdap+0f, 0x1.fffffep-1f, -0x1.02fbd8p+0f);
    check_f(0x1p-126f, 0x1p-24f, 0x1p-149f);
    check_d(0x1.0000000000001p0, 0x1.0000000000001p0, -1.0);
    check_d(0x1p-1022, 0x1p-53, 0x1p-1074);
    check_d(0x1.fffffffffffffp1023, 2.0, -0x1.fffffffffffffp1023);
    for (int i = 0; i < 200000; i++) {
        float a = f_of(0x3f000000u | (uint32_t)(next() & 0x7fffffu));
        float b = f_of(0x3f000000u | (uint32_t)(next() & 0x7fffffu));
        float c = -f_of(bits_f(a * b) + (uint32_t)(next() % 9) - 4u);
        check_f(a, b, c);
        check_f(f_of((uint32_t)next()), f_of((uint32_t)next()), f_of((uint32_t)next()));
        double x = d_of(0x3fe0000000000000ull | (next() & 0xfffffffffffffull));
        double y = d_of(0x3fe0000000000000ull | (next() & 0xfffffffffffffull));
        double z = -d_of(bits_d(x * y) + next() % 9 - 4u);
        check_d(x, y, z);
        check_d(d_of(next()), d_of(next()), d_of(next()));
        if (bad > 0) break;
    }
    printf("%s\n", bad ? "fma/fmaf: not correctly rounded" : "fma/fmaf: correctly rounded");
    return bad ? 1 : 0;
}
#endif

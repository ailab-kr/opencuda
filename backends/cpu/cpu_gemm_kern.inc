/*
 * Inner routines of the CPU backend's GEMM (cpu_gemm.c), compiled once per
 * instruction set: cpu_gemm_generic.c (the build's baseline flags) and, on
 * x86-64, cpu_gemm_avx2.c (AVX2, FMA, F16C). The including file defines
 * GEMM_SUFFIX. Plain C with the GCC/Clang vector extension; the compiler
 * picks the instructions.
 *
 * Panels hold rows of `kcp` floats (a multiple of 8, zero padded), each row
 * contiguous along the inner dimension k, 64-byte aligned.
 */
#include "cpu_gemm.h"

#include <string.h>
#if defined(__F16C__)
#include <immintrin.h>
#endif

#define GEMM_CAT2(a, b) a##_##b
#define GEMM_CAT(a, b)  GEMM_CAT2(a, b)
#define GEMM_FN(name)   GEMM_CAT(name, GEMM_SUFFIX)

typedef float gemm_v8 __attribute__((vector_size(32)));

/* Exact: every binary16 value is a binary32 value. The exponent and
   mantissa bits move into place and a multiply by 2^112 rebiases the
   exponent (subnormal halves become normal floats); infinities and NaNs get
   the all-ones exponent. Branch-free, so the loops below vectorise. */
static inline float gemm_h2f(uint16_t h)
{
    uint32_t mag = (uint32_t)(h & 0x7fffu) << 13;
    float v;
    memcpy(&v, &mag, 4);
    v *= 5.192296858534828e33f; /* 2^112 */
    uint32_t bits;
    memcpy(&bits, &v, 4);
    if ((h & 0x7c00u) == 0x7c00u) bits = 0x7f800000u | mag;
    bits |= (uint32_t)(h & 0x8000u) << 16;
    memcpy(&v, &bits, 4);
    return v;
}

static inline float gemm_bf2f(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

/*
 * Fill `rows_pad` panel rows of `kcp` floats: row r, column kk is element
 * (r0 + r, k0 + kk) of the logical matrix X, where X(r, k) = src[r*ld + k]
 * (kmajor = 1) or src[k*ld + r] (kmajor = 0). Rows past `nrows` and columns
 * past `kc` are zero.
 */
void GEMM_FN(cpu_gemm_panel)(float *dst, int64_t kcp, const void *src, ocu_dtype dt,
                             int64_t ld, int kmajor, int64_t r0, int64_t nrows,
                             int64_t rows_pad, int64_t k0, int64_t kc)
{
    if (kmajor) {
        for (int64_t r = 0; r < nrows; ++r) {
            float *d = dst + r * kcp;
            const int64_t base = (r0 + r) * ld + k0;
            if (dt == OCU_DT_F32) {
                memcpy(d, (const float *)src + base, (size_t)kc * sizeof(float));
            } else if (dt == OCU_DT_F16) {
                const uint16_t *s = (const uint16_t *)src + base;
                int64_t kk = 0;
#if defined(__F16C__)
                /* Hardware conversion, exact like gemm_h2f. */
                for (; kk + 8 <= kc; kk += 8) {
                    _mm256_store_ps(d + kk, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + kk))));
                }
#endif
                for (; kk < kc; ++kk) d[kk] = gemm_h2f(s[kk]);
            } else {
                const uint16_t *s = (const uint16_t *)src + base;
                for (int64_t kk = 0; kk < kc; ++kk) d[kk] = gemm_bf2f(s[kk]);
            }
            for (int64_t kk = kc; kk < kcp; ++kk) d[kk] = 0.0f;
        }
    } else {
        for (int64_t r = 0; r < nrows; ++r) {
            for (int64_t kk = kc; kk < kcp; ++kk) dst[r * kcp + kk] = 0.0f;
        }
        for (int64_t kk = 0; kk < kc; ++kk) {
            const int64_t base = (k0 + kk) * ld + r0;
            if (dt == OCU_DT_F32) {
                const float *s = (const float *)src + base;
                for (int64_t r = 0; r < nrows; ++r) dst[r * kcp + kk] = s[r];
            } else if (dt == OCU_DT_F16) {
                const uint16_t *s = (const uint16_t *)src + base;
                for (int64_t r = 0; r < nrows; ++r) dst[r * kcp + kk] = gemm_h2f(s[r]);
            } else {
                const uint16_t *s = (const uint16_t *)src + base;
                for (int64_t r = 0; r < nrows; ++r) dst[r * kcp + kk] = gemm_bf2f(s[r]);
            }
        }
    }
    for (int64_t r = nrows; r < rows_pad; ++r) memset(dst + r * kcp, 0, (size_t)kcp * sizeof(float));
}

/* Sum of the eight lanes, in a fixed order. */
static inline float gemm_hsum(const gemm_v8 *p)
{
    const gemm_v8 v = *p;
    return ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]));
}

/*
 * acc[i*ldacc + j] += dot(a row i, b row j) over kcp, for i < rows_pad
 * (a multiple of GEMM_MR) and j < cols_pad (a multiple of GEMM_NR).
 * Four rows of A by three rows of B at a time: 12 vector accumulators.
 */
void GEMM_FN(cpu_gemm_dots)(float *acc, int64_t ldacc, const float *ap, const float *bp,
                            int64_t rows_pad, int64_t cols_pad, int64_t kcp)
{
    for (int64_t i = 0; i < rows_pad; i += GEMM_MR) {
        const float *a0 = ap + i * kcp, *a1 = a0 + kcp, *a2 = a1 + kcp, *a3 = a2 + kcp;
        for (int64_t j = 0; j < cols_pad; j += GEMM_NR) {
            const float *b0 = bp + j * kcp, *b1 = b0 + kcp, *b2 = b1 + kcp;
            gemm_v8 c00 = {0}, c01 = {0}, c02 = {0}, c10 = {0}, c11 = {0}, c12 = {0};
            gemm_v8 c20 = {0}, c21 = {0}, c22 = {0}, c30 = {0}, c31 = {0}, c32 = {0};
            for (int64_t k = 0; k < kcp; k += 8) {
                const gemm_v8 x0 = *(const gemm_v8 *)(b0 + k);
                const gemm_v8 x1 = *(const gemm_v8 *)(b1 + k);
                const gemm_v8 x2 = *(const gemm_v8 *)(b2 + k);
                gemm_v8 a = *(const gemm_v8 *)(a0 + k);
                c00 += a * x0; c01 += a * x1; c02 += a * x2;
                a = *(const gemm_v8 *)(a1 + k);
                c10 += a * x0; c11 += a * x1; c12 += a * x2;
                a = *(const gemm_v8 *)(a2 + k);
                c20 += a * x0; c21 += a * x1; c22 += a * x2;
                a = *(const gemm_v8 *)(a3 + k);
                c30 += a * x0; c31 += a * x1; c32 += a * x2;
            }
            float *o = acc + i * ldacc + j;
            o[0] += gemm_hsum(&c00); o[1] += gemm_hsum(&c01); o[2] += gemm_hsum(&c02);
            o += ldacc;
            o[0] += gemm_hsum(&c10); o[1] += gemm_hsum(&c11); o[2] += gemm_hsum(&c12);
            o += ldacc;
            o[0] += gemm_hsum(&c20); o[1] += gemm_hsum(&c21); o[2] += gemm_hsum(&c22);
            o += ldacc;
            o[0] += gemm_hsum(&c30); o[1] += gemm_hsum(&c31); o[2] += gemm_hsum(&c32);
        }
    }
}

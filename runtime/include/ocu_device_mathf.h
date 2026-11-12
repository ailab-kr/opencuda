/*
 * ocu_device_mathf.h -- single-precision device math from float operations
 * only (+ - * /, fmaf, sqrtf, rintf, integer operations and bit casts; no
 * double arithmetic, which runs at a small fraction of the float rate on
 * many GPUs). Included by ocu_device_math.h.
 *
 * Methods (IEEE 754 arithmetic and classic public numerical methods):
 *  - error-free transformations (two-sum, two-product with fma) carry
 *    sums in two floats where one float would lose accuracy;
 *  - exp: x = k ln2 + r, |r| <= ln2/2 (Cody-Waite reduction, ln2 in two
 *    floats), e^r - 1 = r + r^2/2 + r^3 Q(r) in two floats;
 *  - log: x = 2^e m, a 65-entry table gives c ~ 1/m with -ln c in two
 *    floats, r = m c - 1 exactly (two-product), log1p(r) =
 *    r - r^2/2 + r^3 Q(r) with |r| <= 2^-7; the result in two floats is
 *    accurate enough for pow (pow = exp(y log x) in two floats);
 *  - sin, cos, tan: x = k pi/2 + r with pi/2 in three floats below 2^17,
 *    and above by Payne-Hanek reduction with the bits of 2/pi (integer
 *    arithmetic); odd/even polynomials on |r| <= pi/4;
 *  - atan: 1/x above 1, then an odd polynomial on [0, 1]; asin/acos with
 *    asin(x) = pi/2 - 2 asin(sqrt((1 - x)/2)) above 1/2;
 *  - erf: an odd polynomial up to 0.875, then 1 - erfc; erfc(x) =
 *    t exp(-x^2 + P(t)) with t = 1/(1 + x/2) above 1/2 (the form of the
 *    classic Chebyshev fit in Numerical Recipes, with our own coefficients);
 *  - cbrt: polynomial start, Newton steps for x^(-1/3), one last step on
 *    the exact residual.
 * Polynomial coefficients are minimax fits made by tools/gen_mathf_tables.py
 * (Remez exchange in exact rationals), which also makes the tables.
 * OCU_DMF_TABLE is the storage of the log table (device code: global
 * memory, which serves per-thread indices faster than constant memory).
 *
 * No contraction of a*b+c into fma beyond the explicit fmaf calls, so the
 * host build (the accuracy tests) and the device build give the same bits.
 * Clang: OCU_DMF_STRICT at the start of every function, honoured with
 * -ffp-contract=on or fast-honor-pragmas (ocu_cc passes the latter; CUDA's
 * default, fast, ignores it). GCC: an optimize pragma around this file.
 */
#ifndef OCU_DEVICE_MATHF_H
#define OCU_DEVICE_MATHF_H

#include <cstdint>

#ifndef OCU_DMF_TABLE
#define OCU_DMF_TABLE constexpr
#endif
#include "ocu_device_mathf_tables.h"
#include "ocu_fma_exact.h"

#if defined(__clang__)
#define OCU_DMF_STRICT _Pragma("clang fp contract(off)")
#else
#define OCU_DMF_STRICT
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#define OCU_DMF_GCC_POP
#endif
#endif

namespace ocu_dm {
namespace mf {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

OCU_DM_FN u32 fbits(float x) { return __builtin_bit_cast(u32, x); }
OCU_DM_FN float ffrom(u32 b) { return __builtin_bit_cast(float, b); }
OCU_DM_FN float fabs_(float x) { return ffrom(fbits(x) & 0x7fffffffu); }
OCU_DM_FN float copysign_(float x, float y) { return ffrom((fbits(x) & 0x7fffffffu) | (fbits(y) & 0x80000000u)); }
OCU_DM_FN bool isnan_(float x) { return (fbits(x) & 0x7fffffffu) > 0x7f800000u; }
OCU_DM_FN bool isfinite_(float x) { return (fbits(x) & 0x7f800000u) != 0x7f800000u; }
/* Hardware fma in a device pass; on a host without the instruction, exact
   software rather than the C library's fmaf, which is not correctly
   rounded everywhere (ocu_fma_exact.h). */
OCU_DM_FN float fma_(float a, float b, float c) { return ::ocu_fmaf_rn(a, b, c); }
OCU_DM_FN float rint_(float x) { return __builtin_rintf(x); }
OCU_DM_FN float sqrt_(float x) { return __builtin_sqrtf(x); }
OCU_DM_FN float inf_() { return ffrom(0x7f800000u); }
OCU_DM_FN float nan_() { return ffrom(0x7fc00000u); }

/* 2^n for -126 <= n <= 127. */
OCU_DM_FN float pow2(int n) { return ffrom((u32)(n + 127) << 23); }

/* v * 2^n with one rounding, for 2^-26 <= |v| < 4 (any n). */
OCU_DM_FN float scale(float v, int n)
{
    OCU_DMF_STRICT
    if (n > 127) {
        v *= 0x1p127f;
        n -= 127;
        if (n > 127) n = 127;
    } else if (n < -126) {
        v *= 0x1p-100f; /* exact: stays normal */
        n += 100;
        if (n < -126) n = -126;
    }
    return v * pow2(n);
}

/* Exponent of a finite non-zero float: |x| in [2^e, 2^(e+1)). */
OCU_DM_FN int ilogb_(float x)
{
    u32 b = fbits(x) & 0x7fffffffu;
    if (b < 0x00800000u) return 31 - __builtin_clz(b) - 149;
    return (int)(b >> 23) - 127;
}

template <int N>
OCU_DM_FN float horner(float x, const float (&c)[N])
{
    OCU_DMF_STRICT
    float p = c[N - 1];
    for (int i = N - 2; i >= 0; --i) p = fma_(p, x, c[i]);
    return p;
}

/* sum of c[i] x^(i - S) for i >= S */
template <int S, int N>
OCU_DM_FN float horner_tail(float x, const float (&c)[N])
{
    OCU_DMF_STRICT
    float p = c[N - 1];
    for (int i = N - 2; i >= S; --i) p = fma_(p, x, c[i]);
    return p;
}

/* A value hi + lo, |lo| <= ulp(hi)/2 or so. */
struct f2 {
    float hi, lo;
};
OCU_DM_FN f2 two_sum(float a, float b)
{
    OCU_DMF_STRICT
    float s = a + b;
    float bb = s - a;
    return f2{s, (a - (s - bb)) + (b - bb)};
}
OCU_DM_FN f2 fast_two_sum(float a, float b) /* |a| >= |b| or a == 0 */
{
    OCU_DMF_STRICT
    float s = a + b;
    return f2{s, b - (s - a)};
}
OCU_DM_FN f2 neg(f2 a) { return f2{-a.hi, -a.lo}; }

/* ---- exp ------------------------------------------------------------------ */

/* e^(rh + rd) - 1 as two floats, |rh| <= 0.3467, |rd| <= 2^-20 |rh| or so. */
OCU_DM_FN f2 expm1_kernel(float rh, float rd)
{
    OCU_DMF_STRICT
    float zh = rh * rh;
    float zl = fma_(rh, rh, -zh);
    float hz = 0.5f * zh, hzl = 0.5f * zl;
    float c = (rh * zh) * horner(rh, EXP_Q); /* r^3 Q(r) */
    f2 s = fast_two_sum(rh, hz);
    float lo = s.lo + (hzl + (c + fma_(rd, rh, rd)));
    return fast_two_sum(s.hi, lo);
}

/* e^(xh + xl) = 2^k (1 + u), returns u (two floats); |xh| <= 104. */
OCU_DM_FN f2 exp_reduce(float xh, float xl, int *k)
{
    OCU_DMF_STRICT
    float kf = rint_(xh * LOG2E_H);
    float r1 = fma_(-kf, LN2_E_H, xh); /* exact */
    float r2 = fma_(-kf, LN2_E_L, xl);
    f2 r = two_sum(r1, r2);
    *k = (int)kf;
    return expm1_kernel(r.hi, r.lo);
}

OCU_DM_FN f2 one_plus(f2 u) /* |u| < 1 */
{
    OCU_DMF_STRICT
    f2 a = fast_two_sum(1.0f, u.hi);
    return fast_two_sum(a.hi, a.lo + u.lo);
}

/* ---- log ------------------------------------------------------------------ */

/* log1p(rh + rl) as two floats for |r| <= 2^-7 (|rl| tiny). */
OCU_DM_FN f2 log1p_kernel(float rh, float rl)
{
    OCU_DMF_STRICT
    float q = rh * rh;
    float ql = fma_(rh, rh, -q);
    float hz = 0.5f * q;
    float hzl = fma_(0.5f, ql, rh * rl);
    float c = (rh * q) * horner(rh, LOG_Q); /* r^3 Q(r) */
    f2 s = fast_two_sum(rh, -hz);
    return f2{s.hi, s.lo + ((rl - hzl) + c)};
}

/* log(x) as two floats, relative error about 2^-40; x finite and > 0. */
OCU_DM_FN f2 log_kernel(float x)
{
    OCU_DMF_STRICT
    u32 ix = fbits(x);
    int e = 0;
    if (ix < 0x00800000u) { /* subnormal */
        x *= 0x1p24f;
        ix = fbits(x);
        e = -24;
    }
    e += (int)(ix >> 23) - 127;
    u32 mb = ix & 0x007fffffu;
    u32 i = (mb + 0x00010000u) >> 17; /* nearest 1 + i/64, 0..64 */
    float m = ffrom(mb | 0x3f800000u);  /* x = 2^e m, m in [1, 2) */
    if (i >= 27) {                      /* m near 2: use m/2 (the table's c is doubled) */
        m *= 0.5f;
        e += 1;
    }
    const log_entry &t = LOG_TABLE[i];
    float p = m * t.c;
    float pe = fma_(m, t.c, -p);
    f2 r = fast_two_sum(p - 1.0f, pe); /* m c - 1, exactly */
    f2 l = log1p_kernel(r.hi, r.lo);
    float ef = (float)e;
    f2 a = two_sum(ef * LN2_L_H, t.nlh); /* e * LN2_L_H is exact */
    f2 b = two_sum(a.hi, l.hi);
    float low = (a.lo + b.lo) + (fma_(ef, LN2_L_M, t.nll) + fma_(ef, LN2_L_L, l.lo));
    return fast_two_sum(b.hi, low);
}

/* ---- trigonometric reduction -------------------------------------------------------- */

/* |x| >= 2^17: x = k pi/2 + r, |r| <= pi/4 (Payne-Hanek). Returns k. */
OCU_DM_FN int reduce_pio2_large(float x, f2 *rr)
{
    OCU_DMF_STRICT
    u32 ix = fbits(x) & 0x7fffffffu;
    int e = (int)(ix >> 23) - 150; /* |x| = m 2^e, m a 24-bit integer */
    u64 m = (ix & 0x007fffffu) | 0x00800000u;
    /* 96 bits of 2/pi from the bit of value 2^-(e-1) on: bits of higher
       value only add multiples of 4 to x 2/pi. Bit p (value 2^-p) is bit
       p + 31 of the table, counted from the top. */
    int g = e + 30;
    int j = g >> 5, s = g & 31;
    const u32 *w = TWO_OVER_PI_BITS;
    u32 w0 = (u32)((((u64)w[j] << 32) | w[j + 1]) >> (32 - s));
    u32 w1 = (u32)((((u64)w[j + 1] << 32) | w[j + 2]) >> (32 - s));
    u32 w2 = (u32)((((u64)w[j + 2] << 32) | w[j + 3]) >> (32 - s));
    /* m (w0 w1 w2) as 128 bits hi:lo; x 2/pi = that * 2^-94 */
    u64 p2 = m * w2, p1 = m * w1, p0 = m * w0;
    u64 lo = p2 + (p1 << 32);
    u64 hi = p0 + (p1 >> 32) + (lo < p2 ? 1u : 0u);
    int q = (int)((hi >> 30) & 3);
    u64 fh = hi & ((1ull << 30) - 1); /* fraction = (fh 2^64 + lo) 2^-94 */
    bool negf = (fh >> 29) & 1;       /* fraction >= 1/2: round k up */
    if (negf) {
        q += 1;
        lo = ~lo + 1;
        fh = (~fh + (lo == 0 ? 1u : 0u)) & ((1ull << 30) - 1);
    }
    u64 f = (fh << 34) | (lo >> 30); /* top 64 bits of the fraction: f 2^-64 */
    if (f == 0) { *rr = f2{0.0f, 0.0f}; return x < 0 ? -q : q; }
    int lz = __builtin_clzll(f);
    if (lz) f = (f << lz) | ((lo << 34) >> (64 - lz));
    /* fraction = f 2^-(64 + lz), f has its top bit set */
    float fhi = (float)(u32)(f >> 40) * pow2(-24 - lz);
    float flo = (float)(u32)((f >> 16) & 0x00ffffffu) * pow2(-48 - lz);
    float rh = fhi * PIO2_H;
    float rl = fma_(fhi, PIO2_H, -rh) + fma_(flo, PIO2_H, fhi * PIO2_M);
    f2 r = fast_two_sum(rh, rl);
    if (negf) r = neg(r);
    if (x < 0) {
        r = neg(r);
        q = -q;
    }
    *rr = r;
    return q;
}

/* x = k pi/2 + r (r in two floats, |r| < 0.7955); returns k. */
OCU_DM_FN int reduce_pio2(float x, f2 *rr)
{
    OCU_DMF_STRICT
    if (fabs_(x) >= 0x1p17f) return reduce_pio2_large(x, rr);
    float kf = rint_(x * TWO_OVER_PI);
    float r1 = fma_(-kf, PIO2_H, x); /* exact */
    float p = kf * PIO2_M;
    float pe = fma_(kf, PIO2_M, -p);
    f2 s = two_sum(r1, -p);
    float lo = s.lo - fma_(kf, PIO2_L, pe);
    *rr = two_sum(s.hi, lo);
    return (int)kf;
}

/* sin and cos of r = rh + rl, |r| < 0.7955, as two floats. */
OCU_DM_FN f2 sin_kernel(f2 r)
{
    OCU_DMF_STRICT
    float z = r.hi * r.hi;
    float c = fma_(r.hi * z, horner(z, SIN_S), fma_(-0.5f * z, r.lo, r.lo));
    return fast_two_sum(r.hi, c);
}
OCU_DM_FN f2 cos_kernel(f2 r)
{
    OCU_DMF_STRICT
    float zh = r.hi * r.hi;
    float zl = fma_(r.hi, r.hi, -zh);
    float hz = 0.5f * zh;
    float w = 1.0f - hz;
    float ew = (1.0f - w) - hz; /* exact */
    float tail = fma_(zh * zh, horner(zh, COS_C), ew - fma_(0.5f, zl, r.hi * r.lo));
    return fast_two_sum(w, tail);
}

OCU_DM_FN float div2(f2 n, f2 d)
{
    OCU_DMF_STRICT
    float q = n.hi / d.hi;
    float res = fma_(-q, d.hi, n.hi) + fma_(-q, d.lo, n.lo);
    return q + res / d.hi;
}

/* ---- atan ---------------------------------------------------------------------------- */

/* atan(th + tl) - th for 0 <= th <= 1, |tl| <= 2^-23 th. */
OCU_DM_FN float atan_lo(float th, float tl)
{
    OCU_DMF_STRICT
    float zh = th * th;
    float zl = fma_(th, th, -zh);
    float a = horner(zh, ATAN_A);
    float tz = th * zh;
    float tzl = fma_(th, zh, -tz) + th * zl;
    float w = fma_(zh, fma_(zh, 0.2f, -0.7f), 1.0f); /* ~ 1/(1 + z), within 5% */
    return fma_(tz, a, fma_(tzl, a, tl * w));
}

/* ---- erfc ------------------------------------------------------------------------------- */

/* erfc(x) = 2^k (hi + lo) for 0.5 <= x <= 10.1. */
OCU_DM_FN f2 erfc_kernel(float x, int *k)
{
    OCU_DMF_STRICT
    f2 d = two_sum(1.0f, 0.5f * x);
    float th = 1.0f / d.hi;
    float tl = th * (fma_(-th, d.hi, 1.0f) - th * d.lo); /* t = 1/(1 + x/2) = th + tl */
    float u = th - 0.5f;
    /* -x^2 + P(u) = -x^2 + P0 + c1 u + u^2 H(u), the first terms in two
       floats; c1 tl accounts for the low part of t */
    const float c1 = ERFC_P[0];
    float m = c1 * u;
    float ml = fma_(c1, u, -m);
    float h = horner_tail<1>(u, ERFC_P);
    float x2 = x * x;
    float x2l = fma_(x, x, -x2);
    f2 a = two_sum(-x2, ERFC_P0_H);
    f2 b = two_sum(a.hi, m);
    float low = (b.lo + (a.lo - x2l)) + ((ERFC_P0_L + ml) + fma_(u * u, h, c1 * tl));
    f2 v = one_plus(exp_reduce(b.hi, low, k));
    float ph = v.hi * th;
    float pl = fma_(v.hi, th, -ph) + fma_(v.hi, tl, v.lo * th);
    return fast_two_sum(ph, pl);
}

} // namespace mf

/* ---- the functions ------------------------------------------------------------------------ */

OCU_DM_FN float expf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    if (x > 89.0f) return inf_();
    if (x < -104.0f) return 0.0f;
    int k;
    f2 y = one_plus(exp_reduce(x, 0.0f, &k));
    return scale(y.hi, k);
}

OCU_DM_FN float exp2f(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    if (x >= 128.0f) return inf_();
    if (x < -150.0f) return 0.0f;
    float kf = rint_(x);
    float f = x - kf; /* exact */
    float rh = f * LN2_E_H;
    float rd = fma_(f, LN2_E_L, fma_(f, LN2_E_H, -rh));
    f2 y = one_plus(expm1_kernel(rh, rd));
    return scale(y.hi, (int)kf);
}

OCU_DM_FN float exp10f(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    if (x > 38.6f) return inf_();
    if (x < -45.2f) return 0.0f;
    float sh = x * LN10_H;
    float sl = fma_(x, LN10_L, fma_(x, LN10_H, -sh));
    int k;
    f2 y = one_plus(exp_reduce(sh, sl, &k));
    return scale(y.hi, k);
}

OCU_DM_FN float expm1f(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    if (x > 89.0f) return inf_();
    if (x < -18.0f) return -1.0f;
    if (fabs_(x) < 0x1p-25f) return x;
    int k;
    f2 u = exp_reduce(x, 0.0f, &k);
    if (k == 0) return u.hi;
    if (k >= -24 && k <= 24) { /* (2^k - 1) + 2^k u, 2^k - 1 exact */
        float p = pow2(k);
        f2 s = two_sum(p - 1.0f, p * u.hi);
        return s.hi + (s.lo + p * u.lo);
    }
    f2 y = one_plus(u);
    if (k > 24) return scale(y.hi + (y.lo - pow2(k < 126 ? -k : -126)), k);
    return scale(y.hi, k) - 1.0f;
}

OCU_DM_FN float logf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    if (x < 0.0f) return nan_();
    if (x == 0.0f) return -inf_();
    if (!isfinite_(x)) return x;
    return log_kernel(x).hi;
}

OCU_DM_FN float log2f(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    if (x < 0.0f) return nan_();
    if (x == 0.0f) return -inf_();
    if (!isfinite_(x)) return x;
    f2 l = log_kernel(x);
    float ph = l.hi * LOG2E_H;
    float pl = fma_(l.hi, LOG2E_H, -ph) + fma_(l.hi, LOG2E_L, l.lo * LOG2E_H);
    return ph + pl;
}

OCU_DM_FN float log10f(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    if (x < 0.0f) return nan_();
    if (x == 0.0f) return -inf_();
    if (!isfinite_(x)) return x;
    f2 l = log_kernel(x);
    float ph = l.hi * LOG10E_H;
    float pl = fma_(l.hi, LOG10E_H, -ph) + fma_(l.hi, LOG10E_L, l.lo * LOG10E_H);
    return ph + pl;
}

OCU_DM_FN float log1pf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    if (x < -1.0f) return nan_();
    if (x == -1.0f) return -inf_();
    if (!isfinite_(x)) return x;
    float ax = fabs_(x);
    if (ax < 0x1p-25f) return x;
    if (ax <= 0x1p-7f) {
        f2 l = log1p_kernel(x, 0.0f);
        return l.hi + l.lo;
    }
    f2 u = two_sum(1.0f, x);
    f2 l = log_kernel(u.hi);
    return l.hi + (l.lo + u.lo / u.hi);
}

OCU_DM_FN float powf(float x, float y)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (y == 0.0f || x == 1.0f) return 1.0f;
    if (isnan_(x) || isnan_(y)) return x + y;
    float ax = fabs_(x), ay = fabs_(y);
    bool yint = ay >= 0x1p24f || rint_(y) == y; /* infinities count as even */
    bool odd = ay < 0x1p24f && yint && ((int)y & 1);
    if (!isfinite_(y)) {
        if (ax == 1.0f) return 1.0f;
        return (ax > 1.0f) == (y > 0.0f) ? inf_() : 0.0f;
    }
    if (x == 0.0f) {
        if (y < 0.0f) return odd ? copysign_(inf_(), x) : inf_();
        return odd ? x : 0.0f;
    }
    if (!isfinite_(x)) {
        float r = y < 0.0f ? 0.0f : inf_();
        return (x < 0.0f && odd) ? -r : r;
    }
    if (x < 0.0f && !yint) return nan_();
    f2 l = log_kernel(ax);
    float ph = y * l.hi;
    float pl = fma_(y, l.lo, fma_(y, l.hi, -ph));
    float r;
    if (ph > 89.0f) {
        r = inf_();
    } else if (ph < -104.0f) {
        r = 0.0f;
    } else {
        int k;
        f2 v = one_plus(exp_reduce(ph, pl, &k));
        r = scale(v.hi, k);
    }
    return (x < 0.0f && odd) ? -r : r;
}

OCU_DM_FN void sincosf(float x, float *sp, float *cp)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (!isfinite_(x)) {
        *sp = *cp = x - x;
        return;
    }
    if (fabs_(x) < 0x1p-12f) {
        *sp = x;
        *cp = 1.0f;
        return;
    }
    f2 r;
    int k = fabs_(x) <= 0.785398185f ? (r = f2{x, 0.0f}, 0) : reduce_pio2(x, &r);
    float s = sin_kernel(r).hi, c = cos_kernel(r).hi;
    switch (k & 3) {
    case 0: *sp = s; *cp = c; break;
    case 1: *sp = c; *cp = -s; break;
    case 2: *sp = -s; *cp = -c; break;
    default: *sp = -c; *cp = s; break;
    }
}

OCU_DM_FN float sinf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (!isfinite_(x)) return x - x;
    if (fabs_(x) < 0x1p-12f) return x;
    f2 r;
    int k = fabs_(x) <= 0.785398185f ? (r = f2{x, 0.0f}, 0) : reduce_pio2(x, &r);
    float v = (k & 1) ? cos_kernel(r).hi : sin_kernel(r).hi;
    return (k & 2) ? -v : v;
}

OCU_DM_FN float cosf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (!isfinite_(x)) return x - x;
    if (fabs_(x) < 0x1p-12f) return 1.0f;
    f2 r;
    int k = fabs_(x) <= 0.785398185f ? (r = f2{x, 0.0f}, 0) : reduce_pio2(x, &r);
    float v = (k & 1) ? sin_kernel(r).hi : cos_kernel(r).hi;
    return ((k + 1) & 2) ? -v : v;
}

OCU_DM_FN float tanf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (!isfinite_(x)) return x - x;
    if (fabs_(x) < 0x1p-12f) return x;
    f2 r;
    int k = fabs_(x) <= 0.785398185f ? (r = f2{x, 0.0f}, 0) : reduce_pio2(x, &r);
    f2 s = sin_kernel(r), c = cos_kernel(r);
    return (k & 1) ? -div2(c, s) : div2(s, c);
}

OCU_DM_FN float atanf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    float ax = fabs_(x);
    if (ax < 0x1p-12f) return x;
    if (ax > 0x1p27f) return copysign_(PIO2_H, x);
    float r;
    if (ax <= 1.0f) {
        r = ax + atan_lo(ax, 0.0f);
    } else {
        float th = 1.0f / ax;
        float tl = fma_(-th, ax, 1.0f) * th;
        float lo = atan_lo(th, tl);
        f2 a = two_sum(PIO2_H, -th);
        r = a.hi + (a.lo + (PIO2_M - lo));
    }
    return copysign_(r, x);
}

OCU_DM_FN float atan2f(float y, float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x) || isnan_(y)) return x + y;
    float ax = fabs_(x), ay = fabs_(y);
    bool xneg = (fbits(x) >> 31) != 0;
    if (ay == 0.0f) return xneg ? copysign_(PI_H, y) : y;
    if (ax == 0.0f) return copysign_(PIO2_H, y);
    if (!isfinite_(ax)) {
        if (!isfinite_(ay)) return copysign_(xneg ? 0x1.2d97c8p+1f /* 3pi/4 */ : 0x1.921fb6p-1f /* pi/4 */, y);
        return xneg ? copysign_(PI_H, y) : copysign_(0.0f, y);
    }
    if (!isfinite_(ay)) return copysign_(PIO2_H, y);
    bool swap = ay > ax;
    float num = swap ? ax : ay, den = swap ? ay : ax;
    float th, lo = 0.0f;
    int ed = ilogb_(den);
    if (ed - ilogb_(num) > 40) {
        th = num / den; /* atan(t) = t */
    } else {
        num = scale(num, -ed); /* exact: den in [1, 2), num >= 2^-41 */
        den = scale(den, -ed);
        float inv = 1.0f / den;
        th = num * inv;
        float tl = fma_(-th, den, num) * inv;
        lo = atan_lo(th, tl);
    }
    float r;
    if (!swap) {
        if (!xneg) return copysign_(th + lo, y);
        f2 a = two_sum(PI_H, -th); /* pi - atan t */
        r = a.hi + (a.lo + (PI_L - lo));
    } else if (!xneg) {
        f2 a = two_sum(PIO2_H, -th); /* pi/2 - atan t */
        r = a.hi + (a.lo + (PIO2_M - lo));
    } else {
        f2 a = two_sum(PIO2_H, th); /* pi/2 + atan t */
        r = a.hi + (a.lo + (PIO2_M + lo));
    }
    return copysign_(r, y);
}

OCU_DM_FN float asinf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    float ax = fabs_(x);
    if (ax > 1.0f) return nan_();
    if (ax < 0x1p-12f) return x;
    float r;
    if (ax <= 0.5f) {
        float z = ax * ax;
        r = fma_(ax * z, horner(z, ASIN_P), ax);
    } else {
        /* asin x = pi/2 - 2 asin(s), s = sqrt(w), w = (1 - x)/2 exact */
        float w = (1.0f - ax) * 0.5f;
        float sh = sqrt_(w);
        float sl = w == 0.0f ? 0.0f : fma_(-sh, sh, w) / (2.0f * sh);
        float t = fma_(sh * w, horner(w, ASIN_P), sl); /* asin(s) - sh */
        f2 a = two_sum(PIO2_H, -2.0f * sh);
        r = a.hi + (a.lo + (PIO2_M - 2.0f * t));
    }
    return copysign_(r, x);
}

OCU_DM_FN float acosf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    float ax = fabs_(x);
    if (ax > 1.0f) return nan_();
    if (ax <= 0.5f) { /* pi/2 - asin x */
        float z = x * x;
        float t = (x * z) * horner(z, ASIN_P);
        f2 a = two_sum(PIO2_H, -x);
        return a.hi + (a.lo + (PIO2_M - t));
    }
    float w = (1.0f - ax) * 0.5f;
    float sh = sqrt_(w);
    float sl = w == 0.0f ? 0.0f : fma_(-sh, sh, w) / (2.0f * sh);
    float t = fma_(sh * w, horner(w, ASIN_P), sl);
    if (x > 0.0f) return fma_(2.0f, t, 2.0f * sh); /* 2 asin(s) */
    f2 a = two_sum(PI_H, -2.0f * sh);              /* pi - 2 asin(s) */
    return a.hi + (a.lo + (PI_L - 2.0f * t));
}

OCU_DM_FN float sinhf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (!isfinite_(x)) return x;
    float ax = fabs_(x);
    if (ax < 0x1p-12f) return x;
    if (ax < 1.0f) {
        float z = x * x;
        return fma_(x * z, horner(z, SINH_P), x);
    }
    if (ax > 89.5f) return copysign_(inf_(), x);
    int k;
    f2 y = one_plus(exp_reduce(ax, 0.0f, &k)); /* e^|x| = 2^k y */
    /* 2^-k e^-|x| = 2^-2k / y = q + ql */
    float c = k <= 40 ? pow2(-2 * k) : 0.0f;
    float q = c / y.hi;
    float ql = (fma_(-q, y.hi, c) - q * y.lo) / y.hi;
    f2 s = two_sum(y.hi, -q);
    return copysign_(scale(s.hi + (s.lo + (y.lo - ql)), k - 1), x);
}

OCU_DM_FN float coshf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    float ax = fabs_(x);
    if (ax < 0x1p-12f) return 1.0f;
    if (ax > 89.5f) return inf_();
    int k;
    f2 y = one_plus(exp_reduce(ax, 0.0f, &k));
    float c = k <= 40 ? pow2(-2 * k) : 0.0f;
    float q = c / y.hi;
    float ql = (fma_(-q, y.hi, c) - q * y.lo) / y.hi;
    f2 s = two_sum(y.hi, q);
    return scale(s.hi + (s.lo + (y.lo + ql)), k - 1);
}

OCU_DM_FN float tanhf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    float ax = fabs_(x);
    if (ax < 0x1p-12f) return x;
    if (ax < 0.55f) {
        float z = x * x;
        return fma_(x * z, horner(z, TANH_P), x);
    }
    if (ax > 9.1f) return copysign_(1.0f, x);
    /* 1 - 2 / (e^2|x| + 1) */
    int k;
    f2 y = one_plus(exp_reduce(2.0f * ax, 0.0f, &k));
    float p = pow2(k);
    f2 d = fast_two_sum(y.hi * p, 1.0f);
    d.lo += y.lo * p;
    float q = 2.0f / d.hi;
    float ql = (fma_(-q, d.hi, 2.0f) - q * d.lo) * (0.5f * q);
    f2 a = two_sum(1.0f, -q);
    return copysign_(a.hi + (a.lo - ql), x);
}

OCU_DM_FN float cbrtf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (!isfinite_(x) || x == 0.0f) return x + x;
    float ax = fabs_(x);
    int e0 = 0;
    if (fbits(ax) < 0x00800000u) { /* subnormal: cbrt(2^24) = 2^8 */
        ax *= 0x1p24f;
        e0 = -8;
    }
    u32 ix = fbits(ax);
    int e = (int)(ix >> 23) - 127;
    int q = (e + 384) / 3 - 128; /* floor(e / 3) */
    int rem = e - 3 * q;
    float m = ffrom((ix & 0x007fffffu) | ((u32)(127 + rem) << 23)); /* in [1, 8) */
    float t = horner(m, RCBRT_P);                                    /* ~ m^(-1/3) */
    for (int i = 0; i < 2; ++i) t = fma_(t * (1.0f / 3.0f), fma_(-m, t * t * t, 1.0f), t);
    float t2 = t * t;
    float c = m * t2; /* ~ m^(1/3) */
    float c2 = c * c;
    float c2l = fma_(c, c, -c2);
    float res = fma_(c2, c, -m) + c2l * c; /* c^3 - m */
    c = fma_(-res, t2 * (1.0f / 3.0f), c);
    return copysign_(c * pow2(q + e0), x);
}

OCU_DM_FN float hypotf(float x, float y)
{
    OCU_DMF_STRICT
    using namespace mf;
    float ax = fabs_(x), ay = fabs_(y);
    if (ax == inf_() || ay == inf_()) return inf_();
    if (isnan_(x) || isnan_(y)) return x + y;
    float a = ax > ay ? ax : ay, b = ax > ay ? ay : ax;
    if (b == 0.0f) return a;
    int ea = ilogb_(a);
    if (ea - ilogb_(b) > 26) return a + b;
    a = scale(a, -ea); /* exact: a in [1, 2), b >= 2^-27 */
    b = scale(b, -ea);
    float ah = a * a, al = fma_(a, a, -ah);
    float bh = b * b, bl = fma_(b, b, -bh);
    f2 s = two_sum(ah, bh);
    float sl = s.lo + (al + bl);
    float h = sqrt_(s.hi);
    h = fma_(fma_(-h, h, s.hi) + sl, 0.5f / h, h);
    return scale(h, ea);
}

OCU_DM_FN float erff(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    float ax = fabs_(x);
    if (ax <= 0.875f) { /* x + x g(x^2) */
        float z = x * x;
        float zl = fma_(x, x, -z);
        float g = fma_(zl, ERF_G[1], horner(z, ERF_G));
        return fma_(x, g, x);
    }
    if (ax >= 3.92f) return copysign_(1.0f, x);
    int k;
    f2 e = erfc_kernel(ax, &k);
    float p = pow2(k);
    f2 s = two_sum(1.0f, -e.hi * p);
    return copysign_(s.hi + (s.lo - e.lo * p), x);
}

OCU_DM_FN float erfcf(float x)
{
    OCU_DMF_STRICT
    using namespace mf;
    if (isnan_(x)) return x + x;
    float ax = fabs_(x);
    if (ax < 0.5f) { /* (1 - x) - x g(x^2) */
        float z = x * x;
        float zl = fma_(x, x, -z);
        float g = fma_(zl, ERF_G[1], horner(z, ERF_G));
        f2 s = two_sum(1.0f, -x);
        return s.hi + (s.lo - x * g);
    }
    if (x > 10.1f) return 0.0f;
    if (x < -10.1f) return 2.0f;
    int k;
    f2 e = erfc_kernel(ax, &k);
    if (x > 0.0f) return scale(e.hi + e.lo, k);
    float eh = scale(e.hi, k), el = scale(e.lo, k); /* 2 - erfc|x| */
    f2 s = two_sum(2.0f, -eh);
    return s.hi + (s.lo - el);
}

} // namespace ocu_dm

#if defined(OCU_DMF_GCC_POP)
#pragma GCC pop_options
#undef OCU_DMF_GCC_POP
#endif

#endif /* OCU_DEVICE_MATHF_H */

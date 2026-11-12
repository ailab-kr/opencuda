/*
 * ocu_device_math.h -- math functions for device code, built only from
 * IEEE operations every target has (+ - * /, fma, sqrt, rounding to
 * integer, bit casts). Used where no vendor device library may be used
 * (the NVPTX path). The same code compiles for the host, which is how its
 * accuracy is measured against the host C library (tests/runtime).
 *
 * Methods: argument reduction plus polynomial or rational approximation, in
 * the classic public-domain style (the polynomial coefficients are those
 * published with the freely redistributable fdlibm, Sun Microsystems 1993:
 * "Permission to use, copy, modify, and distribute this software is freely
 * granted, provided that this notice is preserved."). Where an
 * approximation alone would lose accuracy, sums are carried in two doubles
 * (pow, log2, log10, exp2, exp10).
 *
 * float functions (ocu_device_mathf.h) use float arithmetic only, except
 * fmodf (exact through double).
 *
 * Known limit: sin, cos and tan reduce their argument with a three-part
 * pi/2 (accurate for |x| < 2^20 * pi/2); larger arguments lose accuracy.
 *
 * C++ only. OCU_DM_FN marks every function (e.g. `static inline __device__`).
 */
#ifndef OCU_DEVICE_MATH_H
#define OCU_DEVICE_MATH_H

#include <cstdint>

#include "ocu_device_math_tables.h"

#ifndef OCU_DM_FN
#define OCU_DM_FN static inline
#endif

namespace ocu_dm {

/* ---- bits ----------------------------------------------------------------- */

OCU_DM_FN std::uint64_t bits(double x) { return __builtin_bit_cast(std::uint64_t, x); }
OCU_DM_FN double from_bits(std::uint64_t b) { return __builtin_bit_cast(double, b); }
OCU_DM_FN std::uint32_t bitsf(float x) { return __builtin_bit_cast(std::uint32_t, x); }
OCU_DM_FN float from_bitsf(std::uint32_t b) { return __builtin_bit_cast(float, b); }

OCU_DM_FN bool isnan(double x) { return (bits(x) & 0x7fffffffffffffffull) > 0x7ff0000000000000ull; }
OCU_DM_FN bool isinf(double x) { return (bits(x) & 0x7fffffffffffffffull) == 0x7ff0000000000000ull; }
OCU_DM_FN bool isfinite(double x) { return (bits(x) & 0x7ff0000000000000ull) != 0x7ff0000000000000ull; }
OCU_DM_FN bool signbit(double x) { return (bits(x) >> 63) != 0; }
OCU_DM_FN double fabs(double x) { return from_bits(bits(x) & 0x7fffffffffffffffull); }
OCU_DM_FN double copysign(double x, double y)
{
    return from_bits((bits(x) & 0x7fffffffffffffffull) | (bits(y) & 0x8000000000000000ull));
}
OCU_DM_FN double inf() { return from_bits(0x7ff0000000000000ull); }
OCU_DM_FN double qnan() { return from_bits(0x7ff8000000000000ull); }
OCU_DM_FN double fma(double a, double b, double c) { return __builtin_fma(a, b, c); }
OCU_DM_FN double sqrt(double x) { return __builtin_sqrt(x); }
OCU_DM_FN double floor(double x) { return __builtin_floor(x); }
OCU_DM_FN double ceil(double x) { return __builtin_ceil(x); }
OCU_DM_FN double trunc(double x) { return __builtin_trunc(x); }
OCU_DM_FN double rint(double x) { return __builtin_rint(x); }
OCU_DM_FN double round(double x) { return __builtin_round(x); }

/* x * 2^n, also for results in the subnormal range and for large n. */
OCU_DM_FN double ldexp(double x, int n)
{
    if (n > 1023) {
        x *= 0x1p1023;
        n -= 1023;
        if (n > 1023) {
            x *= 0x1p1023;
            n -= 1023;
            if (n > 1023) n = 1023;
        }
    } else if (n < -1022) {
        x *= 0x1p-1022 * 0x1p53;
        n += 1022 - 53;
        if (n < -1022) {
            x *= 0x1p-1022 * 0x1p53;
            n += 1022 - 53;
            if (n < -1022) n = -1022;
        }
    }
    return x * from_bits((std::uint64_t)(0x3ff + n) << 52);
}

/* x = m * 2^e with 0.5 <= |m| < 1 (finite, non-zero x). */
OCU_DM_FN double frexp(double x, int *e)
{
    std::uint64_t b = bits(x);
    int ex = (int)((b >> 52) & 0x7ff);
    if (ex == 0) { /* subnormal */
        x *= 0x1p54;
        b = bits(x);
        ex = (int)((b >> 52) & 0x7ff) - 54;
    }
    *e = ex - 1022;
    return from_bits((b & 0x800fffffffffffffull) | 0x3fe0000000000000ull);
}

/* ---- two-double helpers ----------------------------------------------------- */

struct dd {
    double hi, lo;
};
OCU_DM_FN dd two_sum(double a, double b)
{
    double s = a + b;
    double bb = s - a;
    return dd{s, (a - (s - bb)) + (b - bb)};
}
OCU_DM_FN dd fast_two_sum(double a, double b) /* |a| >= |b| */
{
    double s = a + b;
    return dd{s, b - (s - a)};
}
OCU_DM_FN dd two_prod(double a, double b)
{
    double p = a * b;
    return dd{p, fma(a, b, -p)};
}

/* ---- exp ---------------------------------------------------------------------- */

constexpr double LN2_HI = 6.93147180369123816490e-01; /* trailing zeros: k*LN2_HI exact */
constexpr double LN2_LO = 1.90821492927058770002e-10;
constexpr double INV_LN2 = 1.44269504088896338700e+00;

/* e^(r) - 1 core for |r| <= ln2/2, as 1 + r + ... with the fdlibm rational form. */
OCU_DM_FN double exp_reduced(double hi, double lo, int k)
{
    const double P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03,
                 P3 = 6.61375632143793436117e-05, P4 = -1.65339022054652515390e-06,
                 P5 = 4.13813679705723846039e-08;
    double r = hi - lo;
    double t = r * r;
    double c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    double y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi);
    return ldexp(y, k);
}

OCU_DM_FN double exp(double x)
{
    if (isnan(x)) return x;
    if (x > 7.09782712893383973096e+02) return inf();
    if (x < -7.45133219101941108420e+02) return 0.0;
    double kd = rint(x * INV_LN2);
    int k = (int)kd;
    double hi = fma(-kd, LN2_HI, x);
    double lo = kd * LN2_LO;
    return exp_reduced(hi, lo, k);
}

/* exp(hi + lo) for a two-double argument (|lo| tiny relative to hi). */
OCU_DM_FN double exp_dd(double hi, double lo)
{
    if (isnan(hi)) return hi;
    if (hi > 7.09782712893383973096e+02) return inf();
    if (hi < -7.45133219101941108420e+02) return 0.0;
    double kd = rint(hi * INV_LN2);
    int k = (int)kd;
    double rh = fma(-kd, LN2_HI, hi);
    double rl = kd * LN2_LO - lo;
    return exp_reduced(rh, rl, k);
}

OCU_DM_FN double expm1_series(double x) /* |x| <= 1 */
{
    {
        double p = 1.0 / 2432902008176640000.0;   /* 1/20! */
        p = fma(p, x, 1.0 / 121645100408832000.0); /* 1/19! */
        p = fma(p, x, 1.0 / 6402373705728000.0);
        p = fma(p, x, 1.0 / 355687428096000.0);
        p = fma(p, x, 1.0 / 20922789888000.0);
        p = fma(p, x, 1.0 / 1307674368000.0);
        p = fma(p, x, 1.0 / 87178291200.0);
        p = fma(p, x, 1.0 / 6227020800.0);         /* 1/13! */
        p = fma(p, x, 1.0 / 479001600.0);         /* 1/12! */
        p = fma(p, x, 1.0 / 39916800.0);
        p = fma(p, x, 1.0 / 3628800.0);
        p = fma(p, x, 1.0 / 362880.0);
        p = fma(p, x, 1.0 / 40320.0);
        p = fma(p, x, 1.0 / 5040.0);
        p = fma(p, x, 1.0 / 720.0);
        p = fma(p, x, 1.0 / 120.0);
        p = fma(p, x, 1.0 / 24.0);
        p = fma(p, x, 1.0 / 6.0);
        p = fma(p, x, 0.5);
        return fma(p * x, x, x);
    }
}

OCU_DM_FN double expm1(double x)
{
    if (isnan(x)) return x;
    if (fabs(x) <= 1.0) return expm1_series(x);
    if (x > 709.0) return exp(x);
    if (x < -40.0) return -1.0;
    double kd = rint(x * INV_LN2);
    int k = (int)kd;
    double hi = fma(-kd, LN2_HI, x);
    double r = hi - kd * LN2_LO;
    double c = (hi - r) - kd * LN2_LO; /* the part of the reduction r lost */
    double em = expm1_series(r);
    em = fma(em + 1.0, c, em);        /* expm1(r + c) to first order in c */
    if (k > 1020) return ldexp(em + 1.0, k);
    double tk = ldexp(1.0, k);
    return fma(tk, em, tk - 1.0);     /* 2^k - 1 is exact for |k| < 53 */
}

/* ---- log ------------------------------------------------------------------------ */

/* log1p(r) - r + r^2/2 for small |r|: r^3/3 - r^4/4 + ... (|r| < 2^-6). */
OCU_DM_FN double log1p_tail(double z)
{
    double p = -1.0 / 12.0;
    p = fma(p, z, 1.0 / 11.0);
    p = fma(p, z, -1.0 / 10.0);
    p = fma(p, z, 1.0 / 9.0);
    p = fma(p, z, -1.0 / 8.0);
    p = fma(p, z, 1.0 / 7.0);
    p = fma(p, z, -1.0 / 6.0);
    p = fma(p, z, 1.0 / 5.0);
    p = fma(p, z, -1.0 / 4.0);
    p = fma(p, z, 1.0 / 3.0);
    return p * z * z * z;
}

/* log(1 + r) for r = rh + rl, |r| < 2^-6, as two doubles. */
OCU_DM_FN dd log1p_small(double rh, double rl)
{
    dd h = two_prod(rh, 0.5 * rh);           /* rh^2 / 2 exactly */
    double hl = h.lo + rh * rl;              /* + the cross term */
    dd s = two_sum(rh, -h.hi);
    s.lo += (rl - hl) + log1p_tail(rh);
    return fast_two_sum(s.hi, s.lo);
}

OCU_DM_FN dd dd_add(dd a, dd b)
{
    dd s = two_sum(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    return fast_two_sum(s.hi, s.lo);
}

/*
 * log(x) as two doubles, about 2^-100 relative (x finite, > 0).
 * x = 2^k m, m in [1, 2); with a table value c close to 1/m,
 * log x = k log 2 - log c + log(1 + (m c - 1)), and m c - 1 is exact.
 * Near 1 the series runs on x - 1 directly (no cancellation).
 */
OCU_DM_FN dd log_dd(double x)
{
    if (x > 0.9921875 && x < 1.0078125) return log1p_small(x - 1.0, 0.0); /* x - 1 exact */
    int k;
    double m = frexp(x, &k);
    m *= 2.0;
    k -= 1;
    int i = (int)((bits(m) >> 45) & 127);
    const log_entry &t = LOG_TABLE[i];
    dd p = two_prod(m, t.c);
    double rh = p.hi - 1.0;                  /* exact: p.hi is within 2^-7 of 1 */
    dd r = fast_two_sum(rh, p.lo);
    dd l = log1p_small(r.hi, r.lo);
    dd kl = two_prod((double)k, LN2_DD_HI);
    kl.lo += (double)k * LN2_DD_LO;
    dd sum = dd_add(kl, dd{t.neg_log_hi, t.neg_log_lo});
    return dd_add(sum, l);
}

OCU_DM_FN double log(double x)
{
    if (isnan(x)) return x;
    if (x < 0.0) return qnan();
    if (x == 0.0) return -inf();
    if (isinf(x)) return x;
    dd l = log_dd(x);
    return l.hi + l.lo;
}

constexpr double INV_LN2_HI = 1.44269504088896338700e+00;
constexpr double INV_LN2_LO = 2.03552737409310450780e-17;
constexpr double INV_LN10_HI = 4.34294481903251816668e-01;
constexpr double INV_LN10_LO = 1.09831965021676510017e-17;

OCU_DM_FN double log_scaled(double x, double ch, double cl)
{
    if (isnan(x)) return x;
    if (x < 0.0) return qnan();
    if (x == 0.0) return -inf();
    if (isinf(x)) return x;
    dd l = log_dd(x);
    dd p = two_prod(l.hi, ch);
    p.lo += l.hi * cl + l.lo * ch;
    return p.hi + p.lo;
}

OCU_DM_FN double log2(double x)
{
    /* exact for powers of two */
    if (x > 0.0 && isfinite(x) && (bits(x) & 0x000fffffffffffffull) == 0) {
        int e;
        frexp(x, &e);
        return (double)(e - 1);
    }
    return log_scaled(x, INV_LN2_HI, INV_LN2_LO);
}
OCU_DM_FN double log10(double x) { return log_scaled(x, INV_LN10_HI, INV_LN10_LO); }

OCU_DM_FN double log1p(double x)
{
    if (isnan(x)) return x;
    if (x < -1.0) return qnan();
    if (x == -1.0) return -inf();
    if (isinf(x)) return x;
    if (fabs(x) < 0x1p-54) return x;
    double u = 1.0 + x;
    /* log(u) corrected for the rounding of 1 + x */
    double c = (u == 1.0) ? 0.0 : (x - (u - 1.0)) / u;
    return log(u) + c;
}

/* ---- pow ----------------------------------------------------------------------- */

OCU_DM_FN bool is_integer(double y) { return isfinite(y) && rint(y) == y; }
OCU_DM_FN bool is_odd_integer(double y)
{
    if (!is_integer(y) || fabs(y) >= 0x1p53) return false;
    double h = y * 0.5;
    return rint(h) != h;
}

OCU_DM_FN double pow(double x, double y)
{
    if (y == 0.0) return 1.0;
    if (x == 1.0) return 1.0;
    if (isnan(x) || isnan(y)) return x + y;
    double ax = fabs(x);
    if (isinf(y)) {
        if (ax == 1.0) return 1.0;
        return (ax > 1.0) == (y > 0.0) ? inf() : 0.0;
    }
    bool odd = is_odd_integer(y);
    if (x == 0.0) {
        if (y < 0.0) return odd ? copysign(inf(), x) : inf();
        return odd ? x : 0.0;
    }
    if (isinf(x)) {
        double r = y < 0.0 ? 0.0 : inf();
        return (x < 0.0 && odd) ? -r : r;
    }
    if (x < 0.0 && !is_integer(y)) return qnan();
    dd l = log_dd(ax);
    dd p = two_prod(y, l.hi);
    p.lo = fma(y, l.lo, p.lo);
    dd z = fast_two_sum(p.hi, p.lo);
    double r = exp_dd(z.hi, z.lo);
    return (x < 0.0 && odd) ? -r : r;
}

OCU_DM_FN double exp2(double x)
{
    if (isnan(x)) return x;
    if (x >= 1024.0) return inf();
    if (x < -1075.0) return 0.0;
    if (is_integer(x)) return ldexp(1.0, (int)x);
    dd p = two_prod(x, LN2_HI + LN2_LO);
    p.lo = fma(x, LN2_LO - ((LN2_HI + LN2_LO) - LN2_HI), p.lo);
    return exp_dd(p.hi, p.lo);
}

OCU_DM_FN double exp10(double x)
{
    const double LN10_HI = 2.30258509299404590109e+00, LN10_LO = -2.17075622338224935e-16;
    if (isnan(x)) return x;
    if (x > 308.3) return inf();
    if (x < -324.0) return 0.0;
    dd p = two_prod(x, LN10_HI);
    p.lo = fma(x, LN10_LO, p.lo);
    return exp_dd(p.hi, p.lo);
}

/* ---- trigonometry ------------------------------------------------------------- */

/* sin/cos on |x| <= pi/4 for x = xh + xl. */
OCU_DM_FN double kernel_sin(double x, double y)
{
    const double S1 = -1.66666666666666324348e-01, S2 = 8.33333333332248946124e-03,
                 S3 = -1.98412698298579493134e-04, S4 = 2.75573137070700676789e-06,
                 S5 = -2.50507602534068634195e-08, S6 = 1.58969099521155010221e-10;
    double z = x * x, v = z * x;
    double r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    return x - ((z * (0.5 * y - v * r) - y) - v * S1);
}

OCU_DM_FN double kernel_cos(double x, double y)
{
    const double C1 = 4.16666666666666019037e-02, C2 = -1.38888888888741095749e-03,
                 C3 = 2.48015872894767294178e-05, C4 = -2.75573143513906633035e-07,
                 C5 = 2.08757232129817482790e-09, C6 = -1.13596475577881948265e-11;
    double z = x * x, w = z * z;
    double r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
    double hz = 0.5 * z;
    double ww = 1.0 - hz;
    return ww + (((1.0 - ww) - hz) + (z * r - x * y));
}

/* x = k pi/2 + (rh + rl); returns k mod 4. */
OCU_DM_FN int reduce_pio2(double x, double *rh, double *rl)
{
    const double INV_PIO2 = 6.36619772367581382433e-01;
    const double P1 = 1.57079632673412561417e+00, P2 = 6.07710050630396597660e-11,
                 P3 = 2.02226624871116645580e-21, P3T = 8.47842766036889956997e-32;
    double kd = rint(x * INV_PIO2);
    double r = fma(-kd, P1, x);       /* exact while |k| < 2^20 */
    double w = kd * P2;
    double t = r;
    r = t - w;
    double e2 = (t - r) - w;          /* error of the subtraction */
    double w3 = kd * P3;
    double t2 = r;
    r = t2 - w3;
    e2 += (t2 - r) - w3;
    e2 -= kd * P3T;
    dd s = fast_two_sum(r, e2);
    *rh = s.hi;
    *rl = s.lo;
    long long k = (long long)kd;
    return (int)(k & 3);
}

OCU_DM_FN double sin(double x)
{
    if (!isfinite(x)) return qnan();
    if (fabs(x) < 0.785398163397448279) return x == 0.0 ? x : kernel_sin(x, 0.0);
    double rh, rl;
    switch (reduce_pio2(x, &rh, &rl)) {
    case 0: return kernel_sin(rh, rl);
    case 1: return kernel_cos(rh, rl);
    case 2: return -kernel_sin(rh, rl);
    default: return -kernel_cos(rh, rl);
    }
}

OCU_DM_FN double cos(double x)
{
    if (!isfinite(x)) return qnan();
    if (fabs(x) < 0.785398163397448279) return kernel_cos(x, 0.0);
    double rh, rl;
    switch (reduce_pio2(x, &rh, &rl)) {
    case 0: return kernel_cos(rh, rl);
    case 1: return -kernel_sin(rh, rl);
    case 2: return -kernel_cos(rh, rl);
    default: return kernel_sin(rh, rl);
    }
}

OCU_DM_FN double tan(double x)
{
    if (!isfinite(x)) return qnan();
    if (x == 0.0) return x;
    double rh, rl;
    int q = fabs(x) < 0.785398163397448279 ? (rh = x, rl = 0.0, 0) : reduce_pio2(x, &rh, &rl);
    double s = kernel_sin(rh, rl), c = kernel_cos(rh, rl);
    return (q & 1) ? -c / s : s / c;
}

OCU_DM_FN double atan(double x)
{
    const double atanhi[] = {4.63647609000806093515e-01, 7.85398163397448278999e-01,
                             9.82793723247329054082e-01, 1.57079632679489655800e+00};
    const double atanlo[] = {2.26987774529616870924e-17, 3.06161699786838301793e-17,
                             1.39033110312309984516e-17, 6.12323399573676603587e-17};
    const double aT[] = {3.33333333333329318027e-01, -1.99999999998764832476e-01,
                         1.42857142725034663711e-01, -1.11111104054623557880e-01,
                         9.09088713343650656196e-02, -7.69187620504482999495e-02,
                         6.66107313738753120669e-02, -5.83357013379057348645e-02,
                         4.97687799461593236017e-02, -3.65315727442169155270e-02,
                         1.62858201153657823623e-02};
    if (isnan(x)) return x;
    double ax = fabs(x);
    if (ax >= 0x1p66) return copysign(atanhi[3] + atanlo[3], x);
    int id;
    if (ax < 0.4375) {
        if (ax < 0x1p-27) return x;
        id = -1;
    } else if (ax < 1.1875) {
        if (ax < 0.6875) { id = 0; ax = (2.0 * ax - 1.0) / (2.0 + ax); }
        else { id = 1; ax = (ax - 1.0) / (ax + 1.0); }
    } else if (ax < 2.4375) {
        id = 2; ax = (ax - 1.5) / (1.0 + 1.5 * ax);
    } else {
        id = 3; ax = -1.0 / ax;
    }
    double z = ax * ax, w = z * z;
    double s1 = z * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))));
    double s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))));
    if (id < 0) return x - x * (s1 + s2);
    double r = atanhi[id] - ((ax * (s1 + s2) - atanlo[id]) - ax);
    return copysign(r, x);
}

constexpr double PI = 3.14159265358979311600e+00;
constexpr double PI_LO = 1.22464679914735317720e-16;
constexpr double PIO2 = 1.57079632679489655800e+00;
constexpr double PIO2_LO = 6.12323399573676603587e-17;

OCU_DM_FN double atan2(double y, double x)
{
    if (isnan(x) || isnan(y)) return x + y;
    if (y == 0.0) {
        if (signbit(x)) return copysign(PI, y); /* x < 0 or x == -0 */
        return y;
    }
    if (x == 0.0) return copysign(PIO2, y);
    if (isinf(x)) {
        if (isinf(y)) return copysign(signbit(x) ? 2.35619449019234492885e+00 /* 3pi/4 */
                                                 : 7.85398163397448278999e-01 /* pi/4 */, y);
        return signbit(x) ? copysign(PI, y) : copysign(0.0, y);
    }
    if (isinf(y)) return copysign(PIO2, y);
    double a = atan(fabs(y / x));
    if (signbit(x)) a = (PI - a) + PI_LO;
    return copysign(a, y);
}

OCU_DM_FN double asin(double x)
{
    if (isnan(x)) return x;
    double ax = fabs(x);
    if (ax > 1.0) return qnan();
    if (ax == 1.0) return copysign(PIO2, x);
    if (ax < 0x1p-26) return x;
    /* 1 - x^2 = (1 - |x|)(1 + |x|): both factors exact near 1 */
    double w = ax < 0.5 ? fma(-ax, ax, 1.0) : (1.0 - ax) * (1.0 + ax);
    return copysign(atan(ax / sqrt(w)), x);
}

OCU_DM_FN double acos(double x)
{
    if (isnan(x)) return x;
    if (fabs(x) > 1.0) return qnan();
    if (x == 1.0) return 0.0;
    if (x == -1.0) return PI + PI_LO;
    /* acos(x) = 2 atan(sqrt((1 - x) / (1 + x))) */
    return 2.0 * atan(sqrt((1.0 - x) / (1.0 + x)));
}

/* ---- hyperbolic ------------------------------------------------------------------ */

OCU_DM_FN double sinh(double x)
{
    if (!isfinite(x)) return x;
    double ax = fabs(x);
    if (ax < 0.5) {
        double z = x * x;
        double p = 1.0 / 355687428096000.0;          /* 1/17! */
        p = fma(p, z, 1.0 / 1307674368000.0);        /* 1/15! */
        p = fma(p, z, 1.0 / 6227020800.0);
        p = fma(p, z, 1.0 / 39916800.0);
        p = fma(p, z, 1.0 / 362880.0);
        p = fma(p, z, 1.0 / 5040.0);
        p = fma(p, z, 1.0 / 120.0);
        p = fma(p, z, 1.0 / 6.0);
        return fma(p * z, x, x);
    }
    if (ax < 22.0) {
        double t = expm1(ax);
        return copysign(0.5 * (t + t / (t + 1.0)), x);
    }
    if (ax < 709.78) return copysign(0.5 * exp(ax), x);
    double w = exp(0.5 * ax);
    return copysign(0.5 * w * w, x);
}

OCU_DM_FN double cosh(double x)
{
    if (isnan(x)) return x;
    double ax = fabs(x);
    if (isinf(ax)) return ax;
    if (ax < 22.0) {
        double t = exp(ax);
        return 0.5 * t + 0.5 / t;
    }
    if (ax < 709.78) return 0.5 * exp(ax);
    double w = exp(0.5 * ax);
    return 0.5 * w * w;
}

OCU_DM_FN double tanh(double x)
{
    if (isnan(x)) return x;
    double ax = fabs(x);
    if (ax > 22.0) return copysign(1.0, x);
    if (ax < 0x1p-28) return x;
    if (ax < 0.55) {
        double z = x * x;
        double sp = 1.0 / 355687428096000.0;           /* sinh: odd series */
        sp = fma(sp, z, 1.0 / 1307674368000.0);
        sp = fma(sp, z, 1.0 / 6227020800.0);
        sp = fma(sp, z, 1.0 / 39916800.0);
        sp = fma(sp, z, 1.0 / 362880.0);
        sp = fma(sp, z, 1.0 / 5040.0);
        sp = fma(sp, z, 1.0 / 120.0);
        sp = fma(sp, z, 1.0 / 6.0);
        double cp = 1.0 / 6402373705728000.0;          /* cosh - 1: even series */
        cp = fma(cp, z, 1.0 / 20922789888000.0);
        cp = fma(cp, z, 1.0 / 87178291200.0);
        cp = fma(cp, z, 1.0 / 479001600.0);
        cp = fma(cp, z, 1.0 / 3628800.0);
        cp = fma(cp, z, 1.0 / 40320.0);
        cp = fma(cp, z, 1.0 / 720.0);
        cp = fma(cp, z, 1.0 / 24.0);
        cp = fma(cp, z, 0.5);
        /* tanh = (x + x z sp) / (1 + z cp) = x + x z (sp - cp) / (1 + z cp) */
        double den = fma(z, cp, 1.0);
        return fma(x * z, (sp - cp) / den, x);
    }
    double t = expm1(2.0 * ax);
    double r = t / (t + 2.0);
    return copysign(r, x);
}

/* ---- the rest ------------------------------------------------------------------------ */

OCU_DM_FN double cbrt(double x)
{
    if (!isfinite(x) || x == 0.0) return x;
    double ax = fabs(x);
    int e;
    double m = frexp(ax, &e);          /* ax = m 2^e */
    int r = e % 3;
    if (r < 0) r += 3;
    m = ldexp(m, r);                   /* m in [0.5, 4) */
    e -= r;
    double y = 0.6 + 0.4 * m;          /* rough start */
    for (int i = 0; i < 6; ++i) y = y - (y * y * y - m) / (3.0 * y * y);
    /* one correction with the exact residual */
    double y2 = y * y;
    dd c = two_prod(y2, y);
    double res = (c.hi - m) + c.lo;
    y = y - res / (3.0 * y2);
    return copysign(ldexp(y, e / 3), x);
}

OCU_DM_FN double hypot(double x, double y)
{
    if (isinf(x) || isinf(y)) return inf();
    if (isnan(x) || isnan(y)) return qnan();
    double a = fabs(x), b = fabs(y);
    if (a < b) { double t = a; a = b; b = t; }
    if (a == 0.0) return 0.0;
    int e;
    frexp(a, &e);
    double sa = ldexp(a, -e), sb = ldexp(b, -e);
    return ldexp(sqrt(fma(sa, sa, sb * sb)), e);
}

/* Exact remainder x - n y with n = trunc(x / y). */
OCU_DM_FN double fmod(double x, double y)
{
    if (isnan(x) || isnan(y) || isinf(x) || y == 0.0) return qnan();
    if (isinf(y)) return x;
    double ax = fabs(x), ay = fabs(y);
    if (ax < ay) return x;
    int ex, ey;
    frexp(ax, &ex);
    frexp(ay, &ey);
    double r = ax;
    for (int k = ex - ey; k >= 0; --k) {
        double s = ldexp(ay, k);   /* exact */
        if (r >= s) r -= s;         /* exact: s <= r < 2s */
    }
    return copysign(r, x);
}

OCU_DM_FN double fmin(double a, double b) { return isnan(a) ? b : isnan(b) ? a : (a < b || (a == b && signbit(a))) ? a : b; }
OCU_DM_FN double fmax(double a, double b) { return isnan(a) ? b : isnan(b) ? a : (a > b || (a == b && !signbit(a))) ? a : b; }

/* fmod is exact, so the double version rounded is exact too. */
OCU_DM_FN float fmodf(float x, float y) { return (float)fmod((double)x, (double)y); }

} // namespace ocu_dm

/* The float functions, from float operations only. */
#include "ocu_device_mathf.h"

#endif /* OCU_DEVICE_MATH_H */

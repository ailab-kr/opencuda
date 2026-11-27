/*
 * ocu_device_math_ext.h -- more of the CUDA Math API for device code, on top
 * of ocu_device_math.h / ocu_device_mathf.h (same rules: IEEE operations,
 * fma, sqrt, rounding to integer and bit casts only; the same code compiles
 * for the host, where its accuracy is measured).
 *
 *   exact results    modf, logb, ilogb, scalbn, scalbln, nextafter, remainder,
 *                    remquo, lrint / llrint / llround (and f), nan, the
 *                    rounding-mode arithmetic (__fadd_rd ... __dsqrt_ru), the
 *                    64-bit integer conversions (__double2ll_rd ...,
 *                    __ll2float_ru ...): the IEEE result in the named rounding
 *   sinpi, cospi     argument reduced exactly (x = k/2 + r, |r| <= 1/4), pi r
 *                    formed in two doubles, fdlibm's sin / cos kernels
 *   acosh, asinh,    the fdlibm formulas over log / log1p / sqrt
 *   atanh
 *   rcbrt, rhypot,   from cbrt / hypot / sqrt; the norms scaled by a power of
 *   norm*, rnorm*    two (exact) so that no sum of squares overflows
 *   float functions  the double function, rounded once (inputs of a float
 *                    function are exact in double; the double result is far
 *                    inside half a float ulp)
 *
 * Methods: IEEE 754-2019; Sterbenz's lemma for the exact subtractions;
 * error-free transformations (Knuth's two-sum, fma residuals) for the
 * rounding modes; fdlibm (Sun Microsystems 1993, freely redistributable) for
 * the hyperbolic inverses and the sin / cos kernels. C++ only.
 */
#ifndef OCU_DEVICE_MATH_EXT_H
#define OCU_DEVICE_MATH_EXT_H

#include "ocu_device_math.h"
#include "ocu_device_mathf.h"

namespace ocu_dm {

/* ---- bits, neighbours ---------------------------------------------------------------- */

constexpr double DMAX = 1.7976931348623157e308;
constexpr float FMAX = 3.40282347e38f;

OCU_DM_FN bool isnanf_(float x) { return (bitsf(x) & 0x7fffffffu) > 0x7f800000u; }
OCU_DM_FN bool isinff_(float x) { return (bitsf(x) & 0x7fffffffu) == 0x7f800000u; }

/* The next double after x toward +inf (up) or -inf; x not NaN. */
OCU_DM_FN double next_up(double x)
{
    if (x == inf()) return x;
    if (x == 0.0) return from_bits(1ull); /* +-0 -> smallest subnormal */
    std::uint64_t b = bits(x);
    return from_bits(x > 0.0 ? b + 1 : b - 1);
}
OCU_DM_FN double next_down(double x) { return -next_up(-x); }
OCU_DM_FN float next_upf(float x)
{
    if (x == (float)inf()) return x;
    if (x == 0.0f) return from_bitsf(1u);
    std::uint32_t b = bitsf(x);
    return from_bitsf(x > 0.0f ? b + 1 : b - 1);
}
OCU_DM_FN float next_downf(float x) { return -next_upf(-x); }

OCU_DM_FN double nextafter(double x, double y)
{
    if (isnan(x) || isnan(y)) return x + y;
    if (x == y) return y; /* also gives y's sign of zero */
    return y > x ? next_up(x) : next_down(x);
}
OCU_DM_FN float nextafterf(float x, float y)
{
    if (isnanf_(x) || isnanf_(y)) return x + y;
    if (x == y) return y;
    return y > x ? next_upf(x) : next_downf(x);
}

/* ---- exponent, scaling, parts -------------------------------------------------------- */

OCU_DM_FN int ilogb(double x)
{
    if (isnan(x)) return (-2147483647 - 1); /* documented: INT_MIN for NaN and 0, INT_MAX for inf */
    if (x == 0.0) return (-2147483647 - 1);
    if (isinf(x)) return 2147483647;
    int e;
    (void)frexp(x, &e);
    return e - 1;
}
OCU_DM_FN int ilogbf(float x) { return ilogb((double)x); }
OCU_DM_FN double logb(double x)
{
    if (isnan(x)) return x;
    if (x == 0.0) return -inf();
    if (isinf(x)) return inf();
    return (double)ilogb(x);
}
OCU_DM_FN float logbf(float x) { return (float)logb((double)x); }
OCU_DM_FN double scalbn(double x, int n) { return ldexp(x, n); }
OCU_DM_FN double scalbln(double x, long n)
{
    /* beyond +-2200 every finite non-zero x overflows or underflows alike */
    int m = n > 4000 ? 4000 : (n < -4000 ? -4000 : (int)n);
    return ldexp(x, m);
}
OCU_DM_FN float scalbnf(float x, int n) { return (float)ldexp((double)x, n < -400 ? -400 : (n > 400 ? 400 : n)); }
OCU_DM_FN float scalblnf(float x, long n) { return scalbnf(x, n < -400 ? -400 : (n > 400 ? 400 : (int)n)); }
OCU_DM_FN double modf(double x, double *ip)
{
    if (isnan(x)) { *ip = x; return x; }
    double t = trunc(x);
    *ip = t;
    if (isinf(x)) return copysign(0.0, x);
    return copysign(x - t, x); /* exact; the sign of a zero fraction is x's */
}
OCU_DM_FN float modff(float x, float *ip)
{
    double t;
    float f = (float)modf((double)x, &t);
    *ip = (float)t;
    return f;
}
OCU_DM_FN double nan(const char *) { return qnan(); }
OCU_DM_FN float nanf(const char *) { return (float)qnan(); }

/* ---- remainder ------------------------------------------------------------------------ */

/* x REM y (IEEE: x - n y, n the integer nearest x / y, even on a tie) and
   the low three bits of n with the sign of x / y. Exact. */
OCU_DM_FN double remquo(double x, double y, int *quo)
{
    *quo = 0;
    if (isnan(x) || isnan(y)) return x + y;
    if (isinf(x) || y == 0.0) return qnan();
    const bool neg = signbit(x);
    const int qsign = signbit(x) != signbit(y) ? -1 : 1;
    double ax = fabs(x), ay = fabs(y);
    if (isinf(ay)) { return x; }
    double scale = 1.0;
    if (ay > DMAX / 8) {
        if (ax >= ay) { ax *= 0.125; ay *= 0.125; scale = 8.0; } /* both large normals: exact */
    } else if (ax >= 8 * ay) {
        ax = fmod(ax, 8 * ay); /* exact; the quotient changes by a multiple of 8 */
    }
    int q = 0;
    /* ax in [0, 8 ay): Sterbenz makes each subtraction exact */
    if (ax >= 4 * ay) { ax -= 4 * ay; q += 4; }
    if (ax >= 2 * ay) { ax -= 2 * ay; q += 2; }
    if (ax >= ay) { ax -= ay; q += 1; }
    /* ax in [0, ay): round the quotient to nearest, ties to even */
    double rest = ay - ax; /* exact when ax > ay / 2; otherwise >= ax either way */
    if (ax > rest || (ax == rest && (q & 1))) { ax = -rest; q += 1; }
    ax *= scale;
    *quo = qsign * (q & 7);
    double r = neg ? -ax : ax;
    return r == 0.0 ? copysign(0.0, x) : r;
}
OCU_DM_FN double remainder(double x, double y)
{
    int q;
    return remquo(x, y, &q);
}
/* float operands are exact in double and the remainder of two floats is a float */
OCU_DM_FN float remquof(float x, float y, int *quo) { return (float)remquo((double)x, (double)y, quo); }
OCU_DM_FN float remainderf(float x, float y) { return (float)remainder((double)x, (double)y); }

/* ---- integer results -------------------------------------------------------------------- */

/* Saturating conversion of an integral double; NaN gives 0 (as the 32-bit
   conversions here, and the cvt the CUDA intrinsics name). */
OCU_DM_FN long long sat_i64(double r)
{
    if (r != r) return 0;
    if (r >= 9223372036854775807.0) return 9223372036854775807ll; /* 2^63 and up */
    if (r < -9223372036854775807.0) return (-9223372036854775807ll - 1);
    return (long long)r;
}
OCU_DM_FN unsigned sat_u32(double r)
{
    if (r != r || r <= 0.0) return 0u;
    if (r >= 4294967295.0) return 4294967295u;
    return (unsigned)r;
}
OCU_DM_FN unsigned long long sat_u64(double r)
{
    if (r != r || r <= 0.0) return 0ull;
    if (r >= 18446744073709551615.0) return 18446744073709551615ull; /* 2^64 and up */
    return (unsigned long long)r;
}
OCU_DM_FN long long llrint(double x) { return sat_i64(rint(x)); }
OCU_DM_FN long long llround(double x) { return sat_i64(round(x)); }

/* ---- rounding modes --------------------------------------------------------------------- */

enum rmode { RN = 0, RZ = 1, RU = 2, RD = 3 };

/* The double nearest to the exact value v = r + (something of sign s, smaller than
   half an ulp of r) in mode m, given r = v rounded to nearest; finite operands. */
OCU_DM_FN double adjust(double r, int s, int m)
{
    if (m == RN) return r;
    if (isinf(r)) { /* the exact value is finite: beyond the largest double */
        bool pos = r > 0.0;
        if (m == RZ) return pos ? DMAX : -DMAX;
        if (m == RU) return pos ? r : -DMAX;
        return pos ? DMAX : r; /* RD */
    }
    if (s == 0) return r;
    if (m == RU) return s > 0 ? next_up(r) : r;
    if (m == RD) return s < 0 ? next_down(r) : r;
    /* RZ: toward zero when the exact value is nearer zero than r */
    if (r > 0.0 && s < 0) return next_down(r);
    if (r < 0.0 && s > 0) return next_up(r);
    if (r == 0.0) return r;
    return r;
}
OCU_DM_FN int sgn(double v) { return v > 0.0 ? 1 : (v < 0.0 ? -1 : 0); }

/* The float nearest to the exact value hi + (something of sign s, smaller than
   half an ulp of hi in double) in mode m. */
OCU_DM_FN float to_float(double hi, int s, int m)
{
    float f = (float)hi;
    if (m == RN) {
        /* hi is the double nearest the exact value; when hi is exactly halfway
           between two floats the tail decides */
        if ((double)f != hi && s != 0) {
            double other = (double)f < hi ? (double)next_upf(f) : (double)next_downf(f);
            if (other - hi == hi - (double)f) { /* a tie in hi: the exact value is off it */
                bool exact_above = s > 0;
                return exact_above ? ((double)f > hi ? f : (float)other) : ((double)f < hi ? f : (float)other);
            }
        }
        return f;
    }
    if (isinf(f) && !isinf(hi)) {
        bool pos = hi > 0.0;
        if (m == RZ) return pos ? FMAX : -FMAX;
        if (m == RU) return pos ? f : -FMAX;
        return pos ? FMAX : f;
    }
    int t = (double)f != hi ? sgn(hi - (double)f) : s; /* sign of exact - f */
    if (t == 0) return f;
    if (m == RU) return t > 0 ? next_upf(f) : f;
    if (m == RD) return t < 0 ? next_downf(f) : f;
    if (f > 0.0f && t < 0) return next_downf(f);
    if (f < 0.0f && t > 0) return next_upf(f);
    return f;
}

/* An exact zero sum x + y (IEEE 754 6.3): two zeros of one sign keep it;
   otherwise +0, or -0 rounding down. */
OCU_DM_FN double zero_sum(double x, double y, int m)
{
    if (x == 0.0 && y == 0.0 && signbit(x) == signbit(y)) return x;
    return m == RD ? -0.0 : 0.0;
}

/* double arithmetic, each correctly rounded in mode m */
OCU_DM_FN bool any_special(double a, double b) { return !isfinite(a) || !isfinite(b); }
OCU_DM_FN double dadd(double a, double b, int m)
{
    double s = a + b;
    if (any_special(a, b) || m == RN) return s;
    if (s == 0.0) return zero_sum(a, b, m); /* a sum of finite doubles that is 0 is exact */
    dd t = two_sum(a, b);
    return adjust(t.hi, isinf(t.hi) ? 0 : sgn(t.lo), m);
}
/* The sign of a - x y, exactly (finite operands, x y != 0). A plain
   fma(-x, y, a) can round a residual below the smallest subnormal to zero,
   so the operands are split into mantissas in [0.5, 1) and exponents and
   the difference formed near 1. */
OCU_DM_FN int sign_sub_prod(double a, double x, double y)
{
    const int sxy = (signbit(x) != signbit(y)) ? -1 : 1;
    if (a == 0.0) return -sxy;
    int ea, ex, ey;
    double ma = frexp(a, &ea), mx = frexp(x, &ex), my = frexp(y, &ey);
    int d = ea - ex - ey; /* a = ma 2^ea, x y = mx my 2^(ex + ey), |mx my| in [1/4, 1) */
    if (d > 60) return a > 0.0 ? 1 : -1;
    if (d < -60) return -sxy;
    return sgn(fma(-mx, my, ldexp(ma, d))); /* exact operands, one rounding: the sign is exact */
}
OCU_DM_FN double dmul(double a, double b, int m)
{
    double p = a * b;
    if (any_special(a, b) || m == RN || a == 0.0 || b == 0.0) return p;
    if (isinf(p)) return adjust(p, 0, m);
    return adjust(p, -sign_sub_prod(p, a, b), m); /* sign of a b - p, also when p underflowed to 0 */
}
OCU_DM_FN double ddiv(double a, double b, int m)
{
    double q = a / b;
    if (any_special(a, b) || b == 0.0 || m == RN || a == 0.0) return q;
    if (isinf(q)) return adjust(q, 0, m);
    /* sign of a/b - q = sign of (a - q b) times the sign of b; q may have
       underflowed to a signed zero, which adjust moves off */
    int s = (q == 0.0 ? (a > 0.0 ? 1 : -1) : sign_sub_prod(a, q, b)) * (b > 0.0 ? 1 : -1);
    return adjust(q, s, m);
}
OCU_DM_FN double dsqrt(double a, int m)
{
    double s = sqrt(a);
    if (!isfinite(a) || a <= 0.0 || m == RN) return s;
    return adjust(s, sign_sub_prod(a, s, s), m);
}

/* float arithmetic, each correctly rounded in mode m: the exact result as a
   double plus the sign of what is left */
OCU_DM_FN float fadd(float a, float b, int m)
{
    if (m == RN) return a + b;
    if (!isfinite((double)a) || !isfinite((double)b)) return a + b;
    double x = a, y = b;
    dd t = two_sum(x, y);
    if (t.hi == 0.0) return (float)zero_sum(x, y, m);
    return to_float(t.hi, sgn(t.lo), m);
}
OCU_DM_FN float fmul(float a, float b, int m)
{
    if (m == RN) return a * b;
    double p = (double)a * (double)b; /* exact: 48 bits */
    if (!isfinite(p) || p == 0.0) return (float)p;
    return to_float(p, 0, m);
}
OCU_DM_FN float fdiv(float a, float b, int m)
{
    if (m == RN) return a / b;
    double x = a, y = b, q = x / y;
    if (!isfinite(x) || !isfinite(y) || y == 0.0 || x == 0.0) return (float)q;
    return to_float(q, sgn(fma(-q, y, x)) * (y > 0.0 ? 1 : -1), m);
}
OCU_DM_FN float fsqrt(float a, int m)
{
    if (m == RN) return __builtin_sqrtf(a);
    double x = a, s = sqrt(x);
    if (!isfinite(x) || x <= 0.0) return (float)s;
    return to_float(s, sgn(fma(-s, s, x)), m);
}
OCU_DM_FN float ffma(float a, float b, float c, int m)
{
    double p = (double)a * (double)b; /* exact */
    if (!isfinite(p) || !isfinite((double)c)) return (float)(p + (double)c);
    if (m == RN) return __builtin_fmaf(a, b, c);
    dd t = two_sum(p, (double)c);
    if (t.hi == 0.0) return (float)zero_sum(p, (double)c, m); /* p exact: the sum is exact */
    return to_float(t.hi, sgn(t.lo), m);
}
/* 1/sqrt(x) correctly rounded: 1/sqrt in double is within 1.5 double ulps of
   the exact value, and an exact float midpoint is impossible (it would need
   x = 1/m^2 for a 25-bit m), so the nearest float is that of the double result
   except where it lies within 2^-50 of a midpoint: checked exhaustively. */
OCU_DM_FN float frsqrt_rn(float a)
{
    if (isnanf_(a) || a < 0.0f) return (float)qnan();
    if (a == 0.0f) return signbit((double)a) ? -(float)inf() : (float)inf();
    if (isinff_(a)) return 0.0f;
    return (float)(1.0 / sqrt((double)a));
}

/* 64-bit integers to floating point in mode m: v = hi 2^32 + lo exactly, as a
   two-double sum */
OCU_DM_FN dd i64_dd(long long v)
{
    long long hi = v >> 32; /* arithmetic shift: v = hi 2^32 + lo, 0 <= lo < 2^32 */
    unsigned long long lo = (unsigned long long)v & 0xffffffffull;
    return two_sum((double)hi * 4294967296.0, (double)lo);
}
OCU_DM_FN dd u64_dd(unsigned long long v)
{
    return two_sum((double)(v >> 32) * 4294967296.0, (double)(v & 0xffffffffull));
}
OCU_DM_FN double dd_to_double(dd t, int m) { return adjust(t.hi, sgn(t.lo), m); }
OCU_DM_FN float dd_to_float(dd t, int m) { return to_float(t.hi, sgn(t.lo), m); }

/* ---- sinpi, cospi --------------------------------------------------------------------- */

constexpr double PIX_HI = 3.14159265358979311600e+00;
constexpr double PIX_LO = 1.22464679914735317720e-16;

/* x = k/2 + r with |r| <= 1/4 (exact), k mod 4 in *q; pi r = h + l */
OCU_DM_FN void reduce_pi(double x, int *q, double *h, double *l)
{
    double k = rint(2.0 * x); /* 2x exact for |x| < 2^1023 */
    double r = x - 0.5 * k;   /* exact */
    double km = fmod(k, 4.0);
    if (km < 0) km += 4.0;
    *q = (int)km;
    dd p = two_prod(PIX_HI, r);
    *h = p.hi;
    *l = p.lo + PIX_LO * r;
}
OCU_DM_FN double sinpi(double x)
{
    if (!isfinite(x)) return qnan();
    if (fabs(x) >= 0x1p52 || x == trunc(x)) return copysign(0.0, x); /* an integer */
    int q;
    double h, l;
    reduce_pi(x, &q, &h, &l);
    double s;
    switch (q) {
    case 0: s = kernel_sin(h, l); break;
    case 1: s = kernel_cos(h, l); break;
    case 2: s = -kernel_sin(h, l); break;
    default: s = -kernel_cos(h, l); break;
    }
    return s;
}
OCU_DM_FN double cospi(double x)
{
    if (!isfinite(x)) return qnan();
    double ax = fabs(x);
    if (ax >= 0x1p53) return 1.0; /* an even integer */
    if (ax >= 0x1p52) return (fmod(ax, 2.0) == 0.0) ? 1.0 : -1.0;
    int q;
    double h, l;
    reduce_pi(x, &q, &h, &l);
    double c;
    switch (q) {
    case 0: c = kernel_cos(h, l); break;
    case 1: c = -kernel_sin(h, l); break;
    case 2: c = -kernel_cos(h, l); break;
    default: c = kernel_sin(h, l); break;
    }
    return c == 0.0 ? 0.0 : c; /* cospi(n + 1/2) is +0 */
}
OCU_DM_FN float sinpif(float x) { return (float)sinpi((double)x); }
OCU_DM_FN float cospif(float x) { return (float)cospi((double)x); }

/* ---- hyperbolic inverses (fdlibm's formulas) ---------------------------------------------- */

constexpr double LN2 = 6.93147180559945286227e-01;

OCU_DM_FN double acosh(double x)
{
    if (isnan(x)) return x;
    if (x < 1.0) return qnan();
    if (x >= 0x1p28) return isinf(x) ? x : log(x) + LN2;
    if (x == 1.0) return 0.0;
    if (x > 2.0) return log(2.0 * x - 1.0 / (x + sqrt(x * x - 1.0)));
    double t = x - 1.0; /* exact */
    return log1p(t + sqrt(2.0 * t + t * t));
}
OCU_DM_FN double asinh(double x)
{
    if (!isfinite(x)) return x + x; /* NaN, +-inf */
    double ax = fabs(x), r;
    if (ax < 0x1p-28) return x;
    if (ax > 0x1p28) r = log(ax) + LN2;
    else if (ax > 2.0) r = log(2.0 * ax + 1.0 / (sqrt(x * x + 1.0) + ax));
    else {
        double t = x * x;
        r = log1p(ax + t / (1.0 + sqrt(1.0 + t)));
    }
    return copysign(r, x);
}
OCU_DM_FN double atanh(double x)
{
    if (isnan(x)) return x;
    double ax = fabs(x);
    if (ax > 1.0) return qnan();
    if (ax == 1.0) return copysign(inf(), x);
    if (ax < 0x1p-28) return x;
    double r;
    if (ax < 0.5) {
        double t = ax + ax;
        r = 0.5 * log1p(t + t * ax / (1.0 - ax));
    } else {
        r = 0.5 * log1p((ax + ax) / (1.0 - ax));
    }
    return copysign(r, x);
}
OCU_DM_FN float acoshf(float x) { return (float)acosh((double)x); }
OCU_DM_FN float asinhf(float x) { return (float)asinh((double)x); }
OCU_DM_FN float atanhf(float x) { return (float)atanh((double)x); }

/* ---- reciprocal roots, norms -------------------------------------------------------------- */

OCU_DM_FN double rcbrt(double x)
{
    if (isnan(x)) return x;
    if (x == 0.0) return copysign(inf(), x);
    if (isinf(x)) return copysign(0.0, x);
    /* r = 1/cbrt(x), then one Newton step r += r (1 - x r^3) / 3 with the
       residual from exact products (x r and r r: no overflow, x r^3 ~ 1) */
    double r = 1.0 / cbrt(x);
    dd z = two_prod(x, r), y = two_prod(r, r);
    double e = fma(-z.hi, y.hi, 1.0) - (z.hi * y.lo + z.lo * y.hi);
    return fma(r, e * (1.0 / 3.0), r);
}
OCU_DM_FN float rcbrtf(float x) { return (float)rcbrt((double)x); }
OCU_DM_FN double norm_n(int n, const double *a, bool recip);
OCU_DM_FN double rhypot(double x, double y)
{
    if (isinf(x) || isinf(y)) return 0.0;
    if (isnan(x) || isnan(y)) return x + y;
    double v[2] = {x, y}; /* scaled: 1/hypot of huge operands is a (subnormal) non-zero */
    return norm_n(2, v, true);
}
OCU_DM_FN float rhypotf(float x, float y)
{
    double h = sqrt((double)x * x + (double)y * y); /* in range, about 1 double rounding */
    if (isinff_(x) || isinff_(y)) return 0.0f;
    if (isnanf_(x) || isnanf_(y)) return x + y;
    return (float)(1.0 / h);
}

/* sqrt of the sum of squares of n doubles, scaled by a power of two; inf if
   any is infinite (even with a NaN), NaN if one is NaN */
OCU_DM_FN double norm_n(int n, const double *a, bool recip)
{
    double m = 0.0;
    bool nan_seen = false;
    for (int i = 0; i < n; ++i) {
        double v = fabs(a[i]);
        if (isinf(v)) return recip ? 0.0 : inf();
        if (isnan(v)) nan_seen = true;
        else if (v > m) m = v;
    }
    if (nan_seen) return qnan();
    if (m == 0.0) return recip ? inf() : 0.0;
    int e;
    (void)frexp(m, &e);
    /* the sum of squares in two doubles (each square exact as two doubles) */
    dd acc{0.0, 0.0};
    for (int i = 0; i < n; ++i) {
        double v = ldexp(a[i], -e); /* |v| < 1; exact unless below 2^-1074 of the largest */
        acc = dd_add(acc, two_prod(v, v));
    }
    /* sqrt of hi + lo: one Newton correction of the double root */
    double rh = sqrt(acc.hi);
    double rl = (fma(-rh, rh, acc.hi) + acc.lo) / (2.0 * rh);
    if (!recip) return ldexp(rh + rl, e);
    double q = 1.0 / rh;
    double err = fma(-rh, q, 1.0) - rl * q; /* 1 - (rh + rl) q */
    return ldexp(fma(q, err, q), -e);
}
OCU_DM_FN double norm3d(double a, double b, double c)
{
    double v[3] = {a, b, c};
    return norm_n(3, v, false);
}
OCU_DM_FN double norm4d(double a, double b, double c, double d)
{
    double v[4] = {a, b, c, d};
    return norm_n(4, v, false);
}
OCU_DM_FN double rnorm3d(double a, double b, double c)
{
    double v[3] = {a, b, c};
    return norm_n(3, v, true);
}
OCU_DM_FN double rnorm4d(double a, double b, double c, double d)
{
    double v[4] = {a, b, c, d};
    return norm_n(4, v, true);
}
OCU_DM_FN double norm(int n, const double *a) { return norm_n(n, a, false); }
OCU_DM_FN double rnorm(int n, const double *a) { return norm_n(n, a, true); }

/* float norms: squares and sums of floats in double, no scaling needed */
OCU_DM_FN float normf_n(int n, const float *a, bool recip)
{
    double s = 0.0;
    bool nan_seen = false;
    for (int i = 0; i < n; ++i) {
        if (isinff_(a[i])) return recip ? 0.0f : (float)inf();
        if (isnanf_(a[i])) nan_seen = true;
        else s = fma((double)a[i], (double)a[i], s);
    }
    if (nan_seen) return (float)qnan();
    if (s == 0.0) return recip ? (float)inf() : 0.0f;
    double r = sqrt(s);
    return (float)(recip ? 1.0 / r : r);
}
OCU_DM_FN float norm3df(float a, float b, float c)
{
    float v[3] = {a, b, c};
    return normf_n(3, v, false);
}
OCU_DM_FN float norm4df(float a, float b, float c, float d)
{
    float v[4] = {a, b, c, d};
    return normf_n(4, v, false);
}
OCU_DM_FN float rnorm3df(float a, float b, float c)
{
    float v[3] = {a, b, c};
    return normf_n(3, v, true);
}
OCU_DM_FN float rnorm4df(float a, float b, float c, float d)
{
    float v[4] = {a, b, c, d};
    return normf_n(4, v, true);
}
OCU_DM_FN float normf(int n, const float *a) { return normf_n(n, a, false); }
OCU_DM_FN float rnormf(int n, const float *a) { return normf_n(n, a, true); }

/* ---- erf, erfc (double) -------------------------------------------------------------
 * |x| < 1.5: the Taylor series erf(x) = 2/sqrt(pi) sum_n (-1)^n x^(2n+1) / (n! (2n+1))
 *   summed in two doubles, so erfc = 1 - erf keeps its accuracy there as well
 *   (erfc(1.5) = 0.034: one double would lose 5 bits).
 * x >= 1.5: erfc(x) = exp(-x^2) / sqrt(pi) / F, F the continued fraction
 *   x + (1/2) / (x + 1 / (x + (3/2) / (x + 2 / (x + ...)))), evaluated from the
 *   bottom in two doubles at a depth that converges for that x; exp(-x^2) from
 *   x^2 = h + l exactly as exp(-h) (1 - l), so its error does not grow with x.
 *   erf = 1 - erfc there (and +-1 from 6 on, where erfc < 2^-55).
 * Special values as the C library: erf(+-0) = +-0, erf(+-inf) = +-1,
 * erfc(+inf) = 0, erfc(-inf) = 2, NaN in, NaN out. */

constexpr double TWO_OVER_SQRTPI_HI = 1.1283791670955126, TWO_OVER_SQRTPI_LO = 1.533545961316588e-17;
constexpr double ONE_OVER_SQRTPI_HI = 0.5641895835477563, ONE_OVER_SQRTPI_LO = 7.66772980658294e-18;

OCU_DM_FN dd dd_mul(dd a, dd b)
{
    dd p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return fast_two_sum(p.hi, p.lo);
}
OCU_DM_FN dd dd_div(dd a, dd b)
{
    double q1 = a.hi / b.hi;
    dd p = dd_mul(dd{q1, 0.0}, b);
    dd r = dd_add(a, dd{-p.hi, -p.lo});
    double q2 = r.hi / b.hi;
    return fast_two_sum(q1, q2);
}

/* erf(x) for |x| < 1.5 as two doubles */
OCU_DM_FN dd erf_series(double x)
{
    dd x2 = two_prod(x, x);
    dd p = dd{x, 0.0}; /* (-1)^n x^(2n+1) / n! */
    dd s = p;
    for (int n = 1; n < 80; ++n) {
        p = dd_mul(p, x2);
        p = dd_div(dd{-p.hi, -p.lo}, dd{(double)n, 0.0});
        dd t = dd_div(p, dd{(double)(2 * n + 1), 0.0});
        s = dd_add(s, t);
        if (fabs(t.hi) <= 1e-34 * fabs(s.hi)) break;
    }
    return dd_mul(s, dd{TWO_OVER_SQRTPI_HI, TWO_OVER_SQRTPI_LO});
}

/* erfc(x) for x >= 1.5 */
OCU_DM_FN double erfc_cf(double x)
{
    if (x >= 27.3) return 0.0; /* below the smallest subnormal */
    const int depth = (int)(250.0 / (x * x)) + 12;
    dd f = dd{x, 0.0};
    for (int n = depth; n >= 1; --n) f = dd_add(dd{x, 0.0}, dd_div(dd{0.5 * n, 0.0}, f));
    dd x2 = two_prod(x, x);
    double e = exp(-x2.hi) * (1.0 - x2.lo); /* exp(-(h + l)), |l| <= ulp(h) / 2 */
    dd q = dd_div(dd{ONE_OVER_SQRTPI_HI, ONE_OVER_SQRTPI_LO}, f);
    return e * q.hi + e * q.lo;
}

OCU_DM_FN double erf(double x)
{
    if (x != x) return x;
    double ax = fabs(x);
    double r;
    if (ax < 1.5) {
        dd s = erf_series(ax);
        r = s.hi + s.lo;
    } else if (ax < 6.0) {
        r = 1.0 - erfc_cf(ax);
    } else {
        r = 1.0;
    }
    return x < 0.0 ? -r : (x == 0.0 ? x : r);
}

OCU_DM_FN double erfc(double x)
{
    if (x != x) return x;
    double ax = fabs(x);
    if (ax < 1.5) {
        dd s = erf_series(ax); /* erf |x| */
        dd r = x < 0.0 ? dd_add(dd{1.0, 0.0}, s) : dd_add(dd{1.0, 0.0}, dd{-s.hi, -s.lo});
        return r.hi + r.lo;
    }
    if (x > 0.0) return erfc_cf(ax);
    return 2.0 - erfc_cf(ax); /* -inf: 2 */
}

} // namespace ocu_dm

#endif /* OCU_DEVICE_MATH_EXT_H */

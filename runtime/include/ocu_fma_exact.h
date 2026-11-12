/*
 * ocu_fma_exact.h -- fused multiply-add, a*b+c with one rounding, for
 * hosts whose C library fma/fmaf cannot be trusted (MinGW-w64's are not
 * correctly rounded: fmaf(0x1.02fbdap+0, 0x1.fffffep-1, -0x1.02fbd8p+0)
 * gives 0x1.fa085p-25 instead of 0x1.fa084cp-25).
 *
 * IEEE 754-2019 5.4.1 defines fusedMultiplyAdd(x, y, z) as (x*y)+z
 * computed "as if with unbounded range and precision, rounding only once
 * to the destination format"; zero results follow 6.3 (an exact zero sum
 * of operands of opposite sign is +0, or -0 when rounding toward
 * negative; x*y+z with x*y and z zeros of the same sign keeps that sign),
 * invalid operations (0*inf, inf-inf) give a quiet NaN (7.2). The CUDA
 * Math API documents fma/fmaf and __fma_r{n,z,u,d}/__fmaf_r{n,z,u,d} as
 * that single correctly rounded operation, in the named rounding mode.
 *
 * Methods (written from the papers, not from any C library):
 *  - float: the product of two floats (24-bit significands) is exact in
 *    double (53 bits, wider exponent range). The sum p + c is formed as
 *    s + t exactly with the two-sum of Knuth (TAOCP vol. 2, 4.2.2,
 *    Theorem B) / Moller (BIT 5, 1965), then rounded to odd in double:
 *    s, or its neighbour on the side of t when s is even and t != 0.
 *    Boldo and Melquiond, "Emulation of FMA and correctly rounded sums:
 *    proved algorithms using rounding to odd", IEEE Trans. Computers
 *    57(4), 2008, double-rounding theorem: rounding to odd at precision p+k (k >= 2)
 *    followed by rounding to nearest at precision p is the correct
 *    rounding; directed roundings need k >= 1. Double has 53 >= 24 + 2
 *    bits, and every partial result here is far from double's underflow
 *    and overflow, so the final double -> float conversion (including to
 *    float subnormals, infinities and the largest float) rounds the exact
 *    value correctly.
 *  - double, round to nearest, operands between 2^-400 and 2^400 in
 *    magnitude (the common case): the emulated FMA of Boldo and Melquiond
 *    (their FMA emulation algorithm): Dekker's exact product, a two-sum with c, the two low
 *    parts added rounded to odd, one final rounding; the range keeps every
 *    step clear of overflow and underflow, which the proof assumes.
 *  - double, otherwise: exact integer arithmetic on 128 bits. Significands are
 *    normalised (subnormals included), the 106-bit product is formed
 *    exactly, the addend is aligned to it; bits shifted out of the smaller
 *    term are kept by rounding the integer sum to odd (the same Boldo and
 *    Melquiond argument: at least 70 bits lie below the final rounding
 *    position whenever bits were shifted out), and the sum is rounded once
 *    to the double format (IEEE 754-2019 4.3, 7.4 overflow, 7.5 underflow:
 *    subnormal results are rounded at 2^-1074, tininess does not matter
 *    because no flags are kept).
 *
 * Selection: OCU_FMA_BUILTIN = 1 uses the compiler's fma builtin (the
 * instruction, or a call to the C library's fma where the target has
 * none), 0 the software path. Default, first match:
 *  - 1 in an NVIDIA device pass (__NVPTX__ and __CUDA_ARCH__; the CPU path defines only __CUDA_ARCH__) and when the compiler says the
 *    target has a fused multiply-add instruction (__FP_FAST_FMA, __FMA__,
 *    __ARM_FEATURE_FMA);
 *  - OCU_LIBM_FMA_EXACT when defined: the result of the configure check
 *    of the C library (runtime/CMakeLists.txt), passed to every program
 *    that links ocu_cudart;
 *  - 1 with glibc, whose fma/fmaf are correctly rounded and pick the
 *    instruction at run time on processors that have it (the configure
 *    check confirms this on the build host);
 *  - otherwise 0: an unknown C library is not trusted.
 * Directed roundings always use the software path on the host.
 *
 * C and C++. OCU_DM_FN marks every function (as in ocu_device_math.h).
 * Exception flags are not raised (CUDA has none); the rounding mode in
 * effect must be round-to-nearest (the C default).
 */
#ifndef OCU_FMA_EXACT_H
#define OCU_FMA_EXACT_H

#include <stdint.h>

#ifndef OCU_DM_FN
#define OCU_DM_FN static inline
#endif

#ifndef OCU_FMA_BUILTIN
#if (defined(__NVPTX__) && defined(__CUDA_ARCH__)) || defined(__FP_FAST_FMA) || defined(__FMA__) || \
    defined(__ARM_FEATURE_FMA)
#define OCU_FMA_BUILTIN 1
#elif defined(OCU_LIBM_FMA_EXACT)
#define OCU_FMA_BUILTIN OCU_LIBM_FMA_EXACT
#elif defined(__GLIBC__)
#define OCU_FMA_BUILTIN 1
#else
#define OCU_FMA_BUILTIN 0
#endif
#endif

/* The two-sum and the conversions below need double operations rounded
   to double, not to a wider format (x87 extended precision). Such targets
   (and, for double, targets without 128-bit integers) fall back to the C
   library's fma; the software functions are then not declared. */
#if defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__ != 0 && __FLT_EVAL_METHOD__ != 1
#define OCU_FMA_NO_SOFT 1
#endif

#define OCU_FMA_RN 0 /* to nearest, ties to even */
#define OCU_FMA_RZ 1 /* toward zero */
#define OCU_FMA_RU 2 /* toward +infinity */
#define OCU_FMA_RD 3 /* toward -infinity */

#ifdef __cplusplus
#define OCU_FMA_BITS64(x) __builtin_bit_cast(uint64_t, (x))
#define OCU_FMA_DBL(b) __builtin_bit_cast(double, (b))
#define OCU_FMA_BITS32(x) __builtin_bit_cast(uint32_t, (x))
#define OCU_FMA_FLT(b) __builtin_bit_cast(float, (b))
#else
OCU_DM_FN uint64_t ocu_fma_bits64_(double x) { union { double d; uint64_t u; } v; v.d = x; return v.u; }
OCU_DM_FN double ocu_fma_dbl_(uint64_t b) { union { double d; uint64_t u; } v; v.u = b; return v.d; }
OCU_DM_FN uint32_t ocu_fma_bits32_(float x) { union { float f; uint32_t u; } v; v.f = x; return v.u; }
OCU_DM_FN float ocu_fma_flt_(uint32_t b) { union { float f; uint32_t u; } v; v.u = b; return v.f; }
#define OCU_FMA_BITS64(x) ocu_fma_bits64_(x)
#define OCU_FMA_DBL(b) ocu_fma_dbl_(b)
#define OCU_FMA_BITS32(x) ocu_fma_bits32_(x)
#define OCU_FMA_FLT(b) ocu_fma_flt_(b)
#endif

#if !(defined(__NVPTX__) && defined(__CUDA_ARCH__)) && !defined(OCU_FMA_NO_SOFT)

/* ---- float ------------------------------------------------------------------ */

/* a*b+c rounded once to float in `mode`. */
OCU_DM_FN float ocu_fmaf_soft(float a, float b, float c, int mode)
{
    double p = (double)a * (double)b; /* exact (or inf/NaN from inf/NaN inputs) */
    double z = (double)c;
    double s = p + z;
    if ((OCU_FMA_BITS64(s) & 0x7ff0000000000000ull) == 0x7ff0000000000000ull) {
        /* Only from infinite or NaN operands: |p| < 2^256 otherwise. */
        return (float)s;
    }
    if (s == 0.0) {
        /* No double sum of these operands is inexact and zero, so a*b+c is
           exactly zero; the sign under round-to-nearest is that of s. */
        if (mode == OCU_FMA_RD) return (float)-((-p) + (-z)); /* -0 unless both +0 */
        return (float)s;
    }
    /* Two-sum: s + t == p + z exactly. */
    double bv = s - p;
    double t = (p - (s - bv)) + (z - bv);
    if (t != 0.0) {
        /* Round to odd: of the two doubles around p + z (s and the next one
           toward t), the one whose last significand bit is 1. Adjacent
           doubles of one sign have consecutive encodings. */
        uint64_t sb = OCU_FMA_BITS64(s);
        if ((sb & 1u) == 0) {
            sb = ((t > 0.0) == (s > 0.0)) ? sb + 1u : sb - 1u;
            s = OCU_FMA_DBL(sb);
        }
    }
    float f = (float)s; /* round to nearest */
    if (mode == OCU_FMA_RN) return f;
    double fd = (double)f; /* exact */
    uint32_t fb = OCU_FMA_BITS32(f);
    if (mode == OCU_FMA_RZ) {
        if (s > 0.0 ? fd > s : fd < s) fb -= 1u; /* one step toward zero */
    } else if (mode == OCU_FMA_RU) {
        if (fd < s) fb = (s > 0.0) ? fb + 1u : fb - 1u;
    } else { /* OCU_FMA_RD */
        if (fd > s) fb = (s > 0.0) ? fb - 1u : fb + 1u;
    }
    return OCU_FMA_FLT(fb);
}

/* ---- double ----------------------------------------------------------------- */

#if defined(__SIZEOF_INT128__)
#define OCU_FMA_HAVE_SOFT_DOUBLE 1

__extension__ typedef unsigned __int128 ocu_fma_u128;

/* Significand and exponent of a finite nonzero double, x = m * 2^e with
   the top bit of m at bit 52 (subnormals normalised). */
OCU_DM_FN uint64_t ocu_fma_split_(uint64_t u, int *e)
{
    int ef = (int)((u >> 52) & 0x7ffu);
    uint64_t m = u & 0x000fffffffffffffull;
    if (ef != 0) {
        m |= 0x0010000000000000ull;
        *e = ef - 1075;
    } else {
        int sh = __builtin_clzll(m) - 11;
        m <<= sh;
        *e = -1074 - sh;
    }
    return m;
}

/* Zero with the IEEE 754-2019 6.3 sign of an exact zero sum of operands of
   opposite sign. */
OCU_DM_FN double ocu_fma_zero_(int mode) { return mode == OCU_FMA_RD ? -0.0 : 0.0; }

/* x + y rounded to odd (exact two-sum, then the odd neighbour). */
OCU_DM_FN double ocu_fma_add_odd_(double x, double y)
{
    double s = x + y;
    double bv = s - x;
    double t = (x - (s - bv)) + (y - bv);
    uint64_t sb = OCU_FMA_BITS64(s);
    if (t != 0.0 && (sb & 1u) == 0) sb = ((t > 0.0) == (s > 0.0)) ? sb + 1u : sb - 1u;
    return OCU_FMA_DBL(sb);
}

/* Round to nearest when every operand lies in 2^-400 <= |x| < 2^400 (or
   is zero): no overflow, and no step underflows (a nonzero exact result
   is above 2^-906). Boldo and Melquiond 2008, FMA emulation:
   (uh, ul) = a*b exactly by Dekker's product with Veltkamp's splitting
   (Dekker, "A floating-point technique for extending the available
   precision", Numer. Math. 18, 1971), (th, tl) = two-sum(c, uh),
   v = RO(tl + ul), result RN(th + v). */
OCU_DM_FN double ocu_fma_rn_mid_(double a, double b, double c)
{
#if defined(__FP_FAST_FMA) || defined(__FMA__) || defined(__ARM_FEATURE_FMA)
    /* The compiler could contract the products below into fma, which
       Dekker's product must not have; here the instruction exists. */
    return __builtin_fma(a, b, c);
#else
    const double split = 134217729.0; /* 2^27 + 1 */
    double ca = split * a, cb = split * b;
    double ah = ca - (ca - a), al = a - ah;
    double bh = cb - (cb - b), bl = b - bh;
    double uh = a * b;
    double ul = ((ah * bh - uh) + ah * bl + al * bh) + al * bl;
    double th = c + uh;
    double bv = th - c;
    double tl = (c - (th - bv)) + (uh - bv);
    double r = th + ocu_fma_add_odd_(tl, ul);
    /* A nonzero exact result is above 2^-906, so zero is an exact
       cancellation of nonzero terms: +0 (IEEE 754-2019 6.3). */
    return r == 0.0 ? 0.0 : r;
#endif
}

/* a*b+c rounded once to double in `mode`. */
OCU_DM_FN double ocu_fma_soft(double a, double b, double c, int mode)
{
    uint64_t ua = OCU_FMA_BITS64(a), ub = OCU_FMA_BITS64(b), uc = OCU_FMA_BITS64(c);
    const uint64_t EXP = 0x7ff0000000000000ull;
    if (mode == OCU_FMA_RN) {
        /* Biased exponents 623..1422: 2^-400 <= |x| < 2^400, no zeros. */
        uint32_t xa = (uint32_t)((ua >> 52) & 0x7ffu) - 623u;
        uint32_t xb = (uint32_t)((ub >> 52) & 0x7ffu) - 623u;
        uint32_t xc = (uint32_t)((uc >> 52) & 0x7ffu) - 623u;
        if (xa < 800u && xb < 800u && xc < 800u) return ocu_fma_rn_mid_(a, b, c);
    }
    if ((ua & EXP) == EXP || (ub & EXP) == EXP) {
        /* a or b infinite or NaN: the product is exact (inf or NaN), and
           adding c gives the IEEE result (inf - inf is NaN). */
        return a * b + c;
    }
    if ((uc & EXP) == EXP) return c + c; /* finite a*b: the result is c (NaN quieted) */

    int sp = (int)((ua ^ ub) >> 63), sc = (int)(uc >> 63);
    int pzero = (ua << 1) == 0 || (ub << 1) == 0;
    int czero = (uc << 1) == 0;
    if (pzero) {
        if (!czero) return c;          /* exact */
        if (sp == sc) return c;        /* zeros of one sign keep it */
        return ocu_fma_zero_(mode);
    }

    int ea, eb;
    uint64_t ma = ocu_fma_split_(ua, &ea), mb = ocu_fma_split_(ub, &eb);
    /* Product: top bit at 104 or 105, moved to 124 or 125. */
    ocu_fma_u128 r = ((ocu_fma_u128)ma * mb) << 20;
    int e = ea + eb - 20; /* value = r * 2^e */
    int neg = sp;
    if (!czero) {
        int ec;
        uint64_t mc = ocu_fma_split_(uc, &ec);
        ocu_fma_u128 y = (ocu_fma_u128)mc << 72; /* top bit at 124 */
        ec -= 72;
        ocu_fma_u128 x = r;
        int ex = e, sx = sp, sy = sc;
        if (ec > ex) { /* x: the term with the larger exponent */
            ocu_fma_u128 tmp = x;
            x = y;
            y = tmp;
            int ti = ex;
            ex = ec;
            ec = ti;
            sx = sc;
            sy = sp;
        }
        int d = ex - ec;
        /* Both terms have at least 20 trailing zero bits, so `lost` is only
           set when d > 20 and then x is the larger by a factor 2^19. */
        int lost;
        if (d == 0) {
            lost = 0;
        } else if (d >= 128) {
            lost = 1;
            y = 0;
        } else {
            lost = (y << (128 - d)) != 0;
            y >>= d;
        }
        e = ex;
        if (sx == sy) {
            r = x + y; /* < 2^127 */
            if (lost) r |= 1u; /* sum in (x+y, x+y+1): round to odd */
            neg = sx;
        } else if (y > x) { /* only when d <= 1: nothing was lost */
            r = y - x;
            neg = sy;
        } else {
            r = x - y;
            if (lost) r = (r - 1u) | 1u; /* difference in (r-1, r): round to odd */
            neg = sx;
            if (r == 0) return ocu_fma_zero_(mode); /* exact cancellation */
        }
    }

    /* Round r * 2^e (r != 0) to double. */
    uint64_t hi = (uint64_t)(r >> 64);
    int m = hi != 0 ? 127 - __builtin_clzll(hi) : 63 - __builtin_clzll((uint64_t)r);
    int E = m + e; /* 2^E <= |value| < 2^(E+1) */
    uint64_t out;
    int overflow = E > 1023;
    if (!overflow) {
        int lsb = E - 52 < -1074 ? -1074 : E - 52; /* exponent of the result's last bit */
        int sh = lsb - e;
        uint64_t q;
        int up = 0;
        if (sh <= 0) {
            q = (uint64_t)(r << -sh); /* exact */
        } else {
            int inexact, above, tie;
            if (sh > 128) {
                q = 0;
                inexact = 1;
                above = 0;
                tie = 0;
            } else {
                ocu_fma_u128 half = (ocu_fma_u128)1 << (sh - 1);
                ocu_fma_u128 rest = sh == 128 ? r : r & ((half << 1) - 1u);
                q = sh == 128 ? 0 : (uint64_t)(r >> sh);
                inexact = rest != 0;
                above = rest > half;
                tie = rest == half;
            }
            if (mode == OCU_FMA_RN) up = above || (tie && (q & 1u));
            else if (mode == OCU_FMA_RU) up = inexact && !neg;
            else if (mode == OCU_FMA_RD) up = inexact && neg;
        }
        /* Normal: q has its top bit at 52, which adds one to the exponent
           field; subnormal: lsb = -1074 and the field is q's. A carry out
           of the significand moves to the next binade (or to infinity). */
        out = ((uint64_t)(lsb + 1074) << 52) + q + (uint64_t)up;
        overflow = out >= EXP;
    }
    if (overflow) {
        /* IEEE 754-2019 7.4: infinity, or the largest finite double when
           the rounding direction is toward zero for this sign. */
        int to_inf = mode == OCU_FMA_RN || (mode == OCU_FMA_RU && !neg) ||
                     (mode == OCU_FMA_RD && neg);
        out = to_inf ? EXP : 0x7fefffffffffffffull;
    }
    return OCU_FMA_DBL(out | ((uint64_t)neg << 63));
}
#endif /* __SIZEOF_INT128__ */

#endif /* !__NVPTX__ && !OCU_FMA_NO_SOFT */

/* ---- entry points ----------------------------------------------------------- */

OCU_DM_FN float ocu_fmaf_rn(float a, float b, float c)
{
#if OCU_FMA_BUILTIN || (defined(__NVPTX__) && defined(__CUDA_ARCH__))
    return __builtin_fmaf(a, b, c);
#elif defined(OCU_FMA_NO_SOFT)
    return __builtin_fmaf(a, b, c); /* the C library's; see OCU_FMA_NO_SOFT */
#else
    return ocu_fmaf_soft(a, b, c, OCU_FMA_RN);
#endif
}

OCU_DM_FN double ocu_fma_rn(double a, double b, double c)
{
#if OCU_FMA_BUILTIN || (defined(__NVPTX__) && defined(__CUDA_ARCH__))
    return __builtin_fma(a, b, c);
#elif defined(OCU_FMA_HAVE_SOFT_DOUBLE)
    return ocu_fma_soft(a, b, c, OCU_FMA_RN);
#else
    return __builtin_fma(a, b, c); /* the C library's: no 128-bit integers */
#endif
}

#endif /* OCU_FMA_EXACT_H */

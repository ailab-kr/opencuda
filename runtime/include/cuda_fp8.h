/*
 * cuda_fp8.h -- 8-bit floating-point types (E4M3 and E5M2) and their
 * conversions, for host code and device code on every path.
 *
 * Written for this project from the public description of the CUDA FP8
 * intrinsics (CUDA Math API, "FP8 Intrinsics": storage types, the
 * __nv_saturation_t and __nv_fp8_interpretation_t enumerations, the
 * __nv_cvt_* conversion functions and the __nv_fp8_e4m3 / __nv_fp8_e5m2
 * classes) and from the two formats' definitions; not derived from any
 * vendor header. C++ only.
 *
 * Formats:
 *   E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa bits. No infinities;
 *         S.1111.111 is NaN, so the largest finite value is 448 (0x7e).
 *   E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa bits, IEEE-like:
 *         infinities (0x7c) and NaNs; largest finite value 57344 (0x7b).
 * Conversions to FP8 round to nearest even. Out-of-range values (and
 * infinities) become the largest finite value of that sign with
 * __NV_SATFINITE; with __NV_NOSAT they become NaN (E4M3) or infinity
 * (E5M2). A NaN input gives the canonical NaN 0x7f. Conversions from FP8
 * are exact (every FP8 value is a binary16 and a binary32 value).
 * Values are rounded once from the source type (double inputs are not
 * rounded to float first).
 */
#ifndef OCU_CUDA_FP8_H
#define OCU_CUDA_FP8_H

#ifndef __cplusplus
#error "cuda_fp8.h needs C++"
#endif

#include <cstdint>

#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "ocu_vector_types.h"

#if defined(__CUDA__)
#define OCU_FP8_FN inline __attribute__((host)) __attribute__((device))
#else
#define OCU_FP8_FN inline
#endif

typedef unsigned char __nv_fp8_storage_t;
typedef unsigned short __nv_fp8x2_storage_t;
typedef unsigned int __nv_fp8x4_storage_t;

typedef enum __nv_saturation_t { __NV_NOSAT, __NV_SATFINITE } __nv_saturation_t;
typedef enum __nv_fp8_interpretation_t { __NV_E4M3, __NV_E5M2 } __nv_fp8_interpretation_t;

namespace ocu_fp8 {

/* Exact value of FP8 bits as double. */
OCU_FP8_FN double to_double(__nv_fp8_storage_t v, __nv_fp8_interpretation_t f)
{
    const int mbits = f == __NV_E4M3 ? 3 : 2;
    const int bias = f == __NV_E4M3 ? 7 : 15;
    const int emax_field = f == __NV_E4M3 ? 15 : 31;
    const bool neg = (v & 0x80u) != 0;
    const int e = (v & 0x7fu) >> mbits;
    const int m = v & ((1u << mbits) - 1u);
    double r;
    if (f == __NV_E4M3 && (v & 0x7fu) == 0x7fu) {
        r = __builtin_nan("");
    } else if (f == __NV_E5M2 && e == emax_field) {
        r = m == 0 ? __builtin_inf() : __builtin_nan("");
    } else {
        /* value = n * 2^(k): subnormal n = m, k = 1 - bias - mbits;
           normal n = 2^mbits + m, k = e - bias - mbits. */
        const int n = e == 0 ? m : ((1 << mbits) | m);
        const int k = (e == 0 ? 1 : e) - bias - mbits;
        double scale = 1.0;
        for (int i = 0; i < k; ++i) scale *= 2.0;
        for (int i = 0; i > k; --i) scale *= 0.5;
        r = n * scale;
    }
    return neg ? -r : r;
}

/* FP8 bits from a double, rounded to nearest even. */
OCU_FP8_FN __nv_fp8_storage_t from_double(double x, __nv_saturation_t sat,
                                         __nv_fp8_interpretation_t f)
{
    const int mbits = f == __NV_E4M3 ? 3 : 2;
    const int bias = f == __NV_E4M3 ? 7 : 15;
    const unsigned max_code = f == __NV_E4M3 ? 0x7eu : 0x7bu;
    const unsigned overflow_code = sat == __NV_SATFINITE ? max_code
                                   : (f == __NV_E4M3 ? 0x7fu : 0x7cu);
    if (x != x) {
        return 0x7f;
    }
    const unsigned sign = __builtin_signbit(x) ? 0x80u : 0u;
    const double a = sign ? -x : x;
    if (a == __builtin_inf()) {
        return static_cast<__nv_fp8_storage_t>(sign | overflow_code);
    }
    if (a >= 65536.0) { /* above both formats' range, and keeps the loops short */
        return static_cast<__nv_fp8_storage_t>(sign | overflow_code);
    }
    /* Binary exponent of a (a = s * 2^e with 1 <= s < 2). */
    const int emin = 1 - bias;
    int e = 0;
    double s = a;
    if (s != 0.0) {
        while (s >= 2.0) { s *= 0.5; ++e; }
        while (s < 1.0 && e > emin - 1) { s *= 2.0; --e; }
    } else {
        e = emin;
    }
    if (e < emin) e = emin; /* subnormal: same quantum as the smallest normal */
    /* n = a / 2^(e - mbits), rounded to nearest even (exact scaling). */
    double q = a;
    for (int i = 0; i < e - mbits; ++i) q *= 0.5;
    for (int i = 0; i > e - mbits; --i) q *= 2.0;
    double n = __builtin_rint(q); /* default rounding: nearest even */
    unsigned code;
    if (n < static_cast<double>(1 << mbits) && e == emin) {
        code = static_cast<unsigned>(n); /* subnormal (or zero) */
    } else {
        if (n >= static_cast<double>(2 << mbits)) { /* rounded up to the next binade */
            n *= 0.5;
            ++e;
        }
        code = (static_cast<unsigned>(e + bias) << mbits) |
               (static_cast<unsigned>(n) - (1u << mbits));
    }
    if (code > max_code) {
        code = overflow_code;
    }
    return static_cast<__nv_fp8_storage_t>(sign | code);
}

} // namespace ocu_fp8

/* ---- conversion functions ----------------------------------------------- */

OCU_FP8_FN __nv_fp8_storage_t __nv_cvt_double_to_fp8(double x, __nv_saturation_t sat,
                                                    __nv_fp8_interpretation_t f)
{
    return ocu_fp8::from_double(x, sat, f);
}
OCU_FP8_FN __nv_fp8_storage_t __nv_cvt_float_to_fp8(float x, __nv_saturation_t sat,
                                                   __nv_fp8_interpretation_t f)
{
    return ocu_fp8::from_double(static_cast<double>(x), sat, f);
}
OCU_FP8_FN __nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(__half_raw x, __nv_saturation_t sat,
                                                     __nv_fp8_interpretation_t f)
{
    return ocu_fp8::from_double(static_cast<double>(static_cast<float>(__half(x))), sat, f);
}
OCU_FP8_FN __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw x,
                                                         __nv_saturation_t sat,
                                                         __nv_fp8_interpretation_t f)
{
    return ocu_fp8::from_double(static_cast<double>(__bfloat162float(__nv_bfloat16(x))), sat, f);
}

/* Pairs: the first element (.x) goes to the low byte. */
OCU_FP8_FN __nv_fp8x2_storage_t ocu_fp8_pair(__nv_fp8_storage_t lo, __nv_fp8_storage_t hi)
{
    return static_cast<__nv_fp8x2_storage_t>(lo | (static_cast<unsigned>(hi) << 8));
}
OCU_FP8_FN __nv_fp8x2_storage_t __nv_cvt_double2_to_fp8x2(double2 x, __nv_saturation_t sat,
                                                         __nv_fp8_interpretation_t f)
{
    return ocu_fp8_pair(__nv_cvt_double_to_fp8(x.x, sat, f), __nv_cvt_double_to_fp8(x.y, sat, f));
}
OCU_FP8_FN __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2 x, __nv_saturation_t sat,
                                                        __nv_fp8_interpretation_t f)
{
    return ocu_fp8_pair(__nv_cvt_float_to_fp8(x.x, sat, f), __nv_cvt_float_to_fp8(x.y, sat, f));
}
OCU_FP8_FN __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(__half2_raw x, __nv_saturation_t sat,
                                                          __nv_fp8_interpretation_t f)
{
    return ocu_fp8_pair(__nv_cvt_halfraw_to_fp8(__half_raw{x.x}, sat, f),
                        __nv_cvt_halfraw_to_fp8(__half_raw{x.y}, sat, f));
}
OCU_FP8_FN __nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(__nv_bfloat162_raw x,
                                                              __nv_saturation_t sat,
                                                              __nv_fp8_interpretation_t f)
{
    return ocu_fp8_pair(__nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw{x.x}, sat, f),
                        __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw{x.y}, sat, f));
}

OCU_FP8_FN __half_raw __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t x, __nv_fp8_interpretation_t f)
{
    return static_cast<__half_raw>(__half(ocu_fp8::to_double(x, f)));
}
OCU_FP8_FN __half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t x,
                                                 __nv_fp8_interpretation_t f)
{
    __half_raw lo = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(x & 0xffu), f);
    __half_raw hi = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(x >> 8), f);
    return __half2_raw{lo.x, hi.x};
}

/* ---- classes --------------------------------------------------------------
 * Constructors convert with __NV_SATFINITE; conversions out are exact
 * (integers: toward zero, NaN gives 0, saturating).
 */
#define OCU_FP8_SCALAR(NAME, FMT)                                                         \
    struct NAME {                                                                         \
        __nv_fp8_storage_t __x;                                                           \
        NAME() = default;                                                                 \
        OCU_FP8_FN explicit NAME(double v) : __x(ocu_fp8::from_double(v, __NV_SATFINITE, FMT)) {} \
        OCU_FP8_FN explicit NAME(float v)                                                 \
            : __x(ocu_fp8::from_double(static_cast<double>(v), __NV_SATFINITE, FMT)) {}   \
        OCU_FP8_FN explicit NAME(__half v)                                                \
            : __x(__nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(v), __NV_SATFINITE, FMT)) {} \
        OCU_FP8_FN explicit NAME(__nv_bfloat16 v)                                         \
            : __x(__nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16_raw>(v), __NV_SATFINITE, FMT)) {} \
        template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>         \
        OCU_FP8_FN explicit NAME(I v)                                                     \
            : __x(ocu_fp8::from_double(static_cast<double>(v), __NV_SATFINITE, FMT)) {}   \
        OCU_FP8_FN explicit operator double() const { return ocu_fp8::to_double(__x, FMT); } \
        OCU_FP8_FN explicit operator float() const                                        \
        { return static_cast<float>(ocu_fp8::to_double(__x, FMT)); }                      \
        OCU_FP8_FN explicit operator __half() const                                       \
        { return __half(__nv_cvt_fp8_to_halfraw(__x, FMT)); }                             \
        OCU_FP8_FN explicit operator __nv_bfloat16() const                                \
        { return __float2bfloat16(static_cast<float>(ocu_fp8::to_double(__x, FMT))); }    \
        OCU_FP8_FN explicit operator bool() const { return (__x & 0x7fu) != 0; }          \
        template <typename I, typename = std::enable_if_t<std::is_integral_v<I> &&        \
                                                          !std::is_same_v<I, bool>>>      \
        OCU_FP8_FN explicit operator I() const                                            \
        {                                                                                 \
            double d = ocu_fp8::to_double(__x, FMT);                                      \
            if (d != d) return I(0);                                                      \
            d = __builtin_trunc(d);                                                       \
            if (d <= static_cast<double>(std::numeric_limits<I>::min()))                  \
                return std::numeric_limits<I>::min();                                     \
            if (d >= static_cast<double>(std::numeric_limits<I>::max()))                  \
                return std::numeric_limits<I>::max();                                     \
            return static_cast<I>(d);                                                     \
        }                                                                                 \
    };

OCU_FP8_SCALAR(__nv_fp8_e4m3, __NV_E4M3)
OCU_FP8_SCALAR(__nv_fp8_e5m2, __NV_E5M2)
#undef OCU_FP8_SCALAR

#define OCU_FP8_PAIR(NAME, FMT)                                                           \
    struct NAME {                                                                         \
        __nv_fp8x2_storage_t __x;                                                         \
        NAME() = default;                                                                 \
        OCU_FP8_FN explicit NAME(double2 v) : __x(__nv_cvt_double2_to_fp8x2(v, __NV_SATFINITE, FMT)) {} \
        OCU_FP8_FN explicit NAME(float2 v) : __x(__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, FMT)) {} \
        OCU_FP8_FN explicit NAME(__half2 v)                                               \
            : __x(__nv_cvt_halfraw2_to_fp8x2(__half2_raw{v.x.__x, v.y.__x}, __NV_SATFINITE, FMT)) {} \
        OCU_FP8_FN explicit NAME(__nv_bfloat162 v)                                        \
            : __x(__nv_cvt_bfloat16raw2_to_fp8x2(static_cast<__nv_bfloat162_raw>(v), __NV_SATFINITE, FMT)) {} \
        OCU_FP8_FN explicit operator float2() const                                       \
        {                                                                                 \
            return float2{static_cast<float>(ocu_fp8::to_double(                          \
                              static_cast<__nv_fp8_storage_t>(__x & 0xffu), FMT)),        \
                          static_cast<float>(ocu_fp8::to_double(                          \
                              static_cast<__nv_fp8_storage_t>(__x >> 8), FMT))};          \
        }                                                                                 \
        OCU_FP8_FN explicit operator __half2() const                                      \
        {                                                                                 \
            __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(__x, FMT);                         \
            return __half2(__half(__half_raw{r.x}), __half(__half_raw{r.y}));             \
        }                                                                                 \
    };

OCU_FP8_PAIR(__nv_fp8x2_e4m3, __NV_E4M3)
OCU_FP8_PAIR(__nv_fp8x2_e5m2, __NV_E5M2)
#undef OCU_FP8_PAIR

#define OCU_FP8_QUAD(NAME, FMT)                                                           \
    struct NAME {                                                                         \
        __nv_fp8x4_storage_t __x;                                                         \
        NAME() = default;                                                                 \
        OCU_FP8_FN explicit NAME(float4 v)                                                \
            : __x(static_cast<__nv_fp8x4_storage_t>(                                      \
                  __nv_cvt_float2_to_fp8x2(float2{v.x, v.y}, __NV_SATFINITE, FMT) |       \
                  (static_cast<unsigned>(__nv_cvt_float2_to_fp8x2(float2{v.z, v.w},       \
                                                                  __NV_SATFINITE, FMT))   \
                   << 16))) {}                                                            \
        OCU_FP8_FN explicit operator float4() const                                       \
        {                                                                                 \
            float r[4];                                                                   \
            for (int i = 0; i < 4; ++i)                                                   \
                r[i] = static_cast<float>(ocu_fp8::to_double(                             \
                    static_cast<__nv_fp8_storage_t>((__x >> (8 * i)) & 0xffu), FMT));     \
            return float4{r[0], r[1], r[2], r[3]};                                        \
        }                                                                                 \
    };

OCU_FP8_QUAD(__nv_fp8x4_e4m3, __NV_E4M3)
OCU_FP8_QUAD(__nv_fp8x4_e5m2, __NV_E5M2)
#undef OCU_FP8_QUAD

#undef OCU_FP8_FN

#endif /* OCU_CUDA_FP8_H */

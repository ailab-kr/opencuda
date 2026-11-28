/*
 * cuda_fp16.h -- half precision (IEEE 754 binary16) types and functions:
 * __half, __half2 and the documented conversion, arithmetic, comparison
 * and math functions, for host code and CPU-path device code.
 *
 * Written for this project from the public description of the CUDA half
 * precision intrinsics and from IEEE 754; not derived from any vendor
 * header. C++ only.
 *
 * Rounding: conversions from float/double round to nearest even unless the
 * name says otherwise (_rz toward zero, _rd down, _ru up). Arithmetic is
 * done in a wider type and rounded once, which gives the correctly rounded
 * binary16 result (float holds the exact sum/product/quotient closely
 * enough for add, sub, mul and div; fma uses double).
 */
#ifndef OCU_CUDA_FP16_H
#define OCU_CUDA_FP16_H

#ifndef __cplusplus
#error "cuda_fp16.h needs C++"
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

/* Host and device functions in Clang's CUDA mode; plain functions otherwise. */
#if defined(__CUDA__)
#define OCU_F16 __attribute__((host)) __attribute__((device))
#else
#define OCU_F16
#endif

namespace ocu_fp16 {

enum class round { nearest_even, toward_zero, down, up };

/*
 * binary16 bits from a double, with the given rounding: integer operations
 * on the bit pattern only, so host and device give the same result.
 */
OCU_F16 inline std::uint16_t from_double(double v, round mode)
{
    std::uint64_t b = __builtin_bit_cast(std::uint64_t, v);
    std::uint16_t sign = static_cast<std::uint16_t>((b >> 48) & 0x8000u);
    bool neg = sign != 0;
    int e = static_cast<int>((b >> 52) & 0x7ff);
    std::uint64_t man = b & 0xfffffffffffffull;
    if (e == 0x7ff) return static_cast<std::uint16_t>(sign | (man ? 0x7e00u : 0x7c00u));
    if (e == 0 && man == 0) return sign;
    std::uint64_t M = e == 0 ? man : (man | (1ull << 52)); /* value = M 2^(E - 52) */
    int E = e == 0 ? -1022 : e - 1023;
    int eh = E < -14 ? -14 : E;               /* half exponent (subnormals: -14) */
    int shift = 42 + (eh - E);                /* bits of M below the half's last place */
    std::uint64_t q;
    bool round_bit, sticky;
    if (shift >= 64) {
        q = 0;
        round_bit = false;
        sticky = M != 0;
    } else {
        q = M >> shift;
        std::uint64_t rem = M & ((1ull << shift) - 1);
        round_bit = ((rem >> (shift - 1)) & 1u) != 0;
        sticky = (rem & ((1ull << (shift - 1)) - 1)) != 0;
    }
    bool up = false;
    switch (mode) {
    case round::nearest_even: up = round_bit && (sticky || (q & 1u)); break;
    case round::toward_zero:  up = false; break;
    case round::down:         up = neg && (round_bit || sticky); break;
    case round::up:           up = !neg && (round_bit || sticky); break;
    }
    q += up ? 1u : 0u;
    if (q >= 2048u) {                         /* carried into the next binade */
        q >>= 1;
        ++eh;
    }
    if (q < 1024u) return static_cast<std::uint16_t>(sign | q); /* subnormal or zero */
    int biased = eh + 15;
    if (biased >= 31) {                       /* overflow */
        bool to_inf = mode == round::nearest_even || (mode == round::up && !neg) ||
                      (mode == round::down && neg);
        return static_cast<std::uint16_t>(sign | (to_inf ? 0x7c00u : 0x7bffu));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(biased) << 10) |
                                      (q - 1024u));
}

/* Exact value of binary16 bits as float, from the bit pattern. */
OCU_F16 inline float to_float(std::uint16_t h)
{
    std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t man = h & 0x3ffu;
    std::uint32_t bits;
    if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int k = 0;                        /* shifts until the leading bit is bit 10 */
            while ((man & 0x400u) == 0) {
                man <<= 1;
                ++k;
            }
            bits = sign | (static_cast<std::uint32_t>(113 - k) << 23) | ((man & 0x3ffu) << 13);
        }
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return __builtin_bit_cast(float, bits);
}

enum class to_int { trunc, nearest, floor, ceil };

OCU_F16 inline double round_as(double x, to_int m)
{
    switch (m) {
    case to_int::trunc:   return __builtin_trunc(x);
    case to_int::nearest: return __builtin_rint(x);
    case to_int::floor:   return __builtin_floor(x);
    default:              return __builtin_ceil(x);
    }
}

} // namespace ocu_fp16

struct __half_raw {
    unsigned short x;
};

/* Two binary16 bit patterns; .x is the low half of a __half2. */
struct __half2_raw {
    unsigned short x, y;
};

struct __half {
    unsigned short __x;
    __half() = default;
    constexpr __half(__half_raw r) : __x(r.x) {}
    OCU_F16 __half(float f) : __x(ocu_fp16::from_double(f, ocu_fp16::round::nearest_even)) {}
    OCU_F16 __half(double d) : __x(ocu_fp16::from_double(d, ocu_fp16::round::nearest_even)) {}
    /* Any integer type (h = 0; __half g = n;), rounded to nearest even. */
    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    OCU_F16 __half(I i) : __x(ocu_fp16::from_double(static_cast<double>(i), ocu_fp16::round::nearest_even)) {}
    OCU_F16 operator float() const { return ocu_fp16::to_float(__x); }
    OCU_F16 operator __half_raw() const { return __half_raw{__x}; }
    OCU_F16 __half &operator=(float f) { return *this = __half(f); }
    OCU_F16 __half &operator=(double d) { return *this = __half(d); }
    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    OCU_F16 __half &operator=(I i) { return *this = __half(i); }
};

struct __half2 {
    __half x, y;
    __half2() = default;
    OCU_F16 __half2(__half a, __half b) : x(a), y(b) {}
};

typedef __half half;
typedef __half2 half2;

static_assert(sizeof(__half) == 2 && sizeof(__half2) == 4, "binary16 layout");

#define OCU_F16_FN inline __host__ __device__

/* ---- conversions ---------------------------------------------------------- */

OCU_F16_FN __half __ushort_as_half(unsigned short v) { return __half(__half_raw{v}); }
OCU_F16_FN unsigned short __half_as_ushort(__half h) { return h.__x; }
OCU_F16_FN __half __short_as_half(short v)
{
    return __ushort_as_half(__builtin_bit_cast(unsigned short, v));
}
OCU_F16_FN short __half_as_short(__half h)
{
    return __builtin_bit_cast(short, h.__x);
}
OCU_F16_FN float __half2float(__half h) { return ocu_fp16::to_float(h.__x); }

#define OCU_FROM(name, T, mode)                                                    \
    OCU_F16_FN __half name(T v)                                                        \
    {                                                                              \
        return __ushort_as_half(ocu_fp16::from_double(double(v), ocu_fp16::round::mode)); \
    }
OCU_FROM(__float2half, float, nearest_even)
OCU_FROM(__float2half_rn, float, nearest_even)
OCU_FROM(__float2half_rz, float, toward_zero)
OCU_FROM(__float2half_rd, float, down)
OCU_FROM(__float2half_ru, float, up)
OCU_FROM(__double2half, double, nearest_even)
OCU_FROM(__int2half_rn, int, nearest_even)
OCU_FROM(__int2half_rz, int, toward_zero)
OCU_FROM(__uint2half_rn, unsigned, nearest_even)
OCU_FROM(__ll2half_rn, long long, nearest_even)
OCU_FROM(__short2half_rn, short, nearest_even)
#undef OCU_FROM

/* To integers: NaN gives 0, out-of-range values saturate. */
template <typename I>
OCU_F16 inline I ocu_half_to_int(__half h, ocu_fp16::to_int mode)
{
    float f = __half2float(h);
    if (__builtin_isnan(f)) return 0;
    double r = ocu_fp16::round_as(static_cast<double>(f), mode);
    const double lo = static_cast<double>(std::numeric_limits<I>::min());
    const double hi = static_cast<double>(std::numeric_limits<I>::max());
    if (r <= lo) return std::numeric_limits<I>::min();
    if (r >= hi) return std::numeric_limits<I>::max();
    return static_cast<I>(r);
}
OCU_F16_FN int __half2int_rz(__half h) { return ocu_half_to_int<int>(h, ocu_fp16::to_int::trunc); }
OCU_F16_FN int __half2int_rn(__half h) { return ocu_half_to_int<int>(h, ocu_fp16::to_int::nearest); }
OCU_F16_FN int __half2int_rd(__half h) { return ocu_half_to_int<int>(h, ocu_fp16::to_int::floor); }
OCU_F16_FN int __half2int_ru(__half h) { return ocu_half_to_int<int>(h, ocu_fp16::to_int::ceil); }
OCU_F16_FN unsigned __half2uint_rz(__half h) { return ocu_half_to_int<unsigned>(h, ocu_fp16::to_int::trunc); }
OCU_F16_FN short __half2short_rz(__half h) { return ocu_half_to_int<short>(h, ocu_fp16::to_int::trunc); }
OCU_F16_FN long long __half2ll_rz(__half h) { return ocu_half_to_int<long long>(h, ocu_fp16::to_int::trunc); }

/* ---- arithmetic (correctly rounded) ----------------------------------------- */

OCU_F16_FN __half __hadd(__half a, __half b) { return __float2half(__half2float(a) + __half2float(b)); }
OCU_F16_FN __half __hsub(__half a, __half b) { return __float2half(__half2float(a) - __half2float(b)); }
OCU_F16_FN __half __hmul(__half a, __half b) { return __float2half(__half2float(a) * __half2float(b)); }
OCU_F16_FN __half __hdiv(__half a, __half b) { return __float2half(__half2float(a) / __half2float(b)); }
OCU_F16_FN __half __hfma(__half a, __half b, __half c)
{
    return __double2half(double(__half2float(a)) * double(__half2float(b)) + double(__half2float(c)));
}
OCU_F16_FN __half __hneg(__half a) { return __ushort_as_half(static_cast<unsigned short>(a.__x ^ 0x8000u)); }
OCU_F16_FN __half __habs(__half a) { return __ushort_as_half(static_cast<unsigned short>(a.__x & 0x7fffu)); }
OCU_F16_FN __half __hadd_sat(__half a, __half b)
{
    float r = __half2float(__hadd(a, b));
    return __float2half(__builtin_isnan(r) ? 0.0f : __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, r)));
}
OCU_F16_FN __half __hmul_sat(__half a, __half b)
{
    float r = __half2float(__hmul(a, b));
    return __float2half(__builtin_isnan(r) ? 0.0f : __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, r)));
}
/* As documented: a NaN operand is ignored when the other is a number (both
   NaN: a quiet NaN); if both are zero, max gives +0.0 and min gives -0.0. */
OCU_F16_FN __half __hmax(__half a, __half b)
{
    bool na = (a.__x & 0x7fffu) > 0x7c00u, nb = (b.__x & 0x7fffu) > 0x7c00u;
    if (na) return nb ? __ushort_as_half(0x7e00) : b;
    if (nb) return a;
    if (((a.__x | b.__x) & 0x7fffu) == 0) return __ushort_as_half(static_cast<unsigned short>(a.__x & b.__x));
    return __half2float(a) > __half2float(b) ? a : b;
}
OCU_F16_FN __half __hmin(__half a, __half b)
{
    bool na = (a.__x & 0x7fffu) > 0x7c00u, nb = (b.__x & 0x7fffu) > 0x7c00u;
    if (na) return nb ? __ushort_as_half(0x7e00) : b;
    if (nb) return a;
    if (((a.__x | b.__x) & 0x7fffu) == 0) return __ushort_as_half(static_cast<unsigned short>(a.__x | b.__x));
    return __half2float(a) < __half2float(b) ? a : b;
}

/* ---- comparisons ---------------------------------------------------------------
   The plain forms are ordered (false when either value is NaN); the forms
   ending in u are unordered (true when either value is NaN). */

OCU_F16_FN bool __hisnan(__half a) { return (a.__x & 0x7c00u) == 0x7c00u && (a.__x & 0x3ffu) != 0; }
OCU_F16_FN int __hisinf(__half a)
{
    if ((a.__x & 0x7fffu) != 0x7c00u) return 0;
    return (a.__x & 0x8000u) ? -1 : 1;
}
OCU_F16_FN bool __heq(__half a, __half b) { return __half2float(a) == __half2float(b); }
OCU_F16_FN bool __hne(__half a, __half b)
{
    return !__hisnan(a) && !__hisnan(b) && __half2float(a) != __half2float(b);
}
OCU_F16_FN bool __hlt(__half a, __half b) { return __half2float(a) < __half2float(b); }
OCU_F16_FN bool __hle(__half a, __half b) { return __half2float(a) <= __half2float(b); }
OCU_F16_FN bool __hgt(__half a, __half b) { return __half2float(a) > __half2float(b); }
OCU_F16_FN bool __hge(__half a, __half b) { return __half2float(a) >= __half2float(b); }
OCU_F16_FN bool __hequ(__half a, __half b) { return __hisnan(a) || __hisnan(b) || __heq(a, b); }
OCU_F16_FN bool __hneu(__half a, __half b) { return !__heq(a, b); }
OCU_F16_FN bool __hltu(__half a, __half b) { return !__hge(a, b); }
OCU_F16_FN bool __hleu(__half a, __half b) { return !__hgt(a, b); }
OCU_F16_FN bool __hgtu(__half a, __half b) { return !__hle(a, b); }
OCU_F16_FN bool __hgeu(__half a, __half b) { return !__hlt(a, b); }

/* ---- operators ---------------------------------------------------------------- */

OCU_F16_FN __half operator+(__half a, __half b) { return __hadd(a, b); }
OCU_F16_FN __half operator-(__half a, __half b) { return __hsub(a, b); }
OCU_F16_FN __half operator*(__half a, __half b) { return __hmul(a, b); }
OCU_F16_FN __half operator/(__half a, __half b) { return __hdiv(a, b); }
OCU_F16_FN __half operator-(__half a) { return __hneg(a); }
OCU_F16_FN __half operator+(__half a) { return a; }
OCU_F16_FN __half &operator+=(__half &a, __half b) { return a = __hadd(a, b); }
OCU_F16_FN __half &operator-=(__half &a, __half b) { return a = __hsub(a, b); }
OCU_F16_FN __half &operator*=(__half &a, __half b) { return a = __hmul(a, b); }
OCU_F16_FN __half &operator/=(__half &a, __half b) { return a = __hdiv(a, b); }
OCU_F16_FN bool operator==(__half a, __half b) { return __heq(a, b); }
OCU_F16_FN bool operator!=(__half a, __half b) { return __hneu(a, b); }
OCU_F16_FN bool operator<(__half a, __half b) { return __hlt(a, b); }
OCU_F16_FN bool operator<=(__half a, __half b) { return __hle(a, b); }
OCU_F16_FN bool operator>(__half a, __half b) { return __hgt(a, b); }
OCU_F16_FN bool operator>=(__half a, __half b) { return __hge(a, b); }

/* ---- math (through float, then rounded to half) ----------------------------- */

#define OCU_HMATH(name, expr)                                                      \
    OCU_F16_FN __half name(__half a)                                                   \
    {                                                                              \
        float x = __half2float(a);                                                 \
        return __float2half(expr);                                                 \
    }
OCU_HMATH(hsqrt, sqrtf(x))
OCU_HMATH(hrsqrt, 1.0f / sqrtf(x))
OCU_HMATH(hrcp, 1.0f / x)
OCU_HMATH(hexp, expf(x))
OCU_HMATH(hexp2, exp2f(x))
OCU_HMATH(hexp10, powf(10.0f, x))
OCU_HMATH(hlog, logf(x))
OCU_HMATH(hlog2, log2f(x))
OCU_HMATH(hlog10, log10f(x))
OCU_HMATH(hsin, sinf(x))
OCU_HMATH(hcos, cosf(x))
OCU_HMATH(htanh, tanhf(x))
OCU_HMATH(hfloor, __builtin_floorf(x))
OCU_HMATH(hceil, __builtin_ceilf(x))
OCU_HMATH(htrunc, __builtin_truncf(x))
OCU_HMATH(hrint, __builtin_rintf(x))
#undef OCU_HMATH

/* ---- __half2 ------------------------------------------------------------------ */

OCU_F16_FN __half2 __halves2half2(__half a, __half b) { return __half2(a, b); }
OCU_F16_FN __half2 __half2half2(__half a) { return __half2(a, a); }
/* make_half2(x, y): the pair (x, y); x is the low half. */
OCU_F16_FN __half2 make_half2(__half x, __half y) { return __half2(x, y); }
OCU_F16_FN __half2 __float2half2_rn(float a) { return __half2half2(__float2half_rn(a)); }
OCU_F16_FN __half2 __floats2half2_rn(float a, float b)
{
    return __half2(__float2half_rn(a), __float2half_rn(b));
}
OCU_F16_FN __half __low2half(__half2 a) { return a.x; }
OCU_F16_FN __half __high2half(__half2 a) { return a.y; }
OCU_F16_FN float __low2float(__half2 a) { return __half2float(a.x); }
OCU_F16_FN float __high2float(__half2 a) { return __half2float(a.y); }
OCU_F16_FN __half2 __lowhigh2highlow(__half2 a) { return __half2(a.y, a.x); }
OCU_F16_FN __half2 __lows2half2(__half2 a, __half2 b) { return __half2(a.x, b.x); }
OCU_F16_FN __half2 __highs2half2(__half2 a, __half2 b) { return __half2(a.y, b.y); }

#ifdef OCU_VECTOR_TYPES_H
OCU_F16_FN float2 __half22float2(__half2 a) { return make_float2(__half2float(a.x), __half2float(a.y)); }
OCU_F16_FN __half2 __float22half2_rn(float2 a) { return __floats2half2_rn(a.x, a.y); }
#endif

#define OCU_H2(name, op)                                                           \
    OCU_F16_FN __half2 name(__half2 a, __half2 b) { return __half2(op(a.x, b.x), op(a.y, b.y)); }
OCU_H2(__hadd2, __hadd)
OCU_H2(__hsub2, __hsub)
OCU_H2(__hmul2, __hmul)
OCU_H2(__h2div, __hdiv)
OCU_H2(__hmax2, __hmax)
OCU_H2(__hmin2, __hmin)
#undef OCU_H2
OCU_F16_FN __half2 __hfma2(__half2 a, __half2 b, __half2 c)
{
    return __half2(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y));
}
OCU_F16_FN __half2 __hneg2(__half2 a) { return __half2(__hneg(a.x), __hneg(a.y)); }
OCU_F16_FN __half2 __habs2(__half2 a) { return __half2(__habs(a.x), __habs(a.y)); }
OCU_F16_FN __half2 operator+(__half2 a, __half2 b) { return __hadd2(a, b); }
OCU_F16_FN __half2 operator-(__half2 a, __half2 b) { return __hsub2(a, b); }
OCU_F16_FN __half2 operator*(__half2 a, __half2 b) { return __hmul2(a, b); }
OCU_F16_FN __half2 operator/(__half2 a, __half2 b) { return __h2div(a, b); }
OCU_F16_FN __half2 operator-(__half2 a) { return __hneg2(a); }
OCU_F16_FN __half2 operator+(__half2 a) { return a; }
OCU_F16_FN __half2 &operator+=(__half2 &a, __half2 b) { return a = __hadd2(a, b); }
OCU_F16_FN __half2 &operator-=(__half2 &a, __half2 b) { return a = __hsub2(a, b); }
OCU_F16_FN __half2 &operator*=(__half2 &a, __half2 b) { return a = __hmul2(a, b); }
OCU_F16_FN __half2 &operator/=(__half2 &a, __half2 b) { return a = __h2div(a, b); }

/* __half2 comparisons, lane by lane (CUDA Math API, half2 comparison
   functions): __heq2 ... give 1.0 or 0.0 in each half, the _mask forms
   0xFFFF or 0 in each 16-bit half (low lane in the low bits), __hbeq2 ...
   true when both lanes compare true. The u forms are unordered (true when
   either value is NaN). Operators ==, != ... on __half2 are the __hb forms. */
#define OCU_H2CMP(name, op)                                                                     \
    OCU_F16_FN __half2 __h##name##2(__half2 a, __half2 b)                                       \
    {                                                                                           \
        return __half2(__float2half(op(a.x, b.x) ? 1.0f : 0.0f), __float2half(op(a.y, b.y) ? 1.0f : 0.0f)); \
    }                                                                                           \
    OCU_F16_FN unsigned __h##name##2_mask(__half2 a, __half2 b)                                 \
    {                                                                                           \
        return (op(a.x, b.x) ? 0xffffu : 0u) | (op(a.y, b.y) ? 0xffff0000u : 0u);               \
    }                                                                                           \
    OCU_F16_FN bool __hb##name##2(__half2 a, __half2 b) { return op(a.x, b.x) && op(a.y, b.y); }
OCU_H2CMP(eq, __heq)
OCU_H2CMP(ne, __hne)
OCU_H2CMP(lt, __hlt)
OCU_H2CMP(le, __hle)
OCU_H2CMP(gt, __hgt)
OCU_H2CMP(ge, __hge)
OCU_H2CMP(equ, __hequ)
OCU_H2CMP(neu, __hneu)
OCU_H2CMP(ltu, __hltu)
OCU_H2CMP(leu, __hleu)
OCU_H2CMP(gtu, __hgtu)
OCU_H2CMP(geu, __hgeu)
#undef OCU_H2CMP
OCU_F16_FN __half2 __hisnan2(__half2 a)
{
    return __half2(__float2half(__hisnan(a.x) ? 1.0f : 0.0f), __float2half(__hisnan(a.y) ? 1.0f : 0.0f));
}
OCU_F16_FN bool operator==(__half2 a, __half2 b) { return __hbeq2(a, b); }
OCU_F16_FN bool operator!=(__half2 a, __half2 b) { return __hbneu2(a, b); }
OCU_F16_FN bool operator<(__half2 a, __half2 b) { return __hblt2(a, b); }
OCU_F16_FN bool operator<=(__half2 a, __half2 b) { return __hble2(a, b); }
OCU_F16_FN bool operator>(__half2 a, __half2 b) { return __hbgt2(a, b); }
OCU_F16_FN bool operator>=(__half2 a, __half2 b) { return __hbge2(a, b); }

#undef OCU_F16_FN

/* atomicAdd(__half *), atomicAdd(__half2 *) when cuda_runtime.h came first */
#include "ocu_half_atomics.h"

#endif /* OCU_CUDA_FP16_H */

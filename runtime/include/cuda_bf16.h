/*
 * cuda_bf16.h -- bfloat16 types and functions: __nv_bfloat16 (nv_bfloat16),
 * __nv_bfloat162 (nv_bfloat162) and the documented conversion, arithmetic,
 * comparison and pair functions, for host code and CPU-path device code.
 *
 * Written for this project from the public description of the CUDA
 * bfloat16 intrinsics (CUDA Math API, "Bfloat16 Precision Intrinsics") and
 * from the bfloat16 format definition (the upper 16 bits of an IEEE 754
 * binary32); not derived from any vendor header. C++ only.
 *
 * Rounding: conversions from float/double round to nearest even unless the
 * name says otherwise (_rz toward zero, _rd down, _ru up). Conversion to
 * float is exact. Arithmetic is done in float and rounded once: float has
 * more than twice the significand bits of bfloat16 (24 >= 2*8+2), so for
 * add, sub, mul, div and fma the rounded-twice result equals the correctly
 * rounded one. NaN results are quiet NaNs; the sign and payload of a NaN
 * input are not preserved beyond the quiet bit.
 */
#ifndef OCU_CUDA_BF16_H
#define OCU_CUDA_BF16_H

#ifndef __cplusplus
#error "cuda_bf16.h needs C++"
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#if defined(__CUDA__)
#define OCU_BF16 __attribute__((host)) __attribute__((device))
#else
#define OCU_BF16
#endif

namespace ocu_bf16 {

enum class round { nearest_even, toward_zero, down, up };

/* Exact value of bfloat16 bits as float. */
OCU_BF16 inline float to_float(std::uint16_t h)
{
    return __builtin_bit_cast(float, static_cast<std::uint32_t>(h) << 16);
}

/* bfloat16 bits from float bits, with the given rounding. */
OCU_BF16 inline std::uint16_t from_float(float f, round mode)
{
    std::uint32_t u = __builtin_bit_cast(std::uint32_t, f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {                 /* NaN: keep it a quiet NaN */
        return static_cast<std::uint16_t>((u >> 16) | 0x40u);
    }
    std::uint32_t hi = u >> 16, low = u & 0xffffu;
    bool neg = (u >> 31) != 0;
    bool up = false;
    switch (mode) {
    case round::nearest_even: up = low > 0x8000u || (low == 0x8000u && (hi & 1u)); break;
    case round::toward_zero:  up = false; break;
    case round::down:         up = neg && low != 0; break;
    case round::up:           up = !neg && low != 0; break;
    }
    /* Carry into the exponent is the right result; into infinity too. */
    return static_cast<std::uint16_t>(hi + (up ? 1u : 0u));
}

/*
 * bfloat16 bits from a double, rounded to nearest even once. The double is
 * first reduced to a float by round-to-odd (inexact results get an odd last
 * bit), which keeps every bit that matters for the final rounding.
 */
OCU_BF16 inline std::uint16_t from_double(double d)
{
    float f = static_cast<float>(d);                        /* nearest */
    if (d == d && !__builtin_isinf(d) && static_cast<double>(f) != d) {
        /* Inexact: take the float on the zero side of d (a finite
           neighbour, also when f overflowed to infinity) and set its last
           bit. */
        double fd = static_cast<double>(f);
        bool beyond = d > 0 ? fd > d : fd < d;
        std::uint32_t bits = __builtin_bit_cast(std::uint32_t, f);
        if (beyond) bits -= 1u;                             /* one step toward zero */
        f = __builtin_bit_cast(float, bits | 1u);
    }
    return from_float(f, round::nearest_even);
}

} // namespace ocu_bf16

struct __nv_bfloat16_raw {
    unsigned short x;
};

struct __nv_bfloat162_raw {
    unsigned short x;
    unsigned short y;
};

struct __nv_bfloat16 {
    unsigned short __x;
    __nv_bfloat16() = default;
    constexpr __nv_bfloat16(__nv_bfloat16_raw r) : __x(r.x) {}
    OCU_BF16 __nv_bfloat16(float f) : __x(ocu_bf16::from_float(f, ocu_bf16::round::nearest_even)) {}
    OCU_BF16 __nv_bfloat16(double d) : __x(ocu_bf16::from_double(d)) {}
    /* Any integer type, rounded to nearest even (through double: exact for 53 bits). */
    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    OCU_BF16 __nv_bfloat16(I i) : __x(ocu_bf16::from_double(static_cast<double>(i))) {}
    OCU_BF16 operator float() const { return ocu_bf16::to_float(__x); }
    OCU_BF16 operator __nv_bfloat16_raw() const { return __nv_bfloat16_raw{__x}; }
    OCU_BF16 __nv_bfloat16 &operator=(float f) { return *this = __nv_bfloat16(f); }
    OCU_BF16 __nv_bfloat16 &operator=(double d) { return *this = __nv_bfloat16(d); }
    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    OCU_BF16 __nv_bfloat16 &operator=(I i) { return *this = __nv_bfloat16(i); }
};

struct alignas(4) __nv_bfloat162 {
    __nv_bfloat16 x, y;
    __nv_bfloat162() = default;
    OCU_BF16 __nv_bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) : x(a), y(b) {}
    constexpr __nv_bfloat162(__nv_bfloat162_raw r) : x(__nv_bfloat16_raw{r.x}), y(__nv_bfloat16_raw{r.y}) {}
    OCU_BF16 operator __nv_bfloat162_raw() const { return __nv_bfloat162_raw{x.__x, y.__x}; }
};

typedef __nv_bfloat16 nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

static_assert(sizeof(__nv_bfloat16) == 2 && sizeof(__nv_bfloat162) == 4, "bfloat16 layout");
static_assert(alignof(__nv_bfloat162) == 4, "bfloat162 alignment");

#define OCU_BF16_FN inline __host__ __device__

/* ---- conversions ---------------------------------------------------------- */

OCU_BF16_FN __nv_bfloat16 __ushort_as_bfloat16(unsigned short v) { return __nv_bfloat16(__nv_bfloat16_raw{v}); }
OCU_BF16_FN unsigned short __bfloat16_as_ushort(__nv_bfloat16 h) { return h.__x; }
OCU_BF16_FN __nv_bfloat16 __short_as_bfloat16(short v)
{
    return __ushort_as_bfloat16(__builtin_bit_cast(unsigned short, v));
}
OCU_BF16_FN short __bfloat16_as_short(__nv_bfloat16 h) { return __builtin_bit_cast(short, h.__x); }
OCU_BF16_FN float __bfloat162float(__nv_bfloat16 h) { return ocu_bf16::to_float(h.__x); }

OCU_BF16_FN __nv_bfloat16 __float2bfloat16(float f)
{
    return __ushort_as_bfloat16(ocu_bf16::from_float(f, ocu_bf16::round::nearest_even));
}
OCU_BF16_FN __nv_bfloat16 __float2bfloat16_rn(float f) { return __float2bfloat16(f); }
OCU_BF16_FN __nv_bfloat16 __float2bfloat16_rz(float f)
{
    return __ushort_as_bfloat16(ocu_bf16::from_float(f, ocu_bf16::round::toward_zero));
}
OCU_BF16_FN __nv_bfloat16 __float2bfloat16_rd(float f)
{
    return __ushort_as_bfloat16(ocu_bf16::from_float(f, ocu_bf16::round::down));
}
OCU_BF16_FN __nv_bfloat16 __float2bfloat16_ru(float f)
{
    return __ushort_as_bfloat16(ocu_bf16::from_float(f, ocu_bf16::round::up));
}
OCU_BF16_FN __nv_bfloat16 __double2bfloat16(double d) { return __ushort_as_bfloat16(ocu_bf16::from_double(d)); }
OCU_BF16_FN __nv_bfloat16 __int2bfloat16_rn(int v) { return __double2bfloat16(static_cast<double>(v)); }
OCU_BF16_FN __nv_bfloat16 __uint2bfloat16_rn(unsigned v) { return __double2bfloat16(static_cast<double>(v)); }
OCU_BF16_FN __nv_bfloat16 __short2bfloat16_rn(short v) { return __double2bfloat16(static_cast<double>(v)); }
OCU_BF16_FN __nv_bfloat16 __ll2bfloat16_rn(long long v) { return __double2bfloat16(static_cast<double>(v)); }

/* To integers: NaN gives 0, out-of-range values saturate. */
OCU_BF16_FN int __bfloat162int_rz(__nv_bfloat16 h)
{
    float f = __bfloat162float(h);
    if (f != f) return 0;
    if (f >= 2147483648.0f) return 2147483647;
    if (f <= -2147483648.0f) return -2147483647 - 1;
    return static_cast<int>(f);
}
OCU_BF16_FN int __bfloat162int_rn(__nv_bfloat16 h)
{
    float f = __builtin_rintf(__bfloat162float(h));
    if (f != f) return 0;
    if (f >= 2147483648.0f) return 2147483647;
    if (f <= -2147483648.0f) return -2147483647 - 1;
    return static_cast<int>(f);
}

/* ---- arithmetic (correctly rounded) ----------------------------------------- */

OCU_BF16_FN __nv_bfloat16 __hadd(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}
OCU_BF16_FN __nv_bfloat16 __hsub(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __float2bfloat16(__bfloat162float(a) - __bfloat162float(b));
}
OCU_BF16_FN __nv_bfloat16 __hmul(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}
OCU_BF16_FN __nv_bfloat16 __hdiv(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __float2bfloat16(__bfloat162float(a) / __bfloat162float(b));
}
OCU_BF16_FN __nv_bfloat16 __hfma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c)
{
    return __float2bfloat16(__builtin_fmaf(__bfloat162float(a), __bfloat162float(b), __bfloat162float(c)));
}
OCU_BF16_FN __nv_bfloat16 __hneg(__nv_bfloat16 a) { return __ushort_as_bfloat16(static_cast<unsigned short>(a.__x ^ 0x8000u)); }
OCU_BF16_FN __nv_bfloat16 __habs(__nv_bfloat16 a) { return __ushort_as_bfloat16(static_cast<unsigned short>(a.__x & 0x7fffu)); }
OCU_BF16_FN __nv_bfloat16 __hadd_sat(__nv_bfloat16 a, __nv_bfloat16 b)
{
    float r = __bfloat162float(__hadd(a, b));
    return __float2bfloat16(r != r ? 0.0f : __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, r)));
}
OCU_BF16_FN __nv_bfloat16 __hmul_sat(__nv_bfloat16 a, __nv_bfloat16 b)
{
    float r = __bfloat162float(__hmul(a, b));
    return __float2bfloat16(r != r ? 0.0f : __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, r)));
}
/* A NaN operand is ignored if the other is a number (like fmaxf). */
OCU_BF16_FN __nv_bfloat16 __hmax(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __float2bfloat16(__builtin_fmaxf(__bfloat162float(a), __bfloat162float(b)));
}
OCU_BF16_FN __nv_bfloat16 __hmin(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __float2bfloat16(__builtin_fminf(__bfloat162float(a), __bfloat162float(b)));
}

/* ---- comparisons ---------------------------------------------------------------
   The plain forms are ordered (false when either value is NaN); the forms
   ending in u are unordered (true when either value is NaN). */

OCU_BF16_FN bool __hisnan(__nv_bfloat16 a) { return (a.__x & 0x7fffu) > 0x7f80u; }
OCU_BF16_FN int __hisinf(__nv_bfloat16 a)
{
    if ((a.__x & 0x7fffu) != 0x7f80u) return 0;
    return (a.__x & 0x8000u) ? -1 : 1;
}
OCU_BF16_FN bool __heq(__nv_bfloat16 a, __nv_bfloat16 b) { return __bfloat162float(a) == __bfloat162float(b); }
OCU_BF16_FN bool __hne(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return !__hisnan(a) && !__hisnan(b) && __bfloat162float(a) != __bfloat162float(b);
}
OCU_BF16_FN bool __hlt(__nv_bfloat16 a, __nv_bfloat16 b) { return __bfloat162float(a) < __bfloat162float(b); }
OCU_BF16_FN bool __hle(__nv_bfloat16 a, __nv_bfloat16 b) { return __bfloat162float(a) <= __bfloat162float(b); }
OCU_BF16_FN bool __hgt(__nv_bfloat16 a, __nv_bfloat16 b) { return __bfloat162float(a) > __bfloat162float(b); }
OCU_BF16_FN bool __hge(__nv_bfloat16 a, __nv_bfloat16 b) { return __bfloat162float(a) >= __bfloat162float(b); }
OCU_BF16_FN bool __hequ(__nv_bfloat16 a, __nv_bfloat16 b) { return __hisnan(a) || __hisnan(b) || __heq(a, b); }
OCU_BF16_FN bool __hneu(__nv_bfloat16 a, __nv_bfloat16 b) { return !__heq(a, b); }
OCU_BF16_FN bool __hltu(__nv_bfloat16 a, __nv_bfloat16 b) { return !__hge(a, b); }
OCU_BF16_FN bool __hleu(__nv_bfloat16 a, __nv_bfloat16 b) { return !__hgt(a, b); }
OCU_BF16_FN bool __hgtu(__nv_bfloat16 a, __nv_bfloat16 b) { return !__hle(a, b); }
OCU_BF16_FN bool __hgeu(__nv_bfloat16 a, __nv_bfloat16 b) { return !__hlt(a, b); }

/* ---- operators ---------------------------------------------------------------- */

OCU_BF16_FN __nv_bfloat16 operator+(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
OCU_BF16_FN __nv_bfloat16 operator-(__nv_bfloat16 a, __nv_bfloat16 b) { return __hsub(a, b); }
OCU_BF16_FN __nv_bfloat16 operator*(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
OCU_BF16_FN __nv_bfloat16 operator/(__nv_bfloat16 a, __nv_bfloat16 b) { return __hdiv(a, b); }
OCU_BF16_FN __nv_bfloat16 operator-(__nv_bfloat16 a) { return __hneg(a); }
OCU_BF16_FN __nv_bfloat16 operator+(__nv_bfloat16 a) { return a; }
OCU_BF16_FN __nv_bfloat16 &operator+=(__nv_bfloat16 &a, __nv_bfloat16 b) { return a = __hadd(a, b); }
OCU_BF16_FN __nv_bfloat16 &operator-=(__nv_bfloat16 &a, __nv_bfloat16 b) { return a = __hsub(a, b); }
OCU_BF16_FN __nv_bfloat16 &operator*=(__nv_bfloat16 &a, __nv_bfloat16 b) { return a = __hmul(a, b); }
OCU_BF16_FN __nv_bfloat16 &operator/=(__nv_bfloat16 &a, __nv_bfloat16 b) { return a = __hdiv(a, b); }
OCU_BF16_FN bool operator==(__nv_bfloat16 a, __nv_bfloat16 b) { return __heq(a, b); }
OCU_BF16_FN bool operator!=(__nv_bfloat16 a, __nv_bfloat16 b) { return __hneu(a, b); }
OCU_BF16_FN bool operator<(__nv_bfloat16 a, __nv_bfloat16 b) { return __hlt(a, b); }
OCU_BF16_FN bool operator<=(__nv_bfloat16 a, __nv_bfloat16 b) { return __hle(a, b); }
OCU_BF16_FN bool operator>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hgt(a, b); }
OCU_BF16_FN bool operator>=(__nv_bfloat16 a, __nv_bfloat16 b) { return __hge(a, b); }

/* ---- math (through float, then rounded to bfloat16) ------------------------- */

#define OCU_BMATH(name, expr)                                                      \
    OCU_BF16_FN __nv_bfloat16 name(__nv_bfloat16 a)                                \
    {                                                                              \
        float x = __bfloat162float(a);                                             \
        return __float2bfloat16(expr);                                             \
    }
OCU_BMATH(hsqrt, sqrtf(x))
OCU_BMATH(hrsqrt, 1.0f / sqrtf(x))
OCU_BMATH(hrcp, 1.0f / x)
OCU_BMATH(hexp, expf(x))
OCU_BMATH(hexp2, exp2f(x))
OCU_BMATH(hlog, logf(x))
OCU_BMATH(hlog2, log2f(x))
OCU_BMATH(hsin, sinf(x))
OCU_BMATH(hcos, cosf(x))
OCU_BMATH(hfloor, __builtin_floorf(x))
OCU_BMATH(hceil, __builtin_ceilf(x))
OCU_BMATH(htrunc, __builtin_truncf(x))
OCU_BMATH(hrint, __builtin_rintf(x))
#undef OCU_BMATH

/* ---- __nv_bfloat162 ----------------------------------------------------------- */

OCU_BF16_FN __nv_bfloat162 make_bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat162(a, b); }
OCU_BF16_FN __nv_bfloat162 __halves2bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat162(a, b); }
OCU_BF16_FN __nv_bfloat162 __bfloat162bfloat162(__nv_bfloat16 a) { return __nv_bfloat162(a, a); }
OCU_BF16_FN __nv_bfloat162 __float2bfloat162_rn(float a) { return __bfloat162bfloat162(__float2bfloat16_rn(a)); }
OCU_BF16_FN __nv_bfloat162 __floats2bfloat162_rn(float a, float b)
{
    return __nv_bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(b));
}
OCU_BF16_FN __nv_bfloat16 __low2bfloat16(__nv_bfloat162 a) { return a.x; }
OCU_BF16_FN __nv_bfloat16 __high2bfloat16(__nv_bfloat162 a) { return a.y; }
OCU_BF16_FN __nv_bfloat162 __low2bfloat162(__nv_bfloat162 a) { return __nv_bfloat162(a.x, a.x); }
OCU_BF16_FN __nv_bfloat162 __high2bfloat162(__nv_bfloat162 a) { return __nv_bfloat162(a.y, a.y); }
OCU_BF16_FN float __low2float(__nv_bfloat162 a) { return __bfloat162float(a.x); }
OCU_BF16_FN float __high2float(__nv_bfloat162 a) { return __bfloat162float(a.y); }
OCU_BF16_FN __nv_bfloat162 __lowhigh2highlow(__nv_bfloat162 a) { return __nv_bfloat162(a.y, a.x); }
OCU_BF16_FN __nv_bfloat162 __lows2bfloat162(__nv_bfloat162 a, __nv_bfloat162 b) { return __nv_bfloat162(a.x, b.x); }
OCU_BF16_FN __nv_bfloat162 __highs2bfloat162(__nv_bfloat162 a, __nv_bfloat162 b) { return __nv_bfloat162(a.y, b.y); }

#ifdef OCU_VECTOR_TYPES_H
OCU_BF16_FN float2 __bfloat1622float2(__nv_bfloat162 a)
{
    return make_float2(__bfloat162float(a.x), __bfloat162float(a.y));
}
OCU_BF16_FN __nv_bfloat162 __float22bfloat162_rn(float2 a) { return __floats2bfloat162_rn(a.x, a.y); }
#endif

#define OCU_B2(name, op)                                                           \
    OCU_BF16_FN __nv_bfloat162 name(__nv_bfloat162 a, __nv_bfloat162 b)            \
    {                                                                              \
        return __nv_bfloat162(op(a.x, b.x), op(a.y, b.y));                         \
    }
OCU_B2(__hadd2, __hadd)
OCU_B2(__hsub2, __hsub)
OCU_B2(__hmul2, __hmul)
OCU_B2(__h2div, __hdiv)
OCU_B2(__hmax2, __hmax)
OCU_B2(__hmin2, __hmin)
#undef OCU_B2
OCU_BF16_FN __nv_bfloat162 __hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c)
{
    return __nv_bfloat162(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y));
}
OCU_BF16_FN __nv_bfloat162 __hneg2(__nv_bfloat162 a) { return __nv_bfloat162(__hneg(a.x), __hneg(a.y)); }
OCU_BF16_FN __nv_bfloat162 __habs2(__nv_bfloat162 a) { return __nv_bfloat162(__habs(a.x), __habs(a.y)); }
OCU_BF16_FN __nv_bfloat162 operator+(__nv_bfloat162 a, __nv_bfloat162 b) { return __hadd2(a, b); }
OCU_BF16_FN __nv_bfloat162 operator-(__nv_bfloat162 a, __nv_bfloat162 b) { return __hsub2(a, b); }
OCU_BF16_FN __nv_bfloat162 operator*(__nv_bfloat162 a, __nv_bfloat162 b) { return __hmul2(a, b); }
OCU_BF16_FN __nv_bfloat162 operator/(__nv_bfloat162 a, __nv_bfloat162 b) { return __h2div(a, b); }
OCU_BF16_FN __nv_bfloat162 &operator+=(__nv_bfloat162 &a, __nv_bfloat162 b) { return a = __hadd2(a, b); }
OCU_BF16_FN __nv_bfloat162 &operator-=(__nv_bfloat162 &a, __nv_bfloat162 b) { return a = __hsub2(a, b); }
OCU_BF16_FN __nv_bfloat162 &operator*=(__nv_bfloat162 &a, __nv_bfloat162 b) { return a = __hmul2(a, b); }

#undef OCU_BF16_FN

#endif /* OCU_CUDA_BF16_H */

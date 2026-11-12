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

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

namespace ocu_fp16 {

enum class round { nearest_even, toward_zero, down, up };

/* binary16 bits from a double, with the given rounding. */
inline std::uint16_t from_double(double v, round mode)
{
    std::uint16_t sign = std::signbit(v) ? 0x8000u : 0u;
    if (std::isnan(v)) return static_cast<std::uint16_t>(sign | 0x7e00u);
    double a = std::fabs(v);
    bool neg = sign != 0;
    if (std::isinf(a)) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (a == 0.0) return sign;
    // Scale so that one unit in the last place of the result is 1.
    int e;
    std::frexp(a, &e);          // a = m * 2^e, 0.5 <= m < 1
    int exp = e - 1;            // a = 1.f * 2^exp
    if (exp < -14) exp = -14;   // subnormal range: fixed scale 2^-24
    double scaled = std::ldexp(a, 10 - exp); // mantissa with 10 fraction bits
    double whole = std::floor(scaled);
    double frac = scaled - whole;
    bool up = false;
    switch (mode) {
    case round::nearest_even:
        up = frac > 0.5 || (frac == 0.5 && std::fmod(whole, 2.0) != 0.0);
        break;
    case round::toward_zero:
        up = false;
        break;
    case round::down:
        up = neg && frac > 0.0;
        break;
    case round::up:
        up = !neg && frac > 0.0;
        break;
    }
    std::uint32_t m = static_cast<std::uint32_t>(whole) + (up ? 1u : 0u);
    // m is in [2^10, 2^11] for normal numbers, [0, 2^10] for subnormals.
    if (m >= 2048u) {           // rounding carried into the next binade
        m >>= 1;
        ++exp;
    }
    if (m < 1024u) {            // subnormal (or rounded to zero)
        return static_cast<std::uint16_t>(sign | m);
    }
    int biased = exp + 15;
    if (biased >= 31) {         // overflow
        bool to_inf = mode == round::nearest_even || (mode == round::up && !neg) ||
                      (mode == round::down && neg);
        return static_cast<std::uint16_t>(sign | (to_inf ? 0x7c00u : 0x7bffu));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(biased) << 10) |
                                      (m - 1024u));
}

/* Exact value of binary16 bits as float. */
inline float to_float(std::uint16_t h)
{
    std::uint32_t sign = (h & 0x8000u) ? 1u : 0u;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t man = h & 0x3ffu;
    float v;
    if (exp == 0) {
        v = std::ldexp(static_cast<float>(man), -24);
    } else if (exp == 31) {
        v = man ? std::nanf("") : INFINITY;
    } else {
        v = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
    }
    return sign ? -v : v;
}

} // namespace ocu_fp16

struct __half_raw {
    unsigned short x;
};

struct __half {
    unsigned short __x;
    __half() = default;
    constexpr __half(__half_raw r) : __x(r.x) {}
    __half(float f) : __x(ocu_fp16::from_double(f, ocu_fp16::round::nearest_even)) {}
    __half(double d) : __x(ocu_fp16::from_double(d, ocu_fp16::round::nearest_even)) {}
    __half(int i) : __x(ocu_fp16::from_double(i, ocu_fp16::round::nearest_even)) {}
    __half(unsigned i) : __x(ocu_fp16::from_double(i, ocu_fp16::round::nearest_even)) {}
    __half(long long i) : __x(ocu_fp16::from_double(double(i), ocu_fp16::round::nearest_even)) {}
    operator float() const { return ocu_fp16::to_float(__x); }
    operator __half_raw() const { return __half_raw{__x}; }
    __half &operator=(float f) { return *this = __half(f); }
    __half &operator=(double d) { return *this = __half(d); }
};

struct __half2 {
    __half x, y;
    __half2() = default;
    __half2(__half a, __half b) : x(a), y(b) {}
};

typedef __half half;
typedef __half2 half2;

static_assert(sizeof(__half) == 2 && sizeof(__half2) == 4, "binary16 layout");

#define OCU_HD inline __host__ __device__

/* ---- conversions ---------------------------------------------------------- */

OCU_HD __half __ushort_as_half(unsigned short v) { return __half(__half_raw{v}); }
OCU_HD unsigned short __half_as_ushort(__half h) { return h.__x; }
OCU_HD __half __short_as_half(short v)
{
    unsigned short u;
    std::memcpy(&u, &v, sizeof(u));
    return __ushort_as_half(u);
}
OCU_HD short __half_as_short(__half h)
{
    short s;
    std::memcpy(&s, &h.__x, sizeof(s));
    return s;
}
OCU_HD float __half2float(__half h) { return ocu_fp16::to_float(h.__x); }

#define OCU_FROM(name, T, mode)                                                    \
    OCU_HD __half name(T v)                                                        \
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
inline I ocu_half_to_int(__half h, double (*rnd)(double))
{
    float f = __half2float(h);
    if (std::isnan(f)) return 0;
    double r = rnd(static_cast<double>(f));
    const double lo = static_cast<double>(std::numeric_limits<I>::min());
    const double hi = static_cast<double>(std::numeric_limits<I>::max());
    if (r <= lo) return std::numeric_limits<I>::min();
    if (r >= hi) return std::numeric_limits<I>::max();
    return static_cast<I>(r);
}
OCU_HD int __half2int_rz(__half h) { return ocu_half_to_int<int>(h, std::trunc); }
OCU_HD int __half2int_rn(__half h) { return ocu_half_to_int<int>(h, std::nearbyint); }
OCU_HD int __half2int_rd(__half h) { return ocu_half_to_int<int>(h, std::floor); }
OCU_HD int __half2int_ru(__half h) { return ocu_half_to_int<int>(h, std::ceil); }
OCU_HD unsigned __half2uint_rz(__half h) { return ocu_half_to_int<unsigned>(h, std::trunc); }
OCU_HD short __half2short_rz(__half h) { return ocu_half_to_int<short>(h, std::trunc); }
OCU_HD long long __half2ll_rz(__half h) { return ocu_half_to_int<long long>(h, std::trunc); }

/* ---- arithmetic (correctly rounded) ----------------------------------------- */

OCU_HD __half __hadd(__half a, __half b) { return __float2half(__half2float(a) + __half2float(b)); }
OCU_HD __half __hsub(__half a, __half b) { return __float2half(__half2float(a) - __half2float(b)); }
OCU_HD __half __hmul(__half a, __half b) { return __float2half(__half2float(a) * __half2float(b)); }
OCU_HD __half __hdiv(__half a, __half b) { return __float2half(__half2float(a) / __half2float(b)); }
OCU_HD __half __hfma(__half a, __half b, __half c)
{
    return __double2half(double(__half2float(a)) * double(__half2float(b)) + double(__half2float(c)));
}
OCU_HD __half __hneg(__half a) { return __ushort_as_half(static_cast<unsigned short>(a.__x ^ 0x8000u)); }
OCU_HD __half __habs(__half a) { return __ushort_as_half(static_cast<unsigned short>(a.__x & 0x7fffu)); }
OCU_HD __half __hadd_sat(__half a, __half b)
{
    float r = __half2float(__hadd(a, b));
    return __float2half(std::isnan(r) ? 0.0f : std::fmin(1.0f, std::fmax(0.0f, r)));
}
OCU_HD __half __hmul_sat(__half a, __half b)
{
    float r = __half2float(__hmul(a, b));
    return __float2half(std::isnan(r) ? 0.0f : std::fmin(1.0f, std::fmax(0.0f, r)));
}
OCU_HD __half __hmax(__half a, __half b)
{
    return __float2half(std::fmax(__half2float(a), __half2float(b)));
}
OCU_HD __half __hmin(__half a, __half b)
{
    return __float2half(std::fmin(__half2float(a), __half2float(b)));
}

/* ---- comparisons ---------------------------------------------------------------
   The plain forms are ordered (false when either value is NaN); the forms
   ending in u are unordered (true when either value is NaN). */

OCU_HD bool __hisnan(__half a) { return (a.__x & 0x7c00u) == 0x7c00u && (a.__x & 0x3ffu) != 0; }
OCU_HD int __hisinf(__half a)
{
    if ((a.__x & 0x7fffu) != 0x7c00u) return 0;
    return (a.__x & 0x8000u) ? -1 : 1;
}
OCU_HD bool __heq(__half a, __half b) { return __half2float(a) == __half2float(b); }
OCU_HD bool __hne(__half a, __half b)
{
    return !__hisnan(a) && !__hisnan(b) && __half2float(a) != __half2float(b);
}
OCU_HD bool __hlt(__half a, __half b) { return __half2float(a) < __half2float(b); }
OCU_HD bool __hle(__half a, __half b) { return __half2float(a) <= __half2float(b); }
OCU_HD bool __hgt(__half a, __half b) { return __half2float(a) > __half2float(b); }
OCU_HD bool __hge(__half a, __half b) { return __half2float(a) >= __half2float(b); }
OCU_HD bool __hequ(__half a, __half b) { return __hisnan(a) || __hisnan(b) || __heq(a, b); }
OCU_HD bool __hneu(__half a, __half b) { return !__heq(a, b); }
OCU_HD bool __hltu(__half a, __half b) { return !__hge(a, b); }
OCU_HD bool __hleu(__half a, __half b) { return !__hgt(a, b); }
OCU_HD bool __hgtu(__half a, __half b) { return !__hle(a, b); }
OCU_HD bool __hgeu(__half a, __half b) { return !__hlt(a, b); }

/* ---- operators ---------------------------------------------------------------- */

OCU_HD __half operator+(__half a, __half b) { return __hadd(a, b); }
OCU_HD __half operator-(__half a, __half b) { return __hsub(a, b); }
OCU_HD __half operator*(__half a, __half b) { return __hmul(a, b); }
OCU_HD __half operator/(__half a, __half b) { return __hdiv(a, b); }
OCU_HD __half operator-(__half a) { return __hneg(a); }
OCU_HD __half operator+(__half a) { return a; }
OCU_HD __half &operator+=(__half &a, __half b) { return a = __hadd(a, b); }
OCU_HD __half &operator-=(__half &a, __half b) { return a = __hsub(a, b); }
OCU_HD __half &operator*=(__half &a, __half b) { return a = __hmul(a, b); }
OCU_HD __half &operator/=(__half &a, __half b) { return a = __hdiv(a, b); }
OCU_HD bool operator==(__half a, __half b) { return __heq(a, b); }
OCU_HD bool operator!=(__half a, __half b) { return __hneu(a, b); }
OCU_HD bool operator<(__half a, __half b) { return __hlt(a, b); }
OCU_HD bool operator<=(__half a, __half b) { return __hle(a, b); }
OCU_HD bool operator>(__half a, __half b) { return __hgt(a, b); }
OCU_HD bool operator>=(__half a, __half b) { return __hge(a, b); }

/* ---- math (through float, then rounded to half) ----------------------------- */

#define OCU_HMATH(name, expr)                                                      \
    OCU_HD __half name(__half a)                                                   \
    {                                                                              \
        float x = __half2float(a);                                                 \
        return __float2half(expr);                                                 \
    }
OCU_HMATH(hsqrt, std::sqrt(x))
OCU_HMATH(hrsqrt, 1.0f / std::sqrt(x))
OCU_HMATH(hrcp, 1.0f / x)
OCU_HMATH(hexp, std::exp(x))
OCU_HMATH(hexp2, std::exp2(x))
OCU_HMATH(hexp10, std::pow(10.0f, x))
OCU_HMATH(hlog, std::log(x))
OCU_HMATH(hlog2, std::log2(x))
OCU_HMATH(hlog10, std::log10(x))
OCU_HMATH(hsin, std::sin(x))
OCU_HMATH(hcos, std::cos(x))
OCU_HMATH(htanh, std::tanh(x))
OCU_HMATH(hfloor, std::floor(x))
OCU_HMATH(hceil, std::ceil(x))
OCU_HMATH(htrunc, std::trunc(x))
OCU_HMATH(hrint, std::nearbyint(x))
#undef OCU_HMATH

/* ---- __half2 ------------------------------------------------------------------ */

OCU_HD __half2 __halves2half2(__half a, __half b) { return __half2(a, b); }
OCU_HD __half2 __half2half2(__half a) { return __half2(a, a); }
OCU_HD __half2 __float2half2_rn(float a) { return __half2half2(__float2half_rn(a)); }
OCU_HD __half2 __floats2half2_rn(float a, float b)
{
    return __half2(__float2half_rn(a), __float2half_rn(b));
}
OCU_HD __half __low2half(__half2 a) { return a.x; }
OCU_HD __half __high2half(__half2 a) { return a.y; }
OCU_HD float __low2float(__half2 a) { return __half2float(a.x); }
OCU_HD float __high2float(__half2 a) { return __half2float(a.y); }
OCU_HD __half2 __lowhigh2highlow(__half2 a) { return __half2(a.y, a.x); }
OCU_HD __half2 __lows2half2(__half2 a, __half2 b) { return __half2(a.x, b.x); }
OCU_HD __half2 __highs2half2(__half2 a, __half2 b) { return __half2(a.y, b.y); }

#ifdef OCU_VECTOR_TYPES_H
OCU_HD float2 __half22float2(__half2 a) { return make_float2(__half2float(a.x), __half2float(a.y)); }
OCU_HD __half2 __float22half2_rn(float2 a) { return __floats2half2_rn(a.x, a.y); }
#endif

#define OCU_H2(name, op)                                                           \
    OCU_HD __half2 name(__half2 a, __half2 b) { return __half2(op(a.x, b.x), op(a.y, b.y)); }
OCU_H2(__hadd2, __hadd)
OCU_H2(__hsub2, __hsub)
OCU_H2(__hmul2, __hmul)
OCU_H2(__h2div, __hdiv)
OCU_H2(__hmax2, __hmax)
OCU_H2(__hmin2, __hmin)
#undef OCU_H2
OCU_HD __half2 __hfma2(__half2 a, __half2 b, __half2 c)
{
    return __half2(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y));
}
OCU_HD __half2 __hneg2(__half2 a) { return __half2(__hneg(a.x), __hneg(a.y)); }
OCU_HD __half2 __habs2(__half2 a) { return __half2(__habs(a.x), __habs(a.y)); }
OCU_HD __half2 operator+(__half2 a, __half2 b) { return __hadd2(a, b); }
OCU_HD __half2 operator-(__half2 a, __half2 b) { return __hsub2(a, b); }
OCU_HD __half2 operator*(__half2 a, __half2 b) { return __hmul2(a, b); }
OCU_HD __half2 operator/(__half2 a, __half2 b) { return __h2div(a, b); }
OCU_HD __half2 &operator+=(__half2 &a, __half2 b) { return a = __hadd2(a, b); }
OCU_HD __half2 &operator*=(__half2 &a, __half2 b) { return a = __hmul2(a, b); }

#undef OCU_HD

#endif /* OCU_CUDA_FP16_H */

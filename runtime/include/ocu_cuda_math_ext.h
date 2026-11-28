/*
 * ocu_cuda_math_ext.h -- CUDA Math API names over ocu_device_math_ext.h,
 * included near the end of each path's language header, which defines:
 *
 *   OCU_MX_Q              the qualifier of the functions (device inline)
 *   OCU_MX_STD            also the names the C library has (acosh, sinpi,
 *                         remquo, ...): the GPU paths; the CPU path has them
 *                         from <cmath>
 *   OCU_MX_HAS_FSUB_RN    the path already has __fsub_rn
 *   OCU_MX_NEED_DSQRT_RN  the path lacks __dsqrt_rn
 *   OCU_MX_HAS_FMAF_RM    the path already has __fmaf_rz / _ru / _rd
 *   OCU_MX_HAS_I2F_RN     ... __int2float_rn
 *   OCU_MX_HAS_U2F_RN     ... __uint2float_rn
 *   OCU_MX_HAS_F2U32      ... __float2uint_rn / _rz / _ru / _rd
 *
 * Written for this project from the public CUDA Math API reference (names,
 * argument types, special values); not derived from any vendor header.
 */
#ifndef OCU_CUDA_MATH_EXT_H
#define OCU_CUDA_MATH_EXT_H

#include "ocu_device_math_ext.h"

#ifdef OCU_MX_STD
#define OCU_MX1(n) OCU_MX_Q double n(double x) { return ocu_dm::n(x); } \
                   OCU_MX_Q float n##f(float x) { return ocu_dm::n##f(x); }
OCU_MX1(acosh)
OCU_MX1(asinh)
OCU_MX1(atanh)
OCU_MX1(sinpi)
OCU_MX1(cospi)
OCU_MX1(logb)
#undef OCU_MX1
OCU_MX_Q int ilogb(double x) { return ocu_dm::ilogb(x); }
OCU_MX_Q int ilogbf(float x) { return ocu_dm::ilogbf(x); }
OCU_MX_Q double scalbn(double x, int n) { return ocu_dm::scalbn(x, n); }
OCU_MX_Q float scalbnf(float x, int n) { return ocu_dm::scalbnf(x, n); }
OCU_MX_Q double scalbln(double x, long n) { return ocu_dm::scalbln(x, n); }
OCU_MX_Q float scalblnf(float x, long n) { return ocu_dm::scalblnf(x, n); }
OCU_MX_Q double nextafter(double x, double y) { return ocu_dm::nextafter(x, y); }
OCU_MX_Q float nextafterf(float x, float y) { return ocu_dm::nextafterf(x, y); }
OCU_MX_Q double remainder(double x, double y) { return ocu_dm::remainder(x, y); }
OCU_MX_Q float remainderf(float x, float y) { return ocu_dm::remainderf(x, y); }
OCU_MX_Q double remquo(double x, double y, int *q) { return ocu_dm::remquo(x, y, q); }
OCU_MX_Q float remquof(float x, float y, int *q) { return ocu_dm::remquof(x, y, q); }
OCU_MX_Q double modf(double x, double *ip) { return ocu_dm::modf(x, ip); }
OCU_MX_Q float modff(float x, float *ip) { return ocu_dm::modff(x, ip); }
OCU_MX_Q long long llrint(double x) { return ocu_dm::llrint(x); }
OCU_MX_Q long long llrintf(float x) { return ocu_dm::llrint((double)x); }
OCU_MX_Q long long llroundf(float x) { return ocu_dm::llround((double)x); }
OCU_MX_Q long lrintf(float x) { return (long)ocu_dm::llrint((double)x); }
OCU_MX_Q float nanf(const char *t) { return ocu_dm::nanf(t); }
OCU_MX_Q double nan(const char *t) { return ocu_dm::nan(t); }
OCU_MX_Q double erf(double x) { return ocu_dm::erf(x); }
OCU_MX_Q double erfc(double x) { return ocu_dm::erfc(x); }
#endif /* OCU_MX_STD */

/* ---- CUDA's own functions ------------------------------------------------------------- */

OCU_MX_Q double rcbrt(double x) { return ocu_dm::rcbrt(x); }
OCU_MX_Q float rcbrtf(float x) { return ocu_dm::rcbrtf(x); }
OCU_MX_Q double rhypot(double x, double y) { return ocu_dm::rhypot(x, y); }
OCU_MX_Q float rhypotf(float x, float y) { return ocu_dm::rhypotf(x, y); }
OCU_MX_Q void sincospi(double x, double *s, double *c) { *s = ocu_dm::sinpi(x); *c = ocu_dm::cospi(x); }
OCU_MX_Q void sincospif(float x, float *s, float *c) { *s = ocu_dm::sinpif(x); *c = ocu_dm::cospif(x); }
OCU_MX_Q double norm3d(double a, double b, double c) { return ocu_dm::norm3d(a, b, c); }
OCU_MX_Q double norm4d(double a, double b, double c, double d) { return ocu_dm::norm4d(a, b, c, d); }
OCU_MX_Q double rnorm3d(double a, double b, double c) { return ocu_dm::rnorm3d(a, b, c); }
OCU_MX_Q double rnorm4d(double a, double b, double c, double d) { return ocu_dm::rnorm4d(a, b, c, d); }
OCU_MX_Q double norm(int n, const double *a) { return ocu_dm::norm(n, a); }
OCU_MX_Q double rnorm(int n, const double *a) { return ocu_dm::rnorm(n, a); }
OCU_MX_Q float norm3df(float a, float b, float c) { return ocu_dm::norm3df(a, b, c); }
OCU_MX_Q float norm4df(float a, float b, float c, float d) { return ocu_dm::norm4df(a, b, c, d); }
OCU_MX_Q float rnorm3df(float a, float b, float c) { return ocu_dm::rnorm3df(a, b, c); }
OCU_MX_Q float rnorm4df(float a, float b, float c, float d) { return ocu_dm::rnorm4df(a, b, c, d); }
OCU_MX_Q float normf(int n, const float *a) { return ocu_dm::normf(n, a); }
OCU_MX_Q float rnormf(int n, const float *a) { return ocu_dm::rnormf(n, a); }
/* fdividef: the regular division (the documented default; -use_fast_math maps it to __fdividef) */
OCU_MX_Q float fdividef(float x, float y) { return x / y; }
OCU_MX_Q int __finitef(float x) { return (ocu_dm::isnanf_(x) || ocu_dm::isinff_(x)) ? 0 : 1; }
#if !defined(__MINGW32__) /* MinGW-w64's math.h has its own __signbitf (nonzero for a set sign bit, as here) */
OCU_MX_Q int __signbitf(float x) { return (int)(ocu_dm::bitsf(x) >> 31); }
#endif
OCU_MX_Q float __frsqrt_rn(float x) { return ocu_dm::frsqrt_rn(x); }

/* ---- rounding-mode arithmetic ---------------------------------------------------------- */

#define OCU_MX_RM(sfx, M)                                                                                     \
    OCU_MX_Q float __fadd_##sfx(float a, float b) { return ocu_dm::fadd(a, b, M); }                           \
    OCU_MX_Q float __fmul_##sfx(float a, float b) { return ocu_dm::fmul(a, b, M); }                           \
    OCU_MX_Q float __fdiv_##sfx(float a, float b) { return ocu_dm::fdiv(a, b, M); }                           \
    OCU_MX_Q float __frcp_##sfx(float x) { return ocu_dm::fdiv(1.0f, x, M); }                                 \
    OCU_MX_Q float __fsqrt_##sfx(float x) { return ocu_dm::fsqrt(x, M); }                                     \
    OCU_MX_Q float __fsub_##sfx(float a, float b) { return ocu_dm::fadd(a, -b, M); }                          \
    OCU_MX_Q double __dadd_##sfx(double a, double b) { return ocu_dm::dadd(a, b, M); }                        \
    OCU_MX_Q double __dsub_##sfx(double a, double b) { return ocu_dm::dadd(a, -b, M); }                       \
    OCU_MX_Q double __dmul_##sfx(double a, double b) { return ocu_dm::dmul(a, b, M); }                        \
    OCU_MX_Q double __ddiv_##sfx(double a, double b) { return ocu_dm::ddiv(a, b, M); }                        \
    OCU_MX_Q double __drcp_##sfx(double x) { return ocu_dm::ddiv(1.0, x, M); }                                \
    OCU_MX_Q double __dsqrt_##sfx(double x) { return ocu_dm::dsqrt(x, M); }
OCU_MX_RM(rz, ocu_dm::RZ)
OCU_MX_RM(ru, ocu_dm::RU)
OCU_MX_RM(rd, ocu_dm::RD)
#undef OCU_MX_RM
#ifndef OCU_MX_HAS_FMAF_RM
OCU_MX_Q float __fmaf_rz(float a, float b, float c) { return ocu_dm::ffma(a, b, c, ocu_dm::RZ); }
OCU_MX_Q float __fmaf_ru(float a, float b, float c) { return ocu_dm::ffma(a, b, c, ocu_dm::RU); }
OCU_MX_Q float __fmaf_rd(float a, float b, float c) { return ocu_dm::ffma(a, b, c, ocu_dm::RD); }
#endif
#ifndef OCU_MX_HAS_FSUB_RN
OCU_MX_Q float __fsub_rn(float a, float b) { return a - b; }
#endif
OCU_MX_Q double __dsub_rn(double a, double b) { return a - b; }
OCU_MX_Q double __ddiv_rn(double a, double b) { return a / b; }
#ifdef OCU_MX_NEED_DSQRT_RN
OCU_MX_Q double __dsqrt_rn(double x) { return ocu_dm::sqrt(x); }
#endif

/* ---- 64-bit integer conversions (out of range: saturated; NaN: 0) ---------------------- */

#define OCU_MX_CV(sfx, R, M)                                                                                  \
    OCU_MX_Q long long __double2ll_##sfx(double x) { return ocu_dm::sat_i64(R(x)); }                          \
    OCU_MX_Q unsigned long long __double2ull_##sfx(double x) { return ocu_dm::sat_u64(R(x)); }                \
    OCU_MX_Q long long __float2ll_##sfx(float x) { return ocu_dm::sat_i64(R((double)x)); }                    \
    OCU_MX_Q unsigned long long __float2ull_##sfx(float x) { return ocu_dm::sat_u64(R((double)x)); }          \
    OCU_MX_Q float __ll2float_##sfx(long long v) { return ocu_dm::dd_to_float(ocu_dm::i64_dd(v), M); }        \
    OCU_MX_Q float __ull2float_##sfx(unsigned long long v) { return ocu_dm::dd_to_float(ocu_dm::u64_dd(v), M); } \
    OCU_MX_Q double __ll2double_##sfx(long long v) { return ocu_dm::dd_to_double(ocu_dm::i64_dd(v), M); }     \
    OCU_MX_Q double __ull2double_##sfx(unsigned long long v) { return ocu_dm::dd_to_double(ocu_dm::u64_dd(v), M); }
OCU_MX_CV(rn, ocu_dm::rint, ocu_dm::RN)
OCU_MX_CV(rz, ocu_dm::trunc, ocu_dm::RZ)
OCU_MX_CV(ru, ocu_dm::ceil, ocu_dm::RU)
OCU_MX_CV(rd, ocu_dm::floor, ocu_dm::RD)
#undef OCU_MX_CV

/* ---- 32-bit conversions the paths do not all have ---------------------------------------
 * double -> unsigned (saturated, NaN 0), int / unsigned -> float in each mode
 * (the value is exact as a double; one rounding to float), int / unsigned ->
 * double (exact), double -> float in each mode. */
#define OCU_MX_CV32(sfx, R, M)                                                                                \
    OCU_MX_Q unsigned __double2uint_##sfx(double x) { return ocu_dm::sat_u32(R(x)); }                         \
    OCU_MX_Q float __double2float_##sfx(double x) { return ocu_dm::to_float(x, 0, M); }
OCU_MX_CV32(rn, ocu_dm::rint, ocu_dm::RN)
OCU_MX_CV32(rz, ocu_dm::trunc, ocu_dm::RZ)
OCU_MX_CV32(ru, ocu_dm::ceil, ocu_dm::RU)
OCU_MX_CV32(rd, ocu_dm::floor, ocu_dm::RD)
#undef OCU_MX_CV32
#define OCU_MX_I2F(sfx, M)                                                                                    \
    OCU_MX_Q float __int2float_##sfx(int v) { return ocu_dm::to_float((double)v, 0, M); }                     \
    OCU_MX_Q float __uint2float_##sfx(unsigned v) { return ocu_dm::to_float((double)v, 0, M); }
OCU_MX_I2F(rz, ocu_dm::RZ)
OCU_MX_I2F(ru, ocu_dm::RU)
OCU_MX_I2F(rd, ocu_dm::RD)
#undef OCU_MX_I2F
/* the two 32-bit halves of a double's bits, and a double from them */
OCU_MX_Q int __double2hiint(double x) { return (int)(__builtin_bit_cast(unsigned long long, x) >> 32); }
OCU_MX_Q int __double2loint(double x) { return (int)(unsigned)__builtin_bit_cast(unsigned long long, x); }
OCU_MX_Q double __hiloint2double(int hi, int lo)
{
    return __builtin_bit_cast(double, ((unsigned long long)(unsigned)hi << 32) | (unsigned)lo);
}
OCU_MX_Q double __int2double_rn(int v) { return (double)v; }
OCU_MX_Q double __uint2double_rn(unsigned v) { return (double)v; }
#ifndef OCU_MX_HAS_I2F_RN
OCU_MX_Q float __int2float_rn(int v) { return (float)v; }
#endif
#ifndef OCU_MX_HAS_U2F_RN
OCU_MX_Q float __uint2float_rn(unsigned v) { return (float)v; }
#endif
#ifndef OCU_MX_HAS_F2U32
OCU_MX_Q unsigned __float2uint_rn(float x) { return ocu_dm::sat_u32(ocu_dm::rint((double)x)); }
OCU_MX_Q unsigned __float2uint_rz(float x) { return ocu_dm::sat_u32(ocu_dm::trunc((double)x)); }
OCU_MX_Q unsigned __float2uint_ru(float x) { return ocu_dm::sat_u32(ocu_dm::ceil((double)x)); }
OCU_MX_Q unsigned __float2uint_rd(float x) { return ocu_dm::sat_u32(ocu_dm::floor((double)x)); }
#endif

#endif /* OCU_CUDA_MATH_EXT_H */

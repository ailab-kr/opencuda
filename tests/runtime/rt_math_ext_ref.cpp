// Host side of rt_math_ext_gpu.cu: the same functions (ocu_device_math_ext.h
// compiled for the host) for the exact results, and long double references
// for the approximations. Plain C++ (no CUDA).
#include "ocu_device_math_ext.h"

#include <cmath>
#include <cstring>

#include "rt_math_ext_fns.h"

static float fbits(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
static double dbits(unsigned long long u) { double d; std::memcpy(&d, &u, 8); return d; }

void ocu_mx_exact_f(const float *pa, const float *pb, const float *pc, const long long *pv, float *o)
{
    namespace D = ocu_dm;
    using D::RZ; using D::RU; using D::RD;
    float a = *pa, b = *pb, c = *pc;
    long long v = *pv;
    o[E_FADD_RZ] = D::fadd(a, b, RZ); o[E_FADD_RU] = D::fadd(a, b, RU); o[E_FADD_RD] = D::fadd(a, b, RD);
    o[E_FMUL_RZ] = D::fmul(a, b, RZ); o[E_FMUL_RU] = D::fmul(a, b, RU); o[E_FMUL_RD] = D::fmul(a, b, RD);
    o[E_FDIV_RZ] = D::fdiv(a, b, RZ); o[E_FDIV_RU] = D::fdiv(a, b, RU); o[E_FDIV_RD] = D::fdiv(a, b, RD);
    o[E_FSQRT_RZ] = D::fsqrt(a, RZ); o[E_FSQRT_RU] = D::fsqrt(a, RU); o[E_FSQRT_RD] = D::fsqrt(a, RD);
    o[E_FMAF_RZ] = D::ffma(a, b, c, RZ); o[E_FMAF_RU] = D::ffma(a, b, c, RU); o[E_FMAF_RD] = D::ffma(a, b, c, RD);
    o[E_FRSQRT_RN] = D::frsqrt_rn(a);
    o[E_LL2F_RD] = D::dd_to_float(D::i64_dd(v), RD); o[E_LL2F_RU] = D::dd_to_float(D::i64_dd(v), RU);
    o[E_ULL2F_RZ] = D::dd_to_float(D::u64_dd((unsigned long long)v), RZ);
    o[E_F2LL_RD] = (float)D::sat_i64(D::floor((double)a)); o[E_F2ULL_RU] = (float)D::sat_u64(D::ceil((double)a));
    int q;
    o[E_REMQUOF] = D::remquof(a, b, &q); o[E_REMQUOF_Q] = (float)q;
    o[E_NEXTAFTERF] = D::nextafterf(a, b);
    o[E_ILOGBF] = (float)D::ilogbf(a);
    o[E_SCALBNF] = D::scalbnf(a, (int)(v % 300));
    float ip;
    o[E_MODFF] = D::modff(a, &ip); o[E_MODFF_I] = ip;
}

void ocu_mx_exact_d(const double *pa, const double *pb, const long long *pv, double *o)
{
    namespace D = ocu_dm;
    using D::RZ; using D::RU; using D::RD;
    double a = *pa, b = *pb;
    long long v = *pv;
    o[E_DADD_RZ] = D::dadd(a, b, RZ); o[E_DADD_RU] = D::dadd(a, b, RU); o[E_DADD_RD] = D::dadd(a, b, RD);
    o[E_DMUL_RZ] = D::dmul(a, b, RZ); o[E_DMUL_RU] = D::dmul(a, b, RU); o[E_DMUL_RD] = D::dmul(a, b, RD);
    o[E_DDIV_RZ] = D::ddiv(a, b, RZ); o[E_DDIV_RU] = D::ddiv(a, b, RU); o[E_DDIV_RD] = D::ddiv(a, b, RD);
    o[E_DSQRT_RZ] = D::dsqrt(a, RZ); o[E_DSQRT_RU] = D::dsqrt(a, RU); o[E_DSQRT_RD] = D::dsqrt(a, RD);
    o[E_LL2D_RU] = D::dd_to_double(D::i64_dd(v), RU);
    o[E_ULL2D_RD] = D::dd_to_double(D::u64_dd((unsigned long long)v), RD);
    o[E_D2LL_RD] = (double)D::sat_i64(D::floor(a)); o[E_D2ULL_RZ] = (double)D::sat_u64(D::trunc(a));
    int q;
    o[E_REMQUO] = D::remquo(a, b, &q); o[E_REMQUO_Q] = (double)q;
    o[E_REMAINDER] = D::remainder(a, b);
    o[E_NEXTAFTER] = D::nextafter(a, b);
    o[E_ILOGB] = (double)D::ilogb(a);
    o[E_SCALBN] = D::scalbn(a, (int)(v % 3000));
    double ip;
    o[E_MODF] = D::modf(a, &ip); o[E_MODF_I] = ip;
    o[E_LLRINT] = std::fabs(a) < 0x1p62 ? (double)D::llrint(a) : 0.0;
}

static const long double PIL = 3.141592653589793238462643383279502884L;
static long double trig_pi(long double x, bool cosine)
{
    long double k = std::rint(2.0L * x), r = x - 0.5L * k, km = std::fmod(k, 4.0L);
    if (km < 0) km += 4.0L;
    int q = (int)km + (cosine ? 1 : 0);
    if (r == 0) return (q & 1) ? ((q & 2) ? -1.0L : 1.0L) : 0.0L;
    long double s = std::sin(PIL * r), c = std::cos(PIL * r);
    switch (q & 3) { case 0: return s; case 1: return c; case 2: return -s; default: return -c; }
}
static long double ref(int fn, long double a, long double b, long double c, long double d)
{
    switch (fn) {
    case A_SINPI: return trig_pi(a, false);
    case A_COSPI: return trig_pi(a, true);
    case A_ACOSH: return std::acosh(a);
    case A_ASINH: return std::asinh(a);
    case A_ATANH: return std::atanh(a);
    case A_RCBRT: return 1.0L / std::cbrt(a);
    case A_RHYPOT: return 1.0L / std::hypot(a, b);
    case A_NORM3D: return std::sqrt(a * a + b * b + c * c);
    case A_RNORM4D: return 1.0L / std::sqrt(a * a + b * b + c * c + d * d);
    }
    return 0;
}
static double err(long double got, long double want, int mant, int emin)
{
    if (std::isnan(want)) return std::isnan(got) ? 0 : 1e9;
    if (std::isinf(want)) return got == want ? 0 : 1e9;
    if (std::isinf(got)) { /* right when the exact value is beyond the largest finite + half an ulp */
        long double big = std::ldexp(1.0L, emin == -1022 ? 1024 : 128) * (1.0L - std::ldexp(1.0L, -mant - 1));
        return (std::fabs(want) >= big && (got > 0) == (want > 0)) ? 0 : 1e9;
    }
    long double aw = std::fabs(want);
    int e;
    std::frexp(aw, &e);
    if (e - 1 < emin) e = emin + 1;
    long double ulp = std::ldexp(1.0L, e - mant);
    return (double)(std::fabs(got - want) / ulp);
}
double ocu_mx_err_d(int fn, double got, double a, double b, double c, double d)
{
    return err(got, ref(fn, a, b, c, d), 53, -1022);
}
double ocu_mx_err_f(int fn, float got, float a, float b, float c, float d)
{
    return err(got, ref(fn, a, b, c, d), 24, -126);
}

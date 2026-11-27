// Accuracy of ocu_device_math_ext.h, compiled for the host:
//   - exact functions (modf, logb, ilogb, scalbn, scalbln, nextafter,
//     remainder, remquo, llrint, llround and float forms) against the C
//     library, bit for bit;
//   - the rounding-mode arithmetic (__fadd_rd ... __dsqrt_ru: add, sub, mul,
//     div, rcp, sqrt, fma for float; add, mul, div, rcp, sqrt for double)
//     and the 64-bit integer conversions, int / unsigned / double to float
//     and double to unsigned against the host FPU in that rounding mode
//     (fesetround), bit for bit;
//   - __frsqrt_rn on every positive float against 1/sqrtl, bit for bit
//     (OCU_MATH_EXT_FULL=1; otherwise every 61st);
//   - sinpi, cospi, acosh, asinh, atanh, rcbrt, rhypot, norm3d, norm4d,
//     rnorm3d, rnorm4d, norm, rnorm against long double references: double
//     forms within 1.5 ulp (2 for the hyperbolic inverses), float forms
//     within 0.5 ulp + 2^-20 (the double result rounded once), special
//     values exact.
// Build: g++ -O2 -frounding-math -std=c++17 -I runtime/include test_device_math_ext.cpp -pthread
#include "ocu_device_math_ext.h"

#include <cfenv>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static int g_fail;
static long long g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { if (g_fail++ < 40) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

static std::uint64_t bd(double x) { std::uint64_t b; std::memcpy(&b, &x, 8); return b; }
static std::uint32_t bf(float x) { std::uint32_t b; std::memcpy(&b, &x, 4); return b; }
static double dbits(std::uint64_t b) { double x; std::memcpy(&x, &b, 8); return x; }
static float fbits(std::uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
static bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || bd(a) == bd(b); }
static bool samef(float a, float b) { return (std::isnan(a) && std::isnan(b)) || bf(a) == bf(b); }

static std::mt19937_64 rng(12345);
// doubles of all magnitudes and special values
static double rand_double()
{
    switch (rng() % 16) {
    case 0: { static const double sp[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, DBL_MIN, -DBL_MIN, DBL_MAX, -DBL_MAX,
                                          DBL_TRUE_MIN, -DBL_TRUE_MIN, 1.0, -1.0, 0.5, 2.0, 0.25, 1.5, -0.5};
              return sp[rng() % (sizeof sp / sizeof sp[0])]; }
    case 1: return (double)(long long)(rng() % 2000001) / 4 - 250000; // quarter-integers
    case 2: return std::ldexp((double)(rng() >> 11), -(int)(rng() % 60)); // moderate
    default: return dbits(rng());
    }
}
static float rand_float()
{
    switch (rng() % 16) {
    case 0: { static const float sp[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, FLT_MIN, -FLT_MIN, FLT_MAX, -FLT_MAX,
                                         FLT_TRUE_MIN, -FLT_TRUE_MIN, 1.0f, -1.0f, 0.5f, 2.0f, 1.5f, -0.5f};
              return sp[rng() % (sizeof sp / sizeof sp[0])]; }
    case 1: return (float)(long long)(rng() % 20001) / 4 - 2500;
    default: return fbits((std::uint32_t)rng());
    }
}

// ulp of double / float at the exact value v
static long double ulpd(long double v)
{
    long double a = std::fabs(v);
    if (a < 0x1p-1022L) return 0x1p-1074L;
    int e; std::frexp(a, &e);
    return std::ldexp(1.0L, e - 53);
}
static long double ulpf(long double v)
{
    long double a = std::fabs(v);
    if (a < 0x1p-126L) return 0x1p-149L;
    int e; std::frexp(a, &e);
    return std::ldexp(1.0L, e - 24);
}
static double err_d(double got, long double want)
{
    if (std::isnan(want) || std::isinf(want)) return same(got, (double)want) ? 0.0 : 1e9;
    if (std::isinf(got)) return std::fabs(want) > 0x1.fffffffffffffp1023L ? 0.0 : 1e9;
    return (double)(std::fabs((long double)got - want) / ulpd(want));
}
static double err_f(float got, long double want)
{
    if (std::isnan(want) || std::isinf(want)) return samef(got, (float)want) ? 0.0 : 1e9;
    if (std::isinf(got)) return std::fabs(want) > 0x1.fffffep127L ? 0.0 : 1e9;
    return (double)(std::fabs((long double)got - want) / ulpf(want));
}

static const int MODES[4] = {FE_TONEAREST, FE_TOWARDZERO, FE_UPWARD, FE_DOWNWARD};
static const char *MNAME[4] = {"rn", "rz", "ru", "rd"};

static void test_rounding_modes()
{
    for (int m = 0; m < 4; ++m) {
        for (int i = 0; i < 400000; ++i) {
            float a = rand_float(), b = rand_float(), c = rand_float();
            double x = rand_double(), y = rand_double();
            if (i % 3 == 0) { b = std::nextafter(-a, (float)(rng() & 1 ? INFINITY : -INFINITY)); y = -x * (1 + 0x1p-52 * (double)(rng() % 3)); }
            volatile float va = a, vb = b, vc = c;
            volatile double vx = x, vy = y;
            std::fesetround(MODES[m]);
            float fs = va + vb, fd = va - vb, fm = va * vb, fq = va / vb, fr = 1.0f / va, fsq = std::sqrt(va);
            float ff = std::fma(va, vb, vc);
            double ds = vx + vy, dm = vx * vy, dq = vx / vy, dr = 1.0 / vx, dsq = std::sqrt(vx);
            std::fesetround(FE_TONEAREST);
            CHECK(samef(ocu_dm::fadd(a, b, m), fs), "__fadd_%s(%a, %a) = %a want %a", MNAME[m], a, b, ocu_dm::fadd(a, b, m), fs);
            CHECK(samef(ocu_dm::fadd(a, -b, m), fd), "__fsub_%s(%a, %a)", MNAME[m], a, b);
            CHECK(samef(ocu_dm::fmul(a, b, m), fm), "__fmul_%s(%a, %a) = %a want %a", MNAME[m], a, b, ocu_dm::fmul(a, b, m), fm);
            CHECK(samef(ocu_dm::fdiv(a, b, m), fq), "__fdiv_%s(%a, %a) = %a want %a", MNAME[m], a, b, ocu_dm::fdiv(a, b, m), fq);
            CHECK(samef(ocu_dm::fdiv(1.0f, a, m), fr), "__frcp_%s(%a)", MNAME[m], a);
            CHECK(samef(ocu_dm::fsqrt(a, m), fsq), "__fsqrt_%s(%a) = %a want %a", MNAME[m], a, ocu_dm::fsqrt(a, m), fsq);
            CHECK(samef(ocu_dm::ffma(a, b, c, m), ff), "__fmaf_%s(%a, %a, %a) = %a want %a", MNAME[m], a, b, c, ocu_dm::ffma(a, b, c, m), ff);
            CHECK(same(ocu_dm::dadd(x, y, m), ds), "__dadd_%s(%a, %a) = %a want %a", MNAME[m], x, y, ocu_dm::dadd(x, y, m), ds);
            CHECK(same(ocu_dm::dmul(x, y, m), dm), "__dmul_%s(%a, %a) = %a want %a", MNAME[m], x, y, ocu_dm::dmul(x, y, m), dm);
            CHECK(same(ocu_dm::ddiv(x, y, m), dq), "__ddiv_%s(%a, %a) = %a want %a", MNAME[m], x, y, ocu_dm::ddiv(x, y, m), dq);
            CHECK(same(ocu_dm::ddiv(1.0, x, m), dr), "__drcp_%s(%a)", MNAME[m], x);
            CHECK(same(ocu_dm::dsqrt(x, m), dsq), "__dsqrt_%s(%a) = %a want %a", MNAME[m], x, ocu_dm::dsqrt(x, m), dsq);
        }
        // 64-bit integers to float / double
        for (int i = 0; i < 400000; ++i) {
            long long v = (long long)rng() >> (rng() % 64);
            if (rng() & 1) v = -v;
            unsigned long long u = rng() >> (rng() % 64);
            volatile long long vv = v; volatile unsigned long long vu = u;
            std::fesetround(MODES[m]);
            float f1 = (float)vv, f2 = (float)vu; double d1 = (double)vv, d2 = (double)vu;
            std::fesetround(FE_TONEAREST);
            CHECK(samef(ocu_dm::dd_to_float(ocu_dm::i64_dd(v), m), f1), "__ll2float_%s(%lld)", MNAME[m], v);
            CHECK(samef(ocu_dm::dd_to_float(ocu_dm::u64_dd(u), m), f2), "__ull2float_%s(%llu)", MNAME[m], u);
            CHECK(same(ocu_dm::dd_to_double(ocu_dm::i64_dd(v), m), d1), "__ll2double_%s(%lld)", MNAME[m], v);
            CHECK(same(ocu_dm::dd_to_double(ocu_dm::u64_dd(u), m), d2), "__ull2double_%s(%llu)", MNAME[m], u);
        }
        // 32-bit integers and doubles to float (__int2float_r*, __uint2float_r*, __double2float_r*)
        for (int i = 0; i < 400000; ++i) {
            int v = (int)(rng() >> (32 + rng() % 32));
            if (rng() & 1) v = -v;
            unsigned u = (unsigned)(rng() >> (32 + rng() % 32));
            double x = rand_double();
            if (i % 3 == 0) x = (double)rand_float() * (1 + 0x1p-30 * (double)(rng() % 9)); // near floats, ties
            volatile int vv = v; volatile unsigned vu = u; volatile double vx = x;
            std::fesetround(MODES[m]);
            float f1 = (float)vv, f2 = (float)vu, f3 = (float)vx;
            std::fesetround(FE_TONEAREST);
            CHECK(samef(ocu_dm::to_float((double)v, 0, m), f1), "__int2float_%s(%d)", MNAME[m], v);
            CHECK(samef(ocu_dm::to_float((double)u, 0, m), f2), "__uint2float_%s(%u)", MNAME[m], u);
            CHECK(samef(ocu_dm::to_float(x, 0, m), f3), "__double2float_%s(%a) = %a want %a", MNAME[m], x, ocu_dm::to_float(x, 0, m), f3);
        }
    }
    // double to unsigned (__double2uint_r*): saturating, NaN -> 0
    for (int i = 0; i < 400000; ++i) {
        double x = rand_double();
        if (i % 2) x = std::ldexp((double)(long long)rng(), -(int)(rng() % 40) - 24);
        long double r[4] = {std::rint((long double)x), std::trunc((long double)x), std::ceil((long double)x), std::floor((long double)x)};
        double fns[4] = {ocu_dm::rint(x), ocu_dm::trunc(x), ocu_dm::ceil(x), ocu_dm::floor(x)};
        for (int m = 0; m < 4; ++m) {
            long double w = r[m];
            unsigned want = std::isnan(x) || w <= 0 ? 0u : (w >= 0x1p32L ? UINT_MAX : (unsigned)w);
            CHECK(ocu_dm::sat_u32(fns[m]) == want, "__double2uint_%s(%a)", MNAME[m], x);
        }
    }
    // float / double to 64-bit integers: floor / ceil / trunc / rint, saturating, NaN -> 0
    for (int i = 0; i < 400000; ++i) {
        double x = rand_double();
        if (i % 2) x = std::ldexp((double)(long long)rng(), -(int)(rng() % 70));
        long double r[4] = {std::rint((long double)x), std::trunc((long double)x), std::ceil((long double)x), std::floor((long double)x)};
        double fns[4] = {ocu_dm::rint(x), ocu_dm::trunc(x), ocu_dm::ceil(x), ocu_dm::floor(x)};
        for (int m = 0; m < 4; ++m) {
            long double w = r[m];
            long long want = std::isnan(x) ? 0 : (w >= 0x1p63L ? LLONG_MAX : (w < -0x1p63L ? LLONG_MIN : (long long)w));
            unsigned long long wantu = std::isnan(x) || w <= 0 ? 0 : (w >= 0x1p64L ? ULLONG_MAX : (unsigned long long)w);
            CHECK(ocu_dm::sat_i64(fns[m]) == want, "__double2ll_%s(%a)", MNAME[m], x);
            CHECK(ocu_dm::sat_u64(fns[m]) == wantu, "__double2ull_%s(%a)", MNAME[m], x);
        }
    }
}

static void test_exact()
{
    for (int i = 0; i < 2000000; ++i) {
        double x = rand_double(), y = rand_double();
        if (i % 4 == 0) y = x * std::ldexp(1.0, (int)(rng() % 120) - 60) * (1 + (double)(rng() % 7) / 8);
        int q1 = 0, q2 = 0;
        double r1 = ocu_dm::remquo(x, y, &q1), r2 = std::remquo(x, y, &q2);
        CHECK(same(r1, r2) && (std::isnan(r2) || (q1 & 7) == (q2 & 7) * (q2 < 0 ? -1 : 1) * (q2 < 0 ? -1 : 1) && ((q1 < 0) == (q2 < 0) || (q1 & 7) == 0)),
              "remquo(%a, %a) = %a q %d want %a q %d", x, y, r1, q1, r2, q2);
        CHECK(same(ocu_dm::remainder(x, y), std::remainder(x, y)), "remainder(%a, %a)", x, y);
        double ip1, ip2;
        double f1 = ocu_dm::modf(x, &ip1), f2 = std::modf(x, &ip2);
        CHECK(same(f1, f2) && same(ip1, ip2), "modf(%a)", x);
        CHECK(ocu_dm::ilogb(x) == std::ilogb(x), "ilogb(%a) = %d want %d", x, ocu_dm::ilogb(x), std::ilogb(x));
        CHECK(same(ocu_dm::logb(x), std::logb(x)), "logb(%a)", x);
        int n = (int)(rng() % 4400) - 2200;
        CHECK(same(ocu_dm::scalbn(x, n), std::scalbn(x, n)), "scalbn(%a, %d)", x, n);
        long ln = (long)((long long)rng() >> 20);
        CHECK(same(ocu_dm::scalbln(x, ln), std::scalbln(x, ln)), "scalbln(%a, %ld)", x, ln);
        CHECK(same(ocu_dm::nextafter(x, y), std::nextafter(x, y)), "nextafter(%a, %a)", x, y);
        if (std::isfinite(x) && std::fabs(x) < 0x1p62) {
            CHECK(ocu_dm::llrint(x) == std::llrint(x), "llrint(%a)", x);
            CHECK(ocu_dm::llround(x) == std::llround(x), "llround(%a)", x);
        }
        float a = rand_float(), b = rand_float();
        if (i % 4 == 0) b = a * std::ldexp(1.0f, (int)(rng() % 60) - 30);
        int q3, q4;
        float s1 = ocu_dm::remquof(a, b, &q3), s2 = std::remquo(a, b, &q4);
        CHECK(samef(s1, s2) && (std::isnan(s2) || ((q3 % 8) + 8) % 8 == ((q4 % 8) + 8) % 8), "remquof(%a, %a) = %a %d want %a %d", a, b, s1, q3, s2, q4);
        CHECK(samef(ocu_dm::remainderf(a, b), std::remainder(a, b)), "remainderf(%a, %a)", a, b);
        float fi1, fi2;
        CHECK(samef(ocu_dm::modff(a, &fi1), std::modf(a, &fi2)) && samef(fi1, fi2), "modff(%a)", a);
        CHECK(ocu_dm::ilogbf(a) == std::ilogb(a), "ilogbf(%a)", a);
        CHECK(samef(ocu_dm::logbf(a), std::logb(a)), "logbf(%a)", a);
        CHECK(samef(ocu_dm::scalbnf(a, n), std::scalbn(a, n)), "scalbnf(%a, %d)", a, n);
        CHECK(samef(ocu_dm::scalblnf(a, ln), std::scalbln(a, ln)), "scalblnf(%a, %ld)", a, ln);
        CHECK(samef(ocu_dm::nextafterf(a, b), std::nextafter(a, b)), "nextafterf(%a, %a)", a, b);
    }
}

struct worst { double e = 0; long double at = 0, at2 = 0; long long n = 0; };
static void note(worst &w, double e, long double at, long double at2 = 0)
{
    ++w.n;
    if (e > w.e) { w.e = e; w.at = at; w.at2 = at2; }
}
static void report(const char *name, const worst &w, double bound)
{
    bool ok = w.e <= bound;
    if (!ok) ++g_fail;
    ++g_checks;
    std::printf("%-10s max %8.4f ulp (bound %g) at %La %La  (%lld inputs) %s\n", name, w.e, bound, w.at, w.at2, w.n, ok ? "" : "FAIL");
}

static const long double PIL = 3.141592653589793238462643383279502884L;
// x = k/2 + r, |r| <= 1/4, exact in long double for a double x; quadrant k mod 4
static void ref_reduce(long double x, int *q, long double *r)
{
    long double k = std::rint(2.0L * x);
    *r = x - 0.5L * k;
    long double km = std::fmod(k, 4.0L);
    if (km < 0) km += 4.0L;
    *q = (int)km;
}
static long double ref_sinpi(long double x)
{
    int q; long double r;
    ref_reduce(x, &q, &r);
    if (r == 0) return (q & 1) ? (q == 1 ? 1.0L : -1.0L) : 0.0L;
    long double s = std::sin(PIL * r), c = std::cos(PIL * r);
    return q == 0 ? s : q == 1 ? c : q == 2 ? -s : -c;
}
static long double ref_cospi(long double x)
{
    int q; long double r;
    ref_reduce(x, &q, &r);
    if (r == 0) return (q & 1) ? 0.0L : (q == 0 ? 1.0L : -1.0L);
    long double s = std::sin(PIL * r), c = std::cos(PIL * r);
    return q == 0 ? c : q == 1 ? -s : q == 2 ? -c : s;
}

static void test_accuracy(bool full)
{
    worst sp, cp, ach, ash, ath, rc, rh, n3, n4, rn3, rn4, nn, rnn;
    worst spf, cpf, achf, ashf, athf, rcf, rhf, n3f, n4f, rn3f, rn4f;
    const int N = full ? 20000000 : 3000000;
    for (int i = 0; i < N; ++i) {
        double x = rand_double(), y = rand_double(), z = rand_double(), w = rand_double();
        if (i % 3 == 0) { x = std::ldexp((double)(long long)(rng() >> 11), -(int)(rng() % 64)); }
        if (std::isfinite(x)) {
            long double s = ref_sinpi(x), c = ref_cospi(x);
            if (std::fabs(x) < 0x1p52 && x != std::trunc(x)) note(sp, err_d(ocu_dm::sinpi(x), s), x);
            note(cp, err_d(ocu_dm::cospi(x), c), x);
        }
        double xa = 1.0 + std::fabs(x) * (i % 2 ? 1e-9 : 1.0);
        if (std::isfinite(xa)) note(ach, err_d(ocu_dm::acosh(xa), std::acosh((long double)xa)), xa);
        if (std::isfinite(x)) note(ash, err_d(ocu_dm::asinh(x), std::asinh((long double)x)), x);
        double xt = std::fmod(x, 1.0);
        if (std::isfinite(xt)) note(ath, err_d(ocu_dm::atanh(xt), std::atanh((long double)xt)), xt);
        if (std::isfinite(x) && x != 0) note(rc, err_d(ocu_dm::rcbrt(x), 1.0L / std::cbrt((long double)x)), x);
        if (std::isfinite(x) && std::isfinite(y) && (x != 0 || y != 0))
            note(rh, err_d(ocu_dm::rhypot(x, y), 1.0L / std::hypot((long double)x, (long double)y)), x, y);
        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(w) && (x != 0 || y != 0 || z != 0)) {
            // references: long double sums of squares, scaled (exact power of two)
            long double m = std::fmax(std::fmax(std::fabs((long double)x), std::fabs((long double)y)), std::fmax(std::fabs((long double)z), std::fabs((long double)w)));
            int e; std::frexp(m, &e);
            long double X = std::ldexp((long double)x, -e), Y = std::ldexp((long double)y, -e), Z = std::ldexp((long double)z, -e), W = std::ldexp((long double)w, -e);
            long double s3 = std::sqrt(X * X + Y * Y + Z * Z), s4 = std::sqrt(X * X + Y * Y + Z * Z + W * W);
            long double m3 = std::fmax(std::fmax(std::fabs((long double)x), std::fabs((long double)y)), std::fabs((long double)z));
            int e3; std::frexp(m3, &e3);
            long double X3 = std::ldexp((long double)x, -e3), Y3 = std::ldexp((long double)y, -e3), Z3 = std::ldexp((long double)z, -e3);
            s3 = std::sqrt(X3 * X3 + Y3 * Y3 + Z3 * Z3);
            note(n3, err_d(ocu_dm::norm3d(x, y, z), std::ldexp(s3, e3)), x, y);
            note(rn3, err_d(ocu_dm::rnorm3d(x, y, z), std::ldexp(1.0L / s3, -e3)), x, y);
            note(n4, err_d(ocu_dm::norm4d(x, y, z, w), std::ldexp(s4, e)), x, y);
            note(rn4, err_d(ocu_dm::rnorm4d(x, y, z, w), std::ldexp(1.0L / s4, -e)), x, y);
            double v[4] = {x, y, z, w};
            note(nn, err_d(ocu_dm::norm(4, v), std::ldexp(s4, e)), x, y);
            note(rnn, err_d(ocu_dm::rnorm(4, v), std::ldexp(1.0L / s4, -e)), x, y);
        }
        // float forms
        float a = rand_float(), b = rand_float(), c3 = rand_float(), d4 = rand_float();
        if (i % 3 == 0) a = (float)std::ldexp((double)(rng() >> 40), -(int)(rng() % 30));
        if (std::isfinite(a)) {
            if (a != std::trunc(a)) note(spf, err_f(ocu_dm::sinpif(a), ref_sinpi(a)), a);
            note(cpf, err_f(ocu_dm::cospif(a), ref_cospi(a)), a);
            note(ashf, err_f(ocu_dm::asinhf(a), std::asinh((long double)a)), a);
            if (a != 0) note(rcf, err_f(ocu_dm::rcbrtf(a), 1.0L / std::cbrt((long double)a)), a);
            float aa = 1.0f + std::fabs(a);
            if (std::isfinite(aa)) note(achf, err_f(ocu_dm::acoshf(aa), std::acosh((long double)aa)), aa);
            float at = std::fmod(a, 1.0f);
            note(athf, err_f(ocu_dm::atanhf(at), std::atanh((long double)at)), at);
            if (std::isfinite(b) && (a != 0 || b != 0)) note(rhf, err_f(ocu_dm::rhypotf(a, b), 1.0L / std::hypot((long double)a, (long double)b)), a, b);
            if (std::isfinite(b) && std::isfinite(c3) && std::isfinite(d4)) {
                long double S3 = std::sqrt((long double)a * a + (long double)b * b + (long double)c3 * c3);
                long double S4 = std::sqrt((long double)a * a + (long double)b * b + (long double)c3 * c3 + (long double)d4 * d4);
                note(n3f, err_f(ocu_dm::norm3df(a, b, c3), S3), a, b);
                note(n4f, err_f(ocu_dm::norm4df(a, b, c3, d4), S4), a, b);
                if (S3 != 0) note(rn3f, err_f(ocu_dm::rnorm3df(a, b, c3), 1.0L / S3), a, b);
                if (S4 != 0) note(rn4f, err_f(ocu_dm::rnorm4df(a, b, c3, d4), 1.0L / S4), a, b);
            }
        }
    }
    const double FB = 0.5 + 0x1p-20;
    report("sinpi", sp, 1.5); report("cospi", cp, 1.5);
    report("acosh", ach, 3.0); report("asinh", ash, 2.0); report("atanh", ath, 2.0);
    report("rcbrt", rc, 1.5); report("rhypot", rh, 1.5);
    report("norm3d", n3, 1.5); report("norm4d", n4, 1.5); report("rnorm3d", rn3, 1.5); report("rnorm4d", rn4, 1.5);
    report("norm", nn, 1.5); report("rnorm", rnn, 1.5);
    report("sinpif", spf, FB); report("cospif", cpf, FB); report("acoshf", achf, FB); report("asinhf", ashf, FB);
    report("atanhf", athf, FB); report("rcbrtf", rcf, FB); report("rhypotf", rhf, FB);
    report("norm3df", n3f, FB); report("norm4df", n4f, FB); report("rnorm3df", rn3f, FB); report("rnorm4df", rn4f, FB);
}

/* double erf / erfc against long double, within the CUDA Math API maxima
   (erf 2 ulp, erfc 5 ulp): dense in [-6, 6] (where both change), erfc over
   its whole range up to its underflow near 27.3, tiny and negative x */
static void test_erf(bool full)
{
    worst we, wc;
    const int N = full ? 20000000 : 2000000;
    for (int i = 0; i < N; ++i) {
        double x;
        switch (i % 6) {
        case 0: x = ((double)(rng() >> 11) * 0x1p-53) * 12.0 - 6.0; break;       // [-6, 6]
        case 1: x = ((double)(rng() >> 11) * 0x1p-53) * 28.0; break;             // erfc's range
        case 2: x = std::ldexp((double)(rng() >> 11), -(int)(rng() % 1100) - 53); break; // tiny, subnormal
        case 3: x = 1.5 + ((double)(rng() >> 11) * 0x1p-53 - 0.5) * 1e-6; break;  // the switch at 1.5
        case 4: x = -((double)(rng() >> 11) * 0x1p-53) * 6.0; break;            // negative
        default: x = rand_double(); break;
        }
        if (!std::isfinite(x)) continue;
        note(we, err_d(ocu_dm::erf(x), std::erf((long double)x)), x);
        long double rc = std::erfc((long double)x);
        if (rc >= 0x1p-1074L || rc == 0) note(wc, err_d(ocu_dm::erfc(x), rc), x);
    }
    report("erf", we, 2.0);
    report("erfc", wc, 5.0);
}

static void test_special()
{
    CHECK(same(ocu_dm::erf(0.0), 0.0) && same(ocu_dm::erf(-0.0), -0.0), "erf(+-0) = +-0");
    CHECK(ocu_dm::erf(INFINITY) == 1.0 && ocu_dm::erf(-INFINITY) == -1.0 && std::isnan(ocu_dm::erf(NAN)), "erf(+-inf), erf(NaN)");
    CHECK(ocu_dm::erfc(INFINITY) == 0.0 && ocu_dm::erfc(-INFINITY) == 2.0 && std::isnan(ocu_dm::erfc(NAN)) && ocu_dm::erfc(0.0) == 1.0,
          "erfc(+-inf), erfc(NaN), erfc(0)");
    CHECK(ocu_dm::erfc(30.0) == 0.0 && ocu_dm::erfc(-30.0) == 2.0 && ocu_dm::erf(7.0) == 1.0, "erfc / erf far out");
    CHECK(samef(ocu_dm::acoshf(1.0f), 0.0f) && !std::signbit(ocu_dm::acoshf(1.0f)), "acoshf(1) = +0");
    CHECK(std::isnan(ocu_dm::acoshf(0.5f)) && std::isnan(ocu_dm::acoshf(-INFINITY)), "acoshf below 1 is NaN");
    CHECK(samef(ocu_dm::asinhf(-0.0f), -0.0f) && ocu_dm::asinhf(-INFINITY) == -INFINITY, "asinhf(-0), asinhf(-inf)");
    CHECK(ocu_dm::atanhf(1.0f) == INFINITY && ocu_dm::atanhf(-1.0f) == -INFINITY && std::isnan(ocu_dm::atanhf(1.5f)), "atanhf(+-1), outside");
    CHECK(samef(ocu_dm::cospif(0.0f), 1.0f) && samef(ocu_dm::cospif(-0.0f), 1.0f) && std::isnan(ocu_dm::cospif(INFINITY)), "cospif(+-0), inf");
    CHECK(samef(ocu_dm::cospif(0.5f), 0.0f) && samef(ocu_dm::cospif(-1.5f), 0.0f) && samef(ocu_dm::cospif(3.0f), -1.0f), "cospif(half-integers) +0");
    CHECK(samef(ocu_dm::sinpif(-0.0f), -0.0f) && samef(ocu_dm::sinpif(2.0f), 0.0f) && samef(ocu_dm::sinpif(-3.0f), -0.0f), "sinpif zeros");
    CHECK(samef(ocu_dm::sinpif(0.5f), 1.0f) && samef(ocu_dm::sinpif(-0.5f), -1.0f) && samef(ocu_dm::sinpif(1.5f), -1.0f), "sinpif(half)");
    CHECK(ocu_dm::frsqrt_rn(0.0f) == INFINITY && ocu_dm::frsqrt_rn(-0.0f) == -INFINITY && std::isnan(ocu_dm::frsqrt_rn(-1.0f)), "__frsqrt_rn(+-0), < 0");
    CHECK(samef(ocu_dm::frsqrt_rn(INFINITY), 0.0f) && samef(ocu_dm::frsqrt_rn(4.0f), 0.5f), "__frsqrt_rn(inf), (4)");
    CHECK(ocu_dm::rcbrtf(0.0f) == INFINITY && ocu_dm::rcbrtf(-0.0f) == -INFINITY && samef(ocu_dm::rcbrtf(-INFINITY), -0.0f), "rcbrtf(+-0), (-inf)");
    CHECK(samef(ocu_dm::rhypotf(INFINITY, NAN), 0.0f) && std::isnan(ocu_dm::rhypotf(1.0f, NAN)) && ocu_dm::rhypotf(0.0f, -0.0f) == INFINITY, "rhypotf");
    CHECK(ocu_dm::norm3d(1.0, INFINITY, NAN) == INFINITY && std::isnan(ocu_dm::norm4d(1, 2, NAN, 4)) && ocu_dm::rnorm3d(0, -0.0, 0) == INFINITY, "norm specials");
    CHECK(samef(ocu_dm::norm3df(DBL_MAX > 0 ? FLT_MAX : 0, FLT_MAX, 0), (float)(std::sqrt(2.0L) * FLT_MAX)), "norm3df does not overflow early");
    CHECK(ocu_dm::norm3d(DBL_MAX, DBL_MAX, 0) == INFINITY && ocu_dm::norm3d(DBL_MAX / 2, DBL_MAX / 2, 0) == (double)(std::sqrt(2.0L) * (DBL_MAX / 2)), "norm3d overflow threshold");
    CHECK(ocu_dm::ilogbf(0.0f) == INT_MIN && ocu_dm::ilogbf(NAN) == INT_MIN && ocu_dm::ilogbf(INFINITY) == INT_MAX, "ilogbf specials");
    CHECK(std::isnan(ocu_dm::nanf("")) && std::isnan(ocu_dm::nan("1")), "nan");
    float ip; CHECK(samef(ocu_dm::modff(-INFINITY, &ip), -0.0f) && ip == -INFINITY, "modff(-inf)");
}

static void test_frsqrt(bool full)
{
    const std::uint32_t step = full ? 1u : 61u;
    const unsigned nt = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4;
    std::vector<std::thread> th;
    std::vector<long long> bad(nt, 0), n(nt, 0);
    std::vector<std::uint32_t> first(nt, 0);
    for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            for (std::uint64_t b = 1 + (std::uint64_t)t * step; b < 0x7f800000u; b += (std::uint64_t)nt * step) {
                float x = fbits((std::uint32_t)b);
                float want = (float)(1.0L / std::sqrt((long double)x));
                ++n[t];
                if (!samef(ocu_dm::frsqrt_rn(x), want)) { if (!bad[t]++) first[t] = (std::uint32_t)b; }
            }
        });
    for (auto &x : th) x.join();
    long long tb = 0, tn = 0;
    for (unsigned t = 0; t < nt; ++t) { tb += bad[t]; tn += n[t]; if (bad[t]) std::printf("__frsqrt_rn wrong at %a\n", fbits(first[t])); }
    ++g_checks;
    if (tb) ++g_fail;
    std::printf("__frsqrt_rn: %lld of %lld positive floats not correctly rounded %s\n", tb, tn, tb ? "FAIL" : "");
}

int main()
{
    const bool full = std::getenv("OCU_MATH_EXT_FULL") != nullptr;
    test_special();
    test_exact();
    test_rounding_modes();
    test_frsqrt(full);
    test_accuracy(full);
    test_erf(full);
    std::printf("%s: %lld checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

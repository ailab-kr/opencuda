// Accuracy of the float-only functions of ocu_device_math.h (expf, sinf,
// powf, ...), compiled for the host, against the host's double functions
// (whose error, under 1 ulp of double, is 2^-29 ulp of float). Near a
// rounding boundary of float the reference is recomputed in long double.
//
// One-argument functions: every 257th float by default (about 16.7 million
// inputs each); OCU_MATHF_EXHAUSTIVE=1 runs all 2^32 inputs (minutes, all
// cores). Two-argument functions: random and edge-case pairs. Special
// values (NaN, infinities, signed zeros, subnormals) must match exactly.
//
// The bounds are the maximum ulp errors the CUDA math API documentation
// gives for these functions (single precision); the test fails above them.
// It prints, per function, the largest error, where it occurs, and how many
// results are not the correctly rounded float.
#include "ocu_device_math.h"

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static int g_fail;

static float as_float(std::uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

// ulp of float at the exact value v (the spacing of floats in its binade).
static double ulpf(double v)
{
    double a = std::fabs(v);
    if (a < 0x1p-126) return 0x1p-149;
    int e;
    std::frexp(a, &e); // a = m 2^e, m in [0.5, 1)
    return std::ldexp(1.0, e - 24);
}

// Error of `got` in ulps of the exact result `want`; special values exact.
static double err_ulp(float got, long double want)
{
    double w = (double)want;
    if (std::isnan(w) || std::isnan(got)) return (std::isnan(w) && std::isnan(got)) ? 0.0 : 1e9;
    float wr = (float)w;
    if (std::isinf(wr) || std::isinf(got)) {
        if (std::isinf(got) && std::isinf(wr) && (got > 0) == (wr > 0)) return 0.0;
        if (std::isinf(got)) return 1e9;
        return (double)(std::fabs((long double)got - want) / 0x1p104L); // ulp at FLT_MAX
    }
    if (got == 0 && w == 0 && std::signbit(got) != std::signbit(w)) return 1e9;
    return (double)(std::fabs((long double)got - want) / (long double)ulpf(w));
}

// Is `want` (double) so close to a midpoint between floats that the double
// reference cannot decide the rounding?
static bool near_midpoint(double want)
{
    if (!std::isfinite(want)) return false;
    double u = ulpf(want);
    double m = std::fmod(std::fabs(want), u) / u; // position inside the ulp
    return std::fabs(m - 0.5) < 1e-6;
}

struct result {
    double worst = 0;
    float at = 0;
    long long n = 0, not_cr = 0;
    void merge(const result &o)
    {
        if (o.worst > worst) { worst = o.worst; at = o.at; }
        n += o.n;
        not_cr += o.not_cr;
    }
};

static void report(const char *name, const result &r, double bound, const char *at2 = nullptr)
{
    bool ok = r.worst <= bound;
    char where[64];
    if (at2) std::snprintf(where, sizeof where, "%s", at2);
    else std::snprintf(where, sizeof where, "%a", (double)r.at);
    std::printf("%-8s max %7.4f ulp (bound %g) at %-22s  not correctly rounded %10lld of %12lld %s\n",
                name, r.worst, bound, where, r.not_cr, r.n, ok ? "" : "FAIL");
    std::fflush(stdout);
    if (!ok) ++g_fail;
}

static unsigned nthreads()
{
    unsigned n = std::thread::hardware_concurrency();
    if (const char *e = std::getenv("OCU_MATHF_THREADS")) n = (unsigned)std::atoi(e);
    return n ? n : 4;
}

static bool g_exhaustive;

using F1 = float (*)(float);
using R1 = double (*)(double);
using L1 = long double (*)(long double);

// All floats (or every 257th) through ours vs ref.
static void sweep1(const char *name, F1 ours, R1 ref, L1 refl, double bound)
{
    const std::uint64_t step = g_exhaustive ? 1 : 257;
    unsigned nt = nthreads();
    std::vector<result> res(nt);
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t) {
        th.emplace_back([&, t] {
            result r;
            std::uint64_t lo = (0x100000000ull * t / nt) / step * step;
            std::uint64_t hi = 0x100000000ull * (t + 1) / nt;
            for (std::uint64_t b = lo; b < hi; b += step) {
                float x = as_float((std::uint32_t)b);
                float got = ours(x);
                double w = ref((double)x);
                long double wl = w;
                if (near_midpoint(w)) wl = refl((long double)x);
                double e = err_ulp(got, wl);
                ++r.n;
                if ((float)wl != got && !(std::isnan(got) && std::isnan((double)wl))) ++r.not_cr;
                if (e > r.worst) { r.worst = e; r.at = x; }
            }
            res[t] = r;
        });
    }
    for (auto &x : th) x.join();
    result all;
    for (auto &r : res) all.merge(r);
    report(name, all, bound);
}

// Pairs (x, y) from a generator, in parallel.
template <typename Ours, typename Ref, typename Gen>
static void sweep2(const char *name, Ours ours, Ref ref, Gen gen, long long count, double bound)
{
    unsigned nt = nthreads();
    std::vector<result> res(nt);
    std::vector<float> wy(nt), wx(nt);
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t) {
        th.emplace_back([&, t] {
            std::mt19937_64 g(1234 + t);
            result r;
            float bx = 0, by = 0;
            for (long long i = t; i < count; i += nt) {
                float x, y;
                gen(g, i, x, y);
                float got = ours(x, y);
                long double w = ref((long double)x, (long double)y);
                double e = err_ulp(got, w);
                ++r.n;
                if ((float)(double)w != got && !(std::isnan(got) && std::isnan((double)w))) ++r.not_cr;
                if (e > r.worst) { r.worst = e; bx = x; by = y; }
            }
            res[t] = r;
            wx[t] = bx;
            wy[t] = by;
        });
    }
    for (auto &x : th) x.join();
    result all;
    float ax = 0, ay = 0;
    for (unsigned t = 0; t < nt; ++t) {
        if (res[t].worst > all.worst || t == 0) { ax = wx[t]; ay = wy[t]; }
        all.merge(res[t]);
    }
    char at[64];
    std::snprintf(at, sizeof at, "(%a,%a)", (double)ax, (double)ay);
    report(name, all, bound, at);
}

static const float SPECIAL[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f,
                                -2.0f, 3.0f, -3.0f, 1.5f, -1.5f, 0x1p-149f, -0x1p-149f, 0x1p-126f,
                                FLT_MAX, -FLT_MAX, 0x1.000002p0f, 0x1.fffffep-1f, -0x1.fffffep-1f,
                                10.0f, 0.1f, 1e-20f, 1e20f, 0x1p24f, 0x1p25f + 2.0f, 127.0f, 128.0f,
                                -149.0f, -150.0f, 0x1.8p-140f};
static const int NSPECIAL = sizeof SPECIAL / sizeof SPECIAL[0];

// Two-argument special values: every pair of SPECIAL against the host's
// double function (C99 Annex F), which must agree exactly when exact.
template <typename Ours, typename Ref>
static void specials2(const char *name, Ours ours, Ref ref, double bound)
{
    result r;
    for (int i = 0; i < NSPECIAL; ++i)
        for (int j = 0; j < NSPECIAL; ++j) {
            float x = SPECIAL[i], y = SPECIAL[j];
            double e = err_ulp(ours(x, y), (long double)ref((double)x, (double)y));
            ++r.n;
            if (e > r.worst) { r.worst = e; r.at = x; }
        }
    char nm[32];
    std::snprintf(nm, sizeof nm, "%s_sp", name);
    report(nm, r, bound);
}

#if defined(__GLIBC__)
static double ref_exp10(double x) { return ::exp10(x); }
static long double ref_exp10l(long double x) { return ::exp10l(x); }
#else
static double ref_exp10(double x) { return std::pow(10.0, x); }
static long double ref_exp10l(long double x) { return std::pow(10.0L, x); }
#endif

static float log_uniform(std::mt19937_64 &g, float lo2, float hi2)
{
    std::uniform_real_distribution<float> u(lo2, hi2);
    return std::exp2(u(g));
}

int main()
{
    g_exhaustive = std::getenv("OCU_MATHF_EXHAUSTIVE") && std::atoi(std::getenv("OCU_MATHF_EXHAUSTIVE"));
    std::printf("test_device_mathf: %s, %u threads\n",
                g_exhaustive ? "all 2^32 inputs" : "every 257th input", nthreads());
#define ONE(name, b)                                                                                \
    sweep1(#name "f", [](float x) { return ocu_dm::name##f(x); }, [](double x) { return ::name(x); }, \
           [](long double x) { return ::name##l(x); }, b)
    ONE(exp, 2);
    ONE(exp2, 2);
    sweep1("exp10f", [](float x) { return ocu_dm::exp10f(x); }, ref_exp10, ref_exp10l, 2);
    ONE(expm1, 1);
    ONE(log, 1);
    ONE(log2, 1);
    ONE(log10, 2);
    ONE(log1p, 1);
#if defined(_WIN32)
    // MinGW's sin/cos/tan use the x87 instructions, whose argument reduction
    // is wrong for large |x| (sin(0x1.afda28p+127) comes out ~1.5e-7 instead
    // of -0.58), so there is no reference here. Checked on Linux and macOS.
    std::printf("sinf/cosf/tanf/sincosf: not checked (no accurate reference on this host)\n");
#else
    ONE(sin, 2);
    ONE(cos, 2);
    ONE(tan, 4);
    sweep1("sincosf.s", [](float x) { float s, c; ocu_dm::sincosf(x, &s, &c); return s; },
           [](double x) { return ::sin(x); }, [](long double x) { return ::sinl(x); }, 2);
    sweep1("sincosf.c", [](float x) { float s, c; ocu_dm::sincosf(x, &s, &c); return c; },
           [](double x) { return ::cos(x); }, [](long double x) { return ::cosl(x); }, 2);
#endif
    ONE(asin, 2);
    ONE(acos, 2);
    ONE(atan, 2);
    ONE(sinh, 3);
    ONE(cosh, 2);
    ONE(tanh, 2);
    ONE(cbrt, 1);
    ONE(erf, 2);
    ONE(erfc, 4);
#undef ONE

    const long long N2 = g_exhaustive ? 400000000LL : 20000000LL;
    auto powf_ = [](float x, float y) { return ocu_dm::powf(x, y); };
    auto powl_ = [](long double x, long double y) { return (long double)::pow((double)x, (double)y); };
    // powf: |x| over the whole range with y such that the result is in or
    // near the float range; integer y with negative x; x near 1, large y.
    sweep2("powf", powf_, powl_, [](std::mt19937_64 &g, long long i, float &x, float &y) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        switch (i % 4) {
        case 0: {
            x = as_float((std::uint32_t)(g() & 0x7fffffffu));
            if (!std::isfinite(x) || x == 0 || x == 1) x = 3.0f;
            double l = std::log2((double)x);
            y = (float)((-160.0 + 290.0 * u(g)) / (l == 0 ? 1 : l));
            break;
        }
        case 1:
            x = -log_uniform(g, -20, 20);
            y = (float)std::rint(-40.0 + 80.0 * u(g));
            break;
        case 2:
            x = 1.0f + (float)((std::int64_t)(g() % 4001) - 2000) * 0x1p-24f;
            y = (float)((u(g) - 0.5) * 0x1p25);
            break;
        default:
            x = log_uniform(g, -126, 127);
            y = (float)(u(g) * 4.0 - 2.0);
            break;
        }
    }, N2, 4);
    specials2("powf", powf_, [](double x, double y) { return ::pow(x, y); }, 4);

    auto atan2f_ = [](float y, float x) { return ocu_dm::atan2f(y, x); };
    auto atan2l_ = [](long double y, long double x) { return (long double)::atan2((double)y, (double)x); };
    sweep2("atan2f", atan2f_, atan2l_, [](std::mt19937_64 &g, long long i, float &y, float &x) {
        if (i % 2) { // any finite pair
            do { y = as_float((std::uint32_t)g()); } while (!std::isfinite(y));
            do { x = as_float((std::uint32_t)g()); } while (!std::isfinite(x));
        } else { // ratios near 1, and moderate ranges
            x = log_uniform(g, -10, 10) * ((g() & 1) ? 1 : -1);
            y = x * (float)std::exp2(std::uniform_real_distribution<double>(-30, 30)(g)) * ((g() & 1) ? 1 : -1);
        }
    }, N2, 3);
    specials2("atan2f", atan2f_, [](double y, double x) { return ::atan2(y, x); }, 3);

    auto hypotf_ = [](float x, float y) { return ocu_dm::hypotf(x, y); };
    auto hypotl_ = [](long double x, long double y) { return (long double)::hypot((double)x, (double)y); };
    sweep2("hypotf", hypotf_, hypotl_, [](std::mt19937_64 &g, long long i, float &x, float &y) {
        if (i % 2) {
            do { x = as_float((std::uint32_t)g()); } while (!std::isfinite(x));
            do { y = as_float((std::uint32_t)g()); } while (!std::isfinite(y));
        } else {
            x = as_float((std::uint32_t)g());
            if (!std::isfinite(x)) x = 1.0f;
            y = x * (float)std::exp2(std::uniform_real_distribution<double>(-14, 14)(g));
        }
    }, N2, 3);
    specials2("hypotf", hypotf_, [](double x, double y) { return ::hypot(x, y); }, 3);

    std::printf("test_device_mathf: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail ? 1 : 0;
}

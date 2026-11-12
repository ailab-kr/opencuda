// Accuracy of ocu_device_math.h, compiled for the host, against the host's
// long double functions (more precise than double). Errors are measured in
// units in the last place (ulp) of the double result. Prints the largest
// error per function and fails when a function exceeds its documented bound.
#include "ocu_device_math.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

static int g_fail;

static double ulp_of(double r)
{
    double a = std::fabs(r);
    if (a < DBL_MIN) return std::ldexp(1.0, -1074);
    int e;
    std::frexp(a, &e);
    return std::ldexp(1.0, e - 53);
}

// Error of `got` against the exact `want`, in ulps of the correctly
// rounded result. Special values must agree exactly (NaN with NaN).
static double ulps(double got, long double want)
{
    if (std::isnan((double)want) || std::isnan(got)) {
        return (std::isnan((double)want) && std::isnan(got)) ? 0.0 : 1e9;
    }
    double w = (double)want;
    if (std::isinf(w) || std::isinf(got)) return (w == got) ? 0.0 : 1e9;
    return (double)(std::fabs((long double)got - want) / (long double)ulp_of(w));
}

struct err_stat {
    const char *name;
    double worst = 0, at = 0;
    long n = 0;
};

static void report(err_stat &s, double bound)
{
    bool ok = s.worst <= bound;
    std::printf("%-8s max %6.3f ulp (bound %.1f) at %-24.17g over %ld points %s\n", s.name, s.worst,
                bound, s.at, s.n, ok ? "" : "FAIL");
    if (!ok) ++g_fail;
}

template <typename F, typename G>
static void sweep1(const char *name, F ours, G ref, double lo, double hi, bool logscale, double bound,
                   std::mt19937_64 &g, long n = 200000)
{
    err_stat s;
    s.name = name;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (long i = 0; i < n; ++i) {
        double x;
        if (logscale) {
            double e = std::log(lo) + (std::log(hi) - std::log(lo)) * u(g);
            x = std::exp(e);
            if (g() & 1) x = -x;
        } else {
            x = lo + (hi - lo) * u(g);
        }
        double e = ulps(ours(x), ref((long double)x));
        ++s.n;
        if (e > s.worst) { s.worst = e; s.at = x; }
    }
    report(s, bound);
}

int main()
{
    if (LDBL_MANT_DIG <= DBL_MANT_DIG) {
        std::printf("test_device_math: SKIP (long double is no more precise than double here)\n");
        return 0;
    }
    std::mt19937_64 g(2026);
    using namespace ocu_dm;
    // exp, log families
    sweep1("exp", [](double x) { return ocu_dm::exp(x); }, [](long double x) { return expl(x); }, -745, 709.7, false, 1.0, g);
    sweep1("exp_s", [](double x) { return ocu_dm::exp(x); }, [](long double x) { return expl(x); }, -1, 1, false, 1.0, g);
    sweep1("expm1", [](double x) { return ocu_dm::expm1(x); }, [](long double x) { return expm1l(x); }, -1, 1, false, 1.5, g);
    sweep1("log", [](double x) { return ocu_dm::log(std::fabs(x)); }, [](long double x) { return logl(fabsl(x)); }, 1e-300, 1e300, true, 1.0, g);
    sweep1("log_1", [](double x) { return ocu_dm::log(x); }, [](long double x) { return logl(x); }, 0.5, 2.0, false, 1.0, g);
    sweep1("log2", [](double x) { return ocu_dm::log2(std::fabs(x)); }, [](long double x) { return log2l(fabsl(x)); }, 1e-300, 1e300, true, 1.0, g);
    sweep1("log10", [](double x) { return ocu_dm::log10(std::fabs(x)); }, [](long double x) { return log10l(fabsl(x)); }, 1e-300, 1e300, true, 1.0, g);
    sweep1("log1p", [](double x) { return ocu_dm::log1p(x); }, [](long double x) { return log1pl(x); }, -0.9, 10, false, 1.5, g);
    sweep1("exp2", [](double x) { return ocu_dm::exp2(x); }, [](long double x) { return exp2l(x); }, -1070, 1023, false, 1.0, g);
    sweep1("exp10", [](double x) { return ocu_dm::exp10(x); }, [](long double x) { return powl(10.0L, x); }, -300, 300, false, 1.5, g);
    // trigonometry
    sweep1("sin", [](double x) { return ocu_dm::sin(x); }, [](long double x) { return sinl(x); }, -1e5, 1e5, false, 1.0, g);
    sweep1("sin_s", [](double x) { return ocu_dm::sin(x); }, [](long double x) { return sinl(x); }, -4, 4, false, 1.0, g);
    sweep1("cos", [](double x) { return ocu_dm::cos(x); }, [](long double x) { return cosl(x); }, -1e5, 1e5, false, 1.0, g);
    sweep1("tan", [](double x) { return ocu_dm::tan(x); }, [](long double x) { return tanl(x); }, -1e4, 1e4, false, 2.5, g);
    sweep1("atan", [](double x) { return ocu_dm::atan(x); }, [](long double x) { return atanl(x); }, 1e-10, 1e10, true, 1.0, g);
    sweep1("asin", [](double x) { return ocu_dm::asin(x); }, [](long double x) { return asinl(x); }, -1, 1, false, 2.0, g);
    sweep1("acos", [](double x) { return ocu_dm::acos(x); }, [](long double x) { return acosl(x); }, -1, 1, false, 2.0, g);
    // hyperbolic, others
    sweep1("sinh", [](double x) { return ocu_dm::sinh(x); }, [](long double x) { return sinhl(x); }, -700, 700, false, 2.0, g);
    sweep1("cosh", [](double x) { return ocu_dm::cosh(x); }, [](long double x) { return coshl(x); }, -700, 700, false, 2.0, g);
    sweep1("tanh", [](double x) { return ocu_dm::tanh(x); }, [](long double x) { return tanhl(x); }, -20, 20, false, 2.0, g);
    sweep1("cbrt", [](double x) { return ocu_dm::cbrt(x); }, [](long double x) { return cbrtl(x); }, 1e-300, 1e300, true, 1.0, g);
    sweep1("sqrt", [](double x) { return ocu_dm::sqrt(std::fabs(x)); }, [](long double x) { return sqrtl(fabsl(x)); }, 1e-300, 1e300, true, 0.5, g);

    // pow over mixed ranges, including negative bases with integer exponents
    {
        err_stat s;
        s.name = "pow";
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (long i = 0; i < 300000; ++i) {
            double x = std::exp(-20.0 + 40.0 * u(g));
            double y = -30.0 + 60.0 * u(g);
            if (i % 5 == 0) { y = std::rint(y); if (i % 10 == 0) x = -x; }
            double e = ulps(ocu_dm::pow(x, y), powl((long double)x, (long double)y));
            ++s.n;
            if (e > s.worst) { s.worst = e; s.at = x; }
        }
        report(s, 1.0);
    }
    {
        err_stat s;
        s.name = "atan2";
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (long i = 0; i < 200000; ++i) {
            double y = u(g) * 1e3, x = u(g) * 1e3;
            double e = ulps(ocu_dm::atan2(y, x), atan2l((long double)y, (long double)x));
            ++s.n;
            if (e > s.worst) { s.worst = e; s.at = y; }
        }
        report(s, 2.0);
    }
    {
        err_stat s;
        s.name = "fmod";
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (long i = 0; i < 200000; ++i) {
            double x = std::exp(-30 + 60 * u(g)) * ((g() & 1) ? -1 : 1);
            double y = std::exp(-30 + 60 * u(g));
            double e = ulps(ocu_dm::fmod(x, y), (long double)std::fmod(x, y)); // exact in both
            ++s.n;
            if (e > s.worst) { s.worst = e; s.at = x; }
        }
        report(s, 0.0);
    }
    {
        err_stat s;
        s.name = "hypot";
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (long i = 0; i < 200000; ++i) {
            double x = std::exp(-300 + 600 * u(g)), y = std::exp(-300 + 600 * u(g));
            double e = ulps(ocu_dm::hypot(x, y), hypotl((long double)x, (long double)y));
            ++s.n;
            if (e > s.worst) { s.worst = e; s.at = x; }
        }
        report(s, 1.0);
    }

    // float versions: correctly rounded or 1 ulp of float
    {
        int bad = 0;
        std::uniform_real_distribution<float> u(-80.0f, 80.0f);
        for (int i = 0; i < 200000; ++i) {
            float x = u(g);
            float a = ocu_dm::expf(x), b = (float)expl((long double)x);
            float c = ocu_dm::sinf(x * 100.0f), d = (float)sinl((long double)(x * 100.0f));
            float base = std::fabs(x) + 0.5f, ex = x * 0.25f;
            float e = ocu_dm::powf(base, ex), f = (float)powl((long double)base, (long double)ex);
            auto off = [](float p, float q) { return std::fabs(std::nextafter(p, q) - q) > 0 && p != q; };
            bad += off(a, b) || off(c, d) || off(e, f);
        }
        std::printf("float    expf/sinf/powf within 1 float ulp: %s (%d off)\n", bad ? "FAIL" : "ok", bad);
        if (bad) ++g_fail;
    }

    // special values
    {
        int bad = 0;
        bad += !std::isinf(ocu_dm::exp(1000.0)) || ocu_dm::exp(-1000.0) != 0.0 || ocu_dm::exp(0.0) != 1.0;
        bad += ocu_dm::log(1.0) != 0.0 || !std::isinf(ocu_dm::log(0.0)) || !std::isnan(ocu_dm::log(-1.0));
        bad += ocu_dm::pow(-2.0, 3.0) != -8.0 || ocu_dm::pow(-2.0, 2.0) != 4.0 || !std::isnan(ocu_dm::pow(-2.0, 0.5));
        bad += ocu_dm::pow(0.0, -1.0) != INFINITY || ocu_dm::pow(-0.0, -3.0) != -INFINITY || ocu_dm::pow(2.0, 0.0) != 1.0;
        bad += ocu_dm::pow(1.0, NAN) != 1.0 || ocu_dm::pow(NAN, 0.0) != 1.0 || ocu_dm::pow(0.5, INFINITY) != 0.0;
        bad += ocu_dm::sin(0.0) != 0.0 || !std::signbit(ocu_dm::sin(-0.0)) || !std::isnan(ocu_dm::sin(INFINITY));
        bad += ocu_dm::atan2(0.0, -1.0) != ocu_dm::PI || ocu_dm::atan2(-0.0, -1.0) != -ocu_dm::PI;
        bad += std::fabs(ocu_dm::atan2(INFINITY, -INFINITY) - 2.356194490192345) > 1e-15;
        bad += ocu_dm::log2(8.0) != 3.0 || ocu_dm::exp2(10.0) != 1024.0 || ocu_dm::cbrt(-27.0) != -3.0;
        bad += ocu_dm::fmod(5.5, 2.0) != 1.5 || ocu_dm::fmod(-5.5, 2.0) != -1.5 || !std::isnan(ocu_dm::fmod(1.0, 0.0));
        bad += ocu_dm::ldexp(1.0, -1074) != std::ldexp(1.0, -1074) || ocu_dm::ldexp(1.5, 1023) != std::ldexp(1.5, 1023);
        std::printf("special  values: %s (%d wrong)\n", bad ? "FAIL" : "ok", bad);
        if (bad) ++g_fail;
    }
    std::printf("test_device_math: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail ? 1 : 0;
}

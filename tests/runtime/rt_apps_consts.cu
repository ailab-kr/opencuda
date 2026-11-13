// math_constants.h: every CUDART_* constant against an independent reference.
//
// Expectations come from the public CUDA Math API description of the
// constants (each is the named mathematical constant in single or double
// precision; *_HI + *_LO split a constant into its nearest representable
// part and the nearest-representable remainder). The references below are
// decimal strings with 45 digits, computed outside this program with 80-digit
// arithmetic (pi by Machin's formula, logs and roots by decimal arithmetic).
// The strings are parsed with strtod/strtof/strtold, which round correctly,
// so the expected bits are "the constant rounded once". Comparison is by
// bit pattern: one ulp of difference fails.
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint64_t bits(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }

static void report(const char *name, const char *what, double got, double want)
{
    std::printf("FAIL %s %s: got %.17g (0x%016llx) want %.17g (0x%016llx)\n", name, what, got,
                (unsigned long long)bits(got), want, (unsigned long long)bits(want));
}

#define CHECK_D(name, got, want) do { ++g_checks; if (bits((double)(got)) != bits((double)(want))) { ++g_fail; report(name, "(double)", (got), (want)); } } while (0)
#define CHECK_F(name, got, want) do { ++g_checks; if (bits((float)(got)) != bits((float)(want))) { ++g_fail; report(name, "(float)", (got), (want)); } } while (0)

// Reference decimal values (45 digits).
#define REF_SQRT_TWO "1.414213562373095048801688724209698078569671875"
#define REF_SQRT_HALF "0.707106781186547524400844362104849039284835938"
#define REF_THIRD "0.333333333333333333333333333333333333333333333"
#define REF_TWOTHIRD "0.666666666666666666666666666666666666666666667"
#define REF_PIO4 "0.785398163397448309615660845819875721049292350"
#define REF_PIO2 "1.570796326794896619231321691639751442098584700"
#define REF_3PIO4 "2.356194490192344928846982537459627163147877050"
#define REF_2_OVER_PI "0.636619772367581343075535053490057448137838583"
#define REF_PI "3.141592653589793238462643383279502884197169399"
#define REF_SQRT_2PI "2.506628274631000502415765284811045253006986741"
#define REF_SQRT_PIO2 "1.253314137315500251207882642405522626503493370"
#define REF_SQRT_2OPI "0.797884560802865355879892119868763736951717262"
#define REF_L2E "1.442695040888963407359924681001892137426645954"
#define REF_L2T "3.321928094887362347870319429489390175864831393"
#define REF_LG2 "0.301029995663981195213738894724493026768189881"
#define REF_LGE "0.434294481903251827651128918916605082294397006"
#define REF_LN2 "0.693147180559945309417232121458176568075500134"
#define REF_LNT "2.302585092994045684017991454684364207601101489"
#define REF_LNPI "1.144729885849400174143427351353058711647294813"

// Remainders c - double(c) (40 significant digits, exact arithmetic).
#define LO_SQRT_HALF "-4.8336466567264567255273498648834738250649e-17"
#define LO_PIO4 "3.0616169978683830179344100736459915115642e-17"
#define LO_PIO2 "6.1232339957367660358688201472919830231285e-17"
#define LO_PI "1.2246467991473532071737640294583966046257e-16"
#define LO_SQRT_2PI "-1.8328579980459166773380100349471617314187e-16"
#define LO_SQRT_PIO2 "-9.1642899902295833866900501747358086570937e-17"
#define LO_LG2 "-2.8037281277851703937326013621432662713692e-18"

#define EACH_D(X) \
    X(SQRT_TWO) X(SQRT_HALF) X(THIRD) X(TWOTHIRD) X(PIO4) X(PIO2) X(3PIO4) X(2_OVER_PI) \
    X(PI) X(SQRT_2PI) X(SQRT_PIO2) X(SQRT_2OPI) X(L2E) X(L2T) X(LG2) X(LGE) X(LN2) X(LNT) X(LNPI)
#define EACH_F(X) \
    X(SQRT_TWO) X(SQRT_HALF) X(THIRD) X(PIO4) X(PIO2) X(3PIO4) X(2_OVER_PI) X(SQRT_2OPI) \
    X(PI) X(L2E) X(L2T) X(LG2) X(LN2) X(LNPI)
#define EACH_HILO(X) X(SQRT_HALF) X(PIO4) X(PIO2) X(PI) X(SQRT_2PI) X(SQRT_PIO2) X(LG2)

// Constants are macros; they must work in constant expressions and have
// the documented type (single: float, plain: double).
static_assert(std::is_same<decltype(CUDART_PI_F), float>::value, "CUDART_PI_F is float");
static_assert(std::is_same<decltype(CUDART_PI), double>::value, "CUDART_PI is double");
static_assert(std::is_same<decltype(CUDART_INF_F), float>::value, "CUDART_INF_F is float");
static_assert(std::is_same<decltype(CUDART_INF), double>::value, "CUDART_INF is double");
static_assert(std::is_same<decltype(CUDART_NAN_F), float>::value, "CUDART_NAN_F is float");
static_assert(std::is_same<decltype(CUDART_NAN), double>::value, "CUDART_NAN is double");
static_assert(std::is_same<decltype(CUDART_TWO_TO_31_F), float>::value, "float");
static_assert(std::is_same<decltype(CUDART_TWO_TO_31), double>::value, "double");
static_assert(CUDART_ZERO == 0.0 && CUDART_ONE == 1.0 && CUDART_ZERO_F == 0.0f && CUDART_ONE_F == 1.0f,
              "usable in constant expressions");

// Long double is an 80-bit extended on this host; checked so that a
// platform with a narrower one fails loudly instead of testing nothing.
#ifndef __CUDA_ARCH__
static_assert(LDBL_MANT_DIG >= 64, "need an extended long double");
#endif

static void test_values()
{
#define D_(n) CHECK_D("CUDART_" #n, CUDART_##n, std::strtod(REF_##n, nullptr));
    EACH_D(D_)
#undef D_
#define F_(n) CHECK_F("CUDART_" #n "_F", CUDART_##n##_F, std::strtof(REF_##n, nullptr));
    EACH_F(F_)
#undef F_
}

static void test_hilo()
{
    // double: HI is the nearest double, LO the nearest double of the remainder.
#define H_(n)                                                                                       \
    CHECK_D("CUDART_" #n "_HI", CUDART_##n##_HI, std::strtod(REF_##n, nullptr));                     \
    CHECK_D("CUDART_" #n "_LO", CUDART_##n##_LO, std::strtod(LO_##n, nullptr));                      \
    {                                                                                                \
        long double sum = (long double)CUDART_##n##_HI + (long double)CUDART_##n##_LO;               \
        long double ref = std::strtold(REF_##n, nullptr);                                            \
        long double just_hi = std::fabs((long double)CUDART_##n##_HI - ref);                         \
        ++g_checks;                                                                                  \
        if (!(std::fabs(sum - ref) < just_hi)) { ++g_fail; std::printf("FAIL HI+LO of " #n " not closer than HI\n"); } \
    }
    EACH_HILO(H_)
#undef H_
    // The only float split: SQRT_HALF_HI_F is the nearest float and
    // SQRT_HALF_LO_F the nearest float of the remainder.
    float hi = std::strtof(REF_SQRT_HALF, nullptr);
    long double rem = std::strtold(REF_SQRT_HALF, nullptr) - (long double)hi;
    CHECK_F("CUDART_SQRT_HALF_HI_F", CUDART_SQRT_HALF_HI_F, hi);
    CHECK_F("CUDART_SQRT_HALF_LO_F", CUDART_SQRT_HALF_LO_F, (float)rem);
    // HI + LO = constant to better than float precision
    CHECK(std::fabs((long double)CUDART_SQRT_HALF_HI_F + (long double)CUDART_SQRT_HALF_LO_F -
                    std::strtold(REF_SQRT_HALF, nullptr)) < 1e-14L);
}

static void test_special()
{
    // single
    CHECK(std::isinf(CUDART_INF_F) && CUDART_INF_F > 0);
    CHECK(bits(CUDART_INF_F) == 0x7f800000u);
    CHECK(std::isnan(CUDART_NAN_F));
    CHECK(std::signbit(CUDART_NEG_ZERO_F) && CUDART_NEG_ZERO_F == 0.0f);
    CHECK(bits(CUDART_NEG_ZERO_F) == 0x80000000u);
    CHECK(bits(CUDART_ZERO_F) == 0u);
    CHECK(CUDART_ONE_F == 1.0f);
    CHECK(bits(CUDART_MIN_DENORM_F) == 1u);                 // smallest positive subnormal 2^-149
    CHECK(CUDART_MIN_DENORM_F == std::ldexp(1.0f, -149));
    CHECK(CUDART_MAX_NORMAL_F == FLT_MAX);
    CHECK(bits(CUDART_MAX_NORMAL_F) == 0x7f7fffffu);
    CHECK(CUDART_TWO_TO_23_F == 8388608.0f);
    CHECK(CUDART_TWO_TO_24_F == 16777216.0f);
    CHECK(CUDART_TWO_TO_31_F == 2147483648.0f);
    CHECK(CUDART_TWO_TO_32_F == 4294967296.0f);
    // double
    CHECK(std::isinf(CUDART_INF) && CUDART_INF > 0);
    CHECK(bits(CUDART_INF) == 0x7ff0000000000000ull);
    CHECK(std::isnan(CUDART_NAN));
    CHECK(std::signbit(CUDART_NEG_ZERO) && CUDART_NEG_ZERO == 0.0);
    CHECK(bits(CUDART_NEG_ZERO) == 0x8000000000000000ull);
    CHECK(bits(CUDART_ZERO) == 0ull);
    CHECK(CUDART_ONE == 1.0);
    CHECK(bits(CUDART_MIN_DENORM) == 1ull);                 // 2^-1074
    CHECK(CUDART_MAX_NORMAL == DBL_MAX);
    CHECK(bits(CUDART_MAX_NORMAL) == 0x7fefffffffffffffull);
    CHECK(CUDART_TWO_TO_23 == 8388608.0);
    CHECK(CUDART_TWO_TO_24 == 16777216.0);
    CHECK(CUDART_TWO_TO_31 == 2147483648.0);
    CHECK(CUDART_TWO_TO_32 == 4294967296.0);
    // single constant is the double constant rounded (not the other way)
    CHECK(CUDART_PI_F == (float)CUDART_PI);
    CHECK(CUDART_L2E_F == (float)CUDART_L2E);
    // Known literal anchors from the documentation text of the values.
    CHECK(CUDART_PI == 3.1415926535897931);
    CHECK(CUDART_PI_F == 3.14159274f);
    CHECK(CUDART_SQRT_TWO == 1.4142135623730951);
}

// The constants read inside a kernel are the same bits as on the host.
#define DEV_N 26
__global__ void read_consts(double *d, float *f)
{
    d[0] = CUDART_PI; d[1] = CUDART_PIO2; d[2] = CUDART_SQRT_2PI; d[3] = CUDART_L2T;
    d[4] = CUDART_LNPI; d[5] = CUDART_PI_HI; d[6] = CUDART_PI_LO; d[7] = CUDART_INF;
    d[8] = CUDART_NEG_ZERO; d[9] = CUDART_MIN_DENORM; d[10] = CUDART_MAX_NORMAL;
    f[0] = CUDART_PI_F; f[1] = CUDART_L2E_F; f[2] = CUDART_SQRT_HALF_LO_F; f[3] = CUDART_INF_F;
    f[4] = CUDART_NEG_ZERO_F; f[5] = CUDART_MIN_DENORM_F; f[6] = CUDART_MAX_NORMAL_F;
    f[7] = CUDART_TWO_TO_31_F; f[8] = CUDART_THIRD_F;
}

static void test_device()
{
    double *d; float *f;
    CHECK(cudaMalloc(&d, DEV_N * sizeof(double)) == cudaSuccess);
    CHECK(cudaMalloc(&f, DEV_N * sizeof(float)) == cudaSuccess);
    read_consts<<<1, 1>>>(d, f);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    double hd[DEV_N] = {}; float hf[DEV_N] = {};
    CHECK(cudaMemcpy(hd, d, sizeof hd, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(hf, f, sizeof hf, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK_D("dev PI", hd[0], std::strtod(REF_PI, nullptr));
    CHECK_D("dev PIO2", hd[1], std::strtod(REF_PIO2, nullptr));
    CHECK_D("dev SQRT_2PI", hd[2], std::strtod(REF_SQRT_2PI, nullptr));
    CHECK_D("dev L2T", hd[3], std::strtod(REF_L2T, nullptr));
    CHECK_D("dev LNPI", hd[4], std::strtod(REF_LNPI, nullptr));
    CHECK_D("dev PI_HI", hd[5], std::strtod(REF_PI, nullptr));
    CHECK_D("dev PI_LO", hd[6], std::strtod(LO_PI, nullptr));
    CHECK(bits(hd[7]) == 0x7ff0000000000000ull);
    CHECK(bits(hd[8]) == 0x8000000000000000ull);
    CHECK(bits(hd[9]) == 1ull);
    CHECK(hd[10] == DBL_MAX);
    CHECK_F("dev PI_F", hf[0], std::strtof(REF_PI, nullptr));
    CHECK_F("dev L2E_F", hf[1], std::strtof(REF_L2E, nullptr));
    CHECK_F("dev SQRT_HALF_LO_F", hf[2], CUDART_SQRT_HALF_LO_F);
    CHECK(bits(hf[3]) == 0x7f800000u);
    CHECK(bits(hf[4]) == 0x80000000u);
    CHECK(bits(hf[5]) == 1u);
    CHECK(hf[6] == FLT_MAX);
    CHECK(hf[7] == 2147483648.0f);
    CHECK_F("dev THIRD_F", hf[8], std::strtof(REF_THIRD, nullptr));
    cudaFree(d); cudaFree(f);
}

int main(int argc, char **)
{
    test_values();
    test_hilo();
    test_special();
    test_device();
#ifdef OCU_MUTANT
    CHECK_D("mutant", CUDART_PI, 3.1415926535897936); // one ulp up: must fail
#endif
    (void)argc;
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

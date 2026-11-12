// Half precision against an independent oracle: every finite binary16
// value is an integer multiple of 2^-24, so values are compared as exact
// integers (sums and products as multiples of 2^-48 in 128 bits), and the
// nearest representable value is found in a sorted table of all halves.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

typedef __int128 i128;

// Exact value of a finite half, in units of 2^-24.
static long long units24(std::uint16_t h)
{
    long long e = (h >> 10) & 0x1f, m = h & 0x3ff;
    long long v = e == 0 ? m : (m | 0x400) << (e - 1);
    return (h & 0x8000) ? -v : v;
}

struct entry {
    long long v; // units of 2^-24
    std::uint16_t bits;
};
static std::vector<entry> g_table; // finite halves, ascending, -0 left out

static void build_table()
{
    for (unsigned b = 0; b < 65536; ++b) {
        std::uint16_t h = std::uint16_t(b);
        if (((h >> 10) & 0x1f) == 0x1f) continue; // inf, NaN
        if (h == 0x8000) continue;                 // -0: same value as +0
        g_table.push_back({units24(h), h});
    }
    std::sort(g_table.begin(), g_table.end(), [](const entry &a, const entry &b) { return a.v < b.v; });
}

// Nearest half to x * 2^-shift (x exact), ties to even mantissa;
// beyond the largest finite value by half a step or more: infinity.
static std::uint16_t nearest(i128 x, int shift)
{
    // Compare in units of 2^-shift (shift >= 24).
    const i128 scale = i128(1) << (shift - 24);
    const i128 maxv = i128(65504) * (i128(1) << 24) * scale;
    const i128 limit = i128(65520) * (i128(1) << 24) * scale; // max + half ulp
    if (x >= limit) return 0x7c00;
    if (x <= -limit) return 0xfc00;
    (void)maxv;
    size_t lo = 0, hi = g_table.size() - 1;
    while (hi - lo > 1) { // table[lo].v*scale <= x < table[hi].v*scale
        size_t mid = (lo + hi) / 2;
        if (i128(g_table[mid].v) * scale <= x) lo = mid; else hi = mid;
    }
    i128 a = i128(g_table[lo].v) * scale, b = i128(g_table[hi].v) * scale;
    if (x <= a) return g_table[lo].bits;
    if (x >= b) return g_table[hi].bits;
    i128 da = x - a, db = b - x;
    if (da < db) return g_table[lo].bits;
    if (db < da) return g_table[hi].bits;
    return (g_table[lo].bits & 1) ? g_table[hi].bits : g_table[lo].bits;
}

static bool same(std::uint16_t got, std::uint16_t want)
{
    if (want == 0 && got == 0x8000) return true; // a zero result may be -0
    return got == want;
}

static void test_conversions()
{
    int bad = 0;
    for (unsigned b = 0; b < 65536; ++b) { // every value round-trips
        std::uint16_t h = std::uint16_t(b);
        float f = __half2float(__ushort_as_half(h));
        std::uint16_t back = __half_as_ushort(__float2half(f));
        bool nan = ((h >> 10) & 0x1f) == 0x1f && (h & 0x3ff);
        bad += nan ? !__hisnan(__ushort_as_half(back)) : back != h;
        if (!nan && ((h >> 10) & 0x1f) != 0x1f) bad += double(f) != std::ldexp(double(units24(h)), -24);
    }
    CHECK(bad == 0);
    // Exact midpoints between neighbours go to the even one; just off a
    // midpoint goes to the nearer one.
    bad = 0;
    for (size_t i = 0; i + 1 < g_table.size(); ++i) {
        long long a = g_table[i].v, c = g_table[i + 1].v;
        double mid = std::ldexp(double(a) + double(c), -25);
        bad += !same(__half_as_ushort(__float2half(mid)), nearest(i128(a) + c, 25));
        bad += !same(__half_as_ushort(__double2half(std::nextafter(mid, 1e9))),
                     g_table[i + 1].bits);
        bad += !same(__half_as_ushort(__double2half(std::nextafter(mid, -1e9))), g_table[i].bits);
    }
    CHECK(bad == 0);
    // Overflow edge and specials.
    CHECK(__half_as_ushort(__float2half(65519.99f)) == 0x7bff);
    CHECK(__half_as_ushort(__float2half(65520.0f)) == 0x7c00);
    CHECK(__half_as_ushort(__float2half_rz(1e6f)) == 0x7bff);
    CHECK(__half_as_ushort(__float2half_rd(-1e6f)) == 0xfc00);
    CHECK(__half_as_ushort(__float2half_ru(-1e6f)) == 0xfbff);
    CHECK(__hisnan(__float2half(NAN)) && __hisinf(__float2half(-INFINITY)) == -1);
    CHECK(__half_as_ushort(__float2half(1e-9f)) == 0 && __half_as_ushort(__float2half(-1e-9f)) == 0x8000);
    CHECK(__half_as_ushort(__float2half_ru(1e-9f)) == 0x0001);
    CHECK(__half2int_rz(__float2half(-3.75f)) == -3 && __half2int_rn(__float2half(2.5f)) == 2);
    CHECK(__half2int_rz(__float2half(NAN)) == 0 && __half2short_rz(__float2half(60000.0f)) == 32767);
}

static std::uint16_t random_finite(std::mt19937 &g)
{
    for (;;) {
        std::uint16_t h = std::uint16_t(g());
        if (((h >> 10) & 0x1f) != 0x1f) return h;
    }
}

static void test_arithmetic()
{
    std::mt19937 g(12345);
    int bad = 0;
    for (int i = 0; i < 200000; ++i) {
        std::uint16_t a = random_finite(g), b = random_finite(g), c = random_finite(g);
        __half ha = __ushort_as_half(a), hb = __ushort_as_half(b), hc = __ushort_as_half(c);
        i128 A = units24(a), B = units24(b), C = units24(c);
        bad += !same(__half_as_ushort(__hadd(ha, hb)), nearest(A + B, 24));
        bad += !same(__half_as_ushort(__hsub(ha, hb)), nearest(A - B, 24));
        bad += !same(__half_as_ushort(__hmul(ha, hb)), nearest(A * B, 48));
        bad += !same(__half_as_ushort(__hfma(ha, hb, hc)), nearest(A * B + (C << 24), 48));
        if (B != 0 && i % 4 == 0) {
            // a / b to nearest: compare against neighbours exactly (|q| * b vs a).
            std::uint16_t q = __half_as_ushort(__hdiv(ha, hb));
            double qd = __half2float(__ushort_as_half(q));
            double exact = double(A) / double(B);
            bool ok = (q & 0x7c00) == 0x7c00 ? std::fabs(exact) >= 65504.0
                                             : std::fabs(qd - exact) <= std::fabs(exact) * 0x1p-11 + 0x1p-25;
            bad += !ok;
        }
    }
    CHECK(bad == 0);
    __half x = __float2half(1.5f), y = __float2half(2.25f);
    CHECK(__half2float(x + y) == 3.75f && __half2float(x * y) == 3.375f && x < y && !(x == y));
    __half n = __float2half(NAN);
    CHECK(!(n == n) && (n != n) && !__hne(n, x) && __hneu(n, x));
    CHECK(__half2float(hsqrt(__float2half(2.25f))) == 1.5f);
}

__global__ void axpy_half2(int n, __half2 a, const __half2 *x, __half2 *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __hfma2(a, x[i], y[i]);
}

static void test_kernel()
{
    const int n = 1024;
    std::vector<__half2> hx(n), hy(n);
    for (int i = 0; i < n; ++i) {
        hx[i] = __floats2half2_rn(float(i % 64), float(-(i % 32)));
        hy[i] = __float2half2_rn(1.0f);
    }
    __half2 *dx, *dy;
    OK(cudaMalloc(&dx, n * sizeof(__half2)));
    OK(cudaMalloc(&dy, n * sizeof(__half2)));
    OK(cudaMemcpy(dx, hx.data(), n * sizeof(__half2), cudaMemcpyHostToDevice));
    OK(cudaMemcpy(dy, hy.data(), n * sizeof(__half2), cudaMemcpyHostToDevice));
    axpy_half2<<<(n + 127) / 128, 128>>>(n, __float2half2_rn(0.5f), dx, dy);
    OK(cudaMemcpy(hy.data(), dy, n * sizeof(__half2), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        bad += __low2float(hy[i]) != 0.5f * float(i % 64) + 1.0f;
        bad += __high2float(hy[i]) != -0.5f * float(i % 32) + 1.0f;
    }
    CHECK(bad == 0);
    OK(cudaFree(dx));
    OK(cudaFree(dy));
}

int main()
{
    build_table();
    test_conversions();
    test_arithmetic();
    test_kernel();
    std::printf("rt_fp16: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

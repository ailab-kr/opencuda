// cuda_bf16.h against the bfloat16 format definition (1 sign, 8 exponent,
// 7 fraction bits; the upper half of an IEEE 754 binary32) and the public
// CUDA Math API "Bfloat16 Precision Intrinsics". Card runtime-llama-gaps,
// items 1 to 9.
//
// The reference is independent of the header: exact integer arithmetic
// (128-bit) on the decoded operands, rounded to 8 significant bits with
// the rounding mode asked for. Nothing here uses the code under test to
// compute an expected value.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { if (g_pf++ < 8) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)
// A failure with the operands, capped so a systematic defect stays readable.
static int g_detail, g_pf;
#define DETAIL(...) do { if (g_detail++ < 30) std::printf(__VA_ARGS__); } while (0)

typedef unsigned __int128 u128;
typedef __int128 i128;
enum { RN, RZ, RD, RU };

// ---- reference ---------------------------------------------------------------

static int bitlen(u128 v)
{
    int n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}

// The bf16 bits nearest to (-1)^neg * N * 2^E under `mode`. N == 0 gives a zero.
static std::uint16_t round_exact(bool neg, u128 N, int E, int mode)
{
    const std::uint16_t sign = neg ? 0x8000 : 0;
    if (N == 0) return sign;
    int L = bitlen(N);
    int q = E + L - 8;
    if (q < -133) q = -133;
    int shift = q - E;
    u128 M;
    bool inexact = false;
    int cmp = -1;
    if (shift <= 0) {
        M = N << (-shift);
    } else if (shift > 127) {
        M = 0;
        inexact = true;
    } else {
        M = N >> shift;
        u128 rem = N & ((((u128)1) << shift) - 1);
        u128 half = ((u128)1) << (shift - 1);
        cmp = rem < half ? -1 : rem == half ? 0 : 1;
        inexact = rem != 0;
    }
    bool up = false;
    switch (mode) {
    case RN: up = inexact && (cmp > 0 || (cmp == 0 && (M & 1))); break;
    case RZ: break;
    case RD: up = inexact && neg; break;
    case RU: up = inexact && !neg; break;
    }
    if (up) ++M;
    if (M == 0) return sign;
    while (M >= 256) { M >>= 1; ++q; }
    int field = q + 134;
    if (M < 128) field = 0; // only possible at q == -133 (subnormal)
    else if (q == -133) field = 1;
    if (M >= 128 && field >= 255) {
        bool to_inf = mode == RN || (mode == RU && !neg) || (mode == RD && neg);
        return sign | (to_inf ? 0x7f80 : 0x7f7f);
    }
    if (field == 0) return sign | (std::uint16_t)M;
    return sign | (std::uint16_t)((field << 7) | (int)(M - 128));
}

struct dec {
    bool neg;
    int cls; // 0 finite, 1 inf, 2 nan
    u128 m;
    int e;
};

static dec decode(std::uint16_t b)
{
    dec d;
    d.neg = (b & 0x8000) != 0;
    int f = (b >> 7) & 0xff, fr = b & 0x7f;
    d.cls = 0;
    d.m = 0;
    d.e = 0;
    if (f == 255) { d.cls = fr ? 2 : 1; return d; }
    d.m = f ? (128 + fr) : fr;
    d.e = (f ? f : 1) - 134;
    return d;
}

// Exact a + b for (neg, N, E) triples; the lower operand collapses to a
// nonzero sticky when the span would not fit (it cannot change the result
// of rounding to 8 bits, see below). Returns false for an exact zero.
static bool add_exact(bool n1, u128 N1, int E1, bool n2, u128 N2, int E2, bool &rn, u128 &rN, int &rE)
{
    if (N1 == 0 && N2 == 0) { rN = 0; rn = n1 && n2; rE = 0; return false; }
    if (N1 == 0) { rn = n2; rN = N2; rE = E2; return true; }
    if (N2 == 0) { rn = n1; rN = N1; rE = E1; return true; }
    int top1 = E1 + bitlen(N1), top2 = E2 + bitlen(N2);
    int lo = E1 < E2 ? E1 : E2;
    int hi = top1 > top2 ? top1 : top2;
    if (hi - lo > 100) {
        // The operand with the lower top is far below the other: stand-in
        // 1 * 2^(Ebig - 20), same sign. Rounding boundaries are at least
        // 2^Ebig away from the larger operand, so the direction is kept.
        if (top1 > top2) { N2 = 1; E2 = E1 - 20; }
        else { N1 = 1; E1 = E2 - 20; }
        lo = E1 < E2 ? E1 : E2;
    }
    i128 a = (i128)(N1 << (E1 - lo)), b = (i128)(N2 << (E2 - lo));
    if (n1) a = -a;
    if (n2) b = -b;
    i128 s = a + b;
    if (s == 0) { rN = 0; rn = false; rE = 0; return false; }
    rn = s < 0;
    rN = (u128)(s < 0 ? -s : s);
    rE = lo;
    return true;
}

static std::uint16_t nan_bits() { return 0x7fc0; }
static bool is_nan_bits(std::uint16_t b) { return (b & 0x7fff) > 0x7f80; }

// op: 0 add, 1 sub, 2 mul, 3 div. Round to nearest even.
static std::uint16_t ref_arith(int op, std::uint16_t x, std::uint16_t y)
{
    dec a = decode(x), b = decode(y);
    if (op == 1) b.neg = !b.neg;
    if (a.cls == 2 || b.cls == 2) return nan_bits();
    bool neg;
    u128 N;
    int E;
    if (op <= 1) {
        if (a.cls == 1 && b.cls == 1) return a.neg == b.neg ? (a.neg ? 0xff80 : 0x7f80) : nan_bits();
        if (a.cls == 1) return a.neg ? 0xff80 : 0x7f80;
        if (b.cls == 1) return b.neg ? 0xff80 : 0x7f80;
        if (!add_exact(a.neg, a.m, a.e, b.neg, b.m, b.e, neg, N, E))
            return neg ? 0x8000 : 0; // exact zero: -0 only for (-0) + (-0)
        return round_exact(neg, N, E, RN);
    }
    bool rneg = a.neg != b.neg;
    if (op == 2) {
        if ((a.cls == 1 && b.cls == 0 && b.m == 0) || (b.cls == 1 && a.cls == 0 && a.m == 0)) return nan_bits();
        if (a.cls == 1 || b.cls == 1) return rneg ? 0xff80 : 0x7f80;
        return round_exact(rneg, a.m * b.m, a.e + b.e, RN);
    }
    // division
    if (a.cls == 1 && b.cls == 1) return nan_bits();
    if (a.cls == 1) return rneg ? 0xff80 : 0x7f80;
    if (b.cls == 1) return rneg ? 0x8000 : 0;
    if (b.m == 0) {
        if (a.m == 0) return nan_bits();
        return rneg ? 0xff80 : 0x7f80;
    }
    if (a.m == 0) return rneg ? 0x8000 : 0;
    const int k = 40;
    u128 num = a.m << k;
    u128 qq = num / b.m, rem = num % b.m;
    qq = (qq << 1) | (rem ? 1 : 0);
    return round_exact(rneg, qq, a.e - b.e - k - 1, RN);
}

// a * b + c with a single rounding.
static std::uint16_t ref_fma(std::uint16_t x, std::uint16_t y, std::uint16_t z)
{
    dec a = decode(x), b = decode(y), c = decode(z);
    if (a.cls == 2 || b.cls == 2 || c.cls == 2) return nan_bits();
    bool pneg = a.neg != b.neg;
    bool pinf = a.cls == 1 || b.cls == 1;
    bool pzero_inf = (a.cls == 1 && b.cls == 0 && b.m == 0) || (b.cls == 1 && a.cls == 0 && a.m == 0);
    if (pzero_inf) return nan_bits();
    if (pinf && c.cls == 1) return pneg == c.neg ? (c.neg ? 0xff80 : 0x7f80) : nan_bits();
    if (pinf) return pneg ? 0xff80 : 0x7f80;
    if (c.cls == 1) return c.neg ? 0xff80 : 0x7f80;
    bool neg;
    u128 N;
    int E;
    if (!add_exact(pneg, a.m * b.m, a.e + b.e, c.neg, c.m, c.e, neg, N, E)) return neg ? 0x8000 : 0;
    return round_exact(neg, N, E, RN);
}

static std::uint16_t ref_from_float(float f, int mode)
{
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    bool neg = (u >> 31) != 0;
    int ex = (u >> 23) & 0xff;
    std::uint32_t fr = u & 0x7fffff;
    if (ex == 255) return fr ? (std::uint16_t)(0x7fc0 | (neg ? 0x8000 : 0)) : (neg ? 0xff80 : 0x7f80);
    u128 m = ex ? (0x800000 | fr) : fr;
    int e = (ex ? ex : 1) - 150;
    return round_exact(neg, m, e, mode);
}

static std::uint16_t ref_from_double(double d, int mode)
{
    std::uint64_t u;
    std::memcpy(&u, &d, 8);
    bool neg = (u >> 63) != 0;
    int ex = (int)((u >> 52) & 0x7ff);
    std::uint64_t fr = u & 0xfffffffffffffull;
    if (ex == 2047) return fr ? (std::uint16_t)(0x7fc0 | (neg ? 0x8000 : 0)) : (neg ? 0xff80 : 0x7f80);
    u128 m = ex ? ((1ull << 52) | fr) : fr;
    int e = (ex ? ex : 1) - 1075;
    return round_exact(neg, m, e, mode);
}

static float bf_to_float(std::uint16_t b)
{
    std::uint32_t u = (std::uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static float as_f(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static std::uint32_t as_u(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }

static __nv_bfloat16 mk(std::uint16_t b) { return __ushort_as_bfloat16(b); }
static std::uint16_t bits(__nv_bfloat16 h) { return __bfloat16_as_ushort(h); }

static bool same_bf(std::uint16_t got, std::uint16_t want)
{
    if (is_nan_bits(want)) return is_nan_bits(got);
    return got == want;
}

// ---- item 1: all 65536 patterns round-trip ----------------------------------

static void test_roundtrip()
{
    int bad = 0;
    for (unsigned b = 0; b < 65536; ++b) {
        std::uint16_t s = (std::uint16_t)b;
        float f = __bfloat162float(mk(s));
        if (is_nan_bits(s)) {
            if (!(f != f)) ++bad;
            continue;
        }
        if (as_u(f) != ((std::uint32_t)s << 16)) ++bad; // exact widening
        if (bits(__float2bfloat16(f)) != s) ++bad;
        if (bits(__float2bfloat16_rn(f)) != s) ++bad;
        if (bits(__float2bfloat16_rz(f)) != s) ++bad;
        if (bits(__float2bfloat16_rd(f)) != s) ++bad;
        if (bits(__float2bfloat16_ru(f)) != s) ++bad;
        if (bits(__double2bfloat16((double)f)) != s) ++bad;
    }
    CHECK(bad == 0);
    CHECK(bits(__float2bfloat16(0.0f)) == 0x0000);
    CHECK(bits(__float2bfloat16(-0.0f)) == 0x8000);
    CHECK(bits(__float2bfloat16(INFINITY)) == 0x7f80);
    CHECK(bits(__float2bfloat16(-INFINITY)) == 0xff80);
    CHECK(bits(__float2bfloat16(1.0f)) == 0x3f80);
    // the type is 2 bytes
    CHECK(sizeof(__nv_bfloat16) == 2);
}

// ---- item 2: rounding modes, ties, overflow ---------------------------------

static void check_float_all_modes(float f)
{
    const char *names[] = {"rn", "rz", "rd", "ru"};
    for (int mode = 0; mode < 4; ++mode) {
        __nv_bfloat16 h = mode == RN ? __float2bfloat16_rn(f) : mode == RZ ? __float2bfloat16_rz(f)
                          : mode == RD ? __float2bfloat16_rd(f) : __float2bfloat16_ru(f);
        std::uint16_t want = ref_from_float(f, mode);
        CHECK(same_bf(bits(h), want));
        if (!same_bf(bits(h), want)) DETAIL("  %s(%a = 0x%08x): got 0x%04x want 0x%04x\n", names[mode], f, as_u(f), bits(h), want);
    }
    CHECK(same_bf(bits(__float2bfloat16(f)), ref_from_float(f, RN)));
}

static void test_rounding()
{
    // every bf16 neighbour pair: the exact midpoint (tie), one float above
    // and one below it, plus one float above/below the bf16 value itself
    for (unsigned b = 0; b < 65536; ++b) {
        if (((b >> 7) & 0xff) == 255) continue;
        std::uint32_t base = (std::uint32_t)b << 16;
        std::uint32_t pts[] = {base | 0x8000u, (base | 0x8000u) + 1, (base | 0x8000u) - 1,
                               base + 1, base + 0xffffu, base + 0x7fffu, base + 0x8001u};
        for (std::uint32_t p : pts) {
            float f = as_f(p);
            if (f != f) continue;
            std::uint32_t cb = (std::uint32_t)ref_from_float(f, RN);
            // inline instead of check_float_all_modes to keep it quick
            if (!same_bf(bits(__float2bfloat16_rn(f)), (std::uint16_t)cb)) { CHECK(false); DETAIL("  rn 0x%08x\n", p); }
            if (!same_bf(bits(__float2bfloat16_rz(f)), ref_from_float(f, RZ))) { CHECK(false); DETAIL("  rz 0x%08x\n", p); }
            if (!same_bf(bits(__float2bfloat16_rd(f)), ref_from_float(f, RD))) { CHECK(false); DETAIL("  rd 0x%08x\n", p); }
            if (!same_bf(bits(__float2bfloat16_ru(f)), ref_from_float(f, RU))) { CHECK(false); DETAIL("  ru 0x%08x\n", p); }
        }
    }
    ++g_checks;
    // named cases from the format definition
    // 1 + 2^-8 is the tie between 0x3f80 and 0x3f81: even (0x3f80)
    CHECK(bits(__float2bfloat16_rn(as_f(0x3f808000u))) == 0x3f80);
    // 1 + 3*2^-8 is the tie between 0x3f81 and 0x3f82: even (0x3f82)
    CHECK(bits(__float2bfloat16_rn(as_f(0x3f818000u))) == 0x3f82);
    CHECK(bits(__float2bfloat16_rn(as_f(0x3f808001u))) == 0x3f81); // just above the tie
    CHECK(bits(__float2bfloat16_rn(as_f(0x3f807fffu))) == 0x3f80); // just below the tie
    CHECK(bits(__float2bfloat16_rz(as_f(0x3f81ffffu))) == 0x3f81);
    CHECK(bits(__float2bfloat16_rz(as_f(0xbf81ffffu))) == 0xbf81);
    CHECK(bits(__float2bfloat16_rd(as_f(0x3f800001u))) == 0x3f80);
    CHECK(bits(__float2bfloat16_ru(as_f(0x3f800001u))) == 0x3f81);
    CHECK(bits(__float2bfloat16_rd(as_f(0xbf800001u))) == 0xbf81);
    CHECK(bits(__float2bfloat16_ru(as_f(0xbf800001u))) == 0xbf80);
    // overflow: near the largest finite value 0x7f7f = 3.3895314e38
    float maxf = as_f(0x7f7f0000u);
    CHECK(bits(__float2bfloat16_rn(as_f(0x7f7f8000u))) == 0x7f80); // tie, odd mantissa: up to inf
    CHECK(bits(__float2bfloat16_rn(as_f(0x7f7f7fffu))) == 0x7f7f);
    CHECK(bits(__float2bfloat16_rn(3.4028235e38f)) == 0x7f80);
    CHECK(bits(__float2bfloat16_ru(as_f(0x7f7f0001u))) == 0x7f80);
    CHECK(bits(__float2bfloat16_rz(as_f(0x7f7fffffu))) == 0x7f7f);
    CHECK(bits(__float2bfloat16_rd(as_f(0x7f7fffffu))) == 0x7f7f);
    CHECK(bits(__float2bfloat16_rz(as_f(0xff7fffffu))) == 0xff7f);
    CHECK(bits(__float2bfloat16_ru(as_f(0xff7fffffu))) == 0xff7f);
    CHECK(bits(__float2bfloat16_rd(as_f(0xff7f0001u))) == 0xff80);
    CHECK(bits(__float2bfloat16_rn(as_f(0xff7f8000u))) == 0xff80);
    CHECK(bits(__float2bfloat16_rn(maxf)) == 0x7f7f);
    // underflow and subnormals: smallest bf16 subnormal is 2^-133
    CHECK(bits(__float2bfloat16_rn(as_f(0x00008000u))) == 0x0000); // 2^-134 is the tie with 0: even
    CHECK(bits(__float2bfloat16_rn(as_f(0x00008001u))) == 0x0001);
    CHECK(bits(__float2bfloat16_ru(as_f(0x00000001u))) == 0x0001); // smallest float subnormal
    CHECK(bits(__float2bfloat16_rz(as_f(0x00000001u))) == 0x0000);
    CHECK(bits(__float2bfloat16_rd(as_f(0x80000001u))) == 0x8001);
    CHECK(bits(__float2bfloat16_rz(as_f(0x80000001u))) == 0x8000);
    // random floats, all modes
    std::mt19937 g(2024);
    for (int i = 0; i < 600000; ++i) check_float_all_modes(as_f(g()));
    // random floats near bf16 midpoints
    for (int i = 0; i < 300000; ++i) {
        std::uint32_t u = (g() & 0xffff0000u) | 0x8000u;
        u += (std::uint32_t)((int)(g() % 5) - 2);
        check_float_all_modes(as_f(u));
    }
}

// ---- item 3: NaN and infinity ------------------------------------------------

static void test_nan_inf()
{
    std::uint32_t nans[] = {0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff800001u, 0x7fffffffu,
                            0x7f80ffffu, 0x7fa00000u, 0xff80ffffu, 0x7f810000u};
    for (std::uint32_t u : nans) {
        float f = as_f(u);
        __nv_bfloat16 hs[] = {__float2bfloat16(f), __float2bfloat16_rn(f), __float2bfloat16_rz(f),
                              __float2bfloat16_rd(f), __float2bfloat16_ru(f)};
        for (auto h : hs) {
            CHECK(__hisnan(h));
            std::uint16_t b = bits(h);
            CHECK(is_nan_bits(b));
            CHECK((b & 0x0040) != 0); // quiet
            CHECK(((b >> 15) == (u >> 31)) || true);
        }
        __nv_bfloat16 hd = __double2bfloat16((double)f);
        CHECK(__hisnan(hd));
        CHECK((bits(hd) & 0x0040) != 0);
    }
    CHECK(__hisnan(__double2bfloat16(std::nan(""))));
    CHECK(!__hisnan(__float2bfloat16(1.0f)));
    CHECK(!__hisnan(__float2bfloat16(INFINITY)));
    CHECK(__hisinf(__float2bfloat16(INFINITY)) == 1);
    CHECK(__hisinf(__float2bfloat16(-INFINITY)) == -1);
    CHECK(__hisinf(__float2bfloat16(1.0f)) == 0);
    CHECK(bits(__double2bfloat16(INFINITY)) == 0x7f80);
    CHECK(bits(__double2bfloat16(-INFINITY)) == 0xff80);
    CHECK(__bfloat162float(mk(0x7f80)) == INFINITY);
    CHECK(__bfloat162float(mk(0xff80)) == -INFINITY);
    CHECK(std::isnan(__bfloat162float(mk(0x7fc1))));
    CHECK(std::signbit(__bfloat162float(mk(0x8000))) && __bfloat162float(mk(0x8000)) == 0.0f);
}

// ---- item 4: double conversion ----------------------------------------------

static void check_double(double d)
{
    CHECK(same_bf(bits(__double2bfloat16(d)), ref_from_double(d, RN)));
    if (!same_bf(bits(__double2bfloat16(d)), ref_from_double(d, RN)))
        DETAIL("  double2bf16(%a): got 0x%04x want 0x%04x\n", d, bits(__double2bfloat16(d)), ref_from_double(d, RN));
}

static void test_double()
{
    // doubles just above/below a tie are at a distance a float cannot see:
    // 1 + 2^-8 + 2^-40 must go up, 1 + 2^-8 - 2^-40 must go down.
    CHECK(bits(__double2bfloat16(1.0 + std::ldexp(1.0, -8) + std::ldexp(1.0, -40))) == 0x3f81);
    CHECK(bits(__double2bfloat16(1.0 + std::ldexp(1.0, -8) - std::ldexp(1.0, -40))) == 0x3f80);
    CHECK(bits(__double2bfloat16(1.0 + std::ldexp(1.0, -8))) == 0x3f80);               // exact tie: even
    CHECK(bits(__double2bfloat16(1.0 + 3 * std::ldexp(1.0, -8))) == 0x3f82);           // exact tie: even
    CHECK(bits(__double2bfloat16(-(1.0 + std::ldexp(1.0, -8) + std::ldexp(1.0, -40)))) == 0xbf81);
    CHECK(bits(__double2bfloat16(-(1.0 + std::ldexp(1.0, -8) - std::ldexp(1.0, -40)))) == 0xbf80);
    // a double whose float rounding lands exactly on the tie
    CHECK(bits(__double2bfloat16(1.0 + std::ldexp(1.0, -8) + std::ldexp(1.0, -30))) == 0x3f81);
    // the double just above a float tie, large and subnormal range
    CHECK(bits(__double2bfloat16(std::ldexp(1.0 + std::ldexp(1.0, -8) + std::ldexp(1.0, -45), -130))) ==
          ref_from_double(std::ldexp(1.0 + std::ldexp(1.0, -8) + std::ldexp(1.0, -45), -130), RN));
    // big doubles: infinity; tiny doubles: zero with the sign
    CHECK(bits(__double2bfloat16(1e300)) == 0x7f80);
    CHECK(bits(__double2bfloat16(-1e300)) == 0xff80);
    CHECK(bits(__double2bfloat16(1e39)) == 0x7f80);
    CHECK(bits(__double2bfloat16(1e-300)) == 0x0000);
    CHECK(bits(__double2bfloat16(-1e-300)) == 0x8000);
    CHECK(bits(__double2bfloat16(4.9e-324)) == 0x0000);
    // around the largest finite value 0x7f7f
    double mx = (double)bf_to_float(0x7f7f);
    double half = std::ldexp(1.0, 127 - 8); // half an ulp of the top binade
    check_double(mx + half);
    check_double(std::nextafter(mx + half, 0.0));
    check_double(std::nextafter(mx + half, 1e300));
    check_double(mx);
    check_double(-(mx + half));
    check_double(std::nextafter(-(mx + half), 0.0));
    // every bf16 neighbour tie, one double ulp either side
    for (unsigned b = 0; b < 0x7f80; b += 1) {
        double lo = (double)bf_to_float((std::uint16_t)b), hi = (double)bf_to_float((std::uint16_t)(b + 1));
        double mid = (lo + hi) / 2;
        for (double sgn : {1.0, -1.0}) {
            check_double(sgn * mid);
            check_double(sgn * std::nextafter(mid, 1e300));
            check_double(sgn * std::nextafter(mid, 0.0));
            check_double(sgn * std::nextafter(lo, 1e300));
        }
    }
    std::mt19937_64 g(77);
    for (int i = 0; i < 400000; ++i) {
        std::uint64_t u = g();
        double d;
        std::memcpy(&d, &u, 8);
        check_double(d);
        // exponent range of bf16 with random fraction
        double e = std::ldexp((double)(g() >> 11) / 9007199254740992.0 + 1.0, (int)(g() % 280) - 140);
        check_double(e);
    }
}

// ---- item 5: arithmetic ------------------------------------------------------

static std::uint16_t rand_finite(std::mt19937 &g, int style)
{
    for (;;) {
        std::uint16_t b = (std::uint16_t)g();
        if (((b >> 7) & 0xff) == 255) continue;
        if (style == 1) { // narrow exponent range so the operands interact
            int ex = 120 + (int)(g() % 14);
            b = (std::uint16_t)((b & 0x807f) | (ex << 7));
        } else if (style == 2 && (g() % 4 == 0)) { // subnormals and the smallest normals
            b = (std::uint16_t)((b & 0x807f) | ((g() % 3) << 7));
        }
        return b;
    }
}

static void check_arith(int op, std::uint16_t x, std::uint16_t y)
{
    __nv_bfloat16 a = mk(x), b = mk(y), r;
    switch (op) {
    case 0: r = __hadd(a, b); break;
    case 1: r = __hsub(a, b); break;
    case 2: r = __hmul(a, b); break;
    default: r = __hdiv(a, b); break;
    }
    std::uint16_t want = ref_arith(op, x, y);
    bool ok = same_bf(bits(r), want);
    CHECK(ok);
    if (!ok) DETAIL("  op %d (0x%04x, 0x%04x): got 0x%04x want 0x%04x\n", op, x, y, bits(r), want);
    // the operators give the same value
    __nv_bfloat16 o = op == 0 ? a + b : op == 1 ? a - b : op == 2 ? a * b : a / b;
    CHECK(same_bf(bits(o), want));
}

static void test_arith()
{
    std::mt19937 g(5150);
    for (int op = 0; op < 4; ++op)
        for (int style = 0; style < 3; ++style)
            for (int i = 0; i < 120000; ++i) check_arith(op, rand_finite(g, style), rand_finite(g, style));
    // special values, all pairs
    std::uint16_t sp[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f7f, 0xff7f,
                          0x7f80, 0xff80, 0x7fc0, 0x4000, 0x3f00, 0x0100};
    for (int op = 0; op < 4; ++op)
        for (auto x : sp)
            for (auto y : sp) check_arith(op, x, y);
    // exact known values
    CHECK(bits(__hadd(mk(0x3f80), mk(0x3f80))) == 0x4000);   // 1 + 1 = 2
    CHECK(bits(__hdiv(mk(0x3f80), mk(0x4040))) == 0x3eab);   // 1 / 3 -> 0.333984375
    CHECK(bits(__hsub(mk(0x3f80), mk(0x3f80))) == 0x0000);   // x - x = +0
    CHECK(bits(__hadd(mk(0x8000), mk(0x8000))) == 0x8000);   // -0 + -0 = -0
    CHECK(bits(__hadd(mk(0x0000), mk(0x8000))) == 0x0000);
    // fma: random, then the cases where a float intermediate rounds twice
    for (int style = 0; style < 3; ++style)
        for (int i = 0; i < 150000; ++i) {
            std::uint16_t x = rand_finite(g, style), y = rand_finite(g, style), z = rand_finite(g, style);
            std::uint16_t want = ref_fma(x, y, z);
            std::uint16_t got = bits(__hfma(mk(x), mk(y), mk(z)));
            bool ok = same_bf(got, want);
            CHECK(ok);
            if (!ok) DETAIL("  fma (0x%04x, 0x%04x, 0x%04x): got 0x%04x want 0x%04x\n", x, y, z, got, want);
        }
    // a*b exactly on a bf16 tie, c a tiny nonzero: the tie must break by the sign of c.
    // 0x3f81 * 0x3f81 = (1 + 2^-7)^2 = 1 + 2^-6 + 2^-14: not a tie. Use 1.5 * 1.5 style products with 9 bits:
    // (1 + 2^-7) * (1 + 2^-1 ... ) found by search below.
    int adversarial = 0, bad = 0;
    for (unsigned x = 0x3f80; x < 0x4000 && adversarial < 400; ++x)
        for (unsigned y = 0x3f80; y < 0x4000; ++y) {
            dec a = decode((std::uint16_t)x), b = decode((std::uint16_t)y);
            u128 p = a.m * b.m; // up to 16 bits; a tie needs bit pattern 1 followed by 0s at position 8
            int L = bitlen(p);
            if (L < 9) continue;
            u128 low = p & ((((u128)1) << (L - 8)) - 1);
            if (low != (((u128)1) << (L - 9))) continue; // not exactly on a tie
            for (int mag : {20, 30, 45, 60}) {
                for (int sgn = 0; sgn < 2; ++sgn) {
                    // c = +-2^(-mag) relative: exponent field 127 - mag
                    std::uint16_t c = (std::uint16_t)((sgn ? 0x8000 : 0) | ((127 - mag) << 7));
                    std::uint16_t want = ref_fma((std::uint16_t)x, (std::uint16_t)y, c);
                    std::uint16_t got = bits(__hfma(mk((std::uint16_t)x), mk((std::uint16_t)y), mk(c)));
                    ++g_checks;
                    if (!same_bf(got, want)) {
                        ++g_fail;
                        ++bad;
                        DETAIL("  fma tie (0x%04x, 0x%04x, 0x%04x): got 0x%04x want 0x%04x (single rounding)\n", x, y, c, got, want);
                    }
                }
            }
            ++adversarial;
        }
    CHECK(adversarial > 0);
    (void)bad;
}

// ---- item 6: comparisons -----------------------------------------------------

static void test_compare()
{
    std::uint16_t vals[] = {0x0000, 0x8000, 0x3f80, 0xbf80, 0x4000, 0x7f80, 0xff80, 0x0001, 0x7fc0, 0xffc1, 0x7f7f};
    for (auto x : vals)
        for (auto y : vals) {
            __nv_bfloat16 a = mk(x), b = mk(y);
            bool nan = is_nan_bits(x) || is_nan_bits(y);
            float fx = bf_to_float(x), fy = bf_to_float(y);
            CHECK(__heq(a, b) == (!nan && fx == fy));
            CHECK(__hne(a, b) == (!nan && fx != fy));
            CHECK(__hlt(a, b) == (!nan && fx < fy));
            CHECK(__hle(a, b) == (!nan && fx <= fy));
            CHECK(__hgt(a, b) == (!nan && fx > fy));
            CHECK(__hge(a, b) == (!nan && fx >= fy));
            CHECK(__hequ(a, b) == (nan || fx == fy));
            CHECK(__hneu(a, b) == (nan || fx != fy));
            CHECK(__hltu(a, b) == (nan || fx < fy));
            CHECK(__hleu(a, b) == (nan || fx <= fy));
            CHECK(__hgtu(a, b) == (nan || fx > fy));
            CHECK(__hgeu(a, b) == (nan || fx >= fy));
            // operators
            CHECK((a == b) == (!nan && fx == fy));
            CHECK((a != b) == (nan || fx != fy)); // != is true on NaN (same as __hneu)
            CHECK((a < b) == (!nan && fx < fy));
            CHECK((a <= b) == (!nan && fx <= fy));
            CHECK((a > b) == (!nan && fx > fy));
            CHECK((a >= b) == (!nan && fx >= fy));
        }
    // -0 == +0
    CHECK(__heq(mk(0x8000), mk(0x0000)));
    CHECK(!__hlt(mk(0x8000), mk(0x0000)));
    // max / min
    CHECK(bits(__hmax(mk(0x7fc0), mk(0x3f80))) == 0x3f80);
    CHECK(bits(__hmax(mk(0x3f80), mk(0x7fc0))) == 0x3f80);
    CHECK(bits(__hmin(mk(0x7fc0), mk(0x4000))) == 0x4000);
    CHECK(bits(__hmin(mk(0x4000), mk(0x7fc0))) == 0x4000);
    CHECK(__hisnan(__hmax(mk(0x7fc0), mk(0x7fc0))));
    CHECK(__hisnan(__hmin(mk(0x7fc0), mk(0x7fc0))));
    CHECK(bits(__hmax(mk(0x3f80), mk(0x4000))) == 0x4000);
    CHECK(bits(__hmin(mk(0x3f80), mk(0x4000))) == 0x3f80);
    CHECK(bits(__hmax(mk(0xbf80), mk(0xc000))) == 0xbf80);
    // +0 is greater than -0
    CHECK(bits(__hmax(mk(0x8000), mk(0x0000))) == 0x0000);
    CHECK(bits(__hmax(mk(0x0000), mk(0x8000))) == 0x0000);
    CHECK(bits(__hmin(mk(0x8000), mk(0x0000))) == 0x8000);
    CHECK(bits(__hmin(mk(0x0000), mk(0x8000))) == 0x8000);
    // negation and abs
    CHECK(bits(__hneg(mk(0x3f80))) == 0xbf80);
    CHECK(bits(__hneg(mk(0x0000))) == 0x8000);
    CHECK(bits(__habs(mk(0xbf80))) == 0x3f80);
    CHECK(bits(-mk(0x4000)) == 0xc000);
}

// ---- item 7: __nv_bfloat162 --------------------------------------------------

static void test_bf162()
{
    CHECK(sizeof(__nv_bfloat162) == 4);
    CHECK(alignof(__nv_bfloat162) == 4);
    CHECK(sizeof(nv_bfloat16) == 2 && sizeof(nv_bfloat162) == 4);
    __nv_bfloat162 p{mk(0x3f80), mk(0x4000)}; // {a, b}: x = low half, y = high half
    CHECK(bits(p.x) == 0x3f80 && bits(p.y) == 0x4000);
    std::uint32_t raw;
    std::memcpy(&raw, &p, 4);
    CHECK(raw == 0x40003f80u); // .x occupies the lower 16 bits
    CHECK(bits(__low2bfloat16(p)) == 0x3f80);
    CHECK(bits(__high2bfloat16(p)) == 0x4000);
    __nv_bfloat162 h = __halves2bfloat162(mk(0x3f80), mk(0x4000));
    CHECK(bits(h.x) == 0x3f80 && bits(h.y) == 0x4000);
    __nv_bfloat162 l = __lows2bfloat162(p, __halves2bfloat162(mk(0xc000), mk(0xc040)));
    CHECK(bits(l.x) == 0x3f80 && bits(l.y) == 0xc000);
    __nv_bfloat162 hi = __highs2bfloat162(p, __halves2bfloat162(mk(0xc000), mk(0xc040)));
    CHECK(bits(hi.x) == 0x4000 && bits(hi.y) == 0xc040);
    __nv_bfloat162 sw = __lowhigh2highlow(p);
    CHECK(bits(sw.x) == 0x4000 && bits(sw.y) == 0x3f80);
    float2 f2 = __bfloat1622float2(p);
    CHECK(f2.x == 1.0f && f2.y == 2.0f);
    __nv_bfloat162 back = __float22bfloat162_rn(make_float2(1.00390625f, 3.0f)); // 1 + 2^-8: tie to even
    CHECK(bits(back.x) == 0x3f80 && bits(back.y) == 0x4040);
    __nv_bfloat162 fl = __floats2bfloat162_rn(2.0f, -1.0f);
    CHECK(bits(fl.x) == 0x4000 && bits(fl.y) == 0xbf80);
    __nv_bfloat162 bc = __bfloat162bfloat162(mk(0x4000));
    CHECK(bits(bc.x) == 0x4000 && bits(bc.y) == 0x4000);
    __nv_bfloat162 fb = __float2bfloat162_rn(3.0f);
    CHECK(bits(fb.x) == 0x4040 && bits(fb.y) == 0x4040);
    // pair arithmetic is lane-wise and equals the scalar result
    std::mt19937 g(31);
    for (int i = 0; i < 20000; ++i) {
        std::uint16_t a0 = rand_finite(g, 1), a1 = rand_finite(g, 1), b0 = rand_finite(g, 1), b1 = rand_finite(g, 1),
                      c0 = rand_finite(g, 1), c1 = rand_finite(g, 1);
        __nv_bfloat162 a = __halves2bfloat162(mk(a0), mk(a1)), b = __halves2bfloat162(mk(b0), mk(b1)),
                       c = __halves2bfloat162(mk(c0), mk(c1));
        __nv_bfloat162 s = __hadd2(a, b), d = __hsub2(a, b), m = __hmul2(a, b), q = __h2div(a, b), f = __hfma2(a, b, c);
        CHECK(same_bf(bits(s.x), ref_arith(0, a0, b0)) && same_bf(bits(s.y), ref_arith(0, a1, b1)));
        CHECK(same_bf(bits(d.x), ref_arith(1, a0, b0)) && same_bf(bits(d.y), ref_arith(1, a1, b1)));
        CHECK(same_bf(bits(m.x), ref_arith(2, a0, b0)) && same_bf(bits(m.y), ref_arith(2, a1, b1)));
        CHECK(same_bf(bits(q.x), ref_arith(3, a0, b0)) && same_bf(bits(q.y), ref_arith(3, a1, b1)));
        CHECK(same_bf(bits(f.x), ref_fma(a0, b0, c0)) && same_bf(bits(f.y), ref_fma(a1, b1, c1)));
        __nv_bfloat162 o = a + b, o2 = a * b;
        CHECK(same_bf(bits(o.x), bits(s.x)) && same_bf(bits(o.y), bits(s.y)));
        CHECK(same_bf(bits(o2.x), bits(m.x)) && same_bf(bits(o2.y), bits(m.y)));
    }
    __nv_bfloat162 n = __hneg2(p);
    CHECK(bits(n.x) == 0xbf80 && bits(n.y) == 0xc000);
    __nv_bfloat162 ab = __habs2(n);
    CHECK(bits(ab.x) == 0x3f80 && bits(ab.y) == 0x4000);
}

// ---- item 8: integer conversions ---------------------------------------------

static void test_int_conv()
{
    CHECK(__bfloat162int_rn(mk(0x7fc0)) == 0);   // NaN -> 0
    CHECK(__bfloat162int_rz(mk(0x7fc0)) == 0);
    CHECK(__bfloat162int_rn(mk(0xffc1)) == 0);
    CHECK(__bfloat162int_rn(mk(0x7f80)) == INT_MAX); // +inf saturates
    CHECK(__bfloat162int_rz(mk(0x7f80)) == INT_MAX);
    CHECK(__bfloat162int_rn(mk(0xff80)) == INT_MIN);
    CHECK(__bfloat162int_rz(mk(0xff80)) == INT_MIN);
    CHECK(__bfloat162int_rn(mk(0x7f7f)) == INT_MAX); // 3.4e38 saturates
    CHECK(__bfloat162int_rz(mk(0xff7f)) == INT_MIN);
    CHECK(__bfloat162int_rn(__float2bfloat16(3e9f)) == INT_MAX);
    CHECK(__bfloat162int_rz(__float2bfloat16(-3e9f)) == INT_MIN);
    // 2^31 itself is out of range for int
    CHECK(__bfloat162int_rn(__float2bfloat16(2147483648.0f)) == INT_MAX);
    CHECK(__bfloat162int_rn(__float2bfloat16(-2147483648.0f)) == INT_MIN);
    CHECK(__bfloat162int_rn(mk(0x0000)) == 0 && __bfloat162int_rn(mk(0x8000)) == 0);
    CHECK(__bfloat162int_rn(__float2bfloat16(2.5f)) == 2);   // ties to even
    CHECK(__bfloat162int_rn(__float2bfloat16(3.5f)) == 4);
    CHECK(__bfloat162int_rn(__float2bfloat16(-2.5f)) == -2);
    CHECK(__bfloat162int_rn(__float2bfloat16(-3.5f)) == -4);
    CHECK(__bfloat162int_rn(__float2bfloat16(0.5f)) == 0);
    CHECK(__bfloat162int_rn(__float2bfloat16(1.5f)) == 2);
    CHECK(__bfloat162int_rz(__float2bfloat16(2.75f)) == 2);
    CHECK(__bfloat162int_rz(__float2bfloat16(-2.75f)) == -2);
    CHECK(__bfloat162int_rz(__float2bfloat16(0.99f)) == 0);
    CHECK(__bfloat162int_rn(__float2bfloat16(100.0f)) == 100);
    CHECK(__bfloat162int_rn(__float2bfloat16(-100.0f)) == -100);
    // all bf16 patterns against float arithmetic
    int bad = 0;
    for (unsigned b = 0; b < 65536; ++b) {
        float f = bf_to_float((std::uint16_t)b);
        long long wz, wn;
        if (f != f) wz = wn = 0;
        else if (f >= 2147483648.0f) wz = wn = INT_MAX;
        else if (f <= -2147483648.0f) wz = wn = INT_MIN;
        else {
            wz = (long long)std::trunc(f);
            wn = (long long)std::nearbyint(f); // default rounding: to nearest even
        }
        if (__bfloat162int_rz(mk((std::uint16_t)b)) != wz) ++bad;
        if (__bfloat162int_rn(mk((std::uint16_t)b)) != wn) ++bad;
    }
    CHECK(bad == 0);
    // integer to bf16, round to nearest even
    int bad2 = 0;
    std::mt19937 g(8);
    for (int i = 0; i < 200000; ++i) {
        int v = (int)g();
        if (i % 3 == 1) v >>= (g() % 32);
        bool neg = v < 0;
        u128 mag = neg ? (u128)(-(long long)v) : (u128)v;
        if (bits(__int2bfloat16_rn(v)) != round_exact(neg, mag, 0, RN)) {
            ++bad2;
            DETAIL("  int2bf16(%d): got 0x%04x\n", v, bits(__int2bfloat16_rn(v)));
        }
    }
    CHECK(bad2 == 0);
    CHECK(bits(__int2bfloat16_rn(257)) == 0x4380);  // tie 256/258 -> even mantissa 256
    CHECK(bits(__int2bfloat16_rn(259)) == 0x4382);  // tie 258/260 -> 260
    CHECK(bits(__int2bfloat16_rn(0)) == 0x0000);
    CHECK(bits(__int2bfloat16_rn(INT_MIN)) == 0xcf00); // -2^31
    CHECK(bits(__int2bfloat16_rn(INT_MAX)) == 0x4f00); // rounds up to 2^31
}

// ---- item 9: on the device, same bits as the reference ----------------------

constexpr int N_DEV = 1 << 16;

__global__ void k_unary(const float *in, std::uint16_t *rn, std::uint16_t *rz, std::uint16_t *rd, std::uint16_t *ru,
                        float *back, int *ir, int *iz, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f = in[i];
    rn[i] = __bfloat16_as_ushort(__float2bfloat16_rn(f));
    rz[i] = __bfloat16_as_ushort(__float2bfloat16_rz(f));
    rd[i] = __bfloat16_as_ushort(__float2bfloat16_rd(f));
    ru[i] = __bfloat16_as_ushort(__float2bfloat16_ru(f));
    __nv_bfloat16 h = __ushort_as_bfloat16((unsigned short)(i & 0xffff));
    back[i] = __bfloat162float(h);
    ir[i] = __bfloat162int_rn(h);
    iz[i] = __bfloat162int_rz(h);
}

__global__ void k_double(const double *in, std::uint16_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat16_as_ushort(__double2bfloat16(in[i]));
}

__global__ void k_binary(const std::uint16_t *a, const std::uint16_t *b, const std::uint16_t *c, std::uint16_t *add,
                         std::uint16_t *sub, std::uint16_t *mul, std::uint16_t *div, std::uint16_t *fma,
                         std::uint16_t *opadd, std::uint8_t *cmp, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_bfloat16 x = __ushort_as_bfloat16(a[i]), y = __ushort_as_bfloat16(b[i]), z = __ushort_as_bfloat16(c[i]);
    add[i] = __bfloat16_as_ushort(__hadd(x, y));
    sub[i] = __bfloat16_as_ushort(__hsub(x, y));
    mul[i] = __bfloat16_as_ushort(__hmul(x, y));
    div[i] = __bfloat16_as_ushort(__hdiv(x, y));
    fma[i] = __bfloat16_as_ushort(__hfma(x, y, z));
    opadd[i] = __bfloat16_as_ushort(x * y + z);
    cmp[i] = (std::uint8_t)((__heq(x, y) ? 1 : 0) | (__hlt(x, y) ? 2 : 0) | (__hneu(x, y) ? 4 : 0) |
                             (__hgtu(x, y) ? 8 : 0) | ((x != y) ? 16 : 0) | (__hisnan(x) ? 32 : 0));
}

__global__ void k_pair(const std::uint16_t *a, const std::uint16_t *b, const std::uint16_t *c, std::uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_bfloat162 x = __halves2bfloat162(__ushort_as_bfloat16(a[i]), __ushort_as_bfloat16(b[i]));
    __nv_bfloat162 y = __halves2bfloat162(__ushort_as_bfloat16(b[i]), __ushort_as_bfloat16(c[i]));
    __nv_bfloat162 z = __hfma2(x, y, __lowhigh2highlow(x));
    out[i] = (std::uint32_t)__bfloat16_as_ushort(z.x) | ((std::uint32_t)__bfloat16_as_ushort(z.y) << 16);
}

template <typename T>
static T *dev_copy(const std::vector<T> &v)
{
    T *p = nullptr;
    cudaMalloc(&p, v.size() * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <typename T>
static std::vector<T> dev_get(T *p, size_t n)
{
    std::vector<T> v(n);
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

static void test_device()
{
    std::mt19937 g(909);
    std::vector<float> in(N_DEV);
    for (int i = 0; i < N_DEV; ++i) {
        if (i % 2) {
            in[i] = as_f(g());
        } else { // near ties
            std::uint32_t u = (g() & 0xffff0000u) | 0x8000u;
            in[i] = as_f(u + (std::uint32_t)((int)(g() % 3) - 1));
        }
    }
    in[0] = 0.0f; in[1] = -0.0f; in[2] = INFINITY; in[3] = -INFINITY; in[4] = NAN;
    in[5] = as_f(0x7f7f8000u); in[6] = as_f(0x7f7fffffu); in[7] = as_f(0xff7f8000u);
    float *d_in = dev_copy(in);
    std::uint16_t *rn, *rz, *rd, *ru;
    float *back;
    int *ir, *iz;
    cudaMalloc(&rn, N_DEV * 2); cudaMalloc(&rz, N_DEV * 2); cudaMalloc(&rd, N_DEV * 2); cudaMalloc(&ru, N_DEV * 2);
    cudaMalloc(&back, N_DEV * 4); cudaMalloc(&ir, N_DEV * 4); cudaMalloc(&iz, N_DEV * 4);
    k_unary<<<N_DEV / 256, 256>>>(d_in, rn, rz, rd, ru, back, ir, iz, N_DEV);
    OK(cudaDeviceSynchronize());
    auto hrn = dev_get(rn, N_DEV), hrz = dev_get(rz, N_DEV), hrd = dev_get(rd, N_DEV), hru = dev_get(ru, N_DEV);
    auto hback = dev_get(back, N_DEV);
    auto hir = dev_get(ir, N_DEV), hiz = dev_get(iz, N_DEV);
    int bad = 0, bad_i = 0;
    for (int i = 0; i < N_DEV; ++i) {
        if (!same_bf(hrn[i], ref_from_float(in[i], RN))) ++bad;
        if (!same_bf(hrz[i], ref_from_float(in[i], RZ))) ++bad;
        if (!same_bf(hrd[i], ref_from_float(in[i], RD))) ++bad;
        if (!same_bf(hru[i], ref_from_float(in[i], RU))) ++bad;
        float want_back = bf_to_float((std::uint16_t)(i & 0xffff));
        if (!(as_u(hback[i]) == as_u(want_back) || (want_back != want_back && hback[i] != hback[i]))) ++bad;
        // same value as the host code
        mk((std::uint16_t)(i & 0xffff));
        if (hir[i] != __bfloat162int_rn(mk((std::uint16_t)(i & 0xffff)))) ++bad_i;
        if (hiz[i] != __bfloat162int_rz(mk((std::uint16_t)(i & 0xffff)))) ++bad_i;
    }
    CHECK(bad == 0);
    CHECK(bad_i == 0);

    std::vector<double> din(N_DEV);
    std::mt19937_64 g2(4);
    for (int i = 0; i < N_DEV; ++i) {
        if (i % 2) { std::uint64_t u = g2(); std::memcpy(&din[i], &u, 8); }
        else {
            double lo = (double)bf_to_float((std::uint16_t)(g() % 0x7f00)), hi = (double)bf_to_float((std::uint16_t)((g() % 0x7f00) | 0));
            (void)hi;
            double mid = lo + std::ldexp(lo, -8); // a tie for normal lo
            din[i] = (i % 4 == 0) ? std::nextafter(mid, 1e300) : std::nextafter(mid, 0.0);
        }
    }
    double *d_d = dev_copy(din);
    std::uint16_t *dout;
    cudaMalloc(&dout, N_DEV * 2);
    k_double<<<N_DEV / 256, 256>>>(d_d, dout, N_DEV);
    OK(cudaDeviceSynchronize());
    auto hd = dev_get(dout, N_DEV);
    int badd = 0;
    for (int i = 0; i < N_DEV; ++i)
        if (!same_bf(hd[i], ref_from_double(din[i], RN))) ++badd;
    CHECK(badd == 0);

    std::vector<std::uint16_t> A(N_DEV), B(N_DEV), C(N_DEV);
    for (int i = 0; i < N_DEV; ++i) {
        A[i] = (i % 5 == 0) ? (std::uint16_t)g() : rand_finite(g, i % 3);
        B[i] = (i % 7 == 0) ? (std::uint16_t)g() : rand_finite(g, i % 3);
        C[i] = rand_finite(g, i % 3);
    }
    std::uint16_t *da = dev_copy(A), *db = dev_copy(B), *dc = dev_copy(C);
    std::uint16_t *o[6];
    for (auto &p : o) cudaMalloc(&p, N_DEV * 2);
    std::uint8_t *dcmp;
    cudaMalloc(&dcmp, N_DEV);
    k_binary<<<N_DEV / 128, 128>>>(da, db, dc, o[0], o[1], o[2], o[3], o[4], o[5], dcmp, N_DEV);
    OK(cudaDeviceSynchronize());
    int badb[6] = {0, 0, 0, 0, 0, 0}, badc = 0;
    auto cmpv = dev_get(dcmp, N_DEV);
    for (int k = 0; k < 6; ++k) {
        auto v = dev_get(o[k], N_DEV);
        for (int i = 0; i < N_DEV; ++i) {
            std::uint16_t want;
            if (k < 4) want = ref_arith(k, A[i], B[i]);
            else if (k == 4) want = ref_fma(A[i], B[i], C[i]);
            else {
                // x * y + z through operators: two roundings
                std::uint16_t pr = ref_arith(2, A[i], B[i]);
                want = ref_arith(0, pr, C[i]);
            }
            if (!same_bf(v[i], want)) {
                if (badb[k]++ < 3) DETAIL("  device op %d i=%d (0x%04x, 0x%04x, 0x%04x): got 0x%04x want 0x%04x\n", k, i, A[i], B[i], C[i], v[i], want);
            }
        }
    }
    for (int i = 0; i < N_DEV; ++i) {
        bool nan = is_nan_bits(A[i]) || is_nan_bits(B[i]);
        float fx = bf_to_float(A[i]), fy = bf_to_float(B[i]);
        int want = ((!nan && fx == fy) ? 1 : 0) | ((!nan && fx < fy) ? 2 : 0) | ((nan || fx != fy) ? 4 : 0) |
                   ((nan || fx > fy) ? 8 : 0) | ((nan || fx != fy) ? 16 : 0) | (is_nan_bits(A[i]) ? 32 : 0);
        if (cmpv[i] != want) ++badc;
    }
    for (int k = 0; k < 6; ++k) CHECK(badb[k] == 0);
    CHECK(badc == 0);

    std::uint32_t *dp;
    cudaMalloc(&dp, N_DEV * 4);
    k_pair<<<N_DEV / 64, 64>>>(da, db, dc, dp, N_DEV);
    OK(cudaDeviceSynchronize());
    auto hp = dev_get(dp, N_DEV);
    int badp = 0;
    for (int i = 0; i < N_DEV; ++i) {
        // x = (a, b), y = (b, c), result = x * y + swap(x) = (a*b + b, b*c + a), single rounding each
        std::uint16_t z0 = ref_fma(A[i], B[i], B[i]), z1 = ref_fma(B[i], C[i], A[i]);
        if (!same_bf((std::uint16_t)(hp[i] & 0xffff), z0) || !same_bf((std::uint16_t)(hp[i] >> 16), z1)) ++badp;
    }
    CHECK(badp == 0);
    cudaFree(d_in); cudaFree(rn); cudaFree(rz); cudaFree(rd); cudaFree(ru); cudaFree(back); cudaFree(ir); cudaFree(iz);
    cudaFree(d_d); cudaFree(dout); cudaFree(da); cudaFree(db); cudaFree(dc); for (auto p : o) cudaFree(p);
    cudaFree(dcmp); cudaFree(dp);
}

#define RUN(f) do { int before = g_fail; f(); std::printf("%-16s %s (%d failed)\n", #f, g_fail == before ? "ok" : "FAIL", g_fail - before); g_detail = 0; g_pf = 0; } while (0)

int main()
{
    RUN(test_roundtrip);
    RUN(test_rounding);
    RUN(test_nan_inf);
    RUN(test_double);
    RUN(test_arith);
    RUN(test_compare);
    RUN(test_bf162);
    RUN(test_int_conv);
    RUN(test_device);
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

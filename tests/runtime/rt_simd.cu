// SIMD intrinsics of the public CUDA Math API ("SIMD Intrinsics"): the
// per-halfword (…2) and per-byte (…4) functions and the 32-bit / 16x2
// three-operand and predicate forms (__vimax3_*, __viaddmax_*, __vibmax_*,
// the _relu variants).
//
// Expected values: a host reference written lane by lane from the documented
// semantics (wrap-around, signed / unsigned saturation to the lane's range,
// averages rounded away from zero (signed) or up (unsigned), __vhaddu
// rounded toward zero, "set" = 1 per true lane, "cmp" = all ones per true
// lane, __vsads / __vsadu = the sum over lanes of |a - b|), and the examples
// the documentation gives. Random and edge-value inputs. The same source
// runs on every path.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static int g_fail, g_checks, g_pf;
#define CHECK(c) do { ++g_checks; if (!(c)) { if (g_pf++ < 20) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// The functions under test, by number. X(name, expression in a, b, c).
#define OCU_SIMD_LIST(X) \
    X(vabs2, __vabs2(a)) X(vabs4, __vabs4(a)) X(vabsss2, __vabsss2(a)) X(vabsss4, __vabsss4(a)) \
    X(vabsdiffs2, __vabsdiffs2(a, b)) X(vabsdiffs4, __vabsdiffs4(a, b)) \
    X(vabsdiffu2, __vabsdiffu2(a, b)) X(vabsdiffu4, __vabsdiffu4(a, b)) \
    X(vadd2, __vadd2(a, b)) X(vadd4, __vadd4(a, b)) X(vaddss2, __vaddss2(a, b)) X(vaddss4, __vaddss4(a, b)) \
    X(vaddus2, __vaddus2(a, b)) X(vaddus4, __vaddus4(a, b)) \
    X(vsub2, __vsub2(a, b)) X(vsub4, __vsub4(a, b)) X(vsubss2, __vsubss2(a, b)) X(vsubss4, __vsubss4(a, b)) \
    X(vsubus2, __vsubus2(a, b)) X(vsubus4, __vsubus4(a, b)) \
    X(vavgs2, __vavgs2(a, b)) X(vavgs4, __vavgs4(a, b)) X(vavgu2, __vavgu2(a, b)) X(vavgu4, __vavgu4(a, b)) \
    X(vhaddu2, __vhaddu2(a, b)) X(vhaddu4, __vhaddu4(a, b)) \
    X(vcmpeq2, __vcmpeq2(a, b)) X(vcmpeq4, __vcmpeq4(a, b)) X(vcmpne2, __vcmpne2(a, b)) X(vcmpne4, __vcmpne4(a, b)) \
    X(vcmpges2, __vcmpges2(a, b)) X(vcmpges4, __vcmpges4(a, b)) X(vcmpgeu2, __vcmpgeu2(a, b)) X(vcmpgeu4, __vcmpgeu4(a, b)) \
    X(vcmpgts2, __vcmpgts2(a, b)) X(vcmpgts4, __vcmpgts4(a, b)) X(vcmpgtu2, __vcmpgtu2(a, b)) X(vcmpgtu4, __vcmpgtu4(a, b)) \
    X(vcmples2, __vcmples2(a, b)) X(vcmples4, __vcmples4(a, b)) X(vcmpleu2, __vcmpleu2(a, b)) X(vcmpleu4, __vcmpleu4(a, b)) \
    X(vcmplts2, __vcmplts2(a, b)) X(vcmplts4, __vcmplts4(a, b)) X(vcmpltu2, __vcmpltu2(a, b)) X(vcmpltu4, __vcmpltu4(a, b)) \
    X(vseteq2, __vseteq2(a, b)) X(vseteq4, __vseteq4(a, b)) X(vsetne2, __vsetne2(a, b)) X(vsetne4, __vsetne4(a, b)) \
    X(vsetges2, __vsetges2(a, b)) X(vsetges4, __vsetges4(a, b)) X(vsetgeu2, __vsetgeu2(a, b)) X(vsetgeu4, __vsetgeu4(a, b)) \
    X(vsetgts2, __vsetgts2(a, b)) X(vsetgts4, __vsetgts4(a, b)) X(vsetgtu2, __vsetgtu2(a, b)) X(vsetgtu4, __vsetgtu4(a, b)) \
    X(vsetles2, __vsetles2(a, b)) X(vsetles4, __vsetles4(a, b)) X(vsetleu2, __vsetleu2(a, b)) X(vsetleu4, __vsetleu4(a, b)) \
    X(vsetlts2, __vsetlts2(a, b)) X(vsetlts4, __vsetlts4(a, b)) X(vsetltu2, __vsetltu2(a, b)) X(vsetltu4, __vsetltu4(a, b)) \
    X(vmaxs2, __vmaxs2(a, b)) X(vmaxs4, __vmaxs4(a, b)) X(vmaxu2, __vmaxu2(a, b)) X(vmaxu4, __vmaxu4(a, b)) \
    X(vmins2, __vmins2(a, b)) X(vmins4, __vmins4(a, b)) X(vminu2, __vminu2(a, b)) X(vminu4, __vminu4(a, b)) \
    X(vneg2, __vneg2(a)) X(vneg4, __vneg4(a)) X(vnegss2, __vnegss2(a)) X(vnegss4, __vnegss4(a)) \
    X(vsads2, __vsads2(a, b)) X(vsads4, __vsads4(a, b)) X(vsadu2, __vsadu2(a, b)) X(vsadu4, __vsadu4(a, b)) \
    X(vimax3_s32, (unsigned)__vimax3_s32((int)a, (int)b, (int)c)) X(vimax3_u32, __vimax3_u32(a, b, c)) \
    X(vimax3_s16x2, __vimax3_s16x2(a, b, c)) X(vimax3_u16x2, __vimax3_u16x2(a, b, c)) \
    X(vimax3_s32_relu, (unsigned)__vimax3_s32_relu((int)a, (int)b, (int)c)) X(vimax3_s16x2_relu, __vimax3_s16x2_relu(a, b, c)) \
    X(vimin3_s32, (unsigned)__vimin3_s32((int)a, (int)b, (int)c)) X(vimin3_u32, __vimin3_u32(a, b, c)) \
    X(vimin3_s16x2, __vimin3_s16x2(a, b, c)) X(vimin3_u16x2, __vimin3_u16x2(a, b, c)) \
    X(vimin3_s32_relu, (unsigned)__vimin3_s32_relu((int)a, (int)b, (int)c)) X(vimin3_s16x2_relu, __vimin3_s16x2_relu(a, b, c)) \
    X(vimax_s32_relu, (unsigned)__vimax_s32_relu((int)a, (int)b)) X(vimax_s16x2_relu, __vimax_s16x2_relu(a, b)) \
    X(vimin_s32_relu, (unsigned)__vimin_s32_relu((int)a, (int)b)) X(vimin_s16x2_relu, __vimin_s16x2_relu(a, b)) \
    X(viaddmax_s32, (unsigned)__viaddmax_s32((int)a, (int)b, (int)c)) X(viaddmax_u32, __viaddmax_u32(a, b, c)) \
    X(viaddmax_s16x2, __viaddmax_s16x2(a, b, c)) X(viaddmax_u16x2, __viaddmax_u16x2(a, b, c)) \
    X(viaddmax_s32_relu, (unsigned)__viaddmax_s32_relu((int)a, (int)b, (int)c)) X(viaddmax_s16x2_relu, __viaddmax_s16x2_relu(a, b, c)) \
    X(viaddmin_s32, (unsigned)__viaddmin_s32((int)a, (int)b, (int)c)) X(viaddmin_u32, __viaddmin_u32(a, b, c)) \
    X(viaddmin_s16x2, __viaddmin_s16x2(a, b, c)) X(viaddmin_u16x2, __viaddmin_u16x2(a, b, c)) \
    X(viaddmin_s32_relu, (unsigned)__viaddmin_s32_relu((int)a, (int)b, (int)c)) X(viaddmin_s16x2_relu, __viaddmin_s16x2_relu(a, b, c))

enum {
#define OCU_ID(name, expr) F_##name,
    OCU_SIMD_LIST(OCU_ID)
#undef OCU_ID
    NF
};
static const char *const kNames[] = {
#define OCU_NAME(name, expr) #name,
    OCU_SIMD_LIST(OCU_NAME)
#undef OCU_NAME
};

__device__ unsigned apply(int f, unsigned a, unsigned b, unsigned c)
{
    switch (f) {
#define OCU_CASE(name, expr) case F_##name: return (expr);
        OCU_SIMD_LIST(OCU_CASE)
#undef OCU_CASE
    }
    return 0;
}

__global__ void run_all(int n, const unsigned *a, const unsigned *b, const unsigned *c, unsigned *out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int f = 0; f < NF; ++f) out[(size_t)i * NF + f] = apply(f, a[i], b[i], c[i]);
}

// The predicate forms: result and the predicates (bit 0: pred or pred_lo, bit 1: pred_hi).
__global__ void run_pred(int n, const unsigned *a, const unsigned *b, unsigned *out, unsigned *pred)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool p = false, hi = false, lo = false;
    out[i * 8 + 0] = (unsigned)__vibmax_s32((int)a[i], (int)b[i], &p);
    pred[i * 8 + 0] = p;
    out[i * 8 + 1] = __vibmax_u32(a[i], b[i], &p);
    pred[i * 8 + 1] = p;
    out[i * 8 + 2] = (unsigned)__vibmin_s32((int)a[i], (int)b[i], &p);
    pred[i * 8 + 2] = p;
    out[i * 8 + 3] = __vibmin_u32(a[i], b[i], &p);
    pred[i * 8 + 3] = p;
    out[i * 8 + 4] = __vibmax_s16x2(a[i], b[i], &hi, &lo);
    pred[i * 8 + 4] = lo | (hi << 1);
    out[i * 8 + 5] = __vibmax_u16x2(a[i], b[i], &hi, &lo);
    pred[i * 8 + 5] = lo | (hi << 1);
    out[i * 8 + 6] = __vibmin_s16x2(a[i], b[i], &hi, &lo);
    pred[i * 8 + 6] = lo | (hi << 1);
    out[i * 8 + 7] = __vibmin_u16x2(a[i], b[i], &hi, &lo);
    pred[i * 8 + 7] = lo | (hi << 1);
}

// ---- host reference, lane by lane --------------------------------------------

static long long lane_s(unsigned x, int w, int i) { long long v = (x >> (i * w)) & ((1u << w) - 1); return v >= (1ll << (w - 1)) ? v - (1ll << w) : v; }
static long long lane_u(unsigned x, int w, int i) { return (x >> (i * w)) & ((1u << w) - 1); }
static long long sat_s(long long v, int w) { long long lo = -(1ll << (w - 1)), hi = (1ll << (w - 1)) - 1; return v < lo ? lo : v > hi ? hi : v; }
static long long sat_u(long long v, int w) { long long hi = (1ll << w) - 1; return v < 0 ? 0 : v > hi ? hi : v; }
static long long mx(long long x, long long y) { return x > y ? x : y; }
static long long mn(long long x, long long y) { return x < y ? x : y; }
static long long avg_away(long long s) { return s >= 0 ? (s + 1) / 2 : -((-s + 1) / 2); }

// Pack per-lane values (taken modulo the lane width).
template <class F>
static unsigned lanes(int w, F f)
{
    unsigned r = 0;
    for (int i = 0; i < 32 / w; ++i) r |= (unsigned)((unsigned long long)f(i) & ((1ull << w) - 1)) << (i * w);
    return r;
}

static unsigned ref(int f, unsigned a, unsigned b, unsigned c)
{
    const int ia = (int)a, ib = (int)b, ic = (int)c;
    auto S = [](unsigned x, int w, int i) { return lane_s(x, w, i); };
    auto U = [](unsigned x, int w, int i) { return lane_u(x, w, i); };
    auto add_s32 = [](int x, int y) { return (int)((unsigned)x + (unsigned)y); }; // wraps
    switch (f) {
#define W2(e) lanes(16, [&](int i) -> long long { const int w = 16; (void)w; return (e); })
#define W4(e) lanes(8, [&](int i) -> long long { const int w = 8; (void)w; return (e); })
#define BOTH(n2, n4, e) case F_##n2: return W2(e); case F_##n4: return W4(e);
        BOTH(vabs2, vabs4, S(a, w, i) < 0 ? -S(a, w, i) : S(a, w, i))
        BOTH(vabsss2, vabsss4, sat_s(S(a, w, i) < 0 ? -S(a, w, i) : S(a, w, i), w))
        BOTH(vabsdiffs2, vabsdiffs4, S(a, w, i) > S(b, w, i) ? S(a, w, i) - S(b, w, i) : S(b, w, i) - S(a, w, i))
        BOTH(vabsdiffu2, vabsdiffu4, U(a, w, i) > U(b, w, i) ? U(a, w, i) - U(b, w, i) : U(b, w, i) - U(a, w, i))
        BOTH(vadd2, vadd4, U(a, w, i) + U(b, w, i))
        BOTH(vaddss2, vaddss4, sat_s(S(a, w, i) + S(b, w, i), w))
        BOTH(vaddus2, vaddus4, sat_u(U(a, w, i) + U(b, w, i), w))
        BOTH(vsub2, vsub4, U(a, w, i) - U(b, w, i))
        BOTH(vsubss2, vsubss4, sat_s(S(a, w, i) - S(b, w, i), w))
        BOTH(vsubus2, vsubus4, sat_u(U(a, w, i) - U(b, w, i), w))
        BOTH(vavgs2, vavgs4, avg_away(S(a, w, i) + S(b, w, i)))
        BOTH(vavgu2, vavgu4, (U(a, w, i) + U(b, w, i) + 1) / 2)
        BOTH(vhaddu2, vhaddu4, (U(a, w, i) + U(b, w, i)) / 2)
        BOTH(vcmpeq2, vcmpeq4, U(a, w, i) == U(b, w, i) ? -1 : 0)
        BOTH(vcmpne2, vcmpne4, U(a, w, i) != U(b, w, i) ? -1 : 0)
        BOTH(vcmpges2, vcmpges4, S(a, w, i) >= S(b, w, i) ? -1 : 0)
        BOTH(vcmpgeu2, vcmpgeu4, U(a, w, i) >= U(b, w, i) ? -1 : 0)
        BOTH(vcmpgts2, vcmpgts4, S(a, w, i) > S(b, w, i) ? -1 : 0)
        BOTH(vcmpgtu2, vcmpgtu4, U(a, w, i) > U(b, w, i) ? -1 : 0)
        BOTH(vcmples2, vcmples4, S(a, w, i) <= S(b, w, i) ? -1 : 0)
        BOTH(vcmpleu2, vcmpleu4, U(a, w, i) <= U(b, w, i) ? -1 : 0)
        BOTH(vcmplts2, vcmplts4, S(a, w, i) < S(b, w, i) ? -1 : 0)
        BOTH(vcmpltu2, vcmpltu4, U(a, w, i) < U(b, w, i) ? -1 : 0)
        BOTH(vseteq2, vseteq4, U(a, w, i) == U(b, w, i))
        BOTH(vsetne2, vsetne4, U(a, w, i) != U(b, w, i))
        BOTH(vsetges2, vsetges4, S(a, w, i) >= S(b, w, i))
        BOTH(vsetgeu2, vsetgeu4, U(a, w, i) >= U(b, w, i))
        BOTH(vsetgts2, vsetgts4, S(a, w, i) > S(b, w, i))
        BOTH(vsetgtu2, vsetgtu4, U(a, w, i) > U(b, w, i))
        BOTH(vsetles2, vsetles4, S(a, w, i) <= S(b, w, i))
        BOTH(vsetleu2, vsetleu4, U(a, w, i) <= U(b, w, i))
        BOTH(vsetlts2, vsetlts4, S(a, w, i) < S(b, w, i))
        BOTH(vsetltu2, vsetltu4, U(a, w, i) < U(b, w, i))
        BOTH(vmaxs2, vmaxs4, mx(S(a, w, i), S(b, w, i)))
        BOTH(vmaxu2, vmaxu4, mx(U(a, w, i), U(b, w, i)))
        BOTH(vmins2, vmins4, mn(S(a, w, i), S(b, w, i)))
        BOTH(vminu2, vminu4, mn(U(a, w, i), U(b, w, i)))
        BOTH(vneg2, vneg4, -S(a, w, i))
        BOTH(vnegss2, vnegss4, sat_s(-S(a, w, i), w))
#undef BOTH
    case F_vsads2: case F_vsads4: case F_vsadu2: case F_vsadu4: {
        const int w = (f == F_vsads2 || f == F_vsadu2) ? 16 : 8;
        const bool sg = f == F_vsads2 || f == F_vsads4;
        long long s = 0;
        for (int i = 0; i < 32 / w; ++i) {
            const long long x = sg ? S(a, w, i) : U(a, w, i), y = sg ? S(b, w, i) : U(b, w, i);
            s += x > y ? x - y : y - x;
        }
        return (unsigned)s;
    }
    case F_vimax3_s32: return (unsigned)(int)mx(mx(ia, ib), ic);
    case F_vimax3_u32: return (unsigned)mx(mx(a, b), c);
    case F_vimax3_s16x2: return W2(mx(mx(S(a, 16, i), S(b, 16, i)), S(c, 16, i)));
    case F_vimax3_u16x2: return W2(mx(mx(U(a, 16, i), U(b, 16, i)), U(c, 16, i)));
    case F_vimax3_s32_relu: return (unsigned)(int)mx(mx(mx(ia, ib), ic), 0);
    case F_vimax3_s16x2_relu: return W2(mx(mx(mx(S(a, 16, i), S(b, 16, i)), S(c, 16, i)), 0));
    case F_vimin3_s32: return (unsigned)(int)mn(mn(ia, ib), ic);
    case F_vimin3_u32: return (unsigned)mn(mn(a, b), c);
    case F_vimin3_s16x2: return W2(mn(mn(S(a, 16, i), S(b, 16, i)), S(c, 16, i)));
    case F_vimin3_u16x2: return W2(mn(mn(U(a, 16, i), U(b, 16, i)), U(c, 16, i)));
    case F_vimin3_s32_relu: return (unsigned)(int)mx(mn(mn(ia, ib), ic), 0);
    case F_vimin3_s16x2_relu: return W2(mx(mn(mn(S(a, 16, i), S(b, 16, i)), S(c, 16, i)), 0));
    case F_vimax_s32_relu: return (unsigned)(int)mx(mx(ia, ib), 0);
    case F_vimax_s16x2_relu: return W2(mx(mx(S(a, 16, i), S(b, 16, i)), 0));
    case F_vimin_s32_relu: return (unsigned)(int)mx(mn(ia, ib), 0);
    case F_vimin_s16x2_relu: return W2(mx(mn(S(a, 16, i), S(b, 16, i)), 0));
    // a + b wraps in the lane (32 or 16 bits), then max / min with c.
    case F_viaddmax_s32: return (unsigned)(int)mx(add_s32(ia, ib), ic);
    case F_viaddmax_u32: return (unsigned)mx((unsigned)(a + b), c);
    case F_viaddmax_s16x2: return W2(mx(lane_s((unsigned)(S(a, 16, i) + S(b, 16, i)) & 0xffff, 16, 0), S(c, 16, i)));
    case F_viaddmax_u16x2: return W2(mx((U(a, 16, i) + U(b, 16, i)) & 0xffff, U(c, 16, i)));
    case F_viaddmax_s32_relu: return (unsigned)(int)mx(mx(add_s32(ia, ib), ic), 0);
    case F_viaddmax_s16x2_relu: return W2(mx(mx(lane_s((unsigned)(S(a, 16, i) + S(b, 16, i)) & 0xffff, 16, 0), S(c, 16, i)), 0));
    case F_viaddmin_s32: return (unsigned)(int)mn(add_s32(ia, ib), ic);
    case F_viaddmin_u32: return (unsigned)mn((unsigned)(a + b), c);
    case F_viaddmin_s16x2: return W2(mn(lane_s((unsigned)(S(a, 16, i) + S(b, 16, i)) & 0xffff, 16, 0), S(c, 16, i)));
    case F_viaddmin_u16x2: return W2(mn((U(a, 16, i) + U(b, 16, i)) & 0xffff, U(c, 16, i)));
    case F_viaddmin_s32_relu: return (unsigned)(int)mx(mn(add_s32(ia, ib), ic), 0);
    case F_viaddmin_s16x2_relu: return W2(mx(mn(lane_s((unsigned)(S(a, 16, i) + S(b, 16, i)) & 0xffff, 16, 0), S(c, 16, i)), 0));
#undef W2
#undef W4
    }
    return 0xdeadbeefu;
}

// Inputs: lane edge values (0, 1, max, min, min+1, -1 in every lane width)
// mixed with random words.
static std::vector<unsigned> inputs(int n, uint32_t seed)
{
    static const unsigned edge[] = {0u, 1u, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0x80008000u, 0x7fff7fffu,
                                    0x80808080u, 0x7f7f7f7fu, 0x81818181u, 0x80018001u, 0x00ff00ffu, 0xff00ff00u,
                                    0x0000ffffu, 0xffff0000u, 0x01010101u, 0xfefefefeu, 0x807f3302u, 0x7f803301u};
    std::mt19937 g(seed);
    std::vector<unsigned> v(n);
    for (int i = 0; i < n; ++i) {
        const int k = g() % 4;
        if (k == 0) v[i] = edge[g() % (sizeof edge / sizeof edge[0])];
        else if (k == 1) { // per-byte mix of edge bytes
            static const unsigned char eb[] = {0x00, 0x01, 0x7f, 0x80, 0x81, 0xfe, 0xff};
            unsigned w = 0;
            for (int j = 0; j < 4; ++j) w |= (unsigned)eb[g() % 7] << (8 * j);
            v[i] = w;
        } else v[i] = g();
    }
    return v;
}

static void test_all()
{
    const int n = 20000;
    std::vector<unsigned> a = inputs(n, 1), b = inputs(n, 2), c = inputs(n, 3);
    unsigned *da, *db, *dc, *dout;
    cudaMalloc(&da, n * 4); cudaMalloc(&db, n * 4); cudaMalloc(&dc, n * 4);
    cudaMalloc(&dout, (size_t)n * NF * 4);
    cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c.data(), n * 4, cudaMemcpyHostToDevice);
    run_all<<<(n + 127) / 128, 128>>>(n, da, db, dc, dout);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<unsigned> out((size_t)n * NF);
    cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
    int bad[NF] = {};
    for (int i = 0; i < n; ++i)
        for (int f = 0; f < NF; ++f) {
            const unsigned want = ref(f, a[i], b[i], c[i]), got = out[(size_t)i * NF + f];
            ++g_checks;
            if (got != want) {
                if (bad[f]++ == 0 && g_pf++ < 40)
                    std::printf("FAIL %s(0x%08x, 0x%08x, 0x%08x) = 0x%08x, want 0x%08x\n", kNames[f], a[i], b[i], c[i], got, want);
                ++g_fail;
            }
        }
    int nbad = 0;
    for (int f = 0; f < NF; ++f) nbad += bad[f] != 0;
    std::printf("simd: %d functions x %d inputs, %d functions with failures\n", NF, n, nbad);

    // Predicate forms.
    unsigned *dp, *dpo;
    cudaMalloc(&dp, (size_t)n * 8 * 4); cudaMalloc(&dpo, (size_t)n * 8 * 4);
    run_pred<<<(n + 127) / 128, 128>>>(n, da, db, dpo, dp);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<unsigned> po((size_t)n * 8), pp((size_t)n * 8);
    cudaMemcpy(po.data(), dpo, po.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(pp.data(), dp, pp.size() * 4, cudaMemcpyDeviceToHost);
    int pbad = 0;
    for (int i = 0; i < n; ++i) {
        const int x = (int)a[i], y = (int)b[i];
        pbad += po[i * 8 + 0] != (unsigned)(x > y ? x : y) || pp[i * 8 + 0] != (unsigned)(x >= y);
        pbad += po[i * 8 + 1] != (a[i] > b[i] ? a[i] : b[i]) || pp[i * 8 + 1] != (unsigned)(a[i] >= b[i]);
        pbad += po[i * 8 + 2] != (unsigned)(x < y ? x : y) || pp[i * 8 + 2] != (unsigned)(x <= y);
        pbad += po[i * 8 + 3] != (a[i] < b[i] ? a[i] : b[i]) || pp[i * 8 + 3] != (unsigned)(a[i] <= b[i]);
        for (int k = 0; k < 4; ++k) {
            const bool sg = k == 0 || k == 2, isMax = k < 2;
            unsigned r = 0, p = 0;
            for (int l = 0; l < 2; ++l) {
                const long long u = sg ? lane_s(a[i], 16, l) : lane_u(a[i], 16, l);
                const long long v = sg ? lane_s(b[i], 16, l) : lane_u(b[i], 16, l);
                r |= (unsigned)((isMax ? mx(u, v) : mn(u, v)) & 0xffff) << (16 * l);
                p |= (unsigned)(isMax ? u >= v : u <= v) << l; // bit 0: low halfword (pred_lo), bit 1: high (pred_hi)
            }
            pbad += po[i * 8 + 4 + k] != r || pp[i * 8 + 4 + k] != p;
        }
    }
    g_checks += n;
    CHECK(pbad == 0);
    if (pbad) std::printf("predicate forms: %d mismatches\n", pbad);
    cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dout); cudaFree(dp); cudaFree(dpo);
}

// The examples in the documentation.
__global__ void doc_examples(unsigned *o)
{
    o[0] = __vabs2(0x80008000u);
    o[1] = __vabsss2(0x80008000u);
    o[2] = __vabsss4(0x80808080u);
    o[3] = __vcmpeq2(0x12345678u, 0x1234abcdu);
    o[4] = __vcmpgts4(0x807f3302u, 0x7f803301u);
    o[5] = __vcmpges4(0x807f3302u, 0x7f803301u);
    o[6] = __vcmpgeu4(0x807f3302u, 0x7f803301u);
    o[7] = __vsetges4(0x807f3302u, 0x7f803301u);
    o[8] = __vneg2(0x80008000u);
}

static void test_doc_examples()
{
    unsigned *d, h[9];
    cudaMalloc(&d, sizeof h);
    doc_examples<<<1, 1>>>(d);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    CHECK(h[0] == 0x80008000u);
    CHECK(h[1] == 0x7fff7fffu);
    CHECK(h[2] == 0x7f7f7f7fu);
    CHECK(h[3] == 0xffff0000u);
    CHECK(h[4] == 0x00ff00ffu);
    CHECK(h[5] == 0x00ffffffu);
    CHECK(h[6] == 0xff00ffffu);
    CHECK(h[7] == 0x00010101u);
    CHECK(h[8] == 0x80008000u);
    cudaFree(d);
}

int main()
{
    test_doc_examples();
    test_all();
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

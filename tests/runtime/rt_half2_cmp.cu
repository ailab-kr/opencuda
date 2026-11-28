// __half2 comparisons and operators, and the double word intrinsics, in a
// kernel, against values the host computes from the CUDA Math API
// description: __heq2 ... (1.0 / 0.0 per lane), the _mask forms (0xFFFF /
// 0 per 16-bit half, low lane low), __hbeq2 ... (both lanes), the unordered
// u forms (true when either is NaN), __hisnan2, the __half2 operators
// (== != < <= > >= unary - and the compound assignments), __double2hiint /
// __double2loint / __hiloint2double.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_half2_cmp.cu -o rt_half2_cmp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { if (g_fail++ < 20) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

enum { NCMP = 12 };
// per pair: 12 comparisons x (2 lanes as 1/0, mask, both) + isnan2 + 6 operators + arith
struct out_t {
    float lane[NCMP][2];
    unsigned mask[NCMP];
    int both[NCMP];
    float isnan2[2];
    int ops[6];
    float neg[2], sub_assign[2], div_assign[2];
};

__global__ void k(const __half2 *a, const __half2 *b, out_t *o, int n, const double *d, int *hi, int *lo, double *back)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __half2 x = a[i], y = b[i];
        __half2 r[NCMP] = {__heq2(x, y), __hne2(x, y), __hlt2(x, y), __hle2(x, y), __hgt2(x, y), __hge2(x, y),
                           __hequ2(x, y), __hneu2(x, y), __hltu2(x, y), __hleu2(x, y), __hgtu2(x, y), __hgeu2(x, y)};
        unsigned m[NCMP] = {__heq2_mask(x, y), __hne2_mask(x, y), __hlt2_mask(x, y), __hle2_mask(x, y), __hgt2_mask(x, y), __hge2_mask(x, y),
                            __hequ2_mask(x, y), __hneu2_mask(x, y), __hltu2_mask(x, y), __hleu2_mask(x, y), __hgtu2_mask(x, y), __hgeu2_mask(x, y)};
        bool bb[NCMP] = {__hbeq2(x, y), __hbne2(x, y), __hblt2(x, y), __hble2(x, y), __hbgt2(x, y), __hbge2(x, y),
                         __hbequ2(x, y), __hbneu2(x, y), __hbltu2(x, y), __hbleu2(x, y), __hbgtu2(x, y), __hbgeu2(x, y)};
        for (int c = 0; c < NCMP; ++c) {
            o[i].lane[c][0] = __low2float(r[c]);
            o[i].lane[c][1] = __high2float(r[c]);
            o[i].mask[c] = m[c];
            o[i].both[c] = bb[c];
        }
        __half2 nn = __hisnan2(x);
        o[i].isnan2[0] = __low2float(nn); o[i].isnan2[1] = __high2float(nn);
        o[i].ops[0] = x == y; o[i].ops[1] = x != y; o[i].ops[2] = x < y;
        o[i].ops[3] = x <= y; o[i].ops[4] = x > y; o[i].ops[5] = x >= y;
        __half2 ng = -x, s = x, q = x;
        s -= y;
        q /= y;
        o[i].neg[0] = __low2float(ng); o[i].neg[1] = __high2float(ng);
        o[i].sub_assign[0] = __low2float(s); o[i].sub_assign[1] = __high2float(s);
        o[i].div_assign[0] = __low2float(q); o[i].div_assign[1] = __high2float(q);
    }
    if (i < 8) {
        hi[i] = __double2hiint(d[i]);
        lo[i] = __double2loint(d[i]);
        back[i] = __hiloint2double(hi[i], lo[i]);
    }
}

static bool cmp(int c, float x, float y)
{
    bool un = std::isnan(x) || std::isnan(y);
    switch (c) {
    case 0: return x == y;
    case 1: return !un && x != y;
    case 2: return x < y;
    case 3: return x <= y;
    case 4: return x > y;
    case 5: return x >= y;
    case 6: return un || x == y;
    case 7: return un || x != y;
    case 8: return un || x < y;
    case 9: return un || x <= y;
    case 10: return un || x > y;
    default: return un || x >= y;
    }
}

int main()
{
    // values exactly representable in half, both signs, zeros, infinities, NaN
    const float vals[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 65504.0f, -65504.0f, INFINITY, -INFINITY, NAN, 3.0f};
    const int nv = sizeof vals / sizeof vals[0];
    const int n = nv * nv;
    __half2 ha[n], hb[n];
    float fa[n][2], fb[n][2];
    for (int i = 0; i < nv; ++i)
        for (int j = 0; j < nv; ++j) {
            int t = i * nv + j;
            fa[t][0] = vals[i]; fa[t][1] = vals[(i + 3) % nv];
            fb[t][0] = vals[j]; fb[t][1] = vals[(j + 5) % nv];
            ha[t] = __floats2half2_rn(fa[t][0], fa[t][1]);
            hb[t] = __floats2half2_rn(fb[t][0], fb[t][1]);
        }
    const double dv[8] = {0.0, -0.0, 1.0, -2.5, 1e300, 5e-324, INFINITY, 3.141592653589793};
    __half2 *da, *db; out_t *dout; double *dd, *dback; int *dhi, *dlo;
    cudaMalloc(&da, sizeof ha); cudaMalloc(&db, sizeof hb); cudaMalloc(&dout, n * sizeof(out_t));
    cudaMalloc(&dd, sizeof dv); cudaMalloc(&dback, sizeof dv); cudaMalloc(&dhi, 8 * sizeof(int)); cudaMalloc(&dlo, 8 * sizeof(int));
    cudaMemcpy(da, ha, sizeof ha, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof hb, cudaMemcpyHostToDevice);
    cudaMemcpy(dd, dv, sizeof dv, cudaMemcpyHostToDevice);
    k<<<(n + 63) / 64, 64>>>(da, db, dout, n, dd, dhi, dlo, dback);
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "kernel");
    static out_t o[n];
    int hi[8], lo[8]; double back[8];
    cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
    cudaMemcpy(hi, dhi, sizeof hi, cudaMemcpyDeviceToHost);
    cudaMemcpy(lo, dlo, sizeof lo, cudaMemcpyDeviceToHost);
    cudaMemcpy(back, dback, sizeof back, cudaMemcpyDeviceToHost);
    static const char *names[NCMP] = {"eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu"};
    for (int t = 0; t < n; ++t) {
        for (int c = 0; c < NCMP; ++c) {
            bool l0 = cmp(c, fa[t][0], fb[t][0]), l1 = cmp(c, fa[t][1], fb[t][1]);
            CHECK(o[t].lane[c][0] == (l0 ? 1.0f : 0.0f) && o[t].lane[c][1] == (l1 ? 1.0f : 0.0f),
                  "__h%s2(%g,%g ; %g,%g) = %g,%g", names[c], fa[t][0], fa[t][1], fb[t][0], fb[t][1], o[t].lane[c][0], o[t].lane[c][1]);
            CHECK(o[t].mask[c] == ((l0 ? 0xffffu : 0u) | (l1 ? 0xffff0000u : 0u)), "__h%s2_mask = %08x", names[c], o[t].mask[c]);
            CHECK(o[t].both[c] == (l0 && l1), "__hb%s2", names[c]);
        }
        CHECK(o[t].isnan2[0] == (std::isnan(fa[t][0]) ? 1.0f : 0.0f) && o[t].isnan2[1] == (std::isnan(fa[t][1]) ? 1.0f : 0.0f), "__hisnan2");
        const int opc[6] = {0, 7, 2, 3, 4, 5}; // == is hbeq2, != is hbneu2
        for (int p = 0; p < 6; ++p)
            CHECK(o[t].ops[p] == (cmp(opc[p], fa[t][0], fb[t][0]) && cmp(opc[p], fa[t][1], fb[t][1])), "operator %d", p);
        for (int l = 0; l < 2; ++l) {
            float x = fa[t][l], y = fb[t][l];
            float want_neg = -x, got = o[t].neg[l];
            CHECK((std::isnan(want_neg) && std::isnan(got)) || (want_neg == got && std::signbit(want_neg) == std::signbit(got)), "unary - lane %d of %g: %g", l, x, got);
            float ws = __half2float(__float2half(x - y)), wq = __half2float(__float2half(x / y));
            CHECK((std::isnan(ws) && std::isnan(o[t].sub_assign[l])) || ws == o[t].sub_assign[l], "-= lane %d: %g - %g = %g", l, x, y, o[t].sub_assign[l]);
            CHECK((std::isnan(wq) && std::isnan(o[t].div_assign[l])) || wq == o[t].div_assign[l], "/= lane %d: %g / %g = %g", l, x, y, o[t].div_assign[l]);
        }
    }
    for (int i = 0; i < 8; ++i) {
        unsigned long long b; std::memcpy(&b, &dv[i], 8);
        CHECK(hi[i] == (int)(b >> 32) && lo[i] == (int)(unsigned)b, "__double2hiint/loint(%g)", dv[i]);
        CHECK(std::memcmp(&back[i], &dv[i], 8) == 0, "__hiloint2double(%g)", dv[i]);
    }
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

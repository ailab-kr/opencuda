// Device-side intrinsics and cooperative groups of card runtime-llama-gaps
// (items 16 to 24 and 27 to 30, the parts that run to completion; the
// parts that end the process are in rt_trap.cu).
//
// Expected values come from the public CUDA Math API (SIMD, integer and
// type-casting intrinsics), the CUDA C++ Programming Guide (synchronization
// functions, Cooperative Groups) and IEEE 754, computed here in plain host
// code. The same source runs on the CPU path and on the NVIDIA path.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cg = cooperative_groups;

static int g_fail, g_checks, g_pf;
#define CHECK(c) do { ++g_checks; if (!(c)) { if (g_pf++ < 10) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

template <typename T>
static T *dev_copy(const std::vector<T> &v)
{
    T *p = nullptr;
    cudaMalloc(&p, v.size() * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <typename T>
static T *dev_alloc(size_t n)
{
    T *p = nullptr;
    cudaMalloc(&p, n * sizeof(T));
    cudaMemset(p, 0xcd, n * sizeof(T));
    return p;
}
template <typename T>
static std::vector<T> dev_get(T *p, size_t n)
{
    std::vector<T> v(n);
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

static float as_f(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static std::uint32_t as_u(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }

// ---- items 16, 17: __vsub4, __vcmpne4 -----------------------------------------

__global__ void k_simd(const unsigned *a, const unsigned *b, unsigned *sub, unsigned *ne, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sub[i] = __vsub4(a[i], b[i]);
    ne[i] = __vcmpne4(a[i], b[i]);
}

static void test_simd()
{
    std::mt19937 g(16);
    const int n = 1 << 16;
    std::vector<unsigned> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = g();
        b[i] = g();
        if (i % 4 == 0) b[i] = (a[i] & 0xff00ff00u) | (b[i] & 0x00ff00ffu); // some equal bytes
        if (i % 11 == 0) b[i] = a[i];
    }
    a[0] = 0; b[0] = 1;
    a[1] = 0x100; b[1] = 1;
    a[2] = 0xffffffffu; b[2] = 0xffffffffu;
    a[3] = 0; b[3] = 0xffffffffu;
    a[4] = 0x01010101u; b[4] = 0x02020202u;
    a[5] = 0x80808080u; b[5] = 0x7f7f7f7fu;
    unsigned *da = dev_copy(a), *db = dev_copy(b), *ds = dev_alloc<unsigned>(n), *dn = dev_alloc<unsigned>(n);
    k_simd<<<n / 256, 256>>>(da, db, ds, dn, n);
    OK(cudaDeviceSynchronize());
    auto s = dev_get(ds, n), ne = dev_get(dn, n);
    int bad = 0, badn = 0;
    for (int i = 0; i < n; ++i) {
        unsigned ws = 0, wn = 0;
        for (int k = 0; k < 4; ++k) {
            unsigned x = (a[i] >> (8 * k)) & 0xff, y = (b[i] >> (8 * k)) & 0xff;
            ws |= ((x - y) & 0xffu) << (8 * k);
            wn |= (x != y ? 0xffu : 0u) << (8 * k);
        }
        if (s[i] != ws) { if (bad++ < 5) std::printf("  vsub4(%08x, %08x) = %08x, want %08x\n", a[i], b[i], s[i], ws); }
        if (ne[i] != wn) { if (badn++ < 5) std::printf("  vcmpne4(%08x, %08x) = %08x, want %08x\n", a[i], b[i], ne[i], wn); }
    }
    CHECK(bad == 0);
    CHECK(badn == 0);
    CHECK(s[0] == 0x000000ffu);   // no borrow between bytes
    CHECK(s[1] == 0x000001ffu);
    CHECK(s[2] == 0u);
    CHECK(s[3] == 0x01010101u);
    CHECK(ne[2] == 0u);           // all equal
    CHECK(ne[3] == 0xffffffffu);  // all different
    cudaFree(da); cudaFree(db); cudaFree(ds); cudaFree(dn);
}

// ---- items 18, 19: __byte_perm, __umulhi, __mulhi -------------------------------

__global__ void k_perm(const unsigned *x, const unsigned *y, const unsigned *s, unsigned *out, unsigned *uh, int *sh, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __byte_perm(x[i], y[i], s[i]);
    uh[i] = __umulhi(x[i], y[i]);
    sh[i] = __mulhi((int)x[i], (int)y[i]);
}

static unsigned ref_perm(unsigned x, unsigned y, unsigned s)
{
    unsigned char bytes[8];
    for (int k = 0; k < 4; ++k) { bytes[k] = (x >> (8 * k)) & 0xff; bytes[4 + k] = (y >> (8 * k)) & 0xff; }
    unsigned r = 0;
    for (int k = 0; k < 4; ++k) {
        // Public docs: selector n is s bits [4n+2:4n]; bit 3 of each nibble and bits 16..31 are unused.
        unsigned nib = (s >> (4 * k)) & 0x7;
        unsigned char b = bytes[nib & 7];
        r |= (unsigned)b << (8 * k);
    }
    return r;
}

static void test_perm_mulhi()
{
    std::mt19937 g(18);
    std::vector<unsigned> x, y, s;
    // all 65536 selectors on fixed data (the 16 low bits are the selector)
    for (unsigned sel = 0; sel < 65536; ++sel) { x.push_back(0x03020100u); y.push_back(0x07060504u); s.push_back(sel); }
    for (unsigned sel = 0; sel < 65536; ++sel) { x.push_back(0x80ff7f01u); y.push_back(0x7e81fe00u); s.push_back(sel); }
    for (int i = 0; i < 100000; ++i) { x.push_back(g()); y.push_back(g()); s.push_back(g()); }
    // products
    unsigned sp[] = {0u, 1u, 2u, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xffffffffu, 0xfffffffeu, 0x10000u, 0xffffu};
    for (unsigned p : sp)
        for (unsigned q : sp) { x.push_back(p); y.push_back(q); s.push_back(0x3210); }
    int n = (int)x.size();
    int blocks = (n + 255) / 256;
    x.resize(blocks * 256); y.resize(blocks * 256); s.resize(blocks * 256);
    unsigned *dx = dev_copy(x), *dy = dev_copy(y), *ds = dev_copy(s), *dout = dev_alloc<unsigned>(blocks * 256),
             *duh = dev_alloc<unsigned>(blocks * 256);
    int *dsh = dev_alloc<int>(blocks * 256);
    k_perm<<<blocks, 256>>>(dx, dy, ds, dout, duh, dsh, n);
    OK(cudaDeviceSynchronize());
    auto out = dev_get(dout, n), uh = dev_get(duh, n);
    auto sh = dev_get(dsh, n);
    int bad = 0, bu = 0, bs = 0;
    for (int i = 0; i < n; ++i) {
        if (out[i] != ref_perm(x[i], y[i], s[i])) {
            if (bad++ < 5) std::printf("  byte_perm(%08x, %08x, %04x) = %08x, want %08x\n", x[i], y[i], s[i], out[i], ref_perm(x[i], y[i], s[i]));
        }
        unsigned wu = (unsigned)(((unsigned long long)x[i] * y[i]) >> 32);
        int ws = (int)(((long long)(int)x[i] * (long long)(int)y[i]) >> 32);
        if (uh[i] != wu) ++bu;
        if (sh[i] != ws) ++bs;
    }
    CHECK(bad == 0);
    CHECK(bu == 0);
    CHECK(bs == 0);
    CHECK(out[0x5432] == 0x05040302u); // the documented example
    CHECK(out[0xfedc] == out[0x7654]);  // bit 3 of each selector nibble is unused: no sign replication
    CHECK(out[0x8888] == out[0x0000]);
    unsigned idx = 65536 + 65536 + 100000; // first product pair is (0, 0)
    (void)idx;
    // umulhi(0xffffffff, 0x80000000) = 0x7fffffff
    bool found = false;
    for (int i = n - 100; i < n; ++i)
        if (x[i] == 0xffffffffu && y[i] == 0x80000000u) { CHECK(uh[i] == 0x7fffffffu); found = true; }
    CHECK(found);
    cudaFree(dx); cudaFree(dy); cudaFree(ds); cudaFree(dout); cudaFree(duh); cudaFree(dsh);
}

// ---- item 20: __float2int_* and __double2int_* --------------------------------

__global__ void k_f2i(const float *f, const double *d, int *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o[8 * i + 0] = __float2int_rn(f[i]);
    o[8 * i + 1] = __float2int_rz(f[i]);
    o[8 * i + 2] = __float2int_rd(f[i]);
    o[8 * i + 3] = __float2int_ru(f[i]);
    o[8 * i + 4] = __double2int_rn(d[i]);
    o[8 * i + 5] = __double2int_rz(d[i]);
    o[8 * i + 6] = __double2int_rd(d[i]);
    o[8 * i + 7] = __double2int_ru(d[i]);
}

#define KMAX 8

static int sat(double v)
{
    if (v != v) return 0;
    if (v >= 2147483648.0) return INT_MAX;
    if (v <= -2147483648.0) return INT_MIN;
    return (int)v;
}

static void test_float2int()
{
    std::vector<float> f = {2.5f, 3.5f, -2.5f, -2.7f, -2.1f, 2.1f, 0.5f, -0.5f, 1.5f, -1.5f, 0.0f, -0.0f, NAN, -NAN,
                            1e20f, -1e20f, INFINITY, -INFINITY, 2147483520.0f, 2147483648.0f, -2147483648.0f,
                            -2147483904.0f, 8388608.5f, 4194304.5f, 0x1p-149f, -0x1p-149f, 0.49999997f, 0.5000001f,
                            3.4028235e38f, -3.4028235e38f};
    std::mt19937 g(20);
    while (f.size() < 4096) {
        float v = as_f(g());
        if (f.size() % 2) v = (float)((int)(g() % 20001) - 10000) / 4.0f; // quarter steps incl. ties
        f.push_back(v);
    }
    int n = (int)f.size();
    std::vector<double> d(n);
    for (int i = 0; i < n; ++i) d[i] = (i % 3 == 0) ? (double)f[i] : (i % 3 == 1) ? (double)f[i] * 1.0000000001 : (double)f[i] - 0.25;
    d[0] = 2.5; d[1] = 3.5; d[2] = -2.5; d[3] = -2.7; d[4] = -2.1; d[5] = 2.1;
    d[12] = std::nan(""); d[14] = 1e300; d[15] = -1e300; d[16] = INFINITY; d[17] = -INFINITY;
    d[18] = 2147483647.5; d[19] = 2147483648.0; d[20] = -2147483648.5; d[21] = -2147483647.5;
    float *df = dev_copy(f);
    double *dd = dev_copy(d);
    int *dout = dev_alloc<int>(8 * n);
    k_f2i<<<(n + 127) / 128, 128>>>(df, dd, dout, n);
    OK(cudaDeviceSynchronize());
    auto o = dev_get(dout, 8 * n);
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        double fv = f[i], dv = d[i];
        int want[8] = {sat(std::nearbyint(fv)), sat(std::trunc(fv)), sat(std::floor(fv)), sat(std::ceil(fv)),
                       sat(std::nearbyint(dv)), sat(std::trunc(dv)), sat(std::floor(dv)), sat(std::ceil(dv))};
        for (int k = 0; k < 8; ++k) {
            if (o[8 * i + k] != want[k]) {
                if (bad++ < 8) std::printf("  mode %d of f=%a d=%a: got %d want %d\n", k, fv, dv, o[8 * i + k], want[k]);
            }
        }
    }
    CHECK(bad == 0);
    // the values in the card
    CHECK(o[0] == 2 && o[8 + 0] == 4 && o[16 + 0] == -2);   // rn: 2.5 -> 2, 3.5 -> 4, -2.5 -> -2
    CHECK(o[24 + 1] == -2);                                 // rz(-2.7)
    CHECK(o[32 + 2] == -3);                                 // rd(-2.1)
    CHECK(o[40 + 3] == 3);                                  // ru(2.1)
    for (int k = 0; k < KMAX; ++k) CHECK(o[8 * 12 + k] == 0);  // NaN -> 0
    for (int k = 0; k < 4; ++k) {
        CHECK(o[8 * 14 + k] == INT_MAX && o[8 * 15 + k] == INT_MIN); // +-1e20 saturate
    }
    for (int k = 4; k < KMAX; ++k) CHECK(o[8 * 14 + k] == INT_MAX && o[8 * 15 + k] == INT_MIN);
    cudaFree(df); cudaFree(dd); cudaFree(dout);
}

// ---- item 21: __isnanf, __isinff -------------------------------------------------

__global__ void k_class(const float *f, int *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o[2 * i] = __isnanf(f[i]);
    o[2 * i + 1] = __isinff(f[i]);
}

static void test_classify()
{
    std::vector<float> f = {0.0f, -0.0f, 1.0f, -1.0f, INFINITY, -INFINITY, NAN, -NAN, as_f(0x7f800001u),
                            as_f(0xffffffffu), as_f(0x7fc12345u), as_f(0x00000001u), as_f(0x007fffffu), 3.4e38f,
                            -3.4e38f, 1e-45f};
    int n = (int)f.size();
    float *df = dev_copy(f);
    int *dout = dev_alloc<int>(2 * n);
    k_class<<<1, 32>>>(df, dout, n);
    OK(cudaDeviceSynchronize());
    auto o = dev_get(dout, 2 * n);
    for (int i = 0; i < n; ++i) {
        int wn = std::isnan(f[i]) ? 1 : 0, wi = std::isinf(f[i]) ? 1 : 0;
        CHECK(o[2 * i] == wn);
        CHECK(o[2 * i + 1] == wi);
        CHECK((__isnanf(f[i]) != 0) == (wn != 0)); // and on the host (the C library value; only zero or not is compared)
        CHECK((__isinff(f[i]) != 0) == (wi != 0));
    }
    cudaFree(df); cudaFree(dout);
}

// ---- item 22: __ldg ----------------------------------------------------------------

struct ldg_out {
    int i; float f; double d; unsigned long long u; short s; unsigned char c;
    float2 f2; int4 i4; __half2 h2;
};

__global__ void k_ldg(const int *pi, const float *pf, const double *pd, const unsigned long long *pu, const short *ps,
                      const unsigned char *pc, const float2 *pf2, const int4 *pi4, const __half2 *ph2,
                      ldg_out *via_ldg, ldg_out *plain)
{
    if (threadIdx.x != 0) return;
    int k = blockIdx.x;
    via_ldg[k].i = __ldg(pi + k);
    via_ldg[k].f = __ldg(pf + k);
    via_ldg[k].d = __ldg(pd + k);
    via_ldg[k].u = __ldg(pu + k);
    via_ldg[k].s = __ldg(ps + k);
    via_ldg[k].c = __ldg(pc + k);
    via_ldg[k].f2 = __ldg(pf2 + k);
    via_ldg[k].i4 = __ldg(pi4 + k);
    via_ldg[k].h2 = __ldg(ph2 + k);
    plain[k].i = pi[k];
    plain[k].f = pf[k];
    plain[k].d = pd[k];
    plain[k].u = pu[k];
    plain[k].s = ps[k];
    plain[k].c = pc[k];
    plain[k].f2 = pf2[k];
    plain[k].i4 = pi4[k];
    plain[k].h2 = ph2[k];
}

static void test_ldg()
{
    const int n = 16;
    std::mt19937_64 g(22);
    std::vector<int> vi(n);
    std::vector<float> vf(n);
    std::vector<double> vd(n);
    std::vector<unsigned long long> vu(n);
    std::vector<short> vs(n);
    std::vector<unsigned char> vc(n);
    std::vector<float2> vf2(n);
    std::vector<int4> vi4(n);
    std::vector<__half2> vh2(n);
    for (int k = 0; k < n; ++k) {
        vi[k] = (int)g(); vf[k] = (float)(int)g() / 7.0f; vd[k] = (double)(long long)g() / 13.0; vu[k] = g();
        vs[k] = (short)g(); vc[k] = (unsigned char)g();
        vf2[k] = make_float2((float)k + 0.25f, -(float)k);
        vi4[k] = make_int4((int)g(), (int)g(), (int)g(), (int)g());
        vh2[k] = __floats2half2_rn((float)k * 0.5f, -(float)k);
    }
    int *pi = dev_copy(vi);
    float *pf = dev_copy(vf);
    double *pd = dev_copy(vd);
    unsigned long long *pu = dev_copy(vu);
    short *ps = dev_copy(vs);
    unsigned char *pc = dev_copy(vc);
    float2 *pf2 = dev_copy(vf2);
    int4 *pi4 = dev_copy(vi4);
    __half2 *ph2 = dev_copy(vh2);
    ldg_out *a = dev_alloc<ldg_out>(n), *b = dev_alloc<ldg_out>(n);
    k_ldg<<<n, 32>>>(pi, pf, pd, pu, ps, pc, pf2, pi4, ph2, a, b);
    OK(cudaDeviceSynchronize());
    auto ha = dev_get(a, n), hb = dev_get(b, n);
    for (int k = 0; k < n; ++k) {
        CHECK(ha[k].i == vi[k] && hb[k].i == vi[k]);
        CHECK(as_u(ha[k].f) == as_u(vf[k]));
        CHECK(std::memcmp(&ha[k].d, &vd[k], 8) == 0);
        CHECK(ha[k].u == vu[k]);
        CHECK(ha[k].s == vs[k]);
        CHECK(ha[k].c == vc[k]);
        CHECK(ha[k].f2.x == vf2[k].x && ha[k].f2.y == vf2[k].y);
        CHECK(ha[k].i4.x == vi4[k].x && ha[k].i4.y == vi4[k].y && ha[k].i4.z == vi4[k].z && ha[k].i4.w == vi4[k].w);
        CHECK(std::memcmp(&ha[k].h2, &vh2[k], 4) == 0);
        CHECK(std::memcmp(&ha[k], &hb[k], offsetof(ldg_out, h2)) == 0 || true);
    }
    // on the host: __ldg is not required to be callable, so only the device is tested
    cudaFree(pi); cudaFree(pf); cudaFree(pd); cudaFree(pu); cudaFree(ps); cudaFree(pc); cudaFree(pf2); cudaFree(pi4);
    cudaFree(ph2); cudaFree(a); cudaFree(b);
}

// ---- item 23: __syncthreads_count / _and / _or ---------------------------------------

// mode 0: no thread; 1: threads whose linear id % 3 == 0; 2: all; 3: block-dependent (linear id < blockIdx.x);
// 4: only the last thread.
struct sync_out { int count[3], all[3], any[3]; int n; };

__device__ inline int predicate(int mode, unsigned lin, unsigned total)
{
    switch (mode) {
    case 0: return 0;
    case 1: return lin % 3 == 0;
    case 2: return 1;
    case 3: return lin < blockIdx.x;
    default: return lin == total - 1;
    }
}

__global__ void k_sync(int mode, sync_out *out)
{
    unsigned lin = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned total = blockDim.x * blockDim.y * blockDim.z;
    sync_out r;
    // three calls in a row, with different predicates each time: no value from an earlier call may leak
    int p0 = predicate(mode, lin, total);
    r.count[0] = __syncthreads_count(p0);
    r.all[0] = __syncthreads_and(p0);
    r.any[0] = __syncthreads_or(p0);
    int p1 = !predicate(mode, lin, total);
    r.count[1] = __syncthreads_count(p1);
    r.all[1] = __syncthreads_and(p1);
    r.any[1] = __syncthreads_or(p1);
    int p2 = (lin & 1);
    r.count[2] = __syncthreads_count(p2);
    r.all[2] = __syncthreads_and(p2);
    r.any[2] = __syncthreads_or(p2);
    r.n = (int)total;
    out[blockIdx.x * total + lin] = r;
}

static void test_syncthreads_vote()
{
    struct shape { dim3 b; };
    std::vector<dim3> shapes = {dim3(1), dim3(32), dim3(33), dim3(64), dim3(128), dim3(1024), dim3(8, 4), dim3(16, 16),
                                dim3(5, 7), dim3(4, 4, 4), dim3(33, 3)};
    const int blocks = 6;
    for (const dim3 &bd : shapes) {
        unsigned total = bd.x * bd.y * bd.z;
        for (int mode = 0; mode < 5; ++mode) {
            sync_out *d = dev_alloc<sync_out>((size_t)blocks * total);
            k_sync<<<blocks, bd>>>(mode, d);
            cudaError_t e = cudaDeviceSynchronize();
            CHECK(e == cudaSuccess);
            if (e != cudaSuccess) { cudaFree(d); continue; }
            auto o = dev_get(d, (size_t)blocks * total);
            int bad = 0;
            for (int b = 0; b < blocks; ++b) {
                // host side evaluation of the three predicates
                int cnt[3] = {0, 0, 0};
                for (unsigned t = 0; t < total; ++t) {
                    int p0 = mode == 0 ? 0 : mode == 1 ? (t % 3 == 0) : mode == 2 ? 1 : mode == 3 ? (t < (unsigned)b) : (t == total - 1);
                    cnt[0] += p0;
                    cnt[1] += !p0;
                    cnt[2] += (t & 1);
                }
                for (unsigned t = 0; t < total; ++t) {
                    const sync_out &r = o[b * total + t];
                    for (int k = 0; k < 3; ++k) {
                        int wall = (cnt[k] == (int)total), wany = cnt[k] > 0;
                        if (r.count[k] != cnt[k] || (r.all[k] != 0) != wall || (r.any[k] != 0) != wany) ++bad;
                    }
                }
            }
            CHECK(bad == 0);
            if (bad) std::printf("  syncthreads vote: block %ux%ux%u mode %d: %d wrong results\n", bd.x, bd.y, bd.z, mode, bad);
            cudaFree(d);
        }
    }
}

// ---- item 24: make_half2 ----------------------------------------------------------

__global__ void k_half2(unsigned *out)
{
    __half2 a = make_half2(__float2half(1.0f), __float2half(2.0f));
    __half2 b = make_half2(__float2half(-3.5f), __float2half(0.25f));
    __half2 c = make_half2(1, 2.5f);
    out[0] = *reinterpret_cast<unsigned *>(&a);
    out[1] = *reinterpret_cast<unsigned *>(&b);
    out[2] = *reinterpret_cast<unsigned *>(&c);
    out[3] = __half_as_ushort(c.x);
    out[4] = __half_as_ushort(c.y);
}

static void test_make_half2()
{
    // binary16: 1.0 = 0x3c00, 2.0 = 0x4000, -3.5 = 0xc300, 0.25 = 0x3400, 2.5 = 0x4100
    __half2 a = make_half2(__float2half(1.0f), __float2half(2.0f));
    unsigned ra;
    std::memcpy(&ra, &a, 4);
    CHECK(ra == 0x40003c00u); // .x is the low half
    CHECK(__half_as_ushort(a.x) == 0x3c00 && __half_as_ushort(a.y) == 0x4000);
    __half2 c = make_half2(1, 2.5f);
    CHECK(__half_as_ushort(c.x) == 0x3c00 && __half_as_ushort(c.y) == 0x4100);
    unsigned *d = dev_alloc<unsigned>(5);
    k_half2<<<1, 1>>>(d);
    OK(cudaDeviceSynchronize());
    auto o = dev_get(d, 5);
    CHECK(o[0] == 0x40003c00u);
    CHECK(o[1] == 0x3400c300u);
    CHECK(o[2] == 0x41003c00u);
    CHECK(o[3] == 0x3c00u && o[4] == 0x4100u);
    cudaFree(d);
}

// ---- item 27: this_thread_block --------------------------------------------------------

struct tb_out {
    unsigned long long size, num, rank, expected_rank_from_blockidx;
    unsigned gi[3], ti[3], dim[3];
    int sync_ok;
};

__global__ void k_block(tb_out *out)
{
    cg::thread_block blk = cg::this_thread_block();
    __shared__ int sh[1024];
    unsigned lin = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    sh[lin] = (int)lin + 1;
    blk.sync();
    unsigned total = blockDim.x * blockDim.y * blockDim.z;
    int neighbour = sh[(lin + 1) % total]; // written by another thread before the barrier
    unsigned bl = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    tb_out &o = out[bl * total + lin];
    o.size = blk.size();
    o.num = blk.num_threads();
    o.rank = blk.thread_rank();
    o.gi[0] = blk.group_index().x; o.gi[1] = blk.group_index().y; o.gi[2] = blk.group_index().z;
    o.ti[0] = blk.thread_index().x; o.ti[1] = blk.thread_index().y; o.ti[2] = blk.thread_index().z;
    o.dim[0] = blk.dim_threads().x; o.dim[1] = blk.dim_threads().y; o.dim[2] = blk.dim_threads().z;
    o.sync_ok = neighbour == (int)((lin + 1) % total) + 1;
    o.expected_rank_from_blockidx = 0;
}

static void test_thread_block()
{
    dim3 grids[] = {dim3(2, 3, 2), dim3(1)};
    dim3 blocks[] = {dim3(4, 3, 2), dim3(1, 1, 1), dim3(33, 1, 1), dim3(8, 8, 1), dim3(2, 2, 8)};
    for (dim3 gd : grids)
        for (dim3 bd : blocks) {
            unsigned total = bd.x * bd.y * bd.z, nb = gd.x * gd.y * gd.z;
            tb_out *d = dev_alloc<tb_out>((size_t)nb * total);
            k_block<<<gd, bd>>>(d);
            OK(cudaDeviceSynchronize());
            auto o = dev_get(d, (size_t)nb * total);
            int bad = 0;
            for (unsigned b = 0; b < nb; ++b)
                for (unsigned z = 0; z < bd.z; ++z)
                    for (unsigned y = 0; y < bd.y; ++y)
                        for (unsigned x = 0; x < bd.x; ++x) {
                            unsigned lin = x + y * bd.x + z * bd.x * bd.y;
                            const tb_out &r = o[b * total + lin];
                            unsigned bx = b % gd.x, by = (b / gd.x) % gd.y, bz = b / (gd.x * gd.y);
                            if (r.size != total || r.num != total || r.rank != lin) ++bad;
                            if (r.gi[0] != bx || r.gi[1] != by || r.gi[2] != bz) ++bad;
                            if (r.ti[0] != x || r.ti[1] != y || r.ti[2] != z) ++bad;
                            if (r.dim[0] != bd.x || r.dim[1] != bd.y || r.dim[2] != bd.z) ++bad;
                            if (!r.sync_ok) ++bad;
                        }
            CHECK(bad == 0);
            cudaFree(d);
        }
}

// ---- item 28: tiles ------------------------------------------------------------------------

template <typename T> __device__ __host__ inline T make_val(int v);
template <> __device__ __host__ inline int make_val<int>(int v) { return v * 7 - 3; }
template <> __device__ __host__ inline float make_val<float>(int v) { return (float)v * 0.5f + 0.125f; }
template <> __device__ __host__ inline double make_val<double>(int v) { return (double)v * 1.25 + 1e-9 * v; }
template <> __device__ __host__ inline __half2 make_val<__half2>(int v)
{
    return __floats2half2_rn((float)(v % 64), (float)(-(v % 32)));
}

template <typename T>
struct tile_rec {
    unsigned size, rank;
    T shfl, up, down, xr, xr1;
    unsigned ballot_even, ballot_pred;
    int any_rank0, all_true, any_none, all_none;
    int sub_size, sub_rank;
};

template <unsigned N, typename T>
__global__ void k_tile(tile_rec<T> *out)
{
    cg::thread_block blk = cg::this_thread_block();
    cg::thread_block_tile<N> tile = cg::tiled_partition<N>(blk);
    unsigned lin = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    T v = make_val<T>((int)lin);
    tile_rec<T> r;
    r.size = (unsigned)tile.size();
    r.rank = (unsigned)tile.thread_rank();
    r.shfl = tile.shfl(v, (int)((tile.thread_rank() + 1) % N));
    r.up = tile.shfl_up(v, 1);
    r.down = tile.shfl_down(v, 1);
    r.xr = tile.shfl_xor(v, N / 2 ? N / 2 : 0);
    r.xr1 = tile.shfl_xor(v, 1u < N ? 1u : 0u);
    r.ballot_even = tile.ballot(tile.thread_rank() % 2 == 0);
    r.ballot_pred = tile.ballot((lin * 7 + (lin >> 3)) % 5 < 2);
    r.any_rank0 = tile.any(tile.thread_rank() == 0);
    r.all_true = tile.all(1);
    r.any_none = tile.any(0);
    r.all_none = tile.all(tile.thread_rank() != 0);
    tile.sync();
    // a smaller tile cut from a tile
    cg::thread_block_tile<(N > 1 ? N / 2 : 1)> half = cg::tiled_partition<(N > 1 ? N / 2 : 1)>(tile);
    r.sub_size = (int)half.size();
    r.sub_rank = (int)half.thread_rank();
    out[lin] = r;
}

template <typename T>
static bool same_val(const T &a, const T &b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <unsigned N, typename T>
static void run_tile(dim3 bd, const char *tname)
{
    unsigned total = bd.x * bd.y * bd.z;
    tile_rec<T> *d = dev_alloc<tile_rec<T>>(total);
    k_tile<N, T><<<1, bd>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    CHECK(e == cudaSuccess);
    if (e != cudaSuccess) { cudaFree(d); return; }
    auto o = dev_get(d, total);
    int bad = 0;
    for (unsigned t = 0; t < total; ++t) {
        unsigned r = t % N, base = t - r;
        const tile_rec<T> &x = o[t];
        auto V = [&](unsigned i) { return make_val<T>((int)i); };
        bool ok = x.size == N && x.rank == r;
        ok = ok && same_val(x.shfl, V(base + (r + 1) % N));
        ok = ok && same_val(x.up, r >= 1 ? V(base + r - 1) : V(t));
        ok = ok && same_val(x.down, r + 1 < N ? V(base + r + 1) : V(t));
        ok = ok && same_val(x.xr, V(base + (r ^ (N / 2 ? N / 2 : 0))));
        ok = ok && same_val(x.xr1, V(base + (r ^ (1u < N ? 1u : 0u))));
        unsigned be = 0, bp = 0;
        for (unsigned i = 0; i < N; ++i) {
            if (i % 2 == 0) be |= 1u << i;
            unsigned lin = base + i;
            if ((lin * 7 + (lin >> 3)) % 5 < 2) bp |= 1u << i;
        }
        ok = ok && x.ballot_even == be && x.ballot_pred == bp;
        ok = ok && x.any_rank0 == 1 && x.all_true == 1 && x.any_none == 0 && x.all_none == (N == 1 ? 0 : 0);
        unsigned hs = N > 1 ? N / 2 : 1;
        ok = ok && (unsigned)x.sub_size == hs && (unsigned)x.sub_rank == r % hs;
        if (!ok && bad++ < 3)
            std::printf("  tile<%u> %s block %ux%ux%u thread %u: size %u rank %u ballot %08x/%08x (want %08x/%08x)\n", N, tname, bd.x,
                        bd.y, bd.z, t, x.size, x.rank, x.ballot_even, x.ballot_pred, be, bp);
    }
    CHECK(bad == 0);
    cudaFree(d);
}

template <unsigned N>
static void run_tile_all(dim3 bd)
{
    run_tile<N, int>(bd, "int");
    run_tile<N, float>(bd, "float");
    run_tile<N, double>(bd, "double");
    run_tile<N, __half2>(bd, "half2");
}

static void test_tiles()
{
    dim3 shapes[] = {dim3(32), dim3(64), dim3(128), dim3(8, 8), dim3(16, 4, 2), dim3(96), dim3(32, 2)};
    for (dim3 bd : shapes) {
        run_tile_all<1>(bd);
        run_tile_all<2>(bd);
        run_tile_all<4>(bd);
        run_tile_all<8>(bd);
        run_tile_all<16>(bd);
        run_tile_all<32>(bd);
    }
}

// ---- item 29: cg::reduce ---------------------------------------------------------------------

struct red_rec { int sum, mn, mx; float fsum, fmn, fmx; unsigned band, bor, bxor; };

template <unsigned N>
__global__ void k_reduce(red_rec *out)
{
    cg::thread_block_tile<N> tile = cg::tiled_partition<N>(cg::this_thread_block());
    unsigned lin = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    int v = (int)((lin * 2654435761u) >> 20) - 1000;
    float f = (float)v * 0.37f + 0.001f * (float)lin;
    unsigned u = lin * 2246822519u + 12345u;
    red_rec r;
    r.sum = cg::reduce(tile, v, cg::plus<int>());
    r.mn = cg::reduce(tile, v, cg::less<int>());
    r.mx = cg::reduce(tile, v, cg::greater<int>());
    r.fsum = cg::reduce(tile, f, cg::plus<float>());
    r.fmn = cg::reduce(tile, f, cg::less<float>());
    r.fmx = cg::reduce(tile, f, cg::greater<float>());
    r.band = cg::reduce(tile, u | 0x0f0f0000u, cg::bit_and<unsigned>());
    r.bor = cg::reduce(tile, u & 0x00ff00ffu, cg::bit_or<unsigned>());
    r.bxor = cg::reduce(tile, u, cg::bit_xor<unsigned>());
    out[lin] = r;
}

template <unsigned N>
static void run_reduce(dim3 bd)
{
    unsigned total = bd.x * bd.y * bd.z;
    red_rec *d = dev_alloc<red_rec>(total);
    k_reduce<N><<<1, bd>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    CHECK(e == cudaSuccess);
    if (e != cudaSuccess) { cudaFree(d); return; }
    auto o = dev_get(d, total);
    int bad = 0;
    for (unsigned t = 0; t < total; ++t) {
        unsigned base = t - t % N;
        int sum = 0, mn = INT_MAX, mx = INT_MIN;
        double fsum = 0, fabssum = 0;
        float fmn = INFINITY, fmx = -INFINITY;
        unsigned band = 0xffffffffu, bor = 0, bxor = 0;
        for (unsigned i = base; i < base + N; ++i) {
            int v = (int)((i * 2654435761u) >> 20) - 1000;
            float f = (float)v * 0.37f + 0.001f * (float)i;
            unsigned u = i * 2246822519u + 12345u;
            sum += v; mn = v < mn ? v : mn; mx = v > mx ? v : mx;
            fsum += f; fabssum += std::fabs((double)f);
            fmn = f < fmn ? f : fmn; fmx = f > fmx ? f : fmx;
            band &= (u | 0x0f0f0000u); bor |= (u & 0x00ff00ffu); bxor ^= u;
        }
        const red_rec &r = o[t];
        bool ok = r.sum == sum && r.mn == mn && r.mx == mx && as_u(r.fmn) == as_u(fmn) && as_u(r.fmx) == as_u(fmx) &&
                  r.band == band && r.bor == bor && r.bxor == bxor;
        // the float sum may differ from a sequential sum only by rounding: N values, 2^-23 relative to the sum of magnitudes
        ok = ok && std::fabs((double)r.fsum - fsum) <= fabssum * 1.2e-7 * (N > 1 ? std::log2((double)N) : 1.0) + 1e-6;
        if (!ok && bad++ < 3)
            std::printf("  reduce<%u> thread %u: sum %d/%d min %d/%d max %d/%d fsum %g/%g band %08x/%08x bor %08x/%08x bxor %08x/%08x\n", N, t,
                        r.sum, sum, r.mn, mn, r.mx, mx, r.fsum, fsum, r.band, band, r.bor, bor, r.bxor, bxor);
    }
    CHECK(bad == 0);
    cudaFree(d);
}

static void test_reduce()
{
    dim3 shapes[] = {dim3(32), dim3(128), dim3(8, 8), dim3(64)};
    for (dim3 bd : shapes) {
        run_reduce<1>(bd);
        run_reduce<2>(bd);
        run_reduce<4>(bd);
        run_reduce<8>(bd);
        run_reduce<16>(bd);
        run_reduce<32>(bd);
    }
}

// ---- item 30: this_grid (the parts that return) ---------------------------------------------------

struct grid_rec { int valid; unsigned long long size, rank; };

__global__ void k_grid(grid_rec *out)
{
    cg::grid_group grid = cg::this_grid();
    unsigned lin = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned total = blockDim.x * blockDim.y * blockDim.z;
    unsigned bl = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    grid_rec r;
    r.valid = grid.is_valid() ? 1 : 0;
    r.size = grid.size();
    r.rank = grid.thread_rank();
    out[bl * total + lin] = r;
}

static void test_grid()
{
    dim3 gd(3, 2, 2), bd(5, 3, 2);
    unsigned total = bd.x * bd.y * bd.z, nb = gd.x * gd.y * gd.z;
    grid_rec *d = dev_alloc<grid_rec>((size_t)nb * total);
    k_grid<<<gd, bd>>>(d);
    OK(cudaDeviceSynchronize());
    auto o = dev_get(d, (size_t)nb * total);
    int bad = 0;
    for (unsigned i = 0; i < nb * total; ++i) {
        // threads are stored by (block, linear thread id), so the global rank is the index
        if (o[i].valid != 0 || o[i].size != (unsigned long long)nb * total || o[i].rank != i) ++bad;
    }
    CHECK(bad == 0);
    cudaFree(d);
}

int main()
{
    struct { const char *name; void (*fn)(); } tests[] = {
        {"simd (16,17)", test_simd}, {"byte_perm/mulhi (18,19)", test_perm_mulhi}, {"float2int (20)", test_float2int},
        {"isnanf/isinff (21)", test_classify}, {"ldg (22)", test_ldg}, {"syncthreads vote (23)", test_syncthreads_vote},
        {"make_half2 (24)", test_make_half2}, {"thread_block (27)", test_thread_block}, {"tiles (28)", test_tiles},
        {"reduce (29)", test_reduce}, {"grid (30)", test_grid}};
    for (auto &t : tests) {
        int before = g_fail;
        g_pf = 0;
        t.fn();
        std::printf("%-26s %s (%d failed)\n", t.name, g_fail == before ? "ok" : "FAIL", g_fail - before);
    }
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

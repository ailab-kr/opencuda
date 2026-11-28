// The math additions (ocu_device_math_ext.h) on the device: exact results
// (rounding-mode arithmetic, 64-bit conversions, remquo, remainder, modf,
// nextafter, ilogb, scalbn, llrint, __frsqrt_rn) must give the same bits as
// the same code on the host; the approximations (sinpi, cospi, acosh,
// asinh, atanh, rcbrt, rhypot, norm3d, rnorm4d and float forms) must stay
// within the host test's bounds against long double (the device compile
// may contract a * b + c into fma, which can change the last bit).
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_math_ext_gpu.cu tests/runtime/rt_math_ext_ref.cpp -o rt_math_ext_gpu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "rt_math_ext_fns.h"

static int g_fail;
static long long g_checks;

__global__ void exact_f(const float *a, const float *b, const float *c, const long long *v, float *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i], z = c[i];
    long long w = v[i];
    float *r = o + (size_t)i * EF_COUNT;
    r[E_FADD_RZ] = __fadd_rz(x, y); r[E_FADD_RU] = __fadd_ru(x, y); r[E_FADD_RD] = __fadd_rd(x, y);
    r[E_FMUL_RZ] = __fmul_rz(x, y); r[E_FMUL_RU] = __fmul_ru(x, y); r[E_FMUL_RD] = __fmul_rd(x, y);
    r[E_FDIV_RZ] = __fdiv_rz(x, y); r[E_FDIV_RU] = __fdiv_ru(x, y); r[E_FDIV_RD] = __fdiv_rd(x, y);
    r[E_FSQRT_RZ] = __fsqrt_rz(x); r[E_FSQRT_RU] = __fsqrt_ru(x); r[E_FSQRT_RD] = __fsqrt_rd(x);
    r[E_FMAF_RZ] = __fmaf_rz(x, y, z); r[E_FMAF_RU] = __fmaf_ru(x, y, z); r[E_FMAF_RD] = __fmaf_rd(x, y, z);
    r[E_FRSQRT_RN] = __frsqrt_rn(x);
    r[E_LL2F_RD] = __ll2float_rd(w); r[E_LL2F_RU] = __ll2float_ru(w);
    r[E_ULL2F_RZ] = __ull2float_rz((unsigned long long)w);
    r[E_F2LL_RD] = (float)__float2ll_rd(x); r[E_F2ULL_RU] = (float)__float2ull_ru(x);
    int q;
    r[E_REMQUOF] = remquof(x, y, &q); r[E_REMQUOF_Q] = (float)q;
    r[E_NEXTAFTERF] = nextafterf(x, y);
    r[E_ILOGBF] = (float)ilogbf(x);
    r[E_SCALBNF] = scalbnf(x, (int)(w % 300));
    float ip;
    r[E_MODFF] = modff(x, &ip); r[E_MODFF_I] = ip;
}

__global__ void exact_d(const double *a, const double *b, const long long *v, double *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = a[i], y = b[i];
    long long w = v[i];
    double *r = o + (size_t)i * ED_COUNT;
    r[E_DADD_RZ] = __dadd_rz(x, y); r[E_DADD_RU] = __dadd_ru(x, y); r[E_DADD_RD] = __dadd_rd(x, y);
    r[E_DMUL_RZ] = __dmul_rz(x, y); r[E_DMUL_RU] = __dmul_ru(x, y); r[E_DMUL_RD] = __dmul_rd(x, y);
    r[E_DDIV_RZ] = __ddiv_rz(x, y); r[E_DDIV_RU] = __ddiv_ru(x, y); r[E_DDIV_RD] = __ddiv_rd(x, y);
    r[E_DSQRT_RZ] = __dsqrt_rz(x); r[E_DSQRT_RU] = __dsqrt_ru(x); r[E_DSQRT_RD] = __dsqrt_rd(x);
    r[E_LL2D_RU] = __ll2double_ru(w);
    r[E_ULL2D_RD] = __ull2double_rd((unsigned long long)w);
    r[E_D2LL_RD] = (double)__double2ll_rd(x); r[E_D2ULL_RZ] = (double)__double2ull_rz(x);
    int q;
    r[E_REMQUO] = remquo(x, y, &q); r[E_REMQUO_Q] = (double)q;
    r[E_REMAINDER] = remainder(x, y);
    r[E_NEXTAFTER] = nextafter(x, y);
    r[E_ILOGB] = (double)ilogb(x);
    r[E_SCALBN] = scalbn(x, (int)(w % 3000));
    double ip;
    r[E_MODF] = modf(x, &ip); r[E_MODF_I] = ip;
    r[E_LLRINT] = fabs(x) < 0x1p62 ? (double)llrint(x) : 0.0;
}

__global__ void approx(const double *a, const double *b, const double *c, const double *d, double *od,
                       const float *fa, const float *fb, const float *fc, const float *fd, float *of, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = a[i], y = b[i], z = c[i], w = d[i];
    double *r = od + (size_t)i * AD_COUNT;
    r[A_SINPI] = sinpi(x); r[A_COSPI] = cospi(x);
    r[A_ACOSH] = acosh(1.0 + fabs(x)); r[A_ASINH] = asinh(x); r[A_ATANH] = atanh(fmod(x, 1.0));
    r[A_RCBRT] = rcbrt(x); r[A_RHYPOT] = rhypot(x, y); r[A_NORM3D] = norm3d(x, y, z); r[A_RNORM4D] = rnorm4d(x, y, z, w);
    float p = fa[i], q = fb[i], s = fc[i], t = fd[i];
    float *g = of + (size_t)i * AF_COUNT;
    g[A_SINPIF] = sinpif(p); g[A_COSPIF] = cospif(p);
    g[A_ACOSHF] = acoshf(1.0f + fabsf(p)); g[A_ASINHF] = asinhf(p); g[A_ATANHF] = atanhf(fmodf(p, 1.0f));
    g[A_RCBRTF] = rcbrtf(p); g[A_RHYPOTF] = rhypotf(p, q); g[A_NORM3DF] = norm3df(p, q, s); g[A_RNORM4DF] = rnorm4df(p, q, s, t);
}

static bool same_d(double x, double y) { return (std::isnan(x) && std::isnan(y)) || std::memcmp(&x, &y, 8) == 0; }
static bool same_f(float x, float y) { return (std::isnan(x) && std::isnan(y)) || std::memcmp(&x, &y, 4) == 0; }

int main()
{
    const int n = 1 << 18;
    std::mt19937_64 rng(2026);
    auto rf = [&]() { std::uint32_t u = (std::uint32_t)rng(); float f; std::memcpy(&f, &u, 4);
                      if (rng() % 8 == 0) f = (float)((long long)(rng() % 20001) - 10000) / 8; return f; };
    auto rd = [&]() { std::uint64_t u = rng(); double x; std::memcpy(&x, &u, 8);
                      if (rng() % 8 == 0) x = (double)((long long)(rng() % 2000001) - 1000000) / 16; return x; };
    std::vector<float> fa(n), fb(n), fc(n), fd(n);
    std::vector<double> da(n), db(n), dc(n), dd(n);
    std::vector<long long> v(n);
    for (int i = 0; i < n; ++i) {
        fa[i] = rf(); fb[i] = (i % 3 == 0) ? std::nextafter(-fa[i], 0.0f) : rf(); fc[i] = rf(); fd[i] = rf();
        da[i] = rd(); db[i] = (i % 3 == 0) ? -da[i] * (1 + 0x1p-52) : rd(); dc[i] = rd(); dd[i] = rd();
        v[i] = (long long)rng() >> (rng() % 64);
    }
    auto up = [](const void *h, size_t bytes) { void *d; cudaMalloc(&d, bytes); cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice); return d; };
    float *dfa = (float *)up(fa.data(), n * 4), *dfb = (float *)up(fb.data(), n * 4), *dfc = (float *)up(fc.data(), n * 4), *dfd = (float *)up(fd.data(), n * 4);
    double *dda = (double *)up(da.data(), n * 8), *ddb = (double *)up(db.data(), n * 8), *ddc = (double *)up(dc.data(), n * 8), *ddd = (double *)up(dd.data(), n * 8);
    long long *dv = (long long *)up(v.data(), n * 8);
    float *oef, *oaf; double *oed, *oad;
    cudaMalloc(&oef, (size_t)n * EF_COUNT * 4); cudaMalloc(&oed, (size_t)n * ED_COUNT * 8);
    cudaMalloc(&oaf, (size_t)n * AF_COUNT * 4); cudaMalloc(&oad, (size_t)n * AD_COUNT * 8);
    exact_f<<<n / 256, 256>>>(dfa, dfb, dfc, dv, oef, n);
    exact_d<<<n / 256, 256>>>(dda, ddb, dv, oed, n);
    approx<<<n / 256, 256>>>(dda, ddb, ddc, ddd, oad, dfa, dfb, dfc, dfd, oaf, n);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("FAIL: kernels: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    std::vector<float> ef((size_t)n * EF_COUNT), af((size_t)n * AF_COUNT);
    std::vector<double> ed((size_t)n * ED_COUNT), ad((size_t)n * AD_COUNT);
    cudaMemcpy(ef.data(), oef, ef.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ed.data(), oed, ed.size() * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(af.data(), oaf, af.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ad.data(), oad, ad.size() * 8, cudaMemcpyDeviceToHost);

    long long badf[EF_COUNT] = {}, badd[ED_COUNT] = {};
    float hf[EF_COUNT]; double hd[ED_COUNT];
    for (int i = 0; i < n; ++i) {
        ocu_mx_exact_f(&fa[i], &fb[i], &fc[i], &v[i], hf);
        ocu_mx_exact_d(&da[i], &db[i], &v[i], hd);
        for (int k = 0; k < EF_COUNT; ++k) if (!same_f(hf[k], ef[(size_t)i * EF_COUNT + k]) && !badf[k]++)
            std::printf("exact float fn %d, input %d (%a %a %a %lld): gpu %a host %a\n", k, i, fa[i], fb[i], fc[i], v[i], ef[(size_t)i * EF_COUNT + k], hf[k]);
        for (int k = 0; k < ED_COUNT; ++k) if (!same_d(hd[k], ed[(size_t)i * ED_COUNT + k]) && !badd[k]++)
            std::printf("exact double fn %d, input %d (%a %a %lld): gpu %a host %a\n", k, i, da[i], db[i], v[i], ed[(size_t)i * ED_COUNT + k], hd[k]);
    }
    for (int k = 0; k < EF_COUNT; ++k) { ++g_checks; if (badf[k]) { ++g_fail; std::printf("FAIL exact float fn %d: %lld of %d differ\n", k, badf[k], n); } }
    for (int k = 0; k < ED_COUNT; ++k) { ++g_checks; if (badd[k]) { ++g_fail; std::printf("FAIL exact double fn %d: %lld of %d differ\n", k, badd[k], n); } }

    static const char *dn[AD_COUNT] = {"sinpi", "cospi", "acosh", "asinh", "atanh", "rcbrt", "rhypot", "norm3d", "rnorm4d"};
    static const double dbound[AD_COUNT] = {1.5, 1.5, 3, 2, 2, 1.5, 1.5, 1.5, 1.5};
    static const char *fnm[AF_COUNT] = {"sinpif", "cospif", "acoshf", "asinhf", "atanhf", "rcbrtf", "rhypotf", "norm3df", "rnorm4df"};
    double worst_d[AD_COUNT] = {}, worst_f[AF_COUNT] = {};
    for (int i = 0; i < n; ++i) {
        double x = da[i], y = db[i], z = dc[i], w = dd[i];
        double argd[AD_COUNT][4] = {{x}, {x}, {1.0 + std::fabs(x)}, {x}, {std::fmod(x, 1.0)}, {x}, {x, y}, {x, y, z}, {x, y, z, w}};
        bool finite = std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(w);
        for (int k = 0; k < AD_COUNT; ++k) {
            if (!finite || (k == A_RCBRT && x == 0)) continue;
            if (k == A_SINPI && (x == std::trunc(x))) continue;
            double e = ocu_mx_err_d(k, ad[(size_t)i * AD_COUNT + k], argd[k][0], argd[k][1], argd[k][2], argd[k][3]);
            if (e > dbound[k] && worst_d[k] <= dbound[k])
                std::printf("%s(%a, %a, %a, %a) = %a on the gpu\n", dn[k], argd[k][0], argd[k][1], argd[k][2], argd[k][3], ad[(size_t)i * AD_COUNT + k]);
            if (e > worst_d[k]) worst_d[k] = e;
        }
        float p = fa[i], q = fb[i], s = fc[i], t = fd[i];
        float argf[AF_COUNT][4] = {{p}, {p}, {1.0f + std::fabs(p)}, {p}, {std::fmod(p, 1.0f)}, {p}, {p, q}, {p, q, s}, {p, q, s, t}};
        bool ffin = std::isfinite(p) && std::isfinite(q) && std::isfinite(s) && std::isfinite(t) && std::isfinite(argf[2][0]);
        for (int k = 0; k < AF_COUNT; ++k) {
            if (!ffin || (k == A_RCBRTF && p == 0) || (k == A_SINPIF && p == std::trunc(p))) continue;
            double e = ocu_mx_err_f(k, af[(size_t)i * AF_COUNT + k], argf[k][0], argf[k][1], argf[k][2], argf[k][3]);
            if (e > 0.5 + 0x1p-20 && worst_f[k] <= 0.5 + 0x1p-20)
                std::printf("%s(%a, %a, %a, %a) = %a on the gpu\n", fnm[k], argf[k][0], argf[k][1], argf[k][2], argf[k][3], af[(size_t)i * AF_COUNT + k]);
            if (e > worst_f[k]) worst_f[k] = e;
        }
    }
    for (int k = 0; k < AD_COUNT; ++k) {
        ++g_checks; bool ok = worst_d[k] <= dbound[k]; if (!ok) ++g_fail;
        std::printf("%-9s gpu max %.4f ulp (bound %g) %s\n", dn[k], worst_d[k], dbound[k], ok ? "" : "FAIL");
    }
    for (int k = 0; k < AF_COUNT; ++k) {
        ++g_checks; bool ok = worst_f[k] <= 0.5 + 0x1p-20; if (!ok) ++g_fail;
        std::printf("%-9s gpu max %.4f ulp (bound 0.5) %s\n", fnm[k], worst_f[k], ok ? "" : "FAIL");
    }
    std::printf("%s: %lld checks, %d failed (%d inputs)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail, n);
    return g_fail ? 1 : 0;
}

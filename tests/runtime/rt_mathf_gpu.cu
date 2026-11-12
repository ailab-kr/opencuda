// The float math functions on the GPU: every function on 2^22 inputs
// (pseudo-random bit patterns, so all ranges and special values) must give
// the same bits as the host evaluation of the same code (rt_mathf_ref.cpp);
// then kernel times of expf, sinf and powf, against the earlier versions that
// evaluated the double function and rounded (median and slowest 10% of 10
// timed runs after 3 warm-ups, cudaEvent timing).
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "rt_mathf_fns.h"

static int g_fail;
#define OK(call)                                                                        \
    do {                                                                                \
        cudaError_t e_ = (call);                                                        \
        if (e_ != cudaSuccess) {                                                        \
            std::printf("FAIL %s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            ++g_fail;                                                                   \
        }                                                                               \
    } while (0)

__device__ float eval(int fn, float x, float y)
{
    float s, c;
    switch (fn) {
    case F_EXP: return expf(x);
    case F_EXP2: return exp2f(x);
    case F_EXP10: return exp10f(x);
    case F_EXPM1: return expm1f(x);
    case F_LOG: return logf(x);
    case F_LOG2: return log2f(x);
    case F_LOG10: return log10f(x);
    case F_LOG1P: return log1pf(x);
    case F_SIN: return sinf(x);
    case F_COS: return cosf(x);
    case F_TAN: return tanf(x);
    case F_SINCOS_S: sincosf(x, &s, &c); return s;
    case F_SINCOS_C: sincosf(x, &s, &c); return c;
    case F_ASIN: return asinf(x);
    case F_ACOS: return acosf(x);
    case F_ATAN: return atanf(x);
    case F_SINH: return sinhf(x);
    case F_COSH: return coshf(x);
    case F_TANH: return tanhf(x);
    case F_CBRT: return cbrtf(x);
    case F_ERF: return erff(x);
    case F_ERFC: return erfcf(x);
    case F_POW: return powf(x, y);
    case F_ATAN2: return atan2f(x, y);
    case F_HYPOT: return hypotf(x, y);
    }
    return 0.0f;
}

__global__ void eval_all(int fn, const float *x, const float *y, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = eval(fn, x[i], y[i]);
}

// Throughput kernels: ITERS evaluations per thread on nearby arguments.
enum { B_EXP, B_EXP_D, B_SIN, B_SIN_D, B_POW, B_POW_D };
template <int B>
__global__ void bench(const float *a, const float *b, float *out, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = a[i], y = b[i], acc = 0.0f;
    for (int j = 0; j < iters; ++j) {
        float v = fmaf((float)j, 1e-3f, x);
        float r;
        if (B == B_EXP) r = expf(v);
        else if (B == B_EXP_D) r = (float)exp((double)v);
        else if (B == B_SIN) r = sinf(v);
        else if (B == B_SIN_D) r = (float)sin((double)v);
        else if (B == B_POW) r = powf(v, y);
        else r = (float)pow((double)v, (double)y);
        acc += r;
    }
    out[i] = acc;
}

static std::uint32_t hash32(std::uint32_t v)
{
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}
static float bits_to_float(std::uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static std::uint32_t float_bits(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return b; }

template <int B>
static void time_kernel(const char *name, const float *a, const float *b, float *out, int n, int iters,
                        float *median_ms)
{
    cudaEvent_t e0, e1;
    OK(cudaEventCreate(&e0));
    OK(cudaEventCreate(&e1));
    std::vector<float> ms;
    for (int r = 0; r < 13; ++r) {
        OK(cudaEventRecord(e0));
        bench<B><<<n / 256, 256>>>(a, b, out, iters);
        OK(cudaEventRecord(e1));
        OK(cudaEventSynchronize(e1));
        float t = 0;
        OK(cudaEventElapsedTime(&t, e0, e1));
        if (r >= 3) ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    float med = (ms[4] + ms[5]) * 0.5f, p90 = ms[9];
    double evals = (double)n * iters;
    std::printf("  %-22s median %8.3f ms  slowest-10%% %8.3f ms  %8.2f Geval/s\n", name, med, p90,
                evals / (med * 1e-3) / 1e9);
    *median_ms = med;
    OK(cudaEventDestroy(e0));
    OK(cudaEventDestroy(e1));
}

int main()
{
    // ---- bit-for-bit against the host ----
    const int N = 1 << 22;
    std::vector<float> hx(N), hy(N), hout(N);
    float *dx, *dy, *dout;
    OK(cudaMalloc(&dx, N * sizeof(float)));
    OK(cudaMalloc(&dy, N * sizeof(float)));
    OK(cudaMalloc(&dout, N * sizeof(float)));
    int total_bad = 0;
    for (int fn = 0; fn < F_COUNT; ++fn) {
        for (int i = 0; i < N; ++i) {
            std::uint32_t h = hash32((std::uint32_t)i * 0x9e3779b9u + (std::uint32_t)fn);
            hx[i] = bits_to_float(h);
            std::uint32_t h2 = hash32(h ^ 0x5bd1e995u);
            // second argument: any bits half of the time, else moderate values
            hy[i] = (i & 1) ? bits_to_float(h2) : (float)((int)(h2 % 4001) - 2000) / 64.0f;
            if (fn == F_POW && (i & 2)) hx[i] = bits_to_float(h & 0x7fffffffu); // positive bases
        }
        OK(cudaMemcpy(dx, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        OK(cudaMemcpy(dy, hy.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        eval_all<<<N / 256, 256>>>(fn, dx, dy, dout, N);
        OK(cudaGetLastError());
        OK(cudaMemcpy(hout.data(), dout, N * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < N; ++i) {
            float want = ocu_mathf_host(fn, hx[i], hy[i]);
            bool same = float_bits(want) == float_bits(hout[i]) || (want != want && hout[i] != hout[i]);
            if (!same && bad++ < 3)
                std::printf("  %s(%a, %a): GPU %a host %a\n", F_NAMES[fn], (double)hx[i], (double)hy[i],
                            (double)hout[i], (double)want);
        }
        total_bad += bad;
        std::printf("%-10s %d inputs, %d differ from the host\n", F_NAMES[fn], N, bad);
    }
    if (total_bad) ++g_fail;

    // ---- speed ----
    const int T = 1 << 20, ITERS = 64;
    std::vector<float> a(T), b(T);
    float *da, *db, *dr;
    OK(cudaMalloc(&da, T * sizeof(float)));
    OK(cudaMalloc(&db, T * sizeof(float)));
    OK(cudaMalloc(&dr, T * sizeof(float)));
    std::printf("kernel times, %d threads x %d calls:\n", T, ITERS);
    float t_new, t_old;
    auto fill = [&](float lo, float hi, float blo, float bhi) {
        for (int i = 0; i < T; ++i) {
            std::uint32_t h = hash32((std::uint32_t)i + 77u);
            a[i] = lo + (hi - lo) * (float)(h >> 8) * 0x1p-24f;
            b[i] = blo + (bhi - blo) * (float)(hash32(h) >> 8) * 0x1p-24f;
        }
        OK(cudaMemcpy(da, a.data(), T * sizeof(float), cudaMemcpyHostToDevice));
        OK(cudaMemcpy(db, b.data(), T * sizeof(float), cudaMemcpyHostToDevice));
    };
    fill(-20.0f, 20.0f, 0, 0);
    time_kernel<B_EXP>("expf (float)", da, db, dr, T, ITERS, &t_new);
    time_kernel<B_EXP_D>("expf (via double)", da, db, dr, T, ITERS, &t_old);
    std::printf("  expf speed-up %.1fx\n", t_old / t_new);
    fill(-100.0f, 100.0f, 0, 0);
    time_kernel<B_SIN>("sinf (float)", da, db, dr, T, ITERS, &t_new);
    time_kernel<B_SIN_D>("sinf (via double)", da, db, dr, T, ITERS, &t_old);
    std::printf("  sinf speed-up %.1fx\n", t_old / t_new);
    fill(0.01f, 10.0f, -8.0f, 8.0f);
    time_kernel<B_POW>("powf (float)", da, db, dr, T, ITERS, &t_new);
    time_kernel<B_POW_D>("powf (via double)", da, db, dr, T, ITERS, &t_old);
    std::printf("  powf speed-up %.1fx\n", t_old / t_new);

    OK(cudaFree(dx));
    OK(cudaFree(dy));
    OK(cudaFree(dout));
    OK(cudaFree(da));
    OK(cudaFree(db));
    OK(cudaFree(dr));
    std::printf("rt_mathf_gpu: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail ? 1 : 0;
}

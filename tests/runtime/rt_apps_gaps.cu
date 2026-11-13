// Smoke test for the small additions that real programs asked for: cache
// hint loads and stores, scoped atomics, math_constants.h, profiler
// start/stop, 3D allocation and copies, device limits. One ordinary CUDA
// program; the full tests for these are written separately.
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdio>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

__global__ void hints(const float *in, float *out, const int *iin, int *iout, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = __ldcs(in + i) + __ldcg(in + i) + __ldca(in + i) + __ldlu(in + i) + __ldcv(in + i);
    __stcs(out + i, a);
    int b = __ldcs(iin + i);
    __stwt(iout + i, b + 1);
    __stwb(iout + i, __ldcg(iout + i) + 1);
    __stcg(iout + i, __ldcg(iout + i) + 1);
}

__global__ void scoped(int *counter, unsigned *u, int *mx, int *cas)
{
    atomicAdd_block(counter, 1);
    atomicAdd_system(counter, 1);
    atomicMax_block(mx, (int)threadIdx.x);
    atomicMax_system(mx, (int)threadIdx.x + 100);
    atomicInc_block(u, 1000u);
    atomicDec_system(u, 1000u);
    atomicAdd_block(u, 2);
    atomicCAS_system(cas, 0, 7);
    atomicExch_block(cas + 1, 5);
    atomicOr_system(cas + 2, 1 << (threadIdx.x & 7));
}

static void test_hints()
{
    const int n = 256;
    std::vector<float> h(n);
    std::vector<int> hi(n);
    for (int i = 0; i < n; ++i) { h[i] = 0.5f * i; hi[i] = i; }
    float *d, *o;
    int *di, *oi;
    OK(cudaMalloc(&d, n * sizeof(float)));
    OK(cudaMalloc(&o, n * sizeof(float)));
    OK(cudaMalloc(&di, n * sizeof(int)));
    OK(cudaMalloc(&oi, n * sizeof(int)));
    OK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OK(cudaMemcpy(di, hi.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    hints<<<4, 64>>>(d, o, di, oi, n);
    OK(cudaDeviceSynchronize());
    std::vector<float> r(n);
    std::vector<int> ri(n);
    OK(cudaMemcpy(r.data(), o, n * sizeof(float), cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(ri.data(), oi, n * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && r[i] == 5.0f * h[i] && ri[i] == i + 3;
    CHECK(ok);
    cudaFree(d); cudaFree(o); cudaFree(di); cudaFree(oi);
}

static void test_scoped()
{
    int *counter, *mx, *cas;
    unsigned *u;
    OK(cudaMalloc(&counter, sizeof(int)));
    OK(cudaMalloc(&mx, sizeof(int)));
    OK(cudaMalloc(&u, sizeof(unsigned)));
    OK(cudaMalloc(&cas, 3 * sizeof(int)));
    OK(cudaMemset(counter, 0, sizeof(int)));
    OK(cudaMemset(mx, 0, sizeof(int)));
    OK(cudaMemset(u, 0, sizeof(unsigned)));
    OK(cudaMemset(cas, 0, 3 * sizeof(int)));
    scoped<<<2, 32>>>(counter, u, mx, cas);
    OK(cudaDeviceSynchronize());
    int c = 0, m = 0, k[3] = {0, 0, 0};
    unsigned uu = 0;
    OK(cudaMemcpy(&c, counter, sizeof c, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(&m, mx, sizeof m, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(&uu, u, sizeof uu, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(k, cas, sizeof k, cudaMemcpyDeviceToHost));
    CHECK(c == 128);
    CHECK(m == 131);
    CHECK(k[0] == 7 && k[1] == 5 && k[2] == 0xff);
    cudaFree(counter); cudaFree(mx); cudaFree(u); cudaFree(cas);
}

static void test_constants()
{
    CHECK(CUDART_PI_F == 3.14159265358979f);
    CHECK(std::fabs(CUDART_PI - std::acos(-1.0)) == 0.0);
    CHECK(std::isinf(CUDART_INF_F) && std::isnan(CUDART_NAN_F));
    CHECK(CUDART_PI_HI + CUDART_PI_LO == CUDART_PI);
    CHECK(std::fabs(CUDART_SQRT_TWO * CUDART_SQRT_HALF - 1.0) < 1e-15);
    CHECK(std::fabs(CUDART_LN2 - std::log(2.0)) < 1e-16);
    OK(cudaProfilerStart());
    OK(cudaProfilerStop());
}

static void test_3d()
{
    const size_t w = 20, h = 5, d = 4; // bytes, rows, slices
    cudaPitchedPtr p;
    OK(cudaMalloc3D(&p, make_cudaExtent(w, h, d)));
    CHECK(p.pitch >= w && p.xsize == w && p.ysize == h && p.ptr != nullptr);
    OK(cudaMemset3D(p, 0, make_cudaExtent(p.pitch, h, d)));
    std::vector<unsigned char> src(w * h * d), back(w * h * d, 0xee);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (unsigned char)(i * 7 + 1);
    cudaMemcpy3DParms up = {};
    up.srcPtr = make_cudaPitchedPtr(src.data(), w, w, h);
    up.dstPtr = p;
    up.extent = make_cudaExtent(w, h, d);
    up.kind = cudaMemcpyHostToDevice;
    OK(cudaMemcpy3D(&up));
    cudaMemcpy3DParms down = {};
    down.srcPtr = p;
    down.dstPtr = make_cudaPitchedPtr(back.data(), w, w, h);
    down.extent = make_cudaExtent(w, h, d);
    down.kind = cudaMemcpyDeviceToHost;
    OK(cudaMemcpy3D(&down));
    CHECK(back == src);
    cudaMemcpy3DParms arr = up;
    arr.dstArray = reinterpret_cast<cudaArray_t>(0x10);
    CHECK(cudaMemcpy3D(&arr) == cudaErrorNotSupported);
    cudaFree(p.ptr);
}

static void test_limits()
{
    size_t v = 0;
    OK(cudaDeviceGetLimit(&v, cudaLimitStackSize));
    CHECK(v > 0);
    OK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64u << 20));
    OK(cudaDeviceGetLimit(&v, cudaLimitMallocHeapSize));
    CHECK(v == (64u << 20));
    CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 1) == cudaErrorUnsupportedLimit);
    CHECK(cudaDeviceGetLimit(&v, (cudaLimit)99) == cudaErrorInvalidValue);
    CHECK(cudaDeviceGetLimit(nullptr, cudaLimitStackSize) == cudaErrorInvalidValue);
}

int main()
{
    test_hints();
    test_scoped();
    test_constants();
    test_3d();
    test_limits();
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

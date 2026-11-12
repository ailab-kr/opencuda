// Language and API details real programs rely on, as an ordinary CUDA
// program: vector types, built-in variables as variables, literal-0
// kernel arguments, mixed-type min/max, atomics with converted operands,
// pitched allocations, symbol copies on a stream, cache hints, error names,
// and launches written in an included file and in a macro.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "rt_lang_kernels.cuh"

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

__constant__ int c_table[8];

__global__ void builtins(uint3 *tid, dim3 *bdim, int *ws)
{
    unsigned i = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    tid[i] = threadIdx;  // uint3, assignable
    dim3 d = blockDim;   // dim3
    bdim[i] = d;
    ws[i] = warpSize;    // an int, usable in printf-style contexts
}

__global__ void maybe_null(int *out, const int *in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in ? in[i] : -1;
}

__global__ void mixed(unsigned *out, int a, unsigned b, float f, double d, double *dout)
{
    out[0] = max(a, b);
    out[1] = min(a, b);
    dout[0] = max(f, d);
    unsigned *counter = out + 2;
    atomicAdd(counter, 1);          // int literal into an unsigned atomic
    atomicMax(out + 3, threadIdx.x);
}

__global__ void read_table(int *out)
{
    out[threadIdx.x] = c_table[threadIdx.x];
}

__global__ void vec_add(const float4 *a, const float4 *b, float4 *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

#define LAUNCH_FILL(ptr, n, v) fill_value<<<((n) + 63) / 64, 64>>>(ptr, n, v)

static void test_vector_types()
{
    CHECK(sizeof(float4) == 16 && alignof(float4) == 16);
    CHECK(sizeof(float3) == 12 && alignof(float3) == 4);
    CHECK(sizeof(double2) == 16 && alignof(double2) == 16);
    CHECK(sizeof(uchar4) == 4 && alignof(uchar4) == 4);
    CHECK(sizeof(int2) == 8 && alignof(int2) == 8);
    CHECK(sizeof(longlong2) == 16);
    int2 p = make_int2(3, -4);
    CHECK(p.x == 3 && p.y == -4);
    uint3 u = make_uint3(1, 2, 3);
    dim3 d(u);
    CHECK(d.x == 1 && d.y == 2 && d.z == 3);
    dim3 e(5);
    CHECK(e.y == 1 && e.z == 1);

    const int n = 100;
    std::vector<float4> ha(n), hb(n), hc(n);
    for (int i = 0; i < n; ++i) {
        ha[i] = make_float4(float(i), 1, 2, 3);
        hb[i] = make_float4(1, float(i), 0, -3);
    }
    float4 *da, *db, *dc;
    OK(cudaMalloc(&da, n * sizeof(float4)));
    OK(cudaMalloc(&db, n * sizeof(float4)));
    OK(cudaMalloc(&dc, n * sizeof(float4)));
    OK(cudaMemcpy(da, ha.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
    OK(cudaMemcpy(db, hb.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
    vec_add<<<(n + 31) / 32, 32>>>(da, db, dc, n);
    OK(cudaMemcpy(hc.data(), dc, n * sizeof(float4), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i)
        bad += hc[i].x != float(i) + 1 || hc[i].y != float(i) + 1 || hc[i].z != 2 || hc[i].w != 0;
    CHECK(bad == 0);
    OK(cudaFree(da));
    OK(cudaFree(db));
    OK(cudaFree(dc));
}

static void test_builtins()
{
    // Host code may declare locals named like the built-in variables.
    dim3 gridDim(2, 1, 1), blockDim(8, 2, 1);
    const int n = 2 * 16;
    uint3 *tid;
    dim3 *bdim;
    int *ws;
    OK(cudaMalloc(&tid, n * sizeof(uint3)));
    OK(cudaMalloc(&bdim, n * sizeof(dim3)));
    OK(cudaMalloc(&ws, n * sizeof(int)));
    builtins<<<gridDim, blockDim>>>(tid, bdim, ws);
    std::vector<uint3> ht(n);
    std::vector<dim3> hb(n);
    std::vector<int> hw(n);
    OK(cudaMemcpy(ht.data(), tid, n * sizeof(uint3), cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(hb.data(), bdim, n * sizeof(dim3), cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(hw.data(), ws, n * sizeof(int), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        bad += hb[i].x != 8 || hb[i].y != 2 || hb[i].z != 1;
        bad += ht[i].x != unsigned(i % 8) || ht[i].y != unsigned((i / 8) % 2) || ht[i].z != 0;
        bad += hw[i] != 32 && hw[i] != 64;
    }
    CHECK(bad == 0);
    OK(cudaFree(tid));
    OK(cudaFree(bdim));
    OK(cudaFree(ws));
}

static void test_literal_zero_and_mixed()
{
    int *out;
    OK(cudaMalloc(&out, 4 * sizeof(int)));
    maybe_null<<<1, 4>>>(out, 0, 4);       // literal 0 as a null pointer
    maybe_null<<<1, 4>>>(out, NULL, 0);    // NULL, and 0 as an int
    int h[4] = {};
    OK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
    CHECK(h[0] == -1 && h[3] == -1);

    unsigned *u;
    double *dd;
    OK(cudaMalloc(&u, 4 * sizeof(unsigned)));
    OK(cudaMalloc(&dd, sizeof(double)));
    OK(cudaMemset(u, 0, 4 * sizeof(unsigned)));
    mixed<<<1, 64>>>(u, 5, 7u, 1.5f, 2.5, dd);
    unsigned hu[4];
    double hd = 0;
    OK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(&hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
    CHECK(hu[0] == 7 && hu[1] == 5);
    CHECK(hu[2] == 64);
    CHECK(hu[3] == 63);
    CHECK(hd == 2.5);
    OK(cudaFree(out));
    OK(cudaFree(u));
    OK(cudaFree(dd));
}

static void test_included_and_macro_launches()
{
    const int n = 1000;
    int *d;
    OK(cudaMalloc(&d, n * sizeof(int)));
    LAUNCH_FILL(d, n, 7);            // launch inside a macro
    add_one<<<(n + 127) / 128, 128>>>(d, n); // kernel from an included file
    launch_add_one(d, n);            // launch written in the included file
    std::vector<int> h(n);
    OK(cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += h[i] != 9;
    CHECK(bad == 0);
    OK(cudaFree(d));
}

static void test_memory_extras()
{
    char *p = nullptr;
    size_t pitch = 0;
    OK(cudaMallocPitch(&p, &pitch, 100, 7));
    CHECK(pitch >= 100 && pitch % 256 == 0);
    OK(cudaMemset2D(p, pitch, 0x5a, 100, 7));
    std::vector<char> h(100 * 7);
    OK(cudaMemcpy2D(h.data(), 100, p, pitch, 100, 7, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (char c : h) bad += c != 0x5a;
    CHECK(bad == 0);
    CHECK(cudaMemset2D(p, pitch, 0, pitch + 1, 1) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    OK(cudaFree(p));

    void *hp = nullptr;
    OK(cudaHostAlloc(&hp, 1024, cudaHostAllocDefault));
    OK(cudaFreeHost(hp));
    CHECK(cudaHostAlloc(&hp, 1024, 0x100) == cudaErrorInvalidValue);
    (void)cudaGetLastError();

    int table[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    cudaStream_t s;
    OK(cudaStreamCreate(&s));
    OK(cudaMemcpyToSymbolAsync(c_table, table, sizeof(table), 0, cudaMemcpyHostToDevice, s));
    int *out;
    OK(cudaMalloc(&out, sizeof(table)));
    read_table<<<1, 8, 0, s>>>(out);
    int back[8] = {};
    OK(cudaMemcpyAsync(back, out, sizeof(back), cudaMemcpyDeviceToHost, s));
    OK(cudaStreamSynchronize(s));
    CHECK(std::memcmp(back, table, sizeof(table)) == 0);
    int part[2] = {};
    OK(cudaMemcpyFromSymbolAsync(part, c_table, sizeof(part), 3 * sizeof(int),
                                 cudaMemcpyDeviceToHost, s));
    OK(cudaStreamSynchronize(s));
    CHECK(part[0] == 4 && part[1] == 5);
    OK(cudaStreamDestroy(s));
    OK(cudaFree(out));
}

static void test_hints_and_errors()
{
    OK(cudaFuncSetCacheConfig(vec_add, cudaFuncCachePreferL1));
    OK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaFuncCache c = cudaFuncCachePreferNone;
    OK(cudaDeviceGetCacheConfig(&c));
    CHECK(c == cudaFuncCachePreferShared);
    CHECK(cudaDeviceSetCacheConfig(static_cast<cudaFuncCache>(9)) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    CHECK(std::strcmp(cudaGetErrorName(cudaErrorIllegalAddress), "cudaErrorIllegalAddress") == 0);
    CHECK(std::strcmp(cudaGetErrorName(cudaErrorLaunchOutOfResources),
                      "cudaErrorLaunchOutOfResources") == 0);
    CHECK(cudaErrorIllegalAddress == 700 && cudaErrorMissingConfiguration == 52);
    CHECK(std::strlen(cudaGetErrorString(cudaErrorPriorLaunchFailure)) > 0);
    CHECK(std::strlen(cudaGetErrorString(static_cast<cudaError_t>(12345))) > 0);
}

int main()
{
    test_vector_types();
    test_builtins();
    test_literal_zero_and_mixed();
    test_included_and_macro_launches();
    test_memory_extras();
    test_hints_and_errors();
    std::printf("rt_lang: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

// -use_fast_math: built with it, the single-precision functions the CUDA
// Programming Guide lists (expf, exp10f, logf, log2f, log10f, sinf, cosf,
// tanf, powf, sincosf) give their intrinsics' results (__expf ...) where the
// path maps them (the SPIR-V path: OCU_FAST_MATH in device code), and stay
// within the intrinsics' documented error otherwise; double functions do
// not change. For the SPIR-V path (the CPU path lacks some of the
// intrinsics: __exp10f, __tanf).
//   ocu-nvcc --ocu-target=spirv -use_fast_math tests/runtime/rt_fast_math.cu -o rt_fast_math
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

constexpr int NF = 10;
__global__ void k(const float *x, float *a, float *b, double *d, int n, int *mapped)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
#if defined(OCU_FAST_MATH)
        *mapped = 1;
#else
        *mapped = 0;
#endif
    }
    if (i >= n) return;
    float v = x[i], s, c, s2, c2;
    sincosf(v, &s, &c);
    __sincosf(v, &s2, &c2);
    float fa[NF] = {expf(v), exp10f(v * 0.25f), logf(fabsf(v) + 1), log2f(fabsf(v) + 1), log10f(fabsf(v) + 1),
                    sinf(v), cosf(v), tanf(v * 0.1f), powf(fabsf(v) + 1, 1.3f), s + c};
    float fb[NF] = {__expf(v), __exp10f(v * 0.25f), __logf(fabsf(v) + 1), __log2f(fabsf(v) + 1), __log10f(fabsf(v) + 1),
                    __sinf(v), __cosf(v), __tanf(v * 0.1f), __powf(fabsf(v) + 1, 1.3f), s2 + c2};
    for (int f = 0; f < NF; ++f) { a[i * NF + f] = fa[f]; b[i * NF + f] = fb[f]; }
    d[i] = exp((double)v);
}

int main()
{
    const int n = 4096;
    static float x[n], a[n * NF], b[n * NF];
    static double d[n];
    for (int i = 0; i < n; ++i) x[i] = -8.0f + 16.0f * (float)i / n;
    float *dx, *da, *db; double *dd; int *dm, mapped = -1;
    cudaMalloc(&dx, sizeof x); cudaMalloc(&da, sizeof a); cudaMalloc(&db, sizeof b); cudaMalloc(&dd, sizeof d); cudaMalloc(&dm, 4);
    cudaMemcpy(dx, x, sizeof x, cudaMemcpyHostToDevice);
    k<<<n / 256, 256>>>(dx, da, db, dd, n, dm);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(a, da, sizeof a, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, db, sizeof b, cudaMemcpyDeviceToHost);
    cudaMemcpy(d, dd, sizeof d, cudaMemcpyDeviceToHost);
    cudaMemcpy(&mapped, dm, 4, cudaMemcpyDeviceToHost);
    int same = 0, close = 0, dbad = 0;
    for (int i = 0; i < n * NF; ++i) {
        same += std::memcmp(&a[i], &b[i], 4) == 0;
        double r = std::fabs((double)a[i] - b[i]) / (std::fabs((double)b[i]) + 1e-6);
        close += r < 1e-3; // the intrinsics' documented errors are far larger than float rounding
    }
    for (int i = 0; i < n; ++i) dbad += std::fabs(d[i] - std::exp((double)x[i])) > 4.5e-16 * std::exp((double)x[i]); // 2 ulp
    bool ok = e == cudaSuccess && dbad == 0 && (mapped == 1 ? same == n * NF : close == n * NF);
    std::printf("%s: fast-math mapping %s, %d/%d equal to the intrinsics, %d/%d within 1e-3, double exp %d wrong\n",
                ok ? "PASS" : "FAIL", mapped == 1 ? "on" : "off", same, n * NF, close, n * NF, dbad);
    return ok ? 0 : 1;
}

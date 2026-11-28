// double erf / erfc in device code against long double references on the
// host, within the CUDA Math API maxima (erf 2 ulp, erfc 5 ulp): 2^18 inputs
// over [-6, 6] and erfc's range [0, 28], tiny ones, special values.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_erf_gpu.cu -o rt_erf_gpu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

__global__ void k(const double *x, double *e, double *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        e[i] = erf(x[i]);
        c[i] = erfc(x[i]);
    }
}

static double ulps(double got, long double want)
{
    if (std::isnan(want)) return std::isnan(got) ? 0 : 1e9;
    if (std::isinf(want)) return got == (double)want ? 0 : 1e9;
    long double a = std::fabs(want);
    int ex;
    std::frexp(a, &ex);
    long double ulp = a < 0x1p-1022L ? 0x1p-1074L : std::ldexp(1.0L, ex - 53);
    return (double)(std::fabs((long double)got - want) / ulp);
}

int main()
{
    const int n = 1 << 18;
    std::vector<double> x(n);
    uint64_t s = 99;
    for (int i = 0; i < n; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        double u = (double)(s >> 11) * 0x1p-53;
        x[i] = i % 3 == 0 ? u * 12.0 - 6.0 : (i % 3 == 1 ? u * 28.0 : std::ldexp(u, -(int)(s % 900)));
    }
    const double sp[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 1.5, -1.5, 26.6, 27.2, 6.0};
    for (int j = 0; j < 10; ++j) x[j] = sp[j];
    double *dx, *de, *dc;
    cudaMalloc(&dx, n * 8); cudaMalloc(&de, n * 8); cudaMalloc(&dc, n * 8);
    cudaMemcpy(dx, x.data(), n * 8, cudaMemcpyHostToDevice);
    k<<<n / 256, 256>>>(dx, de, dc, n);
    cudaError_t err = cudaDeviceSynchronize();
    std::vector<double> e(n), c(n);
    cudaMemcpy(e.data(), de, n * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(c.data(), dc, n * 8, cudaMemcpyDeviceToHost);
    double we = 0, wc = 0;
    double xe = 0, xc = 0;
    for (int i = 0; i < n; ++i) {
        double ue = ulps(e[i], std::erf((long double)x[i]));
        long double rc = std::erfc((long double)x[i]);
        double uc = (rc >= 0x1p-1074L || rc == 0 || std::isnan(rc)) ? ulps(c[i], rc) : 0;
        if (ue > we) { we = ue; xe = x[i]; }
        if (uc > wc) { wc = uc; xc = x[i]; }
    }
    bool sign0 = std::signbit(e[1]) && !std::signbit(e[0]);
    bool ok = err == cudaSuccess && we <= 2.0 && wc <= 5.0 && sign0;
    std::printf("%s: erf max %.3f ulp (at %a), erfc max %.3f ulp (at %a), erf(-0) %s\n", ok ? "PASS" : "FAIL", we, xe, wc, xc,
                sign0 ? "-0" : "wrong");
    return ok ? 0 : 1;
}

// cub::TransformInputIterator and cub::CountingInputIterator with
// cub::DeviceReduce, and DeviceReduce's accumulator as the CUB
// documentation gives it (Sum starts from the output's value type; the
// accumulator is what the operator gives for (initial value, input)):
// counting non-zeros through a bool iterator into int64_t (HeCBench
// nonzero-cuda), the sum of 0..n-1 from a counting iterator into int64_t
// beyond 2^31, Max through a transform, float data summed into double, and
// the iterators' arithmetic on the host.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_cub_iter.cu -o rt_cub_iter
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { if (g_fail++ < 20) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

struct NonZero {
    __host__ __device__ __forceinline__ bool operator()(const float &a) const { return a != 0.0f; }
};
struct Square {
    __host__ __device__ __forceinline__ float operator()(const float &a) const { return a * a; }
};

template <typename In, typename Out>
static cudaError_t dsum(In in, Out *out, int n)
{
    size_t bytes = 0;
    cub::DeviceReduce::Sum(nullptr, bytes, in, out, n);
    void *tmp = nullptr;
    cudaMalloc(&tmp, bytes);
    cudaError_t e = cub::DeviceReduce::Sum(tmp, bytes, in, out, n);
    cudaDeviceSynchronize();
    cudaFree(tmp);
    return e;
}

int main()
{
    const int n = 1 << 20;
    std::vector<float> h(n);
    long long nz = 0;
    double fsum = 0;
    for (int i = 0; i < n; ++i) {
        h[i] = (i % 7 == 0 || i % 11 == 3) ? 0.0f : (float)((i * 13) % 97) * 0.25f - 6.0f;
        nz += h[i] != 0.0f;
        fsum += h[i];
    }
    float *d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    int64_t *d64;
    double *dd;
    float *df;
    cudaMalloc(&d64, sizeof(int64_t));
    cudaMalloc(&dd, sizeof(double));
    cudaMalloc(&df, sizeof(float));

    // bool iterator into int64_t: a count, not a logical or
    cub::TransformInputIterator<bool, NonZero, float *> nzit(d, NonZero());
    int64_t got = -1;
    CHECK(dsum(nzit, d64, n) == cudaSuccess, "Sum(nonzero)");
    cudaMemcpy(&got, d64, sizeof got, cudaMemcpyDeviceToHost);
    CHECK(got == nz, "non-zeros: %lld want %lld", (long long)got, nz);
    // small input (one block path)
    CHECK(dsum(nzit, d64, 1000) == cudaSuccess, "Sum(nonzero, 1000)");
    cudaMemcpy(&got, d64, sizeof got, cudaMemcpyDeviceToHost);
    long long nz1000 = 0;
    for (int i = 0; i < 1000; ++i) nz1000 += h[i] != 0.0f;
    CHECK(got == nz1000, "non-zeros in 1000: %lld want %lld", (long long)got, nz1000);

    // counting iterator: 0 + 1 + ... + (m-1) into int64_t, beyond 2^31
    const int m = 3000000;
    cub::CountingInputIterator<int64_t> cnt(0);
    CHECK(dsum(cnt, d64, m) == cudaSuccess, "Sum(counting)");
    cudaMemcpy(&got, d64, sizeof got, cudaMemcpyDeviceToHost);
    CHECK(got == (int64_t)m * (m - 1) / 2, "sum of 0..%d: %lld", m - 1, (long long)got);

    // float data into double: accumulated in double
    double gd = 0;
    CHECK(dsum(d, dd, n) == cudaSuccess, "Sum(float -> double)");
    cudaMemcpy(&gd, dd, sizeof gd, cudaMemcpyDeviceToHost);
    CHECK(std::fabs(gd - fsum) <= 1e-9 * (std::fabs(fsum) + n), "float into double: %.17g want %.17g", gd, fsum);

    // Max through a transform
    cub::TransformInputIterator<float, Square, float *> sq(d, Square());
    size_t bytes = 0;
    cub::DeviceReduce::Max(nullptr, bytes, sq, df, n);
    void *tmp;
    cudaMalloc(&tmp, bytes);
    CHECK(cub::DeviceReduce::Max(tmp, bytes, sq, df, n) == cudaSuccess, "Max(square)");
    cudaDeviceSynchronize();
    float gm = 0, wm = 0;
    for (float v : h) wm = v * v > wm ? v * v : wm;
    cudaMemcpy(&gm, df, sizeof gm, cudaMemcpyDeviceToHost);
    CHECK(gm == wm, "max square: %g want %g", gm, wm);
    cudaFree(tmp);

    // host-side iterator arithmetic
    cub::CountingInputIterator<int> c0(10);
    auto c1 = c0 + 5;
    CHECK(*c1 == 15 && c1[2] == 17 && (c1 - c0) == 5 && c0 != c1 && (c1 - 5) == c0, "counting iterator arithmetic");
    ++c1;
    CHECK(*c1 == 16, "counting iterator increment");
    cub::TransformInputIterator<float, Square, const float *> t0(h.data(), Square());
    CHECK(t0[3] == h[3] * h[3] && *(t0 + 4) == h[4] * h[4] && ((t0 + 9) - t0) == 9, "transform iterator on the host");

    cudaFree(d); cudaFree(d64); cudaFree(dd); cudaFree(df);
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

// cub::DeviceSelect::Flagged / If / Unique against host loops: the selected
// items in input order and their count, for n = 0, 1, 1000, 3e6 (several
// scan tiles); flags as char and as a bool TransformInputIterator (the
// nonzero pattern of HeCBench nonzero-cuda), a predicate functor, runs of
// equal values for Unique, the count written as int and as long long.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_cub_select.cu -o rt_cub_select
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { if (g_fail++ < 20) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

struct DivBy3 {
    __host__ __device__ __forceinline__ bool operator()(const float &v) const { return ((int)v) % 3 == 0; }
};
struct NonZero {
    __host__ __device__ __forceinline__ bool operator()(const float &v) const { return v != 0.0f; }
};

template <typename F>
static void with_temp(F f)
{
    size_t bytes = 0;
    f(nullptr, bytes);
    void *tmp = nullptr;
    cudaMalloc(&tmp, bytes ? bytes : 1);
    f(tmp, bytes);
    cudaDeviceSynchronize();
    cudaFree(tmp);
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 3000000};
    for (long long n : sizes) {
        std::vector<float> h(n);
        std::vector<char> fl(n);
        for (long long i = 0; i < n; ++i) {
            h[i] = (float)((i * 7) % 50) - (i % 11 == 0 ? (float)((i * 7) % 50) : 0.0f); // some zeros
            fl[i] = (char)((i % 5 == 1) || (i % 13 == 0));
        }
        std::vector<float> sorted(n);
        for (long long i = 0; i < n; ++i) sorted[i] = (float)(i / 4 + (i % 9 == 0 ? 1 : 0) * 0); // runs of 4
        float *din, *dout, *dsorted;
        char *dfl;
        int *dnum;
        long long *dnum64;
        cudaMalloc(&din, (n ? n : 1) * sizeof(float));
        cudaMalloc(&dout, (n ? n : 1) * sizeof(float));
        cudaMalloc(&dsorted, (n ? n : 1) * sizeof(float));
        cudaMalloc(&dfl, n ? n : 1);
        cudaMalloc(&dnum, sizeof(int));
        cudaMalloc(&dnum64, sizeof(long long));
        if (n) {
            cudaMemcpy(din, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dsorted, sorted.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dfl, fl.data(), n, cudaMemcpyHostToDevice);
        }
        auto check = [&](const char *what, const std::vector<float> &want, bool wide) {
            long long num = -1;
            if (wide) cudaMemcpy(&num, dnum64, sizeof num, cudaMemcpyDeviceToHost);
            else { int v = -1; cudaMemcpy(&v, dnum, sizeof v, cudaMemcpyDeviceToHost); num = v; }
            std::vector<float> got(want.size());
            if (!want.empty()) cudaMemcpy(got.data(), dout, want.size() * sizeof(float), cudaMemcpyDeviceToHost);
            CHECK(num == (long long)want.size() && got == want, "%s n=%lld: count %lld want %zu", what, n, num, want.size());
        };
        // Flagged with char flags
        std::vector<float> want;
        for (long long i = 0; i < n; ++i) if (fl[i]) want.push_back(h[i]);
        with_temp([&](void *t, size_t &b) { cub::DeviceSelect::Flagged(t, b, din, dfl, dout, dnum, n); });
        check("Flagged", want, false);
        // Flagged with a bool transform iterator (non-zeros), count as long long
        want.clear();
        for (long long i = 0; i < n; ++i) if (h[i] != 0.0f) want.push_back(h[i]);
        cub::TransformInputIterator<bool, NonZero, float *> nz(din, NonZero());
        with_temp([&](void *t, size_t &b) { cub::DeviceSelect::Flagged(t, b, din, nz, dout, dnum64, n); });
        check("Flagged(nonzero)", want, true);
        // If with a predicate
        want.clear();
        for (long long i = 0; i < n; ++i) if (DivBy3()(h[i])) want.push_back(h[i]);
        with_temp([&](void *t, size_t &b) { cub::DeviceSelect::If(t, b, din, dout, dnum, n, DivBy3()); });
        check("If", want, false);
        // Unique over runs
        want.clear();
        for (long long i = 0; i < n; ++i) if (i == 0 || sorted[i] != sorted[i - 1]) want.push_back(sorted[i]);
        with_temp([&](void *t, size_t &b) { cub::DeviceSelect::Unique(t, b, dsorted, dout, dnum64, n); });
        check("Unique", want, true);
        cudaFree(din); cudaFree(dout); cudaFree(dsorted); cudaFree(dfl); cudaFree(dnum); cudaFree(dnum64);
    }
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

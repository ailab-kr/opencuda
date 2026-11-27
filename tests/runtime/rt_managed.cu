// Managed memory (cudaMallocManaged) used the way CUDA programs use it: the
// host writes it, kernels read and write it, the host reads it back after a
// synchronize; copies and fills to and from it; the hints
// (cudaMemPrefetchAsync, cudaMemAdvise and their _v2 forms) accept valid
// arguments and refuse ranges outside a managed allocation and unknown
// devices; a 64 MiB allocation touched on both sides; cudaFree.
// On the OpenCL path of an Intel GPU this needs USM shared allocations
// (cl_intel_unified_shared_memory). Same source on every path:
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_managed.cu -o rt_managed
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

__global__ void axpy(float a, const float *x, float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
__global__ void bump(unsigned *p, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] += 1u;
}

int main()
{
    int dev = 0;
    OK(cudaGetDevice(&dev));
    int managed = 0;
    OK(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, dev));
    CHECK(managed == 1);

    const int n = 100000;
    float *x = nullptr, *y = nullptr;
    OK(cudaMallocManaged(&x, n * sizeof(float)));
    OK(cudaMallocManaged(&y, n * sizeof(float), cudaMemAttachGlobal));
    if (!x || !y) { std::printf("FAIL: no managed memory\n"); return 1; }
    for (int i = 0; i < n; ++i) { x[i] = (float)i; y[i] = 1.0f; } // host writes
    OK(cudaMemPrefetchAsync(x, n * sizeof(float), dev, 0));
    OK(cudaMemAdvise(x, n * sizeof(float), cudaMemAdviseSetReadMostly, dev));
    axpy<<<(n + 255) / 256, 256>>>(2.0f, x, y, n);
    OK(cudaDeviceSynchronize());
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += y[i] != 2.0f * i + 1.0f; // host reads
    CHECK(bad == 0);

    // copies and fills with managed memory on either side
    std::vector<float> h(n);
    OK(cudaMemcpy(h.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(h[12345] == 2.0f * 12345 + 1.0f);
    OK(cudaMemcpy(y, x, n * sizeof(float), cudaMemcpyDefault));
    CHECK(y[777] == 777.0f);
    OK(cudaMemset(y, 0, n * sizeof(float)));
    OK(cudaDeviceSynchronize());
    CHECK(y[0] == 0.0f && y[n - 1] == 0.0f);
    float *d = nullptr;
    OK(cudaMalloc(&d, n * sizeof(float)));
    OK(cudaMemcpy(d, x, n * sizeof(float), cudaMemcpyDeviceToDevice));
    OK(cudaMemcpy(y, d, n * sizeof(float), cudaMemcpyDeviceToDevice));
    OK(cudaDeviceSynchronize());
    CHECK(y[n - 1] == (float)(n - 1));

    // pointer attributes
    cudaPointerAttributes a;
    OK(cudaPointerGetAttributes(&a, x + 10));
    CHECK(a.type == cudaMemoryTypeManaged && a.hostPointer == x + 10 && a.devicePointer == x + 10);

    // hints: valid and invalid arguments
    cudaMemLocation loc{cudaMemLocationTypeDevice, dev}, host{cudaMemLocationTypeHost, 0};
    OK(cudaMemPrefetchAsync(x, 4096, cudaCpuDeviceId, 0));
    OK(cudaMemPrefetchAsync_v2(x, 4096, loc, 0, 0));
    OK(cudaMemPrefetchAsync_v2(x, 4096, host, 0, 0));
    OK(cudaMemAdvise(x, 4096, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    OK(cudaMemAdvise(x, 4096, cudaMemAdviseSetAccessedBy, dev));
    OK(cudaMemAdvise_v2(x, 4096, cudaMemAdviseUnsetAccessedBy, loc));
    OK(cudaMemPrefetchAsync(x, 4096, loc, 0));          // CUDA 13 form (C++ overload)
    OK(cudaMemPrefetchAsync(x, 4096, host, 0, 0));
    OK(cudaMemAdvise(x, 4096, cudaMemAdviseSetAccessedBy, loc));
    CHECK(cudaMemPrefetchAsync(d, 4096, dev, 0) == cudaErrorInvalidValue);          // not managed
    CHECK(cudaMemPrefetchAsync(x, (n + 1) * sizeof(float), dev, 0) == cudaErrorInvalidValue); // past the end
    CHECK(cudaMemPrefetchAsync(x, 0, dev, 0) == cudaErrorInvalidValue);
    CHECK(cudaMemPrefetchAsync(x, 4096, 99, 0) == cudaErrorInvalidDevice);
    CHECK(cudaMemAdvise(x, 4096, (cudaMemoryAdvise)42, dev) == cudaErrorInvalidValue);
    CHECK(cudaMemPrefetchAsync_v2(x, 4096, loc, 1u, 0) == cudaErrorInvalidValue);    // flags must be 0
    (void)cudaGetLastError();

    // a large allocation touched by host and device
    const size_t big = (size_t)64 << 20;
    unsigned *p = nullptr;
    OK(cudaMallocManaged(&p, big));
    if (p) {
        const size_t m = big / sizeof(unsigned);
        for (size_t i = 0; i < m; i += 1024) p[i] = (unsigned)i;
        bump<<<1024, 256>>>(p, m);
        OK(cudaDeviceSynchronize());
        size_t wrong = 0;
        for (size_t i = 0; i < m; i += 1024) wrong += p[i] != (unsigned)i + 1u;
        CHECK(wrong == 0);
        OK(cudaFree(p));
    }
    OK(cudaFree(d));
    OK(cudaFree(x));
    OK(cudaFree(y));
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

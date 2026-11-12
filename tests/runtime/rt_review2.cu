// Regression tests for the sixth review.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

static int *g_dev;
static cudaError_t g_inside = cudaSuccess;

static void callback(void *)
{
    // Runtime calls from a host function are not allowed: an error, not a hang.
    g_inside = cudaMemsetAsync(g_dev, 0, sizeof(int), 0);
}

__global__ void noop() {}

int main()
{
    // A host function that calls the runtime returns instead of hanging.
    OK(cudaMalloc(&g_dev, sizeof(int)));
    OK(cudaLaunchHostFunc(0, callback, nullptr));
    CHECK(g_inside == cudaErrorNotPermitted);
    (void)cudaGetLastError();
    OK(cudaMemset(g_dev, 0, sizeof(int))); // the runtime works afterwards
    OK(cudaFree(g_dev));

    // Registering a range that encloses a registered one is refused.
    std::vector<char> buf(8192);
    OK(cudaHostRegister(buf.data() + 1024, 64, 0));
    CHECK(cudaHostRegister(buf.data(), 4096, 0) == cudaErrorHostMemoryAlreadyRegistered);
    CHECK(cudaHostRegister(buf.data() + 1000, 100, 0) == cudaErrorHostMemoryAlreadyRegistered);
    (void)cudaGetLastError();
    OK(cudaHostRegister(buf.data() + 2048, 64, 0)); // disjoint is fine
    OK(cudaHostUnregister(buf.data() + 1024));
    OK(cudaHostUnregister(buf.data() + 2048));
    CHECK(cudaHostRegister(reinterpret_cast<void *>(~std::uintptr_t(0) - 8), 64, 0) ==
          cudaErrorInvalidValue); // would wrap around
    (void)cudaGetLastError();

    // Zero-size allocations; a pointer one past the end is not inside.
    void *z = reinterpret_cast<void *>(1);
    OK(cudaMalloc(&z, 0));
    OK(cudaFree(z));
    char *p = nullptr;
    OK(cudaMalloc(&p, 256));
    cudaPointerAttributes a;
    OK(cudaPointerGetAttributes(&a, p + 256));
    CHECK(a.type == cudaMemoryTypeUnregistered);
    OK(cudaFree(p));
    CHECK(cudaFree(p) != cudaSuccess); // double free is an error
    (void)cudaGetLastError();

    // Occupancy counts whole warps.
    cudaDeviceProp prop;
    OK(cudaGetDeviceProperties(&prop, 0));
    int blocks = 0;
    OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, noop, prop.warpSize + 1, 0));
    CHECK(blocks >= 1 && blocks * 2 * prop.warpSize <= prop.maxThreadsPerMultiProcessor);

    // Integer construction and assignment of __half.
    __half h = 1.0f;
    h = 0;
    CHECK(__half2float(h) == 0.0f);
    h = 3u;
    CHECK(__half2float(h) == 3.0f);
    h = static_cast<short>(-2);
    CHECK(__half2float(h) == -2.0f);
    __half g = 5L;
    __half k = static_cast<std::size_t>(7);
    CHECK(__half2float(g) == 5.0f && __half2float(k) == 7.0f);
    __half big = 100000; // beyond the largest half: infinity
    CHECK(__hisinf(big) == 1);

    std::printf("rt_review2: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

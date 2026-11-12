// Pointer kinds, cudaMemcpyDefault, managed memory and device attributes,
// as an ordinary CUDA program.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

__global__ void scale(float *v, int n, float a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= a;
}

static void test_pointer_attributes()
{
    float *dev = nullptr, *host = nullptr, *managed = nullptr;
    OK(cudaMalloc(&dev, 64 * sizeof(float)));
    OK(cudaMallocHost(&host, 64 * sizeof(float)));
    OK(cudaMallocManaged(&managed, 64 * sizeof(float)));
    std::vector<float> pageable(64);
    cudaPointerAttributes a;
    OK(cudaPointerGetAttributes(&a, dev + 10)); // inside an allocation
    CHECK(a.type == cudaMemoryTypeDevice && a.device == 0 && a.devicePointer == dev + 10);
    OK(cudaPointerGetAttributes(&a, host));
    CHECK(a.type == cudaMemoryTypeHost && a.hostPointer == host);
    OK(cudaPointerGetAttributes(&a, managed));
    CHECK(a.type == cudaMemoryTypeManaged && a.hostPointer == managed && a.devicePointer == managed);
    OK(cudaPointerGetAttributes(&a, pageable.data()));
    CHECK(a.type == cudaMemoryTypeUnregistered);
    CHECK(cudaPointerGetAttributes(nullptr, dev) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    OK(cudaFree(dev));
    OK(cudaPointerGetAttributes(&a, dev)); // freed: no longer known
    CHECK(a.type == cudaMemoryTypeUnregistered);
    OK(cudaFreeHost(host));
    OK(cudaFree(managed));
}

static void test_memcpy_default()
{
    const int n = 256;
    std::vector<float> h(n), back(n, -1), h2(n, -1);
    for (int i = 0; i < n; ++i) h[i] = float(i);
    float *d1, *d2;
    OK(cudaMalloc(&d1, n * sizeof(float)));
    OK(cudaMalloc(&d2, n * sizeof(float)));
    OK(cudaMemcpy(d1, h.data(), n * sizeof(float), cudaMemcpyDefault));      // H2D
    OK(cudaMemcpy(d2, d1, n * sizeof(float), cudaMemcpyDefault));            // D2D
    scale<<<(n + 63) / 64, 64>>>(d2, n, 2.0f);
    OK(cudaMemcpy(back.data(), d2, n * sizeof(float), cudaMemcpyDefault));   // D2H
    OK(cudaMemcpy(h2.data(), back.data(), n * sizeof(float), cudaMemcpyDefault)); // H2H
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += back[i] != 2.0f * float(i) || h2[i] != back[i];
    CHECK(bad == 0);
    // Out of range on the device side is still an error.
    CHECK(cudaMemcpy(d1 + 1, h.data(), n * sizeof(float), cudaMemcpyDefault) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    OK(cudaFree(d1));
    OK(cudaFree(d2));
}

static void test_managed()
{
    const int n = 1000;
    float *m = nullptr;
    OK(cudaMallocManaged(&m, n * sizeof(float)));
    for (int i = 0; i < n; ++i) m[i] = float(i); // the host writes directly
    scale<<<(n + 127) / 128, 128>>>(m, n, 3.0f);
    OK(cudaDeviceSynchronize());
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += m[i] != 3.0f * float(i); // and reads back
    CHECK(bad == 0);
    CHECK(cudaMallocManaged(&m, 16, 0x10) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    OK(cudaFree(m));
}

static void test_attributes()
{
    cudaDeviceProp p;
    OK(cudaGetDeviceProperties(&p, 0));
    int v = -1;
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrWarpSize, 0));
    CHECK(v == p.warpSize);
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerBlock, 0));
    CHECK(v == p.maxThreadsPerBlock);
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    CHECK(v == int(p.sharedMemPerBlock));
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0));
    CHECK(v >= 1);
    int major = 0, minor = -1;
    OK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    OK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
    CHECK(major == p.major && minor == p.minor);
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrManagedMemory, 0));
    CHECK(v == 1);
    CHECK(cudaDeviceGetAttribute(&v, static_cast<cudaDeviceAttr>(9999), 0) == cudaErrorInvalidValue);
    CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrWarpSize, 99) == cudaErrorInvalidDevice);
    CHECK(cudaDeviceGetAttribute(nullptr, cudaDevAttrWarpSize, 0) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
}

int main()
{
    test_pointer_attributes();
    test_memcpy_default();
    test_managed();
    test_attributes();
    std::printf("rt_mem: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

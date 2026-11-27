// Stream-ordered allocation and memory pools, as programs use them
// (cudaMallocAsync / cudaFreeAsync, the default pool and its release
// threshold, pools of their own): kernels see the memory in stream order;
// an allocation after a free on the same stream reuses the memory (100
// cycles do not grow the pool) and the second kernel's values are not
// overwritten by the first; across streams after a synchronize; the
// release threshold (0: cudaDeviceSynchronize gives cached memory back;
// UINT64_MAX: it is kept) and cudaMemPoolTrimTo; cudaFreeAsync of
// cudaMalloc memory and cudaFree of stream-ordered memory; created pools,
// cudaMallocFromPoolAsync, Get / SetMemPool, the attribute set / get rules
// and the documented errors.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_mempool.cu -o rt_mempool
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { if (g_fail++ < 25) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)
#define OK(call) CHECK((call) == cudaSuccess, "%s", #call)

__global__ void fill(int *p, int n, int v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v + i;
}
__global__ void add(const int *a, int *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] += a[i];
}

static uint64_t attr(cudaMemPool_t p, cudaMemPoolAttr a)
{
    uint64_t v = 0;
    cudaMemPoolGetAttribute(p, a, &v);
    return v;
}

int main()
{
    int supported = 0;
    OK(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, 0));
    CHECK(supported == 1, "memory pools supported: %d", supported);
    cudaMemPool_t pool;
    OK(cudaDeviceGetDefaultMemPool(&pool, 0));
    cudaMemPool_t cur = nullptr;
    OK(cudaDeviceGetMemPool(&cur, 0));
    CHECK(cur == pool, "current pool is the default pool");
    uint64_t thr = 0;
    OK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
    CHECK(thr == 0, "default release threshold 0 (%llu)", (unsigned long long)thr);

    const int n = 1 << 16;
    const size_t bytes = n * sizeof(int);
    cudaStream_t s1, s2;
    OK(cudaStreamCreate(&s1));
    OK(cudaStreamCreate(&s2));
    std::vector<int> h(n);

    // basic: allocate, use, free, all in stream order (typed form)
    int *a = nullptr, *b = nullptr;
    OK(cudaMallocAsync(&a, bytes, s1));
    OK(cudaMallocAsync(&b, bytes, s1));
    fill<<<n / 256, 256, 0, s1>>>(a, n, 1);
    fill<<<n / 256, 256, 0, s1>>>(b, n, 100);
    add<<<n / 256, 256, 0, s1>>>(a, b, n);
    OK(cudaMemcpyAsync(h.data(), b, bytes, cudaMemcpyDeviceToHost, s1));
    OK(cudaFreeAsync(a, s1));
    OK(cudaFreeAsync(b, s1));
    OK(cudaStreamSynchronize(s1));
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += h[i] != 101 + 2 * i;
    CHECK(bad == 0, "basic: %d wrong", bad);

    // reuse on one stream: 100 cycles keep the pool's size, values stay in order
    OK(cudaDeviceSynchronize()); // threshold 0: cached memory goes back
    uint64_t base = attr(pool, cudaMemPoolAttrReservedMemCurrent);
    int *keep = nullptr;
    OK(cudaMallocAsync(&keep, bytes, s1));
    for (int r = 0; r < 100; ++r) {
        int *t = nullptr;
        OK(cudaMallocAsync(&t, bytes, s1));
        fill<<<n / 256, 256, 0, s1>>>(t, n, r * 1000);
        add<<<n / 256, 256, 0, s1>>>(t, keep, n); // keep += t, read before the free
        OK(cudaFreeAsync(t, s1));
        if (r == 0) cudaMemsetAsync(keep, 0, bytes, s1); // start the sum after the first round
    }
    OK(cudaMemcpyAsync(h.data(), keep, bytes, cudaMemcpyDeviceToHost, s1));
    OK(cudaStreamSynchronize(s1));
    uint64_t grown = attr(pool, cudaMemPoolAttrReservedMemCurrent) - base;
    CHECK(grown <= 3 * bytes, "100 cycles on one stream: pool grew by %llu bytes", (unsigned long long)grown);
    bad = 0;
    for (int i = 0; i < n; ++i) {
        long long want = 0;
        for (int r = 1; r < 100; ++r) want += r * 1000 + i;
        bad += h[i] != (int)want;
    }
    CHECK(bad == 0, "stream-ordered reuse: %d wrong", bad);
    OK(cudaFreeAsync(keep, s1));

    // across streams after a synchronize
    OK(cudaStreamSynchronize(s1));
    uint64_t before = attr(pool, cudaMemPoolAttrReservedMemCurrent);
    int *c = nullptr;
    OK(cudaMallocAsync(&c, bytes, s2));
    CHECK(attr(pool, cudaMemPoolAttrReservedMemCurrent) == before, "a completed free is reused on another stream");
    fill<<<n / 256, 256, 0, s2>>>(c, n, 7);
    OK(cudaMemcpyAsync(h.data(), c, bytes, cudaMemcpyDeviceToHost, s2));
    OK(cudaFreeAsync(c, s2));
    OK(cudaStreamSynchronize(s2));
    bad = 0;
    for (int i = 0; i < n; ++i) bad += h[i] != 7 + i;
    CHECK(bad == 0, "other stream: %d wrong", bad);

    // release threshold: 0 gives back at device synchronize; UINT64_MAX keeps
    OK(cudaDeviceSynchronize());
    CHECK(attr(pool, cudaMemPoolAttrReservedMemCurrent) == attr(pool, cudaMemPoolAttrUsedMemCurrent),
          "threshold 0: reserved %llu used %llu after synchronize",
          (unsigned long long)attr(pool, cudaMemPoolAttrReservedMemCurrent), (unsigned long long)attr(pool, cudaMemPoolAttrUsedMemCurrent));
    uint64_t big = UINT64_MAX;
    OK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &big));
    int *d = nullptr;
    OK(cudaMallocAsync(&d, bytes, s1));
    OK(cudaFreeAsync(d, s1));
    OK(cudaDeviceSynchronize());
    CHECK(attr(pool, cudaMemPoolAttrReservedMemCurrent) >= attr(pool, cudaMemPoolAttrUsedMemCurrent) + bytes,
          "threshold UINT64_MAX keeps the memory");
    OK(cudaMemPoolTrimTo(pool, 0));
    CHECK(attr(pool, cudaMemPoolAttrReservedMemCurrent) == attr(pool, cudaMemPoolAttrUsedMemCurrent), "trim to 0");
    uint64_t zero = 0;
    OK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &zero));
    CHECK(attr(pool, cudaMemPoolAttrUsedMemHigh) >= 2 * bytes, "used high-water mark");
    OK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
    CHECK(attr(pool, cudaMemPoolAttrUsedMemHigh) == attr(pool, cudaMemPoolAttrUsedMemCurrent), "high-water reset");
    uint64_t one = 1;
    CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &one) == cudaErrorInvalidValue, "high-water set to non-zero");
    CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &zero) == cudaErrorInvalidValue, "read-only attribute");
    (void)cudaGetLastError();

    // cudaFreeAsync of cudaMalloc memory, cudaFree of stream-ordered memory
    int *m = nullptr;
    OK(cudaMalloc(&m, bytes));
    fill<<<n / 256, 256, 0, s1>>>(m, n, 3);
    OK(cudaFreeAsync(m, s1));
    int *e = nullptr;
    OK(cudaMallocAsync(&e, bytes, s1));
    OK(cudaFree(e));
    OK(cudaDeviceSynchronize());
    CHECK(attr(pool, cudaMemPoolAttrUsedMemCurrent) == 0, "nothing in use (%llu)", (unsigned long long)attr(pool, cudaMemPoolAttrUsedMemCurrent));

    // a pool of our own
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = 0;
    cudaMemPool_t mine = nullptr;
    OK(cudaMemPoolCreate(&mine, &props));
    int *f = nullptr;
    OK(cudaMallocFromPoolAsync(&f, bytes, mine, s2));
    CHECK(attr(mine, cudaMemPoolAttrUsedMemCurrent) == bytes, "allocation counted in its pool");
    fill<<<n / 256, 256, 0, s2>>>(f, n, 9);
    OK(cudaMemcpyAsync(h.data(), f, bytes, cudaMemcpyDeviceToHost, s2));
    OK(cudaFreeAsync(f, s2));
    OK(cudaStreamSynchronize(s2));
    CHECK(h[5] == 14 && attr(mine, cudaMemPoolAttrUsedMemCurrent) == 0, "own pool: %d", h[5]);
    OK(cudaDeviceSetMemPool(0, mine));
    int *g = nullptr;
    OK(cudaMallocAsync(&g, 4096, s1));
    CHECK(attr(mine, cudaMemPoolAttrUsedMemCurrent) == 4096, "cudaMallocAsync uses the current pool");
    OK(cudaFreeAsync(g, s1));
    OK(cudaDeviceSetMemPool(0, pool));
    OK(cudaMemPoolDestroy(mine));
    CHECK(cudaMemPoolDestroy(pool) == cudaErrorInvalidValue, "the default pool cannot be destroyed");
    props.handleTypes = cudaMemHandleTypePosixFileDescriptor;
    cudaMemPool_t shared = nullptr;
    CHECK(cudaMemPoolCreate(&shared, &props) == cudaErrorNotSupported, "shareable handles: not supported");
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.id = 99;
    CHECK(cudaMemPoolCreate(&shared, &props) == cudaErrorInvalidDevice, "pool on a device that does not exist");
    CHECK(cudaDeviceGetDefaultMemPool(&shared, 99) == cudaErrorInvalidDevice, "default pool of a device that does not exist");
    CHECK(cudaMallocAsync(nullptr, 16, s1) == cudaErrorInvalidValue, "null pointer");
    void *z = &z;
    OK(cudaMallocAsync(&z, 0, s1));
    CHECK(z == nullptr, "size 0 gives null");
    OK(cudaFreeAsync(nullptr, s1));
    (void)cudaGetLastError();

    OK(cudaStreamDestroy(s1));
    OK(cudaStreamDestroy(s2));
    OK(cudaDeviceSynchronize());
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

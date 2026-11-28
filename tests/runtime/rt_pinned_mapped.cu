// Page-locked host memory that kernels use directly (zero-copy), as CUDA
// programs use cudaHostAlloc(cudaHostAllocMapped) and cudaMallocHost memory:
// cudaHostGetDevicePointer gives the address kernels use; a kernel writes
// results straight into host memory (read with no copy after a
// synchronize), reads input the host wrote, and the host's pointer works as
// a kernel argument too (unified addressing). Copies to and from device
// memory, and async copies on a stream, still take it. Where the path does
// not map page-locked memory (cudaHostGetDevicePointer returns
// cudaErrorNotSupported) only the copies are checked.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_pinned_mapped.cu -o rt_pinned_mapped
#include <cuda_runtime.h>

#include <cstdio>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

__global__ void square_into(const int *in, long long *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (long long)in[i] * in[i];
}

__global__ void add_one(int *p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] += 1;
}

int main()
{
    const int n = 100000;
    int *in = nullptr;
    long long *out = nullptr;
    OK(cudaHostAlloc(&in, n * sizeof(int), cudaHostAllocMapped));
    OK(cudaMallocHost(&out, n * sizeof(long long)));
    for (int i = 0; i < n; ++i) in[i] = i - n / 2;

    // copies take it, whatever the path
    int *dev;
    OK(cudaMalloc(&dev, n * sizeof(int)));
    OK(cudaMemcpy(dev, in, n * sizeof(int), cudaMemcpyHostToDevice));
    cudaStream_t s;
    OK(cudaStreamCreate(&s));
    add_one<<<(n + 255) / 256, 256, 0, s>>>(dev, n);
    OK(cudaMemcpyAsync(in, dev, n * sizeof(int), cudaMemcpyDeviceToHost, s));
    OK(cudaStreamSynchronize(s));
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += in[i] != i - n / 2 + 1;
    CHECK(bad == 0);
    for (int i = 0; i < n; ++i) in[i] = i - n / 2;

    int *din = nullptr;
    cudaError_t e = cudaHostGetDevicePointer((void **)&din, in, 0);
    if (e == cudaErrorNotSupported) {
        (void)cudaGetLastError();
        std::printf("note: this path does not map page-locked memory (zero-copy not checked)\n");
    } else {
        CHECK(e == cudaSuccess && din != nullptr);
        long long *dout = nullptr;
        OK(cudaHostGetDevicePointer((void **)&dout, out, 0));
        square_into<<<(n + 255) / 256, 256>>>(din, dout, n);
        OK(cudaDeviceSynchronize());
        bad = 0;
        for (int i = 0; i < n; ++i) bad += out[i] != (long long)(i - n / 2) * (i - n / 2);
        CHECK(bad == 0);
        // the host pointer itself as the kernel argument; then the host sees the result
        add_one<<<(n + 255) / 256, 256>>>(in, n);
        OK(cudaDeviceSynchronize());
        bad = 0;
        for (int i = 0; i < n; ++i) bad += in[i] != i - n / 2 + 1;
        CHECK(bad == 0);
        cudaPointerAttributes a;
        OK(cudaPointerGetAttributes(&a, in + 5));
        CHECK(a.type == cudaMemoryTypeHost && a.hostPointer == in + 5 && a.devicePointer == in + 5);
        CHECK(cudaFree(in) != cudaSuccess); // not device memory
        (void)cudaGetLastError();
    }

    OK(cudaFree(dev));
    OK(cudaStreamDestroy(s));
    OK(cudaFreeHost(in));
    OK(cudaFreeHost(out));
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

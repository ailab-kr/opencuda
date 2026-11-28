// cudaMalloc / cudaFree in a loop: memory freed and allocated again (the
// OpenCL backend keeps freed device blocks for reuse) holds what kernels,
// copies and memset write to it, is as large as asked and no larger for
// copies, and a block freed while a kernel used it was waited for.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_alloc_reuse.cu -o rt_alloc_reuse
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <vector>

static int checks = 0, failed = 0;
static void check(bool ok, const char *what, int i)
{
    ++checks;
    if (!ok && failed++ < 10) std::printf("FAIL: %s (%d)\n", what, i);
}

__global__ void fill(unsigned *p, size_t n, unsigned seed)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = seed * 2654435761u + (unsigned)i;
}

__global__ void slow_fill(unsigned *p, size_t n, unsigned seed, int never)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned v = (unsigned)i;
    for (int k = 0; k < 20000; ++k) v = v * 1664525u + 1013904223u;
    p[i] = never ? v : seed; // the loop runs, the value is seed
}

int main()
{
    const size_t sizes[] = {4, 1000, 4096, 65536, 1 << 20, (1 << 20) - 100000, 3 << 20, 1 << 20};
    std::vector<unsigned> h;
    for (int it = 0; it < 64; ++it) {
        size_t bytes = sizes[it % 8];
        size_t n = bytes / 4;
        unsigned *a = nullptr, *b = nullptr;
        check(cudaMalloc(&a, bytes) == cudaSuccess && cudaMalloc(&b, bytes) == cudaSuccess, "cudaMalloc", it);
        check(((uintptr_t)a & 255) == 0 && ((uintptr_t)b & 255) == 0 && a != b, "aligned, distinct", it);
        fill<<<(unsigned)((n + 255) / 256), 256>>>(a, n, it);
        check(cudaMemset(b, it & 0xff, bytes) == cudaSuccess, "cudaMemset", it);
        h.assign(n, 0);
        check(cudaMemcpy(h.data(), a, n * 4, cudaMemcpyDeviceToHost) == cudaSuccess, "copy back", it);
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) bad += h[i] != (unsigned)it * 2654435761u + (unsigned)i;
        check(bad == 0, "kernel results in reused memory", it);
        h.assign(n, 0);
        cudaMemcpy(h.data(), b, n * 4, cudaMemcpyDeviceToHost);
        unsigned m = (it & 0xff) * 0x01010101u;
        bad = 0;
        for (size_t i = 0; i < n; ++i) bad += h[i] != m;
        check(bad == 0, "memset in reused memory", it);
        // a copy past the end of what was asked fails, whatever block it was given
        static unsigned big[(3 << 20) / 4 + 1024];
        check(cudaMemcpy(a, big, bytes + 1024, cudaMemcpyHostToDevice) != cudaSuccess, "copy past the end refused", it);
        cudaGetLastError();
        check(cudaFree(a) == cudaSuccess && cudaFree(b) == cudaSuccess, "cudaFree", it);
    }
    // freed while a kernel runs on a stream: the next allocation of that size
    // gets memory the kernel no longer writes
    for (int it = 0; it < 4; ++it) {
        const size_t n = 1 << 18;
        unsigned *p, *q;
        cudaStream_t s;
        cudaStreamCreate(&s);
        cudaMalloc(&p, n * 4);
        slow_fill<<<n / 256, 256, 0, s>>>(p, n, 7, 0);
        cudaFree(p);
        cudaMalloc(&q, n * 4);
        fill<<<n / 256, 256>>>(q, n, 9);
        cudaDeviceSynchronize();
        h.assign(n, 0);
        cudaMemcpy(h.data(), q, n * 4, cudaMemcpyDeviceToHost);
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) bad += h[i] != 9u * 2654435761u + (unsigned)i;
        check(bad == 0, "free waits for the kernel", it);
        cudaFree(q);
        cudaStreamDestroy(s);
    }
    size_t f0, t0, f1, t1;
    cudaMemGetInfo(&f0, &t0);
    void *p;
    cudaMalloc(&p, 64 << 20);
    cudaFree(p);
    cudaMemGetInfo(&f1, &t1);
    check(f1 == f0 && t1 == t0, "cudaMemGetInfo after free as before", 0);
    // the time of a cudaMalloc / cudaFree pair (reported, not checked)
    auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < 2000; ++i) {
        cudaMalloc(&p, 1 << 16);
        cudaFree(p);
    }
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count() / 2000;
    check(cudaGetLastError() == cudaSuccess, "no error left", 0);
    std::printf("%s: %d checks, %d failed (cudaMalloc+cudaFree %.1f us)\n", failed ? "FAIL" : "PASS", checks, failed, us);
    return failed ? 1 : 0;
}

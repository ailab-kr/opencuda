// Regression tests for the fifth review: behaviour real programs depend on.
#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>
#include <sstream>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

// A friend template operator: `operator<< <>` is not a launch.
template <class T> struct Box;
template <class T> std::ostream &operator<<(std::ostream &, const Box<T> &);
template <class T> struct Box {
    T v;
    friend std::ostream &operator<< <>(std::ostream &, const Box<T> &);
};
template <class T> std::ostream &operator<<(std::ostream &o, const Box<T> &b) { return o << b.v; }

// Keeps a stream busy for a while, then ends by itself.
__global__ void busy(volatile unsigned long long *sink, unsigned long long n)
{
    unsigned long long x = 0;
    for (unsigned long long i = 0; i < n; ++i) x += i ^ (x >> 3);
    *sink = x;
}

__global__ void accumulate(int *acc, const int *v) { *acc += *v; }

template <class T>
__global__ void size_of_first(T, int *out) { *out = int(sizeof(T)); }

__global__ void take_pointer(const int *p, int *out) { *out = p == nullptr ? 1 : 0; }

// The common idiom: raw dynamic shared bytes, viewed as another type.
template <class T>
__global__ void through_bytes(T *out)
{
    extern __shared__ unsigned char smem[];
    T *s = reinterpret_cast<T *>(smem);
    T *c = (T *)smem;
    s[threadIdx.x] = T(threadIdx.x + 1);
    __syncthreads();
    out[threadIdx.x] = c[blockDim.x - 1 - threadIdx.x];
}

#if defined(OCU_CPU_PATH)
__global__ void broken_warp()
{
    // Lane 0 waits for its warp while the others wait for the block: the
    // CPU backend reports a launch failure instead of hanging. (On a GPU
    // this is undefined behaviour and may hang, so GPUs use a trap.)
    if (threadIdx.x == 0) {
        __syncwarp();
    } else {
        __syncthreads();
    }
}
#else
__global__ void broken_warp() { __trap(); }
#endif

static void test_pageable_async()
{
    // Host-to-device from pageable memory: the source is read before the
    // call returns, even while the stream is still busy.
    unsigned long long *sink;
    int *acc, *dv;
    OK(cudaMalloc(&sink, sizeof(*sink)));
    OK(cudaMalloc(&acc, sizeof(int)));
    OK(cudaMalloc(&dv, 4 * sizeof(int)));
    cudaStream_t s;
    OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    // On s itself: a non-blocking stream is not ordered after stream 0.
    OK(cudaMemsetAsync(acc, 0, sizeof(int), s));
    busy<<<1, 1, 0, s>>>(sink, 20000000ull);
    int h = 0;
    for (int i = 1; i <= 4; ++i) {
        h = i;
        OK(cudaMemcpyAsync(dv + i - 1, &h, sizeof(int), cudaMemcpyHostToDevice, s));
        accumulate<<<1, 1, 0, s>>>(acc, dv + i - 1);
    }
    h = 1000;
    int sum = 0;
    OK(cudaMemcpyAsync(&sum, acc, sizeof(int), cudaMemcpyDeviceToHost, s));
    // Device-to-host into pageable memory is complete on return.
    CHECK(sum == 10);
    OK(cudaStreamSynchronize(s));
    CHECK(sum == 10);
    OK(cudaStreamDestroy(s));
    OK(cudaFree(sink));
    OK(cudaFree(acc));
    OK(cudaFree(dv));
}

static void test_literal_zero_and_null()
{
    int *out;
    OK(cudaMalloc(&out, sizeof(int)));
    int h = 0;
    size_of_first<<<1, 1>>>(0, out); // T deduces int, as in a direct call
    OK(cudaMemcpy(&h, out, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(h == int(sizeof(int)));
    size_of_first<<<1, 1>>>(0.0, out);
    OK(cudaMemcpy(&h, out, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(h == int(sizeof(double)));
    take_pointer<<<1, 1, 0, NULL>>>(0, out); // NULL stream, 0 pointer
    OK(cudaMemcpy(&h, out, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(h == 1);
    take_pointer<<<1, 1, 0, 0>>>(NULL, out);
    OK(cudaMemcpy(&h, out, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(h == 1);
    OK(cudaFree(out));
}

static void test_shared_bytes()
{
    float *out;
    OK(cudaMalloc(&out, 32 * sizeof(float)));
    through_bytes<<<1, 32, 32 * sizeof(float)>>>(out);
    float h[32];
    OK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 32; ++i) bad += h[i] != float(32 - i);
    CHECK(bad == 0);
    OK(cudaFree(out));
}

#if !defined(OCU_CPU_PATH)
static void test_sticky_failure()
{
    // A kernel fault on a GPU: reported, and stays reported. The device's
    // context is lost for the process, so reset is not part of this test.
    broken_warp<<<1, 32>>>();
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaErrorLaunchFailure);
    CHECK(cudaDeviceSynchronize() == cudaErrorLaunchFailure);
    CHECK(cudaStreamSynchronize(0) == cudaErrorLaunchFailure);
}
#else
static void test_sticky_failure()
{
    broken_warp<<<1, 32>>>();
    CHECK(cudaGetLastError() == cudaSuccess); // the launch itself was valid
    CHECK(cudaDeviceSynchronize() == cudaErrorLaunchFailure);
    // Still visible to later synchronising calls, and new work is refused.
    CHECK(cudaDeviceSynchronize() == cudaErrorLaunchFailure);
    CHECK(cudaStreamSynchronize(0) == cudaErrorLaunchFailure);
    int *p = nullptr;
    OK(cudaMalloc(&p, 4));
    CHECK(cudaMemset(p, 0, 4) == cudaErrorLaunchFailure);
    (void)cudaGetLastError();
    // Reset clears it.
    OK(cudaDeviceReset());
    OK(cudaMemset(p, 0, 4));
    OK(cudaDeviceSynchronize());
    OK(cudaFree(p));
}
#endif

int main()
{
    std::ostringstream text;
    text << Box<int>{7};
    CHECK(text.str() == "7");
    test_pageable_async();
    test_literal_zero_and_null();
    test_shared_bytes();
    test_sticky_failure(); // last: it resets the device
    std::printf("rt_review: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

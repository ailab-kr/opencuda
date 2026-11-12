// Runtime API behaviour, written as an ordinary CUDA program (no ocu
// names): built by ocu_cc.py --target cpu and linked with ocu_cudart.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

__constant__ float c_coeff[4];
__device__ int d_counter;

__global__ void scale(float *v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = v[i] * c_coeff[0] + c_coeff[1];
}

__global__ void count(int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&d_counter, 1);
}

// Spins until *flag is set: lets the test see work still running.
__global__ void wait_flag(volatile int *flag)
{
    while (*flag == 0) {
    }
}

__global__ void store(int *out, int v) { *out = v; }

__global__ void block_reverse(int *data)
{
    extern __shared__ int tile[];
    int t = threadIdx.x;
    tile[t] = data[blockIdx.x * blockDim.x + t];
    __syncthreads();
    data[blockIdx.x * blockDim.x + t] = tile[blockDim.x - 1 - t];
}

static void test_devices()
{
    int n = -1;
    OK(cudaGetDeviceCount(&n));
    CHECK(n >= 1);
    cudaDeviceProp p;
    OK(cudaGetDeviceProperties(&p, 0));
    CHECK(p.warpSize == 32 || p.warpSize == 64);
    CHECK(p.maxThreadsPerBlock == 1024);
    CHECK(std::strlen(p.name) > 0);
    CHECK(cudaGetDeviceProperties(&p, n) == cudaErrorInvalidDevice);
    CHECK(cudaGetLastError() == cudaErrorInvalidDevice);
    CHECK(cudaGetLastError() == cudaSuccess); // read and cleared
    CHECK(cudaSetDevice(-1) == cudaErrorInvalidDevice);
    CHECK(cudaPeekAtLastError() == cudaErrorInvalidDevice);
    CHECK(cudaPeekAtLastError() == cudaErrorInvalidDevice); // peek keeps it
    (void)cudaGetLastError();
    OK(cudaSetDevice(0));
    int d = -1;
    OK(cudaGetDevice(&d));
    CHECK(d == 0);
    int v = 0;
    OK(cudaRuntimeGetVersion(&v));
    CHECK(v == CUDART_VERSION);
    CHECK(std::strcmp(cudaGetErrorName(cudaErrorInvalidValue), "cudaErrorInvalidValue") == 0);
    CHECK(std::strlen(cudaGetErrorString(cudaErrorMemoryAllocation)) > 0);
}

static void test_memory_and_launch()
{
    const int n = 1000;
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = float(i);
    float *d = nullptr;
    OK(cudaMalloc(&d, n * sizeof(float)));
    OK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    float coeff[4] = {2.0f, 1.0f, 0, 0};
    OK(cudaMemcpyToSymbol(c_coeff, coeff, sizeof(coeff)));
    float back[4] = {};
    OK(cudaMemcpyFromSymbol(back, c_coeff, sizeof(back)));
    CHECK(back[0] == 2.0f && back[1] == 1.0f);
    scale<<<(n + 255) / 256, 256>>>(d, n);
    OK(cudaGetLastError());
    OK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += h[i] != 2.0f * float(i) + 1.0f;
    CHECK(bad == 0);

    int zero = 0, got = -1;
    OK(cudaMemcpyToSymbol(d_counter, &zero, sizeof(zero)));
    count<<<40, 128>>>(5000);
    OK(cudaDeviceSynchronize());
    OK(cudaMemcpyFromSymbol(&got, d_counter, sizeof(got)));
    CHECK(got == 5000);

    // Bad launch configuration: reported by cudaGetLastError.
    scale<<<1, 2048>>>(d, n);
    CHECK(cudaGetLastError() == cudaErrorInvalidConfiguration);
    OK(cudaGetLastError());

    // memset, device-to-device, host-to-host, 2-D copies.
    OK(cudaMemset(d, 0, n * sizeof(float)));
    float *d2 = nullptr;
    OK(cudaMalloc(&d2, n * sizeof(float)));
    OK(cudaMemcpy(d2, d, n * sizeof(float), cudaMemcpyDeviceToDevice));
    OK(cudaMemcpy(h.data(), d2, n * sizeof(float), cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < n; ++i) bad += h[i] != 0.0f;
    CHECK(bad == 0);
    float a[3] = {1, 2, 3}, b[3] = {};
    OK(cudaMemcpy(b, a, sizeof(a), cudaMemcpyHostToHost));
    CHECK(b[2] == 3);
    std::vector<int> grid(6 * 10, -1), got2(4 * 3, 0);
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 10; ++c) grid[r * 10 + c] = r * 100 + c;
    int *dpitch = nullptr;
    OK(cudaMalloc(&dpitch, 16 * 6 * sizeof(int)));
    OK(cudaMemcpy2D(dpitch, 16 * sizeof(int), grid.data(), 10 * sizeof(int),
                    4 * sizeof(int), 3, cudaMemcpyHostToDevice));
    OK(cudaMemcpy2D(got2.data(), 4 * sizeof(int), dpitch, 16 * sizeof(int),
                    4 * sizeof(int), 3, cudaMemcpyDeviceToHost));
    bad = 0;
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) bad += got2[r * 4 + c] != r * 100 + c;
    CHECK(bad == 0);
    CHECK(cudaMemcpy2D(dpitch, 4, grid.data(), 40, 8, 1, cudaMemcpyHostToDevice) ==
          cudaErrorInvalidValue); // width > dpitch
    (void)cudaGetLastError();

    // Out-of-range copy is an error, not memory corruption.
    CHECK(cudaMemcpy(d, h.data(), (n + 1) * sizeof(float), cudaMemcpyHostToDevice) ==
          cudaErrorInvalidValue);
    (void)cudaGetLastError();
    CHECK(cudaFree(h.data()) == cudaErrorInvalidValue);
    (void)cudaGetLastError();

    // Dynamic shared memory through the launch configuration.
    std::vector<int> hr(64);
    for (int i = 0; i < 64; ++i) hr[i] = i;
    int *dr = nullptr;
    OK(cudaMalloc(&dr, 64 * sizeof(int)));
    OK(cudaMemcpy(dr, hr.data(), 64 * sizeof(int), cudaMemcpyHostToDevice));
    block_reverse<<<2, 32, 32 * sizeof(int)>>>(dr);
    OK(cudaMemcpy(hr.data(), dr, 64 * sizeof(int), cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < 64; ++i) bad += hr[i] != (i / 32) * 32 + (31 - i % 32);
    CHECK(bad == 0);

    size_t freeb = 0, total = 0;
    OK(cudaMemGetInfo(&freeb, &total));
    CHECK(total > 0 && freeb <= total);
    void *pinned = nullptr;
    OK(cudaMallocHost(&pinned, 4096));
    OK(cudaFreeHost(pinned));
    OK(cudaFree(d));
    OK(cudaFree(d2));
    OK(cudaFree(dpitch));
    OK(cudaFree(dr));
    OK(cudaFree(nullptr));
}

static void test_streams_and_events()
{
    int *flag = nullptr, *out = nullptr;
    OK(cudaMallocHost(&flag, sizeof(int)));
    OK(cudaMalloc(&out, sizeof(int)));
    *flag = 0;
    cudaStream_t s1, s2;
    OK(cudaStreamCreate(&s1));
    OK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    cudaEvent_t e0, e1, e2;
    OK(cudaEventCreate(&e0));
    OK(cudaEventCreate(&e1));
    OK(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&e2, 0x80) == cudaErrorInvalidValue);
    (void)cudaGetLastError();

    OK(cudaEventRecord(e0, s1));
    wait_flag<<<1, 1, 0, s1>>>(flag);
    store<<<1, 1, 0, s1>>>(out, 7);
    OK(cudaEventRecord(e1, s1));
    CHECK(cudaStreamQuery(s1) == cudaErrorNotReady);
    CHECK(cudaEventQuery(e1) == cudaErrorNotReady);
    CHECK(cudaGetLastError() == cudaSuccess); // not-ready is not an error
    // A non-blocking stream does not wait for s1.
    OK(cudaMemsetAsync(out, 0, 0, s2));
    OK(cudaStreamSynchronize(s2));
    *flag = 1;
    OK(cudaEventSynchronize(e1));
    float ms = -1;
    OK(cudaEventElapsedTime(&ms, e0, e1));
    CHECK(ms >= 0.0f);
    CHECK(cudaEventElapsedTime(&ms, e0, e2) == cudaErrorInvalidResourceHandle);
    (void)cudaGetLastError();
    int got = 0;
    OK(cudaMemcpy(&got, out, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(got == 7);

    // Legacy default stream: work on stream 0 waits for blocking streams.
    *flag = 0;
    wait_flag<<<1, 1, 0, s1>>>(flag);
    store<<<1, 1, 0, s1>>>(out, 9);
    OK(cudaEventRecord(e2, s1));
    *flag = 1; // release s1 before the default-stream copy waits on it
    OK(cudaMemcpy(&got, out, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(got == 9);
    OK(cudaStreamWaitEvent(s2, e2, 0));
    OK(cudaStreamSynchronize(cudaStreamLegacy));

    OK(cudaStreamDestroy(s1));
    OK(cudaStreamDestroy(s2));
    CHECK(cudaStreamDestroy(nullptr) == cudaErrorInvalidResourceHandle);
    (void)cudaGetLastError();
    OK(cudaEventDestroy(e0));
    OK(cudaEventDestroy(e1));
    OK(cudaEventDestroy(e2));
    OK(cudaFreeHost(flag));
    OK(cudaFree(out));
    OK(cudaThreadSynchronize());
    OK(cudaDeviceReset());
}

int main()
{
    test_devices();
    test_memory_and_launch();
    test_streams_and_events();
    std::printf("rt_api: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

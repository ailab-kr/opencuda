// Streams with priorities and flags, capture status, host functions in a
// stream, registered host memory, device flags, peers, function attributes
// and occupancy, as an ordinary CUDA program.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

__global__ void set_value(int *p, int v) { *p = v; }

__global__ void add_one(float *v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] += 1.0f;
}

struct seen {
    int *device_value;
    int observed;
    int calls;
};

static void on_host(void *p)
{
    seen *s = static_cast<seen *>(p);
    s->observed = *s->device_value; // the kernel before it has finished
    ++s->calls;
}

static void test_streams()
{
    int least = -1, greatest = -1;
    OK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CHECK(least >= greatest);
    cudaStream_t s;
    OK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, greatest - 5)); // clamped
    int prio = 99;
    OK(cudaStreamGetPriority(s, &prio));
    CHECK(prio >= greatest && prio <= least);
    unsigned flags = 0;
    OK(cudaStreamGetFlags(s, &flags));
    CHECK(flags == cudaStreamNonBlocking);
    OK(cudaStreamGetFlags(0, &flags));
    CHECK(flags == cudaStreamDefault);
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusActive;
    OK(cudaStreamIsCapturing(s, &cs));
    CHECK(cs == cudaStreamCaptureStatusNone);
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal) == cudaErrorNotSupported);
    (void)cudaGetLastError();

    // A host function runs after the work before it.
    int *mem;
    OK(cudaMallocManaged(&mem, sizeof(int)));
    *mem = 0;
    seen state{mem, -1, 0};
    set_value<<<1, 1, 0, s>>>(mem, 42);
    OK(cudaLaunchHostFunc(s, on_host, &state));
    OK(cudaStreamSynchronize(s));
    CHECK(state.calls == 1 && state.observed == 42);
    CHECK(cudaLaunchHostFunc(s, nullptr, nullptr) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    OK(cudaFree(mem));
    OK(cudaStreamDestroy(s));
}

static void test_host_register()
{
    const int n = 512;
    std::vector<float> h(n, 1.0f);
    OK(cudaHostRegister(h.data(), n * sizeof(float), cudaHostRegisterDefault));
    cudaPointerAttributes a;
    OK(cudaPointerGetAttributes(&a, h.data() + 3));
    CHECK(a.type == cudaMemoryTypeHost);
    CHECK(cudaHostRegister(h.data(), n * sizeof(float), 0) == cudaErrorHostMemoryAlreadyRegistered);
    (void)cudaGetLastError();
    float *dp = nullptr;
    OK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&dp), h.data(), 0));
    add_one<<<(n + 127) / 128, 128>>>(dp, n); // the device works on host memory
    OK(cudaDeviceSynchronize());
    int bad = 0;
    for (float x : h) bad += x != 2.0f;
    CHECK(bad == 0);
    OK(cudaHostUnregister(h.data()));
    CHECK(cudaHostUnregister(h.data()) == cudaErrorHostMemoryNotRegistered);
    (void)cudaGetLastError();
    CHECK(cudaHostRegister(h.data(), n * sizeof(float), 0x40) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
}

static void test_device_and_functions()
{
    OK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost));
    CHECK(cudaSetDeviceFlags(0x3) == cudaErrorInvalidValue); // two schedules
    (void)cudaGetLastError();
    int can = -1;
    OK(cudaDeviceCanAccessPeer(&can, 0, 0));
    CHECK(can == 0);
    CHECK(cudaDeviceCanAccessPeer(&can, 0, 7) == cudaErrorInvalidDevice);
    (void)cudaGetLastError();

    cudaDeviceProp p;
    OK(cudaGetDeviceProperties(&p, 0));
    OK(cudaFuncSetAttribute(add_one, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            int(p.sharedMemPerBlockOptin)));
    CHECK(cudaFuncSetAttribute(add_one, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               int(p.sharedMemPerBlockOptin) + 1) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    cudaFuncAttributes fa;
    OK(cudaFuncGetAttributes(&fa, add_one));
    CHECK(fa.maxThreadsPerBlock == p.maxThreadsPerBlock);

    int blocks = -1;
    OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, add_one, 256, 0));
    CHECK(blocks >= 1 && blocks * 256 <= p.maxThreadsPerMultiProcessor);
    OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, add_one, 2048, 0));
    CHECK(blocks == 0); // a block that cannot run
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, add_one, 0, 0) ==
          cudaErrorInvalidValue);
    (void)cudaGetLastError();
}

int main()
{
    test_streams();
    test_host_register();
    test_device_and_functions();
    std::printf("rt_misc: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

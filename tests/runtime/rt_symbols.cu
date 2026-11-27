// __constant__ and __device__ variables used through the runtime's symbol
// API the way CUDA programs use them: constant tables written with
// cudaMemcpyToSymbol (also at an offset, also async on a stream) and read by
// kernels, a struct in constant memory, a __device__ array with an initial
// value (read back before any launch, and by a kernel), a __device__
// counter bumped with atomics and read with cudaMemcpyFromSymbol, a variable
// written by one kernel and read by another, access through a device
// function, cudaGetSymbolAddress (the address works as a kernel argument and
// for cudaMemcpy) and cudaGetSymbolSize. Same source on every path:
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_symbols.cu -o rt_symbols
#include <cuda_runtime.h>

#include <cstdio>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

struct params { int n; float scale; float bias[3]; };

__constant__ float c_coeff[16];
__constant__ params c_p;
__device__ int d_counter;
__device__ float d_table[4] = {1.0f, 2.0f, 3.0f, 4.0f};
__device__ unsigned d_shared_flag;
__device__ double d_big[4096];

__device__ float through_helper(int i) { return c_coeff[i & 15] * d_table[i & 3] + c_p.bias[i % 3]; }

__global__ void apply(float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = through_helper(i) * c_p.scale + (float)c_p.n;
        atomicAdd(&d_counter, 1);
    }
}
__global__ void writer(unsigned v) { if (threadIdx.x == 0) d_shared_flag = v; }
__global__ void reader(unsigned *out) { if (threadIdx.x == 0) *out = d_shared_flag; }
__global__ void via_pointer(double *p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 2.0 * i;
}
__global__ void read_big(double *out) { out[threadIdx.x] = d_big[threadIdx.x * 10]; }

int main()
{
    // initial value, before any kernel ran
    float t0[4] = {0, 0, 0, 0};
    OK(cudaMemcpyFromSymbol(t0, d_table, sizeof t0));
    CHECK(t0[0] == 1.0f && t0[3] == 4.0f);

    float coeff[16];
    for (int i = 0; i < 16; ++i) coeff[i] = 0.5f * i;
    OK(cudaMemcpyToSymbol(c_coeff, coeff, 8 * sizeof(float)));                          // first half
    OK(cudaMemcpyToSymbol(c_coeff, coeff + 8, 8 * sizeof(float), 8 * sizeof(float)));   // at an offset
    float back[16] = {};
    OK(cudaMemcpyFromSymbol(back, c_coeff, sizeof back));
    int same = 0;
    for (int i = 0; i < 16; ++i) same += back[i] == coeff[i];
    CHECK(same == 16);
    params p{7, 2.0f, {0.25f, 0.5f, 0.75f}};
    cudaStream_t s;
    OK(cudaStreamCreate(&s));
    OK(cudaMemcpyToSymbolAsync(c_p, &p, sizeof p, 0, cudaMemcpyHostToDevice, s));
    OK(cudaStreamSynchronize(s));
    int zero = 0;
    OK(cudaMemcpyToSymbol(d_counter, &zero, sizeof zero));

    const int n = 5000;
    float *out;
    OK(cudaMalloc(&out, n * sizeof(float)));
    apply<<<(n + 255) / 256, 256>>>(out, n);
    OK(cudaDeviceSynchronize());
    static float h[n];
    OK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        float want = (coeff[i & 15] * (float)((i & 3) + 1) + p.bias[i % 3]) * p.scale + (float)p.n;
        bad += h[i] != want;
    }
    CHECK(bad == 0);
    int count = -1;
    OK(cudaMemcpyFromSymbol(&count, d_counter, sizeof count));
    CHECK(count == n);

    // a variable shared by two kernels
    unsigned *flag;
    OK(cudaMalloc(&flag, sizeof(unsigned)));
    writer<<<1, 32>>>(0xC0FFEEu);
    reader<<<1, 32>>>(flag);
    unsigned got = 0;
    OK(cudaMemcpy(&got, flag, sizeof got, cudaMemcpyDeviceToHost));
    CHECK(got == 0xC0FFEEu);

    // symbol address and size (the CPU path, whose device variables are host
    // variables, knows no symbol sizes: those checks are for the GPU paths)
    void *addr = nullptr;
    size_t size = 0;
    OK(cudaGetSymbolAddress(&addr, d_big));
    const bool sized = cudaGetSymbolSize(&size, d_big) == cudaSuccess;
    (void)cudaGetLastError();
    if (sized) CHECK(addr != nullptr && size == sizeof(double) * 4096);
    else std::printf("note: no symbol sizes on this path (size, bounds and cudaMemcpy on a symbol address not checked)\n");
    via_pointer<<<16, 256>>>(static_cast<double *>(addr), 4096);
    OK(cudaDeviceSynchronize());
    double *dout;
    OK(cudaMalloc(&dout, 256 * sizeof(double)));
    read_big<<<1, 256>>>(dout);
    double hb[256];
    OK(cudaMemcpy(hb, dout, sizeof hb, cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < 256; ++i) bad += hb[i] != 2.0 * (i * 10);
    CHECK(bad == 0);
    if (sized) {
        double first[2];
        OK(cudaMemcpy(first, addr, sizeof first, cudaMemcpyDeviceToHost));
        CHECK(first[0] == 0.0 && first[1] == 2.0);
        CHECK(cudaMemcpyToSymbol(c_coeff, coeff, sizeof coeff + 4) == cudaErrorInvalidValue); // past the end
        (void)cudaGetLastError();
    }

    OK(cudaFree(out)); OK(cudaFree(flag)); OK(cudaFree(dout));
    OK(cudaStreamDestroy(s));
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

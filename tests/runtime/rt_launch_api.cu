// cudaLaunchKernel: launching through the API with an array of argument
// pointers, as libraries do.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

struct params {
    float scale;
    int offset;
};

__global__ void affine(float *out, const float *in, int n, params p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * p.scale + float(p.offset);
}

__global__ void shared_sum(const int *in, int *out)
{
    extern __shared__ int s[];
    s[threadIdx.x] = in[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0;
        for (unsigned i = 0; i < blockDim.x; ++i) t += s[i];
        *out = t;
    }
}

__global__ void no_args(int *flag);

int main()
{
    const int n = 300;
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = float(i);
    float *in, *out;
    OK(cudaMalloc(&in, n * sizeof(float)));
    OK(cudaMalloc(&out, n * sizeof(float)));
    OK(cudaMemcpy(in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    int count = n;
    params p{2.0f, 5};
    void *args[] = {&out, &in, &count, &p};
    cudaStream_t s;
    OK(cudaStreamCreate(&s));
    OK(cudaLaunchKernel(affine, dim3((n + 127) / 128), dim3(128), args, 0, s));
    // The values were copied at the call: changing them now has no effect.
    p.scale = 100.0f;
    count = 0;
    OK(cudaStreamSynchronize(s));
    std::vector<float> r(n);
    OK(cudaMemcpy(r.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += r[i] != 2.0f * float(i) + 5.0f;
    CHECK(bad == 0);

    // Dynamic shared memory through the API.
    int *vals, *sum;
    std::vector<int> hv(64);
    for (int i = 0; i < 64; ++i) hv[i] = i + 1;
    OK(cudaMalloc(&vals, 64 * sizeof(int)));
    OK(cudaMalloc(&sum, sizeof(int)));
    OK(cudaMemcpy(vals, hv.data(), 64 * sizeof(int), cudaMemcpyHostToDevice));
    void *args2[] = {&vals, &sum};
    OK(cudaLaunchKernel(shared_sum, dim3(1), dim3(64), args2, 64 * sizeof(int)));
    int hs = 0;
    OK(cudaMemcpy(&hs, sum, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(hs == 64 * 65 / 2);

    // Errors: a bad configuration, missing arguments.
    CHECK(cudaLaunchKernel(affine, dim3(1), dim3(4096), args) == cudaErrorInvalidConfiguration);
    CHECK(cudaGetLastError() == cudaErrorInvalidConfiguration);
    CHECK(cudaLaunchKernel(affine, dim3(1), dim3(1), nullptr) != cudaSuccess);
    (void)cudaGetLastError();

    OK(cudaStreamDestroy(s));
    OK(cudaFree(in));
    OK(cudaFree(out));
    OK(cudaFree(vals));
    OK(cudaFree(sum));
    std::printf("rt_launch_api: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

__global__ void no_args(int *flag) { *flag = 1; }

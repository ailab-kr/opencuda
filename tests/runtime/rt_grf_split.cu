// OpenCL back end on Intel GPUs: the register file is chosen per kernel
// (default one, or the large one for a kernel that spills there), which
// builds a module twice. What must hold whichever build a kernel comes from:
//   1. constant tables read the same from every kernel;
//   2. a __device__ variable written by one kernel and read by another is one
//      variable (such a module is built once);
//   3. a kernel with many live values (it spills with the default register
//      file at 32 lanes) computes the right result.
// Run on an Intel GPU:
//   ocu-nvcc --ocu-target=spirv tests/runtime/rt_grf_split.cu tests/runtime/rt_grf_split_vars.cu -o rt_grf_split
//   OCU_BACKEND=opencl OCU_OPENCL_VERBOSE=1 ./rt_grf_split
// (with VERBOSE the back end says which kernels took the large register file).
// It also runs, trivially, on the CPU path.
#include <cuda_runtime.h>

#include <cstdio>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static __device__ const unsigned table[8] = {3, 1, 4, 1, 5, 9, 2, 6};

// Constant tables only in this group of kernels (the split applies if the
// whole translation unit has no variable that can change; see 2 below).
__global__ void light(unsigned *o) { o[threadIdx.x] = table[threadIdx.x & 7] * 2u; }

constexpr int kLive = 160; // live floats per lane: 160 x 32 lanes x 4 B = 160 registers

__global__ void heavy(const float *in, float *out, int n)
{
    float acc[kLive];
#pragma unroll
    for (int j = 0; j < kLive; ++j) acc[j] = in[(threadIdx.x + j) % n];
    for (int r = 0; r < 8; ++r) {
#pragma unroll
        for (int j = 0; j < kLive; ++j) acc[j] = acc[j] * 0.5f + acc[(j + 1) % kLive] * 0.25f + (float)table[j & 7];
    }
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < kLive; ++j) s += acc[j] * (float)(j + 1);
    s += __shfl_xor_sync(0xffffffffu, s, 1); // a warp-wide operation: 32 lanes per sub-group
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// 2 is in rt_grf_split_vars.cu: a translation unit is one module, and this
// one must have constant tables only.
unsigned grf_split_counter_test(); // 5 x 64 bumps of a shared __device__ variable

static float heavy_ref(const float *in, int tid, int n)
{
    float acc[kLive];
    for (int j = 0; j < kLive; ++j) acc[j] = in[(tid + j) % n];
    for (int r = 0; r < 8; ++r)
        for (int j = 0; j < kLive; ++j) acc[j] = acc[j] * 0.5f + acc[(j + 1) % kLive] * 0.25f + (float)table[j & 7];
    float s = 0.0f;
    for (int j = 0; j < kLive; ++j) s += acc[j] * (float)(j + 1);
    return s;
}

int main()
{
    unsigned *d_u = nullptr, h_u[64];
    cudaMalloc(&d_u, sizeof h_u);
    light<<<1, 64>>>(d_u);
    CHECK(cudaMemcpy(h_u, d_u, sizeof h_u, cudaMemcpyDeviceToHost) == cudaSuccess);
    static const unsigned t[8] = {3, 1, 4, 1, 5, 9, 2, 6};
    int bad = 0;
    for (int i = 0; i < 64; ++i) bad += h_u[i] != t[i & 7] * 2u;
    CHECK(bad == 0);

    const int n = 97, threads = 64, blocks = 4;
    float h_in[n], h_out[threads * blocks];
    for (int i = 0; i < n; ++i) h_in[i] = (float)((i * 7) % 13) - 6.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof h_in);
    cudaMalloc(&d_out, sizeof h_out);
    cudaMemcpy(d_in, h_in, sizeof h_in, cudaMemcpyHostToDevice);
    heavy<<<blocks, threads>>>(d_in, d_out, n);
    CHECK(cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost) == cudaSuccess);
    bad = 0;
    for (int b = 0; b < blocks; ++b)
        for (int i = 0; i < threads; ++i) {
            // lane pairs (i, i ^ 1) add their sums
            const float want = heavy_ref(h_in, i, n) + heavy_ref(h_in, i ^ 1, n);
            const float got = h_out[b * threads + i];
            const float tol = 1e-4f * (want < 0 ? -want : want) + 1e-3f;
            bad += !(got - want <= tol && want - got <= tol);
        }
    CHECK(bad == 0);
    if (bad) std::printf("heavy: %d of %d wrong\n", bad, threads * blocks);

    CHECK(grf_split_counter_test() == 5u * 64u);

    cudaFree(d_u); cudaFree(d_in); cudaFree(d_out);
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

// atomicAdd on __half, __half2, __nv_bfloat16 and __nv_bfloat162 (CUDA C++
// Programming Guide, "atomicAdd"; the two-element forms are atomic per
// element). Checked against exact expectations:
//   - counters: N threads each add 1 to a counter; the sum is exact (N <= 256
//     for bf16, 2048 for half) and the old values returned are exactly
//     0, 1, ..., N-1 in some order (each add saw a distinct old value);
//   - both halves of a 32-bit word: neighbouring 16-bit counters at the two
//     alignments do not disturb each other;
//   - rounding: one add of 1 + 2^-10 ... values whose float sum is not a
//     half: the result is the round-to-nearest-even of the exact sum.
// The same source runs on every path:
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_half_atomics.cu -o rt_half_atomics
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// counters[0..3] are four 16-bit values in two 32-bit words; thread t adds 1
// to counter t % 4 and records the old value.
__global__ void count_half(__half *c, float *olds, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) olds[t] = __half2float(atomicAdd(&c[t % 4], __float2half(1.0f)));
}
__global__ void count_bf16(__nv_bfloat16 *c, float *olds, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) olds[t] = __bfloat162float(atomicAdd(&c[t % 4], __float2bfloat16(1.0f)));
}
__global__ void count_half2(__half2 *c, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) atomicAdd(c, __half2(__float2half(1.0f), __float2half(2.0f)));
}
__global__ void count_bf162(__nv_bfloat162 *c, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) atomicAdd(c, __nv_bfloat162(__float2bfloat16(1.0f), __float2bfloat16(-1.0f)));
}
__global__ void one_add(__half *h, __half v, __half *old)
{
    *old = atomicAdd(h, v);
}

static bool olds_are_permutation(const std::vector<float> &o, int lane, int per)
{
    std::vector<int> seen(per, 0);
    for (size_t t = lane; t < o.size(); t += 4) {
        int v = (int)o[t];
        if ((float)v != o[t] || v < 0 || v >= per || seen[v]++) return false;
    }
    return true;
}

int main()
{
    const int n = 4 * 512; // 512 per counter: exact in half (integers up to 2048 are)
    float *d_olds;
    cudaMalloc(&d_olds, n * sizeof(float));
    std::vector<float> olds(n);

    { // half counters
        __half *c; cudaMalloc(&c, 4 * sizeof(__half)); cudaMemset(c, 0, 4 * sizeof(__half));
        count_half<<<(n + 255) / 256, 256>>>(c, d_olds, n);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        __half h[4]; cudaMemcpy(h, c, sizeof h, cudaMemcpyDeviceToHost);
        cudaMemcpy(olds.data(), d_olds, n * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 4; ++i) {
            CHECK(__half2float(h[i]) == 512.0f);
            CHECK(olds_are_permutation(olds, i, 512));
        }
        cudaFree(c);
    }
    { // bf16 counters: 256 per counter is the most that stays exact (8 bits)
        const int nb = 4 * 256;
        __nv_bfloat16 *c; cudaMalloc(&c, 4 * sizeof(__nv_bfloat16)); cudaMemset(c, 0, 4 * sizeof(__nv_bfloat16));
        count_bf16<<<(nb + 255) / 256, 256>>>(c, d_olds, nb);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        __nv_bfloat16 b[4]; cudaMemcpy(b, c, sizeof b, cudaMemcpyDeviceToHost);
        std::vector<float> ob(nb);
        cudaMemcpy(ob.data(), d_olds, nb * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 4; ++i) {
            CHECK(__bfloat162float(b[i]) == 256.0f);
            CHECK(olds_are_permutation(ob, i, 256));
        }
        cudaFree(c);
    }
    { // half2 / bf162: both elements, independent
        __half2 *c; cudaMalloc(&c, sizeof(__half2)); cudaMemset(c, 0, sizeof(__half2));
        count_half2<<<4, 256>>>(c, 1000);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        __half2 h; cudaMemcpy(&h, c, sizeof h, cudaMemcpyDeviceToHost);
        CHECK(__half2float(h.x) == 1000.0f && __half2float(h.y) == 2000.0f);
        cudaFree(c);
        __nv_bfloat162 *b; cudaMalloc(&b, sizeof(__nv_bfloat162)); cudaMemset(b, 0, sizeof(__nv_bfloat162));
        count_bf162<<<1, 200>>>(b, 200);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        __nv_bfloat162 r; cudaMemcpy(&r, b, sizeof r, cudaMemcpyDeviceToHost);
        CHECK(__bfloat162float(r.x) == 200.0f && __bfloat162float(r.y) == -200.0f);
        cudaFree(b);
    }
    { // rounding of one add: exact sum rounded to nearest even (half: 11 bits)
        struct { float a, b, want; } cases[] = {
            {1.0f, 1.0f / 2048, 1.0f},                // tie: 1 + half an ulp -> even (1.0)
            {1.0f + 1.0f / 1024, 1.0f / 2048, 1.0f + 2.0f / 1024}, // tie -> even (up)
            {2048.0f, 1.0f, 2048.0f},                 // ulp 2: tie -> even
            {2050.0f, 1.0f, 2052.0f},                 // tie -> even (up)
            {65504.0f, 8.0f, 65504.0f},               // below the overflow threshold (65520)
            {-3.0f, 3.0f, 0.0f},
        };
        __half *h, *old;
        cudaMalloc(&h, sizeof(__half));
        cudaMalloc(&old, sizeof(__half));
        for (auto &c : cases) {
            __half a = __float2half(c.a);
            cudaMemcpy(h, &a, sizeof a, cudaMemcpyHostToDevice);
            one_add<<<1, 1>>>(h, __float2half(c.b), old);
            __half r, o;
            cudaMemcpy(&r, h, sizeof r, cudaMemcpyDeviceToHost);
            cudaMemcpy(&o, old, sizeof o, cudaMemcpyDeviceToHost);
            CHECK(__half2float(r) == c.want);
            CHECK(__half_as_ushort(o) == __half_as_ushort(a));
        }
        cudaFree(h); cudaFree(old);
    }
    cudaFree(d_olds);
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

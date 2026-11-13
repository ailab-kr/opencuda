// A small program for the build-variant tests (variants.cmake): -O0, -g,
// OCU_CPU_LOOPS=off, a wrong plugin path, -ccbin, and objects with and
// without the whole-block pass linked together. Linked with cases_ext.cu.
// Launches: 1 kernel without a barrier (loop when converted), 1 with a
// barrier (always fibers), the inline kernel of inline_kernel.cuh from this
// object and from cases_ext.cu. Prints "variant ok" and exits 0 when all
// results are right.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

void launch_inl_ext(int *o, int n);
#include "inline_kernel.cuh"

__global__ void k_lp_v1(int *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = i * 3 - 1;
}
__global__ void k_fb_v2(int *o)
{
    __shared__ int s[64];
    s[threadIdx.x] = (int)threadIdx.x * 2;
    __syncthreads();
    o[threadIdx.x] = s[63 - threadIdx.x];
}

int main()
{
    int bad = 0;
    int *d = nullptr;
    cudaMalloc((void **)&d, 256 * 4);
    cudaMemset(d, 0, 256 * 4);
    std::vector<int> h(256);
    k_lp_v1<<<4, 64>>>(d, 200);
    cudaMemcpy(h.data(), d, 256 * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 200; ++i) bad += h[i] != i * 3 - 1;
    k_fb_v2<<<1, 64>>>(d);
    cudaMemcpy(h.data(), d, 64 * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 64; ++i) bad += h[i] != (63 - i) * 2;
    launch_inl(d, 128);
    cudaMemcpy(h.data(), d, 128 * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 128; ++i) bad += h[i] != i * 11 + 3;
    cudaMemset(d, 0, 256 * 4);
    launch_inl_ext(d, 128);
    cudaMemcpy(h.data(), d, 128 * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 128; ++i) bad += h[i] != i * 11 + 3;
    cudaFree(d);
    std::printf(bad ? "variant BAD %d\n" : "variant ok\n", bad);
    return bad ? 1 : 0;
}

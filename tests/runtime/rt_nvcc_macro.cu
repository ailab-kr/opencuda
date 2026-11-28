// nvcc defines __NVCC__ (with __CUDACC__) in every pass of a CUDA compile,
// and programs gate their execution-space keywords on it, e.g.
//   #if !defined(__NVCC__) && !defined(__HIPCC__)
//   #define __device__
//   #define __host__
//   #endif
// (HeCBench xsbench-cuda, gpp-cuda). ocu-nvcc defines both in the host and
// the device passes. Same source on every path:
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_nvcc_macro.cu -o rt_nvcc_macro
#include <cuda_runtime.h>

#include <cstdio>

#ifndef __NVCC__
#error "__NVCC__ is not defined in this pass"
#endif
#ifndef __CUDACC__
#error "__CUDACC__ is not defined in this pass"
#endif

#if !defined(__NVCC__) && !defined(__HIPCC__)
#define __device__
#define __host__
#endif

// declared __host__ __device__ through the idiom above, defined later
__host__ __device__ int twice(int x);

__global__ void k(int *o)
{
#ifdef __NVCC__
    o[threadIdx.x] = twice((int)threadIdx.x) + 1000;
#else
    o[threadIdx.x] = -1;
#endif
}

__host__ __device__ int twice(int x) { return 2 * x; }

int main()
{
    int *d, h[32];
    cudaMalloc(&d, sizeof h);
    k<<<1, 32>>>(d);
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < 32; ++i) bad += h[i] != 2 * i + 1000;
    bad += twice(21) != 42; // the same function from host code
    cudaFree(d);
    std::printf("%s: 33 checks, %d failed\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}

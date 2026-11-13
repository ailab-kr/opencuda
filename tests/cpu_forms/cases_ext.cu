// The second translation unit of the cases program: a device function and a
// kernel that cases.cu uses from outside, and the inline kernel of
// inline_kernel.cuh launched from here.
#include <cuda_runtime.h>

#include "inline_kernel.cuh"

__device__ int ext_dev(int x) { return x * 3 + 1; }

__global__ void k_ext(int *o)
{
    o[blockIdx.x * blockDim.x + threadIdx.x] = (int)threadIdx.x * 7 + (int)blockIdx.x;
}

void launch_inl_ext(int *o, int n) { launch_inl(o, n); }

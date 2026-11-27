// Part 2 of rt_grf_split.cu: a __device__ variable written by one kernel and
// read by another must be one variable (this module is built once).
#include <cuda_runtime.h>

__device__ unsigned shared_counter;
__global__ void reset() { shared_counter = 0; }
__global__ void bump() { atomicAdd(&shared_counter, 1u); }
__global__ void read_counter(unsigned *o) { *o = shared_counter; }

unsigned grf_split_counter_test()
{
    unsigned *d = nullptr, h = 0;
    cudaMalloc(&d, sizeof h);
    reset<<<1, 1>>>();
    for (int i = 0; i < 5; ++i) bump<<<2, 32>>>();
    read_counter<<<1, 1>>>(d);
    cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost);
    cudaFree(d);
    return h;
}

// Kernels and a launch in an included file (found next to the including
// source), as many programs are organised.
#pragma once

__global__ void fill_value(int *p, int n, int v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void add_one(int *p, int n)
{
    extern __shared__ int unused_dynamic[]; // rewritten here too
    (void)unused_dynamic;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] += 1;
}

static inline void launch_add_one(int *p, int n)
{
    add_one << < (n + 255) / 256, 256 >> > (p, n); // spaced chevrons, as some editors format them
}

// Kernels in a shared library; the scale factor comes from the build.
#include <cuda_runtime.h>

__global__ void scale_add(int n, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += SCALE * x[i];
}

extern "C" int run_scale_add(int n, const float *x, float *y)
{
    scale_add<<<(n + 127) / 128, 128>>>(n, x, y);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
}

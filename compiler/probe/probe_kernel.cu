#include "probe_defs.h"

__global__ void saxpy(int n, float a, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void block_sum(const float *in, float *out)
{
    __shared__ float tile[256];
    unsigned t = threadIdx.x;
    tile[t] = in[blockIdx.x * blockDim.x + t];
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            tile[t] += tile[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        out[blockIdx.x] = tile[0];
    }
}

void run(int n, float a, const float *x, float *y)
{
    saxpy<<<(n + 255) / 256, 256>>>(n, a, x, y);
}

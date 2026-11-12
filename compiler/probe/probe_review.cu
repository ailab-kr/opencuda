// Probe input: the source patterns the 2026-11-12 review found the CPU path
// mishandled. Written for the probe; every path compiles it unchanged.
#include "probe_defs.h"

// File-scope dynamic shared memory, used by more than one kernel.
extern __shared__ float scratch[];

__global__ void rotate_left(float *data)
{
    int t = threadIdx.x;
    int n = blockDim.x;
    int base = blockIdx.x * n;
    scratch[t] = data[base + t];
    __syncthreads();
    data[base + t] = scratch[(t + 1) % n];
}

// Template kernel launched without explicit template arguments.
template <typename T>
__global__ void fill(T *out, T value, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

// Kernel that takes an optional pointer: NULL means "skip".
__global__ void maybe_scale(float *v, const float *factor, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = factor ? v[i] * factor[0] : -v[i];
}

#define BLOCK 32
#define LAUNCH_FILL(ptr, val, n) \
    fill<<<((n) + BLOCK - 1) / BLOCK, \
           BLOCK>>>(ptr, val, n)

void run_rotate(float *data, int blocks)
{
    wchar_t tag = L'r'; // a prefixed character literal next to a launch
    (void)tag;
    rotate_left /* kernel */ <<<blocks, BLOCK, BLOCK * sizeof(float)>>> /* go */ (data);
}

void run_fill(int *a, double *b, int n)
{
    LAUNCH_FILL(a, 7, n);
    fill<<<(n + 63) / 64, 64>>>(b, 2.5, n);
}

void run_maybe_scale(float *v, const float *factor, int n)
{
    if (factor) {
        maybe_scale<<<(n + 31) / 32, 32>>>(v, factor, n);
    } else {
        maybe_scale<<<(n + 31) / 32, 32>>>(v, NULL, n);
    }
}

// Probe input: language features beyond probe_kernel.cu.
// Written for the probe; the CPU and GPU paths compile it unchanged.
#include "probe_defs.h"

template <typename T>
__device__ T square(T v) { return v * v; }

// Template kernel, 2-D grid and block, __device__ helper.
template <typename T>
__global__ void square_2d(const T *in, T *out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        out[y * width + x] = square(in[y * width + x]);
    }
}

// Dynamic shared memory and a barrier: reverse each block's slice.
__global__ void reverse_block(int *data)
{
    extern __shared__ int buf[];
    int t = threadIdx.x;
    int n = blockDim.x;
    int base = blockIdx.x * n;
    buf[t] = data[base + t];
    __syncthreads();
    data[base + t] = buf[n - 1 - t];
}

// Atomics across blocks: histogram of values 0..15.
__global__ void histogram(const int *values, int n, unsigned *bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&bins[values[i] & 15], 1u);
    }
}

// Struct argument passed by value.
struct Affine { float scale; float shift; };

__global__ void affine(Affine f, float *v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = v[i] * f.scale + f.shift;
}

// Host functions that launch, in the shapes real code uses.
void run_square(const float *in, float *out, int w, int h)
{
    dim3 block(16, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    square_2d<float><<<grid, block>>>(in, out, w, h);
}

void run_square_int(const int *in, int *out, int w, int h)
{
    // Spread over lines, with a comment containing <<<not a launch>>>.
    square_2d<int>
        <<< dim3((w + 15) / 16, (h + 15) / 16),
            dim3(16, 16) >>>
        (in, out,
         w, h);
}

void run_reverse(int *data, int blocks, int threads)
{
    const char *note = "a string with <<<chevrons>>> stays as is";
    (void)note;
    reverse_block<<<blocks, threads, threads * sizeof(int)>>>(data);
}

void run_histogram(const int *values, int n, unsigned *bins)
{
    histogram<<<(n + 127) / 128, 128, 0, 0>>>(values, n, bins);
}

void run_affine(float *v, int n)
{
    Affine f = {2.0f, -1.0f};
    ::affine<<<(n + 63) / 64, 64>>>(f, v, n);
}

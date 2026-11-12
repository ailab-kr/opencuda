// Kernels for the cu2msl translator features (tests/metal/test_metal_features.c
// runs them on an Apple GPU; tests/metal/msl_check.py checks the generated
// MSL on any host). Written for this project.
#include "cu2msl_features.cuh"

#define TILE 64
#ifndef SCALE_NUM
#define SCALE_NUM 1
#endif

enum Mode { MODE_ADD = 1, MODE_MUL = 2 };
static const int kBias = 3;

// ---- device functions called with device, threadgroup and thread pointers ----------
__device__ float sum3(const float *p)
{
    return p[0] + p[1] + p[2];
}

__device__ void bump(float &x, float by)
{
    x += by;
}

// returns a pointer into whatever memory it was given
__device__ float *advance(float *p, int n)
{
    return p + n;
}

__global__ void spaces(const float *in, float *out)
{
    __shared__ float tile[TILE];
    int t = threadIdx.x;
    tile[t] = in[blockIdx.x * TILE + t];
    __syncthreads();
    float local[3] = {1.0f, 2.0f, 3.0f};
    float r = 0.0f;
    if (t + 2 < TILE) {
        r = sum3(&tile[t]) + sum3(in + blockIdx.x * TILE + t) + sum3(local);
    }
    __syncthreads();          // every read of tile is done
    bump(r, 0.5f);            // thread reference
    bump(tile[t], 1.0f);      // threadgroup reference
    float *q = advance(tile, t);
    float *o = advance(out, blockIdx.x * TILE);
    o[t] = r + *q;
}

// ---- templates -----------------------------------------------------------------------
template <typename T, int N, bool Twice>
__global__ void scale_n(T *data, T factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        T v = data[i] * factor;
        data[i] = Twice ? v + v : v;
    }
}

template <typename T>
__device__ T clamp_to(T v, T hi)
{
    return v > hi ? hi : v;
}

__global__ void clamp_all(int *a, float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = clamp_to<int>(a[i], 100);
        b[i] = clamp_to(b[i], 0.5f);
    }
}

void launch_templates(float *f, int *k)
{
    scale_n<float, 1000, true><<<4, 256>>>(f, 1.5f);
    scale_n<int, 500, false><<<2, 256>>>(k, 3);
}

// ---- __constant__ and __device__ variables ------------------------------------------------
__constant__ float c_weights[4] = {0.5f, -1.0f, 2.0f};
__constant__ int c_offset = 7;
__device__ unsigned int d_counter;
__device__ float d_table[8];

__global__ void use_globals(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * c_weights[i % 4] + c_offset + d_table[i % 8];
        if (in[i] > 0.0f) atomicAdd(&d_counter, 1u);
    }
}

// ---- struct arguments with mixed field sizes ------------------------------------------------
__global__ void with_struct(Params p, int n, Pair q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = p.src[i];
        if (p.mode == MODE_ADD) v += p.scale * p.tag + p.shift.y;
        else v *= p.scale;
        p.dst[i] = v + q.a + (float)q.b + kBias + SCALE_NUM;
    }
}

// ---- dynamic shared memory seen from a device function -----------------------------------
__device__ float *scratch()
{
    extern __shared__ float dyn[];
    return dyn;
}

__global__ void reverse_block(const float *in, float *out)
{
    float *s = scratch();
    int t = threadIdx.x;
    s[t] = in[blockIdx.x * blockDim.x + t];
    __syncthreads();
    out[blockIdx.x * blockDim.x + t] = s[blockDim.x - 1 - t];
}

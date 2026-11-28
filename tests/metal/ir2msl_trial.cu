// Kernels for the IR -> MSL trial. k_hard uses what the source translator
// (cu2msl) refuses: a namespace, a class template with a member function,
// if constexpr, a constexpr function that calls itself, and a function
// chosen by a constexpr function.
__global__ void k_saxpy(int n, float a, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

namespace traits {
template <int kind> struct cfg {
    static constexpr int scale = kind == 0 ? 1 : 3;
    float bias;
    __device__ float apply(float v) const { return v * scale + bias; }
};
}
static constexpr __host__ __device__ int steps(int kind, int depth = 0)
{
    return depth >= 2 ? kind + 2 : steps(kind, depth + 1);
}
typedef float (*op_t)(float, float);
static __device__ float op_add(float a, float b) { return a + b; }
static __device__ float op_mul(float a, float b) { return a * b; }
static constexpr __host__ __device__ op_t pick(int kind) { return kind == 0 ? op_add : op_mul; }

template <int kind, bool clampit>
__global__ void k_hard(int n, const float *x, const int *idx, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr int ns = steps(kind);
    constexpr op_t op = pick(kind);
    traits::cfg<kind> c{0.5f};
    float acc = x[idx[i]];
    for (int s = 0; s < ns; ++s) {
        acc = op(acc, c.apply(x[(i + s) % n]));
        if (acc > 1e6f) break;
    }
    if constexpr (clampit) {
        acc = acc < -100.0f ? -100.0f : (acc > 100.0f ? 100.0f : acc);
    }
    switch (i & 3) {
        case 0: acc += 1.0f; break;
        case 1: acc -= 1.0f; break;
        case 2: acc = -acc; break;
        default: break;
    }
    y[i] = acc;
}

// Block sum with shared memory, a barrier and a warp reduction.
__global__ void k_block_sum(const float *x, float *out)
{
    __shared__ float part[8];
    float v = x[blockIdx.x * blockDim.x + threadIdx.x];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off, 32);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (unsigned w = 0; w < blockDim.x / 32; ++w) s += part[w];
        out[blockIdx.x] = s;
    }
}

void host(int n, float *x, int *idx, float *y)
{
    k_saxpy<<<1, 32>>>(n, 2.0f, x, y);
    k_hard<0, false><<<1, 32>>>(n, x, idx, y);
    k_hard<1, true><<<1, 32>>>(n, x, idx, y);
    k_block_sum<<<1, 256>>>(x, y);
}

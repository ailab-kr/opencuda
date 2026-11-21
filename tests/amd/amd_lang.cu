/*
 * Device-language checks for the AMD path (runtime/include/ocu_cuda_lang_amdgcn.h):
 * every kernel writes what each thread saw; tests/amd/check_amd_lang.py
 * compares it with the meaning the CUDA programming guide documents.
 * Plain CUDA, no host code: compiled by tools/amdgpu/ocu_amd_cc.py and run
 * through the HIP runtime.
 */

/* 1. built-in variables: one record of 12 values per thread */
__global__ void builtins(unsigned *out)
{
    unsigned t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned b = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    unsigned *o = out + 12 * (b * blockDim.x * blockDim.y * blockDim.z + t);
    o[0] = threadIdx.x; o[1] = threadIdx.y; o[2] = threadIdx.z;
    o[3] = blockIdx.x;  o[4] = blockIdx.y;  o[5] = blockIdx.z;
    o[6] = blockDim.x;  o[7] = blockDim.y;  o[8] = blockDim.z;
    o[9] = gridDim.x;   o[10] = gridDim.y;  o[11] = gridDim.z;
}

/* 2. shuffles: kind (0 idx, 1 up, 2 down, 3 xor), argument, width */
__global__ void shuffles(int *out, int kind, int arg, int width)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int v = 1000 * blockIdx.x + threadIdx.x;
    int r;
    switch (kind) {
    case 0: r = __shfl_sync(0xffffffffu, v, arg, width); break;
    case 1: r = __shfl_up_sync(0xffffffffu, v, (unsigned)arg, width); break;
    case 2: r = __shfl_down_sync(0xffffffffu, v, (unsigned)arg, width); break;
    default: r = __shfl_xor_sync(0xffffffffu, v, arg, width); break;
    }
    out[t] = r;
}

/* 64-bit and float shuffles */
__global__ void shuffles_wide(long long *o64, float *of)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    long long v = ((long long)(threadIdx.x + 1) << 40) | (unsigned)(7 * threadIdx.x);
    o64[t] = __shfl_xor_sync(0xffffffffu, v, 5);
    of[t] = __shfl_down_sync(0xffffffffu, (float)threadIdx.x * 0.5f, 3);
}

/* 3. votes and warp information */
__global__ void votes(unsigned *out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x % 32;
    int p = (lane % 3) == 0;
    unsigned *o = out + 6 * t;
    o[0] = __ballot_sync(0xffffffffu, p);
    o[1] = __any_sync(0xffffffffu, lane == 17);
    o[2] = __all_sync(0xffffffffu, lane < 32);
    o[3] = __all_sync(0xffffffffu, lane != 9);
    o[4] = __match_any_sync(0xffffffffu, lane / 8);
    o[5] = __activemask();
}

/* 4. atomics on global memory */
__global__ void atomics(int *ci, unsigned *cu, unsigned long long *cull, float *cf, double *cd, unsigned *inc,
                        unsigned *dec, int *mx, int *mn, unsigned *bits, int *cas)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(ci, 1);
    atomicSub(ci + 1, 2);
    atomicAdd(cu, (unsigned)t);
    atomicAdd(cull, 1ull << 33);
    atomicAdd(cf, 0.5f);
    atomicAdd(cd, 0.25);
    atomicInc(inc, 9u);
    atomicDec(dec, 9u);
    atomicMax(mx, t * 7 % 1013);
    atomicMin(mn, -(t * 5 % 977));
    atomicOr(bits, 1u << (t % 32));
    atomicXor(bits + 1, 1u << (t % 5));
    atomicAnd(bits + 2, ~(1u << (t % 7)));
    int seen = *cas;
    while (true) {
        int got = atomicCAS(cas, seen, seen + 3);
        if (got == seen) break;
        seen = got;
    }
}

/* 5. block-wide predicates and shared memory atomics */
__global__ void block_votes(int *out)
{
    __shared__ int s;
    if (threadIdx.x == 0) s = 0;
    __syncthreads();
    atomicAdd(&s, (int)threadIdx.x);
    int c = __syncthreads_count(threadIdx.x % 4 == 1);
    int a = __syncthreads_and(threadIdx.x < 1000);
    int a2 = __syncthreads_and(threadIdx.x != 77);
    int o = __syncthreads_or(threadIdx.x == 130);
    int o2 = __syncthreads_or(0);
    if (threadIdx.x == 0) {
        int *r = out + 6 * blockIdx.x;
        r[0] = c; r[1] = a; r[2] = a2; r[3] = o; r[4] = o2; r[5] = s;
    }
}

/* 6. integer and conversion intrinsics on given inputs */
__global__ void intrinsics(const int *a, const int *b, const float *f, const double *d, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int *o = out + 24 * i;
    unsigned ua = (unsigned)a[i], ub = (unsigned)b[i];
    o[0] = __popc(ua);
    o[1] = __clz(a[i]);
    o[2] = __ffs(a[i]);
    o[3] = (int)__brev(ua);
    o[4] = __mul24(a[i], b[i]);
    o[5] = (int)__umul24(ua, ub);
    o[6] = __mulhi(a[i], b[i]);
    o[7] = (int)__umulhi(ua, ub);
    o[8] = (int)__byte_perm(ua, ub, (unsigned)(i * 0x1357) & 0x7777u);
    o[9] = __float2int_rn(f[i]);
    o[10] = __float2int_rz(f[i]);
    o[11] = __float2int_rd(f[i]);
    o[12] = __float2int_ru(f[i]);
    o[13] = (int)__float2uint_rn(f[i]);
    o[14] = __double2int_rn(d[i]);
    o[15] = __float_as_int(__saturatef(f[i]));
    o[16] = __popcll(((unsigned long long)ua << 32) | ub);
    o[17] = __clzll((long long)a[i]);
    long long mh = __mul64hi((long long)a[i] * 65537, (long long)b[i] * 8191);
    o[18] = (int)(mh & 0xffffffff);
    o[19] = (int)(mh >> 32);
    o[20] = (int)__sad(a[i], b[i], 5u);
    o[21] = min(a[i], b[i]);
    o[22] = max(a[i], b[i]);
    o[23] = __float_as_int(fminf(f[i], 1.5f));
}

/* 7. math functions (OpenCuda's device math library) */
__global__ void mathf(const float *x, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float *o = out + 12 * i;
    o[0] = expf(v);
    o[1] = logf(fabsf(v) + 1e-3f);
    o[2] = sinf(v);
    o[3] = cosf(v);
    o[4] = tanhf(v);
    o[5] = sqrtf(fabsf(v));
    o[6] = powf(fabsf(v) + 0.5f, 1.75f);
    o[7] = atan2f(v, 0.75f);
    o[8] = erff(v);
    o[9] = exp2f(v);
    o[10] = log2f(fabsf(v) + 1e-3f);
    o[11] = fmodf(v * 13.0f, 3.25f);
}

/* 8. a warp-level reduction the way CUDA programs write it */
__global__ void warp_reduce(const float *in, float *out)
{
    float v = in[blockIdx.x * blockDim.x + threadIdx.x];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) out[(blockIdx.x * blockDim.x + threadIdx.x) / 32] = v;
}

/* 9. __device__ variable and atomic on it, template kernel */
__device__ unsigned long long g_total;
template <typename T, int N>
__global__ void sum_into(const T *in)
{
    T acc = 0;
    for (int k = 0; k < N; ++k) acc += in[(blockIdx.x * blockDim.x + threadIdx.x) * N + k];
    atomicAdd(&g_total, (unsigned long long)acc);
}
template __global__ void sum_into<int, 4>(const int *);

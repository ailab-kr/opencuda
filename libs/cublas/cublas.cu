/*
 * The BLAS library API (cublas_v2.h) on the runtime API.
 *
 * One CUDA source for every backend: the host side is chip-neutral C++
 * that only calls the runtime API, and the arithmetic is done by the CUDA
 * kernels below, compiled by the project's toolchain for each target (see
 * libs/cublas/CMakeLists.txt). Nothing here knows which chip it runs on.
 *
 * Semantics follow the public cuBLAS documentation and, where it defers to
 * it, the reference BLAS: column-major matrices; vector increments where a
 * negative value starts at the far end; quick returns for empty problems;
 * alpha == 0 (or k == 0) never reads A and B; beta == 0 never reads C.
 *
 * Precision: every real compute type accumulates in at least the
 * precision it names (16F and 32F accumulate in binary32, 64F in
 * binary64), so results are never less accurate than the documented
 * compute type. Combinations that would need something else (integer and
 * complex types, CUBLAS_COMPUTE_16F_PEDANTIC, which prescribes binary16
 * arithmetic) return CUBLAS_STATUS_NOT_SUPPORTED.
 */
#include <cuda_runtime.h>

#include "cublas_v2.h"

#include <math.h>
#include <new>
#include <stdint.h>
#include <stdlib.h>

#define CB_HD __host__ __device__

struct ocu_cublas_context {
    cudaStream_t stream;
    cublasPointerMode_t pointer_mode;
    cublasAtomicsMode_t atomics;
    cublasMath_t math;
    void *workspace;
    size_t workspace_bytes;
    float *scratch; /* device memory for host-mode scalar results */
    int sm_count;
};

namespace {

/* ---- element types ------------------------------------------------------ */

struct f16 { uint16_t v; };  /* IEEE binary16 bits (CUDA_R_16F, __half) */
struct bf16 { uint16_t v; }; /* bfloat16 bits (CUDA_R_16BF) */

CB_HD inline float bits_to_float(uint32_t b) { return __builtin_bit_cast(float, b); }
CB_HD inline uint32_t float_to_bits(float f) { return __builtin_bit_cast(uint32_t, f); }

/* Exact: every binary16 value is a binary32 value. Moving the exponent and
   mantissa bits into place and scaling by 2^112 is exact for normal and
   subnormal halves alike; infinities and NaNs get the all-ones exponent.
   No branches, which matters inside the GEMM loads. */
CB_HD inline float half_to_float(uint16_t h)
{
    const uint32_t mag = uint32_t(h & 0x7fffu) << 13;
    float v = bits_to_float(mag) * 5.192296858534828e33f; /* 2^112 */
    const uint32_t special = (h & 0x7c00u) == 0x7c00u ? (0x7f800000u | mag) : float_to_bits(v);
    return bits_to_float(special | (uint32_t(h & 0x8000u) << 16));
}

/* Round to nearest even. Normal results: add the rounding bias to the bit
   pattern (carries into the exponent are correct, up to infinity from
   65520 on). Subnormal results: one binary32 addition of 0.5, whose own
   rounding to nearest even is the rounding wanted. Short and nearly
   branch-free: it is inlined many times into the GEMM epilogue. */
CB_HD inline uint16_t float_to_half(float x)
{
    uint32_t f = float_to_bits(x);
    const uint32_t sign = f & 0x80000000u;
    f ^= sign;
    uint32_t o;
    if (f >= 0x47800000u) { /* >= 65536: infinity, or NaN */
        o = f > 0x7f800000u ? 0x7e00u : 0x7c00u;
    } else if (f < 0x38800000u) { /* below 2^-14 */
        o = float_to_bits(bits_to_float(f) + 0.5f) - 0x3f000000u;
    } else {
        f += 0xc8000fffu + ((f >> 13) & 1u); /* rebias exponent, round half to even */
        o = f >> 13;
    }
    return uint16_t(o | (sign >> 16));
}

CB_HD inline float bf16_to_float(uint16_t b) { return bits_to_float(uint32_t(b) << 16); }

CB_HD inline uint16_t float_to_bf16(float f)
{
    uint32_t x = float_to_bits(f);
    if ((x & 0x7fffffffu) > 0x7f800000u) return uint16_t((x >> 16) | 0x40u); /* quiet NaN */
    return uint16_t((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}

template <class Acc> CB_HD inline Acc ld(const float *p) { return Acc(*p); }
template <class Acc> CB_HD inline Acc ld(const double *p) { return Acc(*p); }
template <class Acc> CB_HD inline Acc ld(const f16 *p) { return Acc(half_to_float(p->v)); }
template <class Acc> CB_HD inline Acc ld(const bf16 *p) { return Acc(bf16_to_float(p->v)); }

template <class Acc> CB_HD inline void st(float *p, Acc v) { *p = float(v); }
template <class Acc> CB_HD inline void st(double *p, Acc v) { *p = double(v); }
template <class Acc> CB_HD inline void st(f16 *p, Acc v) { p->v = float_to_half(float(v)); }
template <class Acc> CB_HD inline void st(bf16 *p, Acc v) { p->v = float_to_bf16(float(v)); }

/* ---- GEMM kernel -------------------------------------------------------- */

struct gemm_params {
    int m, n, k;
    int opa, opb; /* 0: as stored, 1: transposed */
    const void *a, *b;
    void *c;
    const void *const *a_arr; /* pointer-array batches (else NULL) */
    const void *const *b_arr;
    void *const *c_arr;
    long long lda, ldb, ldc;
    long long sa, sb, sc; /* strided batches */
    double alpha, beta;              /* host pointer mode */
    const void *alpha_dev, *beta_dev; /* device pointer mode (else NULL) */
    int batch_offset;
};

/*
 * C = alpha op(A) op(B) + beta C for one BM x BN tile of one batch.
 * 256 threads; thread (tx, ty) owns TM x TN outputs strided by the thread
 * grid (rows tx + i*RX, columns ty + j*RY), so shared-memory reads do not
 * conflict and global stores are contiguous along columns. Tiles of A and
 * B are staged in shared memory in the accumulation type, with the load
 * order chosen per transpose so neighbouring threads read neighbouring
 * addresses. Elements outside the matrix read as zero.
 */
template <class TA, class TC, class TS, class Acc, int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(256) gemm_kernel(gemm_params p)
{
    constexpr int RX = BM / TM, RY = BN / TN, NT = RX * RY;
    static_assert(NT == 256, "256 threads per block");
    static_assert((BM * BK) % NT == 0 && (BN * BK) % NT == 0, "whole tiles per thread");
    __shared__ Acc As[BK][BM + 1];
    __shared__ Acc Bs[BK][BN + 1];

    const long long batch = (long long)blockIdx.z + p.batch_offset;
    const TA *A, *B;
    TC *C;
    if (p.a_arr != nullptr) {
        A = static_cast<const TA *>(p.a_arr[batch]);
        B = static_cast<const TA *>(p.b_arr[batch]);
        C = static_cast<TC *>(p.c_arr[batch]);
    } else {
        A = static_cast<const TA *>(p.a) + batch * p.sa;
        B = static_cast<const TA *>(p.b) + batch * p.sb;
        C = static_cast<TC *>(p.c) + batch * p.sc;
    }
    const Acc alpha = p.alpha_dev ? ld<Acc>(static_cast<const TS *>(p.alpha_dev)) : Acc(p.alpha);
    const Acc beta = p.beta_dev ? ld<Acc>(static_cast<const TS *>(p.beta_dev)) : Acc(p.beta);

    const int tid = threadIdx.x, tx = tid % RX, ty = tid / RX;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    Acc acc[TM][TN]; /* fully unrolled loops only: stays in registers */
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = Acc(0);

    const bool product = alpha != Acc(0) && p.k > 0; /* same for the whole block */
    constexpr int LA = BM * BK / NT, LB = BN * BK / NT;
    /* The next tiles, read while the current one is used. Kept as stored
       (converted when written to shared memory), so the loads stay in
       flight during the arithmetic instead of waiting for a conversion. */
    TA ra[LA], rb[LB];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int r = 0; r < LA; ++r) {
            const int idx = tid + r * NT;
            int i, kk;
            if (p.opa == 0) { i = idx % BM; kk = idx / BM; }
            else { kk = idx % BK; i = idx / BK; }
            const int gi = m0 + i, gk = k0 + kk;
            ra[r] = gi < p.m && gk < p.k ? A[p.opa == 0 ? gi + gk * p.lda : gk + gi * p.lda] : TA{};
        }
#pragma unroll
        for (int r = 0; r < LB; ++r) {
            const int idx = tid + r * NT;
            int j, kk;
            if (p.opb == 0) { kk = idx % BK; j = idx / BK; }
            else { j = idx % BN; kk = idx / BN; }
            const int gj = n0 + j, gk = k0 + kk;
            rb[r] = gj < p.n && gk < p.k ? B[p.opb == 0 ? gk + gj * p.ldb : gj + gk * p.ldb] : TA{};
        }
    };
    if (product) {
        fetch(0);
        for (int k0 = 0; k0 < p.k; k0 += BK) {
#pragma unroll
            for (int r = 0; r < LA; ++r) {
                const int idx = tid + r * NT;
                if (p.opa == 0) As[idx / BM][idx % BM] = ld<Acc>(&ra[r]);
                else As[idx % BK][idx / BK] = ld<Acc>(&ra[r]);
            }
#pragma unroll
            for (int r = 0; r < LB; ++r) {
                const int idx = tid + r * NT;
                if (p.opb == 0) Bs[idx % BK][idx / BK] = ld<Acc>(&rb[r]);
                else Bs[idx / BN][idx % BN] = ld<Acc>(&rb[r]);
            }
            __syncthreads();
            if (k0 + BK < p.k) fetch(k0 + BK);
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                Acc a[TM], b[TN];
#pragma unroll
                for (int i = 0; i < TM; ++i) a[i] = As[kk][tx + i * RX];
#pragma unroll
                for (int j = 0; j < TN; ++j) b[j] = Bs[kk][ty + j * RY];
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int gj = n0 + ty + j * RY;
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int gi = m0 + tx + i * RX;
            if (gi < p.m && gj < p.n) {
                TC *out = C + gi + gj * p.ldc;
                Acc r = product ? alpha * acc[i][j] : Acc(0);
                if (beta != Acc(0)) r += beta * ld<Acc>(out); /* beta == 0: C is not read */
                st<Acc>(out, r);
            }
        }
    }
}

/* ---- level 1 and 2 kernels ---------------------------------------------- */

CB_HD inline long long vec_index(long long i, int n, int inc)
{
    return inc >= 0 ? i * inc : (long long)(n - 1 - i) * -(long long)inc;
}

__global__ void axpy_kernel(int n, float alpha, const float *alpha_dev, const float *x,
                            int incx, float *y, int incy)
{
    const float a = alpha_dev ? *alpha_dev : alpha;
    if (a == 0.0f) return; /* reference BLAS: y unchanged */
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        float *yi = y + vec_index(i, n, incy);
        *yi = a * x[vec_index(i, n, incx)] + *yi;
    }
}

__global__ void scal_kernel(int n, float alpha, const float *alpha_dev, float *x, int incx)
{
    const float a = alpha_dev ? *alpha_dev : alpha;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        x[i * incx] *= a;
    }
}

constexpr int RED_THREADS = 256;

/* One block: each thread sums a strided share, then a tree in shared memory. */
__global__ void dot_kernel(int n, const float *x, int incx, const float *y, int incy,
                           float *result)
{
    __shared__ float part[RED_THREADS];
    float s = 0.0f;
    for (long long i = threadIdx.x; i < n; i += RED_THREADS)
        s += x[vec_index(i, n, incx)] * y[vec_index(i, n, incy)];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int w = RED_THREADS / 2; w > 0; w /= 2) {
        if ((int)threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = part[0];
}

/*
 * Euclidean norm without overflow or underflow: each thread keeps a scale
 * (largest |x| seen) and a sum of squares relative to it, as LAPACK's
 * slassq does; pairs are merged in shared memory. Infinities and NaNs are
 * tracked separately (NaN wins).
 */
__global__ void nrm2_kernel(int n, const float *x, int incx, float *result)
{
    __shared__ float sc[RED_THREADS], sq[RED_THREADS];
    __shared__ int special[RED_THREADS]; /* bit 0: inf seen, bit 1: NaN seen */
    float scale = 0.0f, ssq = 1.0f;
    int flags = 0;
    for (long long i = threadIdx.x; i < n; i += RED_THREADS) {
        const float v = x[i * incx];
        const float a = v < 0.0f ? -v : v;
        if (a != a) { flags |= 2; continue; }
        if (a > 3.402823466e38f) { flags |= 1; continue; }
        if (a == 0.0f) continue;
        if (scale < a) {
            const float r = scale / a;
            ssq = 1.0f + ssq * r * r;
            scale = a;
        } else {
            const float r = a / scale;
            ssq += r * r;
        }
    }
    sc[threadIdx.x] = scale;
    sq[threadIdx.x] = ssq;
    special[threadIdx.x] = flags;
    __syncthreads();
    for (int w = RED_THREADS / 2; w > 0; w /= 2) {
        if ((int)threadIdx.x < w) {
            const int o = threadIdx.x + w;
            float s1 = sc[threadIdx.x], q1 = sq[threadIdx.x], s2 = sc[o], q2 = sq[o];
            if (s2 > s1) { float t = s1; s1 = s2; s2 = t; t = q1; q1 = q2; q2 = t; }
            if (s2 > 0.0f) { const float r = s2 / s1; q1 += q2 * r * r; }
            sc[threadIdx.x] = s1;
            sq[threadIdx.x] = q1;
            special[threadIdx.x] |= special[o];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int f = special[0];
        float r;
        if (f & 2) r = bits_to_float(0x7fc00000u);
        else if (f & 1) r = bits_to_float(0x7f800000u);
        else r = sc[0] == 0.0f ? 0.0f : sc[0] * sqrtf(sq[0]);
        *result = r;
    }
}

/* y = alpha A x + beta y, one thread per row. */
__global__ void gemv_n_kernel(int m, int n, float alpha, const float *alpha_dev,
                              const float *a, long long lda, const float *x, int incx,
                              float beta, const float *beta_dev, float *y, int incy)
{
    const float al = alpha_dev ? *alpha_dev : alpha;
    const float be = beta_dev ? *beta_dev : beta;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    float s = 0.0f;
    if (al != 0.0f)
        for (int j = 0; j < n; ++j) s += a[i + j * lda] * x[vec_index(j, n, incx)];
    float *yi = y + vec_index(i, m, incy);
    float r = al != 0.0f ? al * s : 0.0f;
    if (be != 0.0f) r += be * *yi;
    *yi = r;
}

/* y = alpha A^T x + beta y, one block per output element. */
__global__ void gemv_t_kernel(int m, int n, float alpha, const float *alpha_dev,
                              const float *a, long long lda, const float *x, int incx,
                              float beta, const float *beta_dev, float *y, int incy)
{
    __shared__ float part[RED_THREADS];
    const float al = alpha_dev ? *alpha_dev : alpha;
    const float be = beta_dev ? *beta_dev : beta;
    const long long j = blockIdx.x;
    float s = 0.0f;
    if (al != 0.0f)
        for (int i = threadIdx.x; i < m; i += RED_THREADS)
            s += a[i + j * lda] * x[vec_index(i, m, incx)];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int w = RED_THREADS / 2; w > 0; w /= 2) {
        if ((int)threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float *yj = y + vec_index(j, n, incy);
        float r = al != 0.0f ? al * part[0] : 0.0f;
        if (be != 0.0f) r += be * *yj;
        *yj = r;
    }
}

/*
 * Triangular solve, one batch per blockIdx.z (+ offset), one thread per
 * independent right-hand side: a column of B (left side: op(A) X = alpha B)
 * or a row of B (right side: X op(A) = alpha B). Plain substitution in the
 * order the triangle requires; X overwrites B. alpha == 0 sets B to zero
 * without reading A or B, as the reference BLAS does.
 */
struct trsm_params {
    int left, upper_op, unit; /* upper_op: op(A) is upper triangular */
    int trans;                /* op(A)(i, j) = trans ? A(j, i) : A(i, j) */
    int m, n;
    float alpha;
    const float *alpha_dev;
    const float *const *a_arr;
    float *const *b_arr;
    long long lda, ldb;
    int batch_offset;
};

__global__ void trsm_kernel(trsm_params p)
{
    const long long batch = (long long)blockIdx.z + p.batch_offset;
    const float *A = p.a_arr[batch];
    float *B = p.b_arr[batch];
    const float alpha = p.alpha_dev ? *p.alpha_dev : p.alpha;
    const int rhs = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = p.left ? p.m : p.n; /* order of A */
    if (rhs >= (p.left ? p.n : p.m)) return;
    /* element e of this right-hand side */
    const long long base = p.left ? (long long)rhs * p.ldb : rhs;
    const long long step = p.left ? 1 : p.ldb;
#define OCU_T(i, j) (p.trans ? A[(j) + (long long)(i) * p.lda] : A[(i) + (long long)(j) * p.lda])
    if (alpha == 0.0f) {
        for (int e = 0; e < size; ++e) B[base + e * step] = 0.0f;
        return;
    }
    for (int e = 0; e < size; ++e) B[base + e * step] *= alpha;
    /* Left: T x = b. Right: x T = b, i.e. sum_i x_i T(i, j) = b_j. */
    const bool forward = p.left ? !p.upper_op : p.upper_op;
    for (int s = 0; s < size; ++s) {
        const int e = forward ? s : size - 1 - s;
        float v = B[base + e * step];
        if (forward) {
            for (int q = 0; q < e; ++q) v -= (p.left ? OCU_T(e, q) : OCU_T(q, e)) * B[base + q * step];
        } else {
            for (int q = e + 1; q < size; ++q) v -= (p.left ? OCU_T(e, q) : OCU_T(q, e)) * B[base + q * step];
        }
        if (!p.unit) v /= OCU_T(e, e);
        B[base + e * step] = v;
    }
#undef OCU_T
}

/* ---- host side ---------------------------------------------------------- */

cublasStatus_t from_cuda(cudaError_t e)
{
    switch (e) {
    case cudaSuccess: return CUBLAS_STATUS_SUCCESS;
    case cudaErrorMemoryAllocation: return CUBLAS_STATUS_ALLOC_FAILED;
    case cudaErrorNotSupported: return CUBLAS_STATUS_NOT_SUPPORTED;
    default: return CUBLAS_STATUS_EXECUTION_FAILED;
    }
}

bool device_mode(cublasHandle_t h) { return h->pointer_mode == CUBLAS_POINTER_MODE_DEVICE; }

/* Optional override for tuning: OCU_CUBLAS_TILE=64 or 128. */
int tile_override()
{
    static const int v = [] {
        const char *s = getenv("OCU_CUBLAS_TILE");
        return s ? atoi(s) : 0;
    }();
    return v;
}

template <class TA, class TC, class TS, class Acc, int BM, int BN, int BK, int TM, int TN>
cublasStatus_t launch_gemm_cfg(cublasHandle_t h, gemm_params p, int batch)
{
    const long long gx = (p.m + BM - 1) / BM, gy = (p.n + BN - 1) / BN;
    if (gx > 0x7fffffffLL || gy > 65535) return CUBLAS_STATUS_NOT_SUPPORTED;
    for (int off = 0; off < batch; off += 65535) {
        p.batch_offset = off;
        const int nz = batch - off < 65535 ? batch - off : 65535;
        void *args[] = {&p};
        cudaError_t e = cudaLaunchKernel(gemm_kernel<TA, TC, TS, Acc, BM, BN, BK, TM, TN>,
                                         dim3((unsigned)gx, (unsigned)gy, (unsigned)nz),
                                         dim3(256), args, 0, h->stream);
        if (e != cudaSuccess) return from_cuda(e);
    }
    return CUBLAS_STATUS_SUCCESS;
}

/* Large tiles (more arithmetic per load) once they keep at least half the
   multiprocessors busy; small ones for small problems. Measured on a 70-SM
   GPU: 1024^2 is 1.7x faster with large tiles (64 blocks), 512^2 2x faster
   with small ones (16 large blocks), 768^2 the same. */
template <class TA, class TC, class TS, class Acc>
cublasStatus_t launch_gemm(cublasHandle_t h, const gemm_params &p, int batch)
{
    const long long big_tiles = (long long)((p.m + 127) / 128) * ((p.n + 127) / 128) * batch;
    bool big = p.m >= 128 && p.n >= 128 && 2 * big_tiles >= h->sm_count;
    if (tile_override() == 64) big = false;
    if (tile_override() == 128) big = true;
    if (big) return launch_gemm_cfg<TA, TC, TS, Acc, 128, 128, 16, 8, 8>(h, p, batch);
    return launch_gemm_cfg<TA, TC, TS, Acc, 64, 64, 16, 4, 4>(h, p, batch);
}

enum class combo { none, s, d, h16, h_s, h_f, bf_s, bf_f };

bool valid_op(cublasOperation_t o)
{
    return o == CUBLAS_OP_N || o == CUBLAS_OP_T || o == CUBLAS_OP_C; /* C == T for real types */
}

bool valid_algo(cublasGemmAlgo_t a)
{
    const int v = (int)a;
    return (v >= -1 && v <= 23) || (v >= 99 && v <= 115);
}

combo pick(cudaDataType_t at, cudaDataType_t bt, cudaDataType_t ct, int compute)
{
    /* older cudaDataType_t compute values */
    if (compute == CUDA_R_16F) compute = CUBLAS_COMPUTE_16F;
    else if (compute == CUDA_R_32F) compute = CUBLAS_COMPUTE_32F;
    else if (compute == CUDA_R_64F) compute = CUBLAS_COMPUTE_64F;
    if (at != bt) return combo::none;
    switch (compute) {
    case CUBLAS_COMPUTE_16F:
        return at == CUDA_R_16F && ct == CUDA_R_16F ? combo::h16 : combo::none;
    case CUBLAS_COMPUTE_32F:
    case CUBLAS_COMPUTE_32F_PEDANTIC:
        if (at == CUDA_R_32F && ct == CUDA_R_32F) return combo::s;
        if (at == CUDA_R_16F && ct == CUDA_R_16F) return combo::h_s;
        if (at == CUDA_R_16F && ct == CUDA_R_32F) return combo::h_f;
        if (at == CUDA_R_16BF && ct == CUDA_R_16BF) return combo::bf_s;
        if (at == CUDA_R_16BF && ct == CUDA_R_32F) return combo::bf_f;
        return combo::none;
    case CUBLAS_COMPUTE_32F_FAST_16F:
    case CUBLAS_COMPUTE_32F_FAST_16BF:
    case CUBLAS_COMPUTE_32F_FAST_TF32:
        return at == CUDA_R_32F && ct == CUDA_R_32F ? combo::s : combo::none;
#ifndef OCU_CUBLAS_NO_FP64
    case CUBLAS_COMPUTE_64F:
    case CUBLAS_COMPUTE_64F_PEDANTIC:
        return at == CUDA_R_64F && ct == CUDA_R_64F ? combo::d : combo::none;
#endif
    default:
        return combo::none; /* 16F_PEDANTIC, integer and complex compute */
    }
}

enum class scale { f32, f64, f16 };

scale scale_of(combo c)
{
    return c == combo::d ? scale::f64 : c == combo::h16 ? scale::f16 : scale::f32;
}

double host_scalar(scale s, const void *p)
{
    switch (s) {
    case scale::f64: return *static_cast<const double *>(p);
    case scale::f16: return half_to_float(static_cast<const f16 *>(p)->v);
    default: return *static_cast<const float *>(p);
    }
}

/* Every GEMM entry point ends here. Pointer arrays (a_arr...) or strides. */
cublasStatus_t gemm_any(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k, const void *alpha, const void *a,
                        const void *const *a_arr, cudaDataType_t at, int lda, long long sa,
                        const void *b, const void *const *b_arr, cudaDataType_t bt, int ldb,
                        long long sb, const void *beta, void *c, void *const *c_arr,
                        cudaDataType_t ct, int ldc, long long sc, int batch, int compute,
                        cublasGemmAlgo_t algo)
{
    if (h == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!valid_op(transa) || !valid_op(transb) || m < 0 || n < 0 || k < 0 || batch < 0 ||
        !valid_algo(algo))
        return CUBLAS_STATUS_INVALID_VALUE;
    const int opa = transa == CUBLAS_OP_N ? 0 : 1, opb = transb == CUBLAS_OP_N ? 0 : 1;
    const int arows = opa == 0 ? m : k, brows = opb == 0 ? k : n;
    if (lda < (arows > 1 ? arows : 1) || ldb < (brows > 1 ? brows : 1) || ldc < (m > 1 ? m : 1))
        return CUBLAS_STATUS_INVALID_VALUE;
    const combo cb = pick(at, bt, ct, compute);
    if (cb == combo::none) return CUBLAS_STATUS_NOT_SUPPORTED;
    if (alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0 || batch == 0) return CUBLAS_STATUS_SUCCESS;

    gemm_params p{};
    p.m = m; p.n = n; p.k = k; p.opa = opa; p.opb = opb;
    p.a = a; p.b = b; p.c = c; p.a_arr = a_arr; p.b_arr = b_arr; p.c_arr = c_arr;
    p.lda = lda; p.ldb = ldb; p.ldc = ldc; p.sa = sa; p.sb = sb; p.sc = sc;
    if (device_mode(h)) {
        p.alpha_dev = alpha;
        p.beta_dev = beta;
    } else {
        p.alpha = host_scalar(scale_of(cb), alpha);
        p.beta = host_scalar(scale_of(cb), beta);
        if ((p.alpha == 0.0 || k == 0) && p.beta == 1.0) return CUBLAS_STATUS_SUCCESS;
    }
    switch (cb) {
    case combo::s: return launch_gemm<float, float, float, float>(h, p, batch);
#ifndef OCU_CUBLAS_NO_FP64
    case combo::d: return launch_gemm<double, double, double, double>(h, p, batch);
#endif
    case combo::h16: return launch_gemm<f16, f16, f16, float>(h, p, batch);
    case combo::h_s: return launch_gemm<f16, f16, float, float>(h, p, batch);
    case combo::h_f: return launch_gemm<f16, float, float, float>(h, p, batch);
    case combo::bf_s: return launch_gemm<bf16, bf16, float, float>(h, p, batch);
    case combo::bf_f: return launch_gemm<bf16, float, float, float>(h, p, batch);
    default: return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}

/* Scalar results: straight to the caller's device memory, or through the
   handle's scratch to host memory (then the call waits for the result). */
cublasStatus_t finish_scalar(cublasHandle_t h, float *result, float *dev)
{
    if (device_mode(h)) return CUBLAS_STATUS_SUCCESS;
    cudaError_t e = cudaMemcpyAsync(result, dev, sizeof(float), cudaMemcpyDeviceToHost, h->stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(h->stream);
    return from_cuda(e);
}

cublasStatus_t zero_scalar(cublasHandle_t h, float *result)
{
    if (!device_mode(h)) {
        *result = 0.0f;
        return CUBLAS_STATUS_SUCCESS;
    }
    return from_cuda(cudaMemsetAsync(result, 0, sizeof(float), h->stream));
}

unsigned blocks_for(long long n)
{
    long long b = (n + 255) / 256;
    return (unsigned)(b > 4096 ? 4096 : b < 1 ? 1 : b);
}

} // namespace

/* ---- helper functions --------------------------------------------------- */

extern "C" {

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle)
{
    if (handle == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    *handle = nullptr;
    ocu_cublas_context *h = new (std::nothrow) ocu_cublas_context();
    if (h == nullptr) return CUBLAS_STATUS_ALLOC_FAILED;
    h->stream = nullptr;
    h->pointer_mode = CUBLAS_POINTER_MODE_HOST;
    h->atomics = CUBLAS_ATOMICS_NOT_ALLOWED;
    h->math = CUBLAS_DEFAULT_MATH;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&h->scratch), 256);
    if (e != cudaSuccess) {
        delete h;
        if (e == cudaErrorMemoryAllocation) return CUBLAS_STATUS_ALLOC_FAILED;
        return CUBLAS_STATUS_NOT_INITIALIZED; /* no device, no driver */
    }
    int dev = 0, sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        sms < 1)
        sms = 1;
    h->sm_count = sms;
    *handle = h;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    cudaError_t e = cudaFree(handle->scratch); /* waits for work using it */
    delete handle;
    return e == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (version == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    *version = CUBLAS_VERSION;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value)
{
    if (value == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    switch (type) {
    case MAJOR_VERSION: *value = CUBLAS_VER_MAJOR; return CUBLAS_STATUS_SUCCESS;
    case MINOR_VERSION: *value = CUBLAS_VER_MINOR; return CUBLAS_STATUS_SUCCESS;
    case PATCH_LEVEL: *value = CUBLAS_VER_PATCH; return CUBLAS_STATUS_SUCCESS;
    default: return CUBLAS_STATUS_INVALID_VALUE;
    }
}

size_t cublasGetCudartVersion(void) { return CUDART_VERSION; }

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    handle->stream = streamId;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (streamId == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    *streamId = handle->stream;
    return CUBLAS_STATUS_SUCCESS;
}

/* Accepted and kept; the kernels here need no workspace. */
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace,
                                     size_t workspaceSizeInBytes)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    handle->workspace = workspace;
    handle->workspace_bytes = workspace ? workspaceSizeInBytes : 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (mode == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = handle->pointer_mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (mode != CUBLAS_POINTER_MODE_HOST && mode != CUBLAS_POINTER_MODE_DEVICE)
        return CUBLAS_STATUS_INVALID_VALUE;
    handle->pointer_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (mode == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = handle->atomics;
    return CUBLAS_STATUS_SUCCESS;
}

/* Kept; no kernel here uses atomics, so results never depend on it. */
cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (mode != CUBLAS_ATOMICS_NOT_ALLOWED && mode != CUBLAS_ATOMICS_ALLOWED)
        return CUBLAS_STATUS_INVALID_VALUE;
    handle->atomics = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (mode == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = handle->math;
    return CUBLAS_STATUS_SUCCESS;
}

/* Kept. Modes only permit reduced precision; the kernels here always use
   the full precision of the compute type, which every mode allows. */
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    const int base = (int)mode & ~(int)CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
    if (base < CUBLAS_DEFAULT_MATH || base > CUBLAS_TF32_TENSOR_OP_MATH)
        return CUBLAS_STATUS_INVALID_VALUE;
    handle->math = mode;
    return CUBLAS_STATUS_SUCCESS;
}

const char *cublasGetStatusName(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "CUBLAS_STATUS_UNKNOWN";
}

const char *cublasGetStatusString(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "the library was not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "resource allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "an invalid value was passed";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "the device does not support this operation";
    case CUBLAS_STATUS_MAPPING_ERROR: return "access to device memory failed";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "the device program failed to execute";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "an internal operation failed";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "the functionality requested is not supported";
    case CUBLAS_STATUS_LICENSE_ERROR: return "license error";
    }
    return "unknown error";
}

/* ---- level 1 ------------------------------------------------------------ */

cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha,
                              const float *x, int incx, float *y, int incy)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (alpha == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;
    if (incy == 0 && n > 1) return CUBLAS_STATUS_NOT_SUPPORTED; /* every element writes y[0] */
    float a = 0.0f;
    const float *a_dev = nullptr;
    if (device_mode(handle)) a_dev = alpha;
    else if ((a = *alpha) == 0.0f) return CUBLAS_STATUS_SUCCESS;
    void *args[] = {&n, &a, &a_dev, &x, &incx, &y, &incy};
    return from_cuda(cudaLaunchKernel(axpy_kernel, dim3(blocks_for(n)), dim3(256), args, 0,
                                      handle->stream));
}

cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha, float *x,
                              int incx)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (alpha == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n <= 0 || incx <= 0) return CUBLAS_STATUS_SUCCESS; /* reference BLAS */
    float a = 0.0f;
    const float *a_dev = nullptr;
    if (device_mode(handle)) a_dev = alpha;
    else a = *alpha;
    void *args[] = {&n, &a, &a_dev, &x, &incx};
    return from_cuda(cudaLaunchKernel(scal_kernel, dim3(blocks_for(n)), dim3(256), args, 0,
                                      handle->stream));
}

cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx,
                             const float *y, int incy, float *result)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (result == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n <= 0) return zero_scalar(handle, result);
    float *dev = device_mode(handle) ? result : handle->scratch;
    void *args[] = {&n, &x, &incx, &y, &incy, &dev};
    cudaError_t e = cudaLaunchKernel(dot_kernel, dim3(1), dim3(RED_THREADS), args, 0,
                                     handle->stream);
    if (e != cudaSuccess) return from_cuda(e);
    return finish_scalar(handle, result, dev);
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx,
                              float *result)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (result == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n <= 0 || incx <= 0) return zero_scalar(handle, result);
    float *dev = device_mode(handle) ? result : handle->scratch;
    void *args[] = {&n, &x, &incx, &dev};
    cudaError_t e = cudaLaunchKernel(nrm2_kernel, dim3(1), dim3(RED_THREADS), args, 0,
                                     handle->stream);
    if (e != cudaSuccess) return from_cuda(e);
    return finish_scalar(handle, result, dev);
}

/* ---- level 2 ------------------------------------------------------------ */

cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                              const float *alpha, const float *A, int lda, const float *x,
                              int incx, const float *beta, float *y, int incy)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!valid_op(trans) || m < 0 || n < 0 || lda < (m > 1 ? m : 1) || incx == 0 || incy == 0 ||
        alpha == nullptr || beta == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    float al = 0.0f, be = 0.0f;
    const float *al_dev = nullptr, *be_dev = nullptr;
    if (device_mode(handle)) {
        al_dev = alpha;
        be_dev = beta;
    } else {
        al = *alpha;
        be = *beta;
        if (al == 0.0f && be == 1.0f) return CUBLAS_STATUS_SUCCESS;
    }
    long long ld = lda;
    void *args[] = {&m, &n, &al, &al_dev, &A, &ld, &x, &incx, &be, &be_dev, &y, &incy};
    cudaError_t e;
    if (trans == CUBLAS_OP_N) {
        e = cudaLaunchKernel(gemv_n_kernel, dim3((unsigned)((m + 255) / 256)), dim3(256), args, 0,
                             handle->stream);
    } else {
        e = cudaLaunchKernel(gemv_t_kernel, dim3((unsigned)n), dim3(RED_THREADS), args, 0,
                             handle->stream);
    }
    return from_cuda(e);
}

/* ---- level 3 ------------------------------------------------------------ */

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo, cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n, const float *alpha,
                                  const float *const A[], int lda, float *const B[], int ldb,
                                  int batchCount)
{
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    const int order = side == CUBLAS_SIDE_LEFT ? m : n;
    if ((side != CUBLAS_SIDE_LEFT && side != CUBLAS_SIDE_RIGHT) ||
        (uplo != CUBLAS_FILL_MODE_LOWER && uplo != CUBLAS_FILL_MODE_UPPER) || !valid_op(trans) ||
        (diag != CUBLAS_DIAG_NON_UNIT && diag != CUBLAS_DIAG_UNIT) || m < 0 || n < 0 ||
        batchCount < 0 || lda < (order > 1 ? order : 1) || ldb < (m > 1 ? m : 1) ||
        alpha == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0 || batchCount == 0) return CUBLAS_STATUS_SUCCESS;
    trsm_params p{};
    p.left = side == CUBLAS_SIDE_LEFT;
    p.trans = trans != CUBLAS_OP_N;
    p.upper_op = (uplo == CUBLAS_FILL_MODE_UPPER) != (p.trans != 0);
    p.unit = diag == CUBLAS_DIAG_UNIT;
    p.m = m;
    p.n = n;
    if (device_mode(handle)) p.alpha_dev = alpha;
    else p.alpha = *alpha;
    p.a_arr = A;
    p.b_arr = B;
    p.lda = lda;
    p.ldb = ldb;
    const int rhs = p.left ? n : m;
    for (int off = 0; off < batchCount; off += 65535) {
        p.batch_offset = off;
        const int nz = batchCount - off < 65535 ? batchCount - off : 65535;
        void *args[] = {&p};
        cudaError_t e = cudaLaunchKernel(trsm_kernel, dim3((unsigned)((rhs + 63) / 64), 1, (unsigned)nz),
                                         dim3(64), args, 0, handle->stream);
        if (e != cudaSuccess) return from_cuda(e);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda, const float *B,
                              int ldb, const float *beta, float *C, int ldc)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, A, nullptr, CUDA_R_32F, lda, 0, B,
                    nullptr, CUDA_R_32F, ldb, 0, beta, C, nullptr, CUDA_R_32F, ldc, 0, 1,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const double *alpha, const double *A, int lda,
                              const double *B, int ldb, const double *beta, double *C,
                              int ldc)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, A, nullptr, CUDA_R_64F, lda, 0, B,
                    nullptr, CUDA_R_64F, ldb, 0, beta, C, nullptr, CUDA_R_64F, ldc, 0, 1,
                    CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const __half *alpha, const __half *A, int lda, const __half *B,
                           int ldb, const __half *beta, __half *C, int ldc)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, A, nullptr, CUDA_R_16F, lda, 0, B,
                    nullptr, CUDA_R_16F, ldb, 0, beta, C, nullptr, CUDA_R_16F, ldc, 0, 1,
                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
}

#define OCU_ARR(p) reinterpret_cast<const void *const *>(p)
#define OCU_OUT_ARR(p) reinterpret_cast<void *const *>(p)

cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const float *alpha, const float *const Aarray[], int lda,
                                  const float *const Barray[], int ldb, const float *beta,
                                  float *const Carray[], int ldc, int batchCount)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, nullptr, OCU_ARR(Aarray),
                    CUDA_R_32F, lda, 0, nullptr, OCU_ARR(Barray), CUDA_R_32F, ldb, 0, beta,
                    nullptr, OCU_OUT_ARR(Carray), CUDA_R_32F, ldc, 0, batchCount,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const double *alpha, const double *const Aarray[],
                                  int lda, const double *const Barray[], int ldb,
                                  const double *beta, double *const Carray[], int ldc,
                                  int batchCount)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, nullptr, OCU_ARR(Aarray),
                    CUDA_R_64F, lda, 0, nullptr, OCU_ARR(Barray), CUDA_R_64F, ldb, 0, beta,
                    nullptr, OCU_OUT_ARR(Carray), CUDA_R_64F, ldc, 0, batchCount,
                    CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const __half *alpha, const __half *const Aarray[], int lda,
                                  const __half *const Barray[], int ldb, const __half *beta,
                                  __half *const Carray[], int ldc, int batchCount)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, nullptr, OCU_ARR(Aarray),
                    CUDA_R_16F, lda, 0, nullptr, OCU_ARR(Barray), CUDA_R_16F, ldb, 0, beta,
                    nullptr, OCU_OUT_ARR(Carray), CUDA_R_16F, ldc, 0, batchCount,
                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const float *alpha, const float *A, int lda,
                                         long long int strideA, const float *B, int ldb,
                                         long long int strideB, const float *beta, float *C,
                                         int ldc, long long int strideC, int batchCount)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, A, nullptr, CUDA_R_32F, lda,
                    strideA, B, nullptr, CUDA_R_32F, ldb, strideB, beta, C, nullptr,
                    CUDA_R_32F, ldc, strideC, batchCount, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const double *alpha, const double *A, int lda,
                                         long long int strideA, const double *B, int ldb,
                                         long long int strideB, const double *beta,
                                         double *C, int ldc, long long int strideC,
                                         int batchCount)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, A, nullptr, CUDA_R_64F, lda,
                    strideA, B, nullptr, CUDA_R_64F, ldb, strideB, beta, C, nullptr,
                    CUDA_R_64F, ldc, strideC, batchCount, CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const __half *alpha, const __half *A, int lda,
                                         long long int strideA, const __half *B, int ldb,
                                         long long int strideB, const __half *beta,
                                         __half *C, int ldc, long long int strideC,
                                         int batchCount)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, A, nullptr, CUDA_R_16F, lda,
                    strideA, B, nullptr, CUDA_R_16F, ldb, strideB, beta, C, nullptr,
                    CUDA_R_16F, ldc, strideC, batchCount, CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void *alpha, const void *A, cudaDataType_t Atype, int lda,
                            const void *B, cudaDataType_t Btype, int ldb, const void *beta,
                            void *C, cudaDataType_t Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t algo)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, A, nullptr, Atype, lda, 0, B,
                    nullptr, Btype, ldb, 0, beta, C, nullptr, Ctype, ldc, 0, 1,
                    (int)computeType, algo);
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                   cublasOperation_t transb, int m, int n, int k,
                                   const void *alpha, const void *const Aarray[],
                                   cudaDataType_t Atype, int lda, const void *const Barray[],
                                   cudaDataType_t Btype, int ldb, const void *beta,
                                   void *const Carray[], cudaDataType_t Ctype, int ldc,
                                   int batchCount, cublasComputeType_t computeType,
                                   cublasGemmAlgo_t algo)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, nullptr, Aarray, Atype, lda, 0,
                    nullptr, Barray, Btype, ldb, 0, beta, nullptr, Carray, Ctype, ldc, 0,
                    batchCount, (int)computeType, algo);
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                          cublasOperation_t transb, int m, int n, int k,
                                          const void *alpha, const void *A,
                                          cudaDataType_t Atype, int lda,
                                          long long int strideA, const void *B,
                                          cudaDataType_t Btype, int ldb,
                                          long long int strideB, const void *beta, void *C,
                                          cudaDataType_t Ctype, int ldc,
                                          long long int strideC, int batchCount,
                                          cublasComputeType_t computeType,
                                          cublasGemmAlgo_t algo)
{
    return gemm_any(handle, transa, transb, m, n, k, alpha, A, nullptr, Atype, lda, strideA, B,
                    nullptr, Btype, ldb, strideB, beta, C, nullptr, Ctype, ldc, strideC,
                    batchCount, (int)computeType, algo);
}

} /* extern "C" */

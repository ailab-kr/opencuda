/*
 * ocu_cuda_lang_spirv.h -- CUDA language support for SPIR-V devices through
 * OpenCL (Clang in HIP mode, --offload=spirv64 device pass; compiler/spirv).
 *
 * Adapted from ocu_cuda_lang_amdgcn.h: the same CUDA surface, with the AMD
 * builtins replaced by OpenCL C built-in functions (get_local_id, barrier,
 * sub_group_shuffle, sub_group_ballot, native_exp2 ...) declared with their
 * C++ (Itanium) names, which llvm-spirv turns into SPIR-V built-ins and
 * instructions. Atomics use Clang's scoped atomic builtins (SPIR-V scopes).
 *
 * Warp functions need 32-wide sub-groups: compiler/spirv/cu2spv.py gives
 * every kernel that uses sub-group operations intel_reqd_sub_group_size(32).
 */
#ifndef OCU_CUDA_LANG_SPIRV_H
#define OCU_CUDA_LANG_SPIRV_H

#if defined(__HIP_DEVICE_COMPILE__)
#define OCU_SPV_DEVICE 1
/* CUDA programs select device code with __CUDA_ARCH__. The AMD device pass
   reports the CUDA feature level it implements (OCU_SPV_CUDA_ARCH, default
   800: warp-synchronous intrinsics, atomics on 64-bit values, fp16). */
#ifndef OCU_SPV_CUDA_ARCH
#define OCU_SPV_CUDA_ARCH 800 /* the build's -arch sets it (compiler/spirv) */
#endif
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ OCU_SPV_CUDA_ARCH
#endif
#endif
#ifndef __CUDACC__
#define __CUDACC__ 1
#endif

/*
 * Device math first, as on the NVPTX path: in the device pass the
 * compiler's __builtin_<name> math calls (used by standard library headers,
 * when there are any) are pointed at the project's own library.
 */
#define OCU_DM_FN static inline __attribute__((device))
#define OCU_DMF_TABLE static __attribute__((device))
#include "ocu_device_math.h"
#if defined(OCU_SPV_DEVICE)
#define __builtin_acos(x) ::ocu_dm::acos(x)
#define __builtin_acosf(x) ::ocu_dm::acosf(x)
#define __builtin_asin(x) ::ocu_dm::asin(x)
#define __builtin_asinf(x) ::ocu_dm::asinf(x)
#define __builtin_atan(x) ::ocu_dm::atan(x)
#define __builtin_atanf(x) ::ocu_dm::atanf(x)
#define __builtin_cos(x) ::ocu_dm::cos(x)
#define __builtin_cosf(x) ::ocu_dm::cosf(x)
#define __builtin_cosh(x) ::ocu_dm::cosh(x)
#define __builtin_coshf(x) ::ocu_dm::coshf(x)
#define __builtin_exp(x) ::ocu_dm::exp(x)
#define __builtin_expf(x) ::ocu_dm::expf(x)
#define __builtin_log(x) ::ocu_dm::log(x)
#define __builtin_logf(x) ::ocu_dm::logf(x)
#define __builtin_log10(x) ::ocu_dm::log10(x)
#define __builtin_log10f(x) ::ocu_dm::log10f(x)
#define __builtin_sin(x) ::ocu_dm::sin(x)
#define __builtin_sinf(x) ::ocu_dm::sinf(x)
#define __builtin_sinh(x) ::ocu_dm::sinh(x)
#define __builtin_sinhf(x) ::ocu_dm::sinhf(x)
#define __builtin_tan(x) ::ocu_dm::tan(x)
#define __builtin_tanf(x) ::ocu_dm::tanf(x)
#define __builtin_tanh(x) ::ocu_dm::tanh(x)
#define __builtin_tanhf(x) ::ocu_dm::tanhf(x)
#define __builtin_cbrt(x) ::ocu_dm::cbrt(x)
#define __builtin_cbrtf(x) ::ocu_dm::cbrtf(x)
#define __builtin_erff(x) ::ocu_dm::erff(x)
#define __builtin_erfcf(x) ::ocu_dm::erfcf(x)
#define __builtin_exp2(x) ::ocu_dm::exp2(x)
#define __builtin_exp2f(x) ::ocu_dm::exp2f(x)
#define __builtin_expm1(x) ::ocu_dm::expm1(x)
#define __builtin_expm1f(x) ::ocu_dm::expm1f(x)
#define __builtin_log1p(x) ::ocu_dm::log1p(x)
#define __builtin_log1pf(x) ::ocu_dm::log1pf(x)
#define __builtin_log2(x) ::ocu_dm::log2(x)
#define __builtin_log2f(x) ::ocu_dm::log2f(x)
#define __builtin_atan2(x, y) ::ocu_dm::atan2(x, y)
#define __builtin_atan2f(x, y) ::ocu_dm::atan2f(x, y)
#define __builtin_fmod(x, y) ::ocu_dm::fmod(x, y)
#define __builtin_fmodf(x, y) ::ocu_dm::fmodf(x, y)
#define __builtin_hypot(x, y) ::ocu_dm::hypot(x, y)
#define __builtin_hypotf(x, y) ::ocu_dm::hypotf(x, y)
#define __builtin_pow(x, y) ::ocu_dm::pow(x, y)
#define __builtin_powf(x, y) ::ocu_dm::powf(x, y)
#define __builtin_ldexp(x, n) ::ocu_dm::ldexp(x, n)
#define __builtin_ldexpf(x, n) ((float)::ocu_dm::ldexp((double)(x), n))
#endif

/* The C and C++ library headers a .cu file expects, where the machine has
   them. The AMD device pass needs none of them: a machine with only the
   GPU driver still compiles kernels. The driver's compiler library carries
   a few libc++ headers but not the library (<cstddef> is there and fails,
   <utility> is not), so the C++ library counts as present only with
   <utility>. */
#if __has_include(<utility>) && __has_include(<type_traits>)
#define OCU_SPV_HAVE_CXXLIB 1
#endif
#if __has_include(<math.h>) && defined(OCU_SPV_HAVE_CXXLIB)
#include <math.h>
#endif
#if __has_include(<stdio.h>)
#include <stdio.h>
#endif
#if __has_include(<stdlib.h>)
#include <stdlib.h>
#endif
#if __has_include(<string.h>)
#include <string.h>
#endif
#if defined(OCU_SPV_HAVE_CXXLIB)
#include <cstdint>
#include <cstddef>
#include <climits>
#include <new>
#include <type_traits>
#include <utility>
#endif
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

/* ---- declaration specifiers -------------------------------------------- */

#define __host__        __attribute__((host))
#define __device__      __attribute__((device))
#define __global__      __attribute__((global))
#define __shared__      __attribute__((shared))
#define __constant__    __attribute__((constant))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __noinline__
#define __align__(n)    __attribute__((aligned(n)))
#define __restrict__    __restrict
#define __launch_bounds__(...)
#define OCU_LB_FIRST(n, ...) (n)

#define OCU_D  static __device__ inline __attribute__((always_inline))
#define OCU_HD static __host__ __device__ inline

#define OCU_LANG_FN static __device__ inline __attribute__((always_inline))
#include "ocu_cuda_lang_common.h"

/* ---- launch ABI the compiler looks up for <<<>>> in HIP mode -------------
   Only declared: in the device pass host code is parsed, never generated. */
extern "C" int __hipPushCallConfiguration(dim3 grid, dim3 block, size_t shared = 0, void *stream = 0);
extern "C" int __hipPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shared, void *stream);
extern "C" int hipConfigureCall(dim3 grid, dim3 block, size_t shared = 0, void *stream = 0);

/* ---- OpenCL built-in functions (device pass only) ------------------------------- */

#if defined(OCU_SPV_DEVICE)
typedef unsigned int ocu_spv_u4 __attribute__((ext_vector_type(4)));
__device__ unsigned long get_local_id(unsigned);
__device__ unsigned long get_group_id(unsigned);
__device__ unsigned long get_local_size(unsigned);
__device__ unsigned long get_num_groups(unsigned);
__device__ unsigned get_sub_group_local_id(void);
__device__ void barrier(unsigned);
__device__ void sub_group_barrier(unsigned);
__device__ int sub_group_shuffle(int, unsigned);
__device__ ocu_spv_u4 sub_group_ballot(int);
__device__ float native_exp2(float);
__device__ float native_log2(float);
__device__ float native_sin(float);
__device__ float native_cos(float);
__device__ float native_recip(float);
__device__ long mul_hi(long, long);
__device__ unsigned long mul_hi(unsigned long, unsigned long);
/* cl_khr_integer_dot_product: the GPU's DP4A instruction (no saturation, as __dp4a). */
__device__ int dot_4x8packed_ss_int(unsigned, unsigned);
__device__ unsigned dot_4x8packed_uu_uint(unsigned, unsigned);
#define OCU_SPV_DIM_x 0
#define OCU_SPV_DIM_y 1
#define OCU_SPV_DIM_z 2
#endif

/* ---- built-in variables ----------------------------------------------------- */

#if defined(OCU_SPV_DEVICE)
#define OCU_AMD_TID(d) ((unsigned)get_local_id(OCU_SPV_DIM_##d))
#define OCU_AMD_BID(d) ((unsigned)get_group_id(OCU_SPV_DIM_##d))
#define OCU_AMD_BDIM(d) ((unsigned)get_local_size(OCU_SPV_DIM_##d))
#define OCU_AMD_GDIM(d) ((unsigned)get_num_groups(OCU_SPV_DIM_##d))
#else
#define OCU_AMD_TID(d) 0u /* host pass: never executed */
#define OCU_AMD_BID(d) 0u
#define OCU_AMD_BDIM(d) 0u
#define OCU_AMD_GDIM(d) 0u
#endif

#define OCU_BUILTIN_VAR(TYPE, VAR, GET, CONV)                                      \
    struct TYPE {                                                                  \
        __declspec(property(get = get_x)) unsigned x;                              \
        __declspec(property(get = get_y)) unsigned y;                              \
        __declspec(property(get = get_z)) unsigned z;                              \
        static __device__ __forceinline__ unsigned get_x() { return GET(x); }      \
        static __device__ __forceinline__ unsigned get_y() { return GET(y); }      \
        static __device__ __forceinline__ unsigned get_z() { return GET(z); }      \
        __device__ __forceinline__ operator CONV() const { return CONV{get_x(), get_y(), get_z()}; } \
    };                                                                             \
    extern const __device__ TYPE VAR;

OCU_BUILTIN_VAR(ocu_thread_idx_t, threadIdx, OCU_AMD_TID, uint3)
OCU_BUILTIN_VAR(ocu_block_idx_t, blockIdx, OCU_AMD_BID, uint3)
OCU_BUILTIN_VAR(ocu_block_dim_t, blockDim, OCU_AMD_BDIM, dim3)
OCU_BUILTIN_VAR(ocu_grid_dim_t, gridDim, OCU_AMD_GDIM, dim3)
#undef OCU_BUILTIN_VAR

static constexpr int warpSize = 32;

/* ---- synchronisation and fences ---------------------------------------------- */

OCU_D void __syncthreads()
{
#if defined(OCU_SPV_DEVICE)
    barrier(3u);
#endif
}
OCU_D void __threadfence_block()
{
#if defined(OCU_SPV_DEVICE)
    __scoped_atomic_thread_fence(__ATOMIC_SEQ_CST, __MEMORY_SCOPE_WRKGRP);
#endif
}
OCU_D void __threadfence()
{
#if defined(OCU_SPV_DEVICE)
    __scoped_atomic_thread_fence(__ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
#endif
}
OCU_D void __threadfence_system()
{
#if defined(OCU_SPV_DEVICE)
    __scoped_atomic_thread_fence(__ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM);
#endif
}
[[noreturn]] OCU_D void __trap() { __builtin_trap(); }

/* ---- device library pieces the programs call ------------------------------------ */

/* Declared so that host code parses; a device call has no definition and
   stops the link (device printf is not provided on this path). */
/* Device printf is not provided on this path yet: a device-side overload that
   prints nothing keeps such code building (the host's printf is untouched). */
static __device__ inline int printf(const char *, ...) { return 0; }
/* memcpy / memset in device code (the C library's are host functions) */
static __device__ inline void *memcpy(void *d, const void *s, size_t n) { return __builtin_memcpy(d, s, n); }
static __device__ inline void *memset(void *d, int v, size_t n) { return __builtin_memset(d, v, n); }
OCU_D void *ocu_amd_memcpy(void *d, const void *s, size_t n) { return __builtin_memcpy(d, s, n); }
OCU_D void *ocu_amd_memset(void *d, int v, size_t n) { return __builtin_memset(d, v, n); }

/* ---- atomics (device scope, relaxed, as documented) ------------------------------ */

#define OCU_AMD_AGENT __HIP_MEMORY_SCOPE_AGENT
#define OCU_ATOMIC_INT(T)                                                                          \
    OCU_D T atomicAdd(T *a, T v) { return __hip_atomic_fetch_add(a, v, __ATOMIC_RELAXED, OCU_AMD_AGENT); } \
    OCU_D T atomicSub(T *a, T v) { return __hip_atomic_fetch_add(a, (T)(0 - v), __ATOMIC_RELAXED, OCU_AMD_AGENT); } \
    OCU_D T atomicExch(T *a, T v) { return __hip_atomic_exchange(a, v, __ATOMIC_RELAXED, OCU_AMD_AGENT); } \
    OCU_D T atomicAnd(T *a, T v) { return __hip_atomic_fetch_and(a, v, __ATOMIC_RELAXED, OCU_AMD_AGENT); } \
    OCU_D T atomicOr(T *a, T v) { return __hip_atomic_fetch_or(a, v, __ATOMIC_RELAXED, OCU_AMD_AGENT); }  \
    OCU_D T atomicXor(T *a, T v) { return __hip_atomic_fetch_xor(a, v, __ATOMIC_RELAXED, OCU_AMD_AGENT); } \
    OCU_D T atomicMin(T *a, T v) { return __hip_atomic_fetch_min(a, v, __ATOMIC_RELAXED, OCU_AMD_AGENT); } \
    OCU_D T atomicMax(T *a, T v) { return __hip_atomic_fetch_max(a, v, __ATOMIC_RELAXED, OCU_AMD_AGENT); } \
    OCU_D T atomicCAS(T *a, T c, T v)                                                              \
    {                                                                                              \
        __hip_atomic_compare_exchange_strong(a, &c, v, __ATOMIC_RELAXED, __ATOMIC_RELAXED, OCU_AMD_AGENT); \
        return c;                                                                                  \
    }
OCU_ATOMIC_INT(int)
OCU_ATOMIC_INT(unsigned int)
OCU_ATOMIC_INT(unsigned long long)
OCU_ATOMIC_INT(long long)
#undef OCU_ATOMIC_INT
OCU_D float atomicAdd(float *a, float v) { return __hip_atomic_fetch_add(a, v, __ATOMIC_RELAXED, OCU_AMD_AGENT); }
OCU_D double atomicAdd(double *a, double v) { return __hip_atomic_fetch_add(a, v, __ATOMIC_RELAXED, OCU_AMD_AGENT); }
OCU_D float atomicExch(float *a, float v)
{
    return __builtin_bit_cast(float, atomicExch(reinterpret_cast<int *>(a), __builtin_bit_cast(int, v)));
}
/* ((old >= v) ? 0 : old + 1) and (((old == 0) || (old > v)) ? v : old - 1), atomically */
OCU_D unsigned atomicInc(unsigned *a, unsigned v)
{
#if defined(OCU_SPV_DEVICE)
    unsigned old = *(volatile unsigned *)a, assumed;
    do { assumed = old; old = atomicCAS(a, assumed, assumed >= v ? 0u : assumed + 1u); } while (old != assumed);
    return old;
#else
    return 0;
#endif
}
OCU_D unsigned atomicDec(unsigned *a, unsigned v)
{
#if defined(OCU_SPV_DEVICE)
    unsigned old = *(volatile unsigned *)a, assumed;
    do { assumed = old; old = atomicCAS(a, assumed, (assumed == 0u || assumed > v) ? v : assumed - 1u); } while (old != assumed);
    return old;
#else
    return 0;
#endif
}

/* ---- warp functions (wave32) ------------------------------------------------------- */

OCU_D unsigned __ocu_lane()
{
#if defined(OCU_SPV_DEVICE)
    return get_sub_group_local_id();
#else
    return 0;
#endif
}
OCU_D unsigned __ocu_ballot(bool p)
{
#if defined(OCU_SPV_DEVICE)
    return sub_group_ballot(p ? 1 : 0).x;
#else
    return 0;
#endif
}
/* The lanes executing this instruction together. */
OCU_D unsigned __activemask() { return __ocu_ballot(true); }
OCU_D void __syncwarp(unsigned = 0xffffffffu)
{
#if defined(OCU_SPV_DEVICE)
    sub_group_barrier(3u);
#endif
}
OCU_D unsigned __ballot_sync(unsigned m, int p) { return __ocu_ballot(p != 0) & m; }
OCU_D int __any_sync(unsigned m, int p) { return (__ocu_ballot(p != 0) & m) != 0; }
OCU_D int __all_sync(unsigned m, int p) { return (__ocu_ballot(p == 0) & m) == 0; }
/* The lanes of `m` whose value equals this lane's. */
OCU_D unsigned __match_any_sync(unsigned m, unsigned v)
{
    unsigned left = m & __activemask(), same = 0;
    while (left) {
        unsigned first = (unsigned)__builtin_ctz(left);
        unsigned theirs;
#if defined(OCU_SPV_DEVICE)
        theirs = (unsigned)sub_group_shuffle((int)v, first);
#else
        theirs = v;
#endif
        unsigned group = __ocu_ballot(v == theirs) & left;
        if (v == theirs) same = group;
        left &= ~group;
    }
    return same;
}

/* One 32-bit word from lane `src` (any lane of the wave). */
OCU_D int __ocu_permute(int src, int x)
{
#if defined(OCU_SPV_DEVICE)
    return sub_group_shuffle(x, (unsigned)src);
#else
    return x;
#endif
}

/* The lane a shuffle reads, as documented for each kind; `w` is the segment
   width (a power of two up to 32). An out-of-range source reads the
   calling lane's own value. */
OCU_D int __ocu_shfl_src(int kind, int arg, int w)
{
    int lane = (int)__ocu_lane();
    int base = lane & ~(w - 1);
    switch (kind) {
    case 0: return base + (arg & (w - 1));                                  /* idx */
    case 1: return (lane - base) < arg ? lane : lane - arg;                 /* up */
    case 2: return (lane - base) + arg >= w ? lane : lane + arg;            /* down */
    default: {                                                              /* xor */
        int t = lane ^ arg;
        return t >= base + w ? lane : t;
    }
    }
}

template <typename T>
OCU_D T __ocu_shfl(int kind, T v, int arg, int w)
{
    static_assert(sizeof(T) <= 4 || sizeof(T) == 8, "shuffles move values of at most 8 bytes");
    int src = __ocu_shfl_src(kind, arg, w);
    if constexpr (sizeof(T) == 4) {
        return __builtin_bit_cast(T, __ocu_permute(src, __builtin_bit_cast(int, v)));
    } else if constexpr (sizeof(T) < 4) {
        int x = 0;
        __builtin_memcpy(&x, &v, sizeof(T));
        x = __ocu_permute(src, x);
        T r;
        __builtin_memcpy(&r, &x, sizeof(T));
        return r;
    } else {
        long long x = __builtin_bit_cast(long long, v);
        int lo = __ocu_permute(src, (int)(x & 0xffffffff)), hi = __ocu_permute(src, (int)(x >> 32));
        return __builtin_bit_cast(T, ((long long)hi << 32) | (unsigned)lo);
    }
}
template <typename T>
OCU_D T __shfl_sync(unsigned, T v, int src, int width = 32) { return __ocu_shfl(0, v, src, width); }
template <typename T>
OCU_D T __shfl_up_sync(unsigned, T v, unsigned d, int width = 32) { return __ocu_shfl(1, v, (int)d, width); }
template <typename T>
OCU_D T __shfl_down_sync(unsigned, T v, unsigned d, int width = 32) { return __ocu_shfl(2, v, (int)d, width); }
template <typename T>
OCU_D T __shfl_xor_sync(unsigned, T v, int x, int width = 32) { return __ocu_shfl(3, v, x, width); }

/* ---- block-wide predicates ------------------------------------------------------------ */

OCU_D int __ocu_sync_reduce(int p, int kind)
{
    __shared__ int ocu_sync_word;
    unsigned t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    __syncthreads();
    if (t == 0) ocu_sync_word = kind == 1 ? 1 : 0;
    __syncthreads();
    if (kind == 0 && p) __hip_atomic_fetch_add(&ocu_sync_word, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    if (kind == 1 && !p) __hip_atomic_store(&ocu_sync_word, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    if (kind == 2 && p) __hip_atomic_store(&ocu_sync_word, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    __syncthreads();
    int r = ocu_sync_word;
    __syncthreads();
    return r;
}
OCU_D int __syncthreads_count(int p) { return __ocu_sync_reduce(p, 0); }
OCU_D int __syncthreads_and(int p) { return __ocu_sync_reduce(p, 1); }
OCU_D int __syncthreads_or(int p) { return __ocu_sync_reduce(p, 2); }

/* ---- integer and bit intrinsics ----------------------------------------------------- */

OCU_D int __popc(unsigned x) { return __builtin_popcount(x); }
OCU_D int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
OCU_D int __clz(int x) { return x == 0 ? 32 : __builtin_clz((unsigned)x); }
OCU_D int __clzll(long long x) { return x == 0 ? 64 : __builtin_clzll((unsigned long long)x); }
OCU_D int __ffs(int x) { return __builtin_ffs(x); }
OCU_D int __ffsll(long long x) { return __builtin_ffsll(x); }
OCU_D unsigned __brev(unsigned x) { return __builtin_bitreverse32(x); }
OCU_D unsigned long long __brevll(unsigned long long x) { return __builtin_bitreverse64(x); }
/* the low 24 bits of each operand (sign-extended for the signed form) */
OCU_D int __mul24(int a, int b) { return ((a << 8) >> 8) * ((b << 8) >> 8); }
OCU_D unsigned __umul24(unsigned a, unsigned b) { return (a & 0xffffffu) * (b & 0xffffffu); }
OCU_D int __mulhi(int a, int b) { return (int)(((long long)a * b) >> 32); }
OCU_D unsigned __umulhi(unsigned a, unsigned b) { return (unsigned)(((unsigned long long)a * b) >> 32); }
OCU_D int __dp4a(int a, int b, int c) { return c + dot_4x8packed_ss_int((unsigned)a, (unsigned)b); }
OCU_D unsigned __dp4a(unsigned a, unsigned b, unsigned c) { return c + dot_4x8packed_uu_uint(a, b); }
OCU_D long long __mul64hi(long long a, long long b) { return (long long)mul_hi((long)a, (long)b); }
OCU_D unsigned long long __umul64hi(unsigned long long a, unsigned long long b)
{
    return (unsigned long long)mul_hi((unsigned long)a, (unsigned long)b);
}
/* Byte i of the result is byte (s >> 4i) & 7 of the eight bytes {b:a}. */
OCU_D unsigned __byte_perm(unsigned a, unsigned b, unsigned s)
{
    unsigned long long x = ((unsigned long long)b << 32) | a;
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) r |= (unsigned)((x >> (8 * ((s >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
    return r;
}
OCU_D int __float_as_int(float x) { return __builtin_bit_cast(int, x); }
OCU_D unsigned __float_as_uint(float x) { return __builtin_bit_cast(unsigned, x); }
OCU_D float __int_as_float(int x) { return __builtin_bit_cast(float, x); }
OCU_D float __uint_as_float(unsigned x) { return __builtin_bit_cast(float, x); }
OCU_D long long __double_as_longlong(double x) { return __builtin_bit_cast(long long, x); }
OCU_D double __longlong_as_double(long long x) { return __builtin_bit_cast(double, x); }
OCU_D float __saturatef(float x) { return x > 0.0f ? (x < 1.0f ? x : 1.0f) : 0.0f; } /* NaN -> 0 */
/* |a - b| in unsigned arithmetic: the difference of two ints fits in 32 unsigned bits */
OCU_D unsigned __sad(int a, int b, unsigned c)
{
    return (a > b ? (unsigned)a - (unsigned)b : (unsigned)b - (unsigned)a) + c;
}
OCU_D unsigned __usad(unsigned a, unsigned b, unsigned c) { return (a > b ? a - b : b - a) + c; }

/* Conversions to an integer saturate to the destination's range; NaN gives 0
   (documented behaviour of the cvt the CUDA intrinsics name). */
OCU_D int __ocu_sat_i32(double r)
{
    if (r != r) return 0;
    if (r >= 2147483647.0) return 2147483647;
    if (r <= -2147483648.0) return (-2147483647 - 1);
    return (int)r;
}
OCU_D unsigned __ocu_sat_u32(double r)
{
    if (r != r || r <= 0.0) return 0u;
    if (r >= 4294967295.0) return 4294967295u;
    return (unsigned)r;
}
OCU_D int __float2int_rn(float x) { return __ocu_sat_i32(__builtin_rintf(x)); }
OCU_D int __float2int_rz(float x) { return __ocu_sat_i32(__builtin_truncf(x)); }
OCU_D int __float2int_rd(float x) { return __ocu_sat_i32(__builtin_floorf(x)); }
OCU_D int __float2int_ru(float x) { return __ocu_sat_i32(__builtin_ceilf(x)); }
OCU_D unsigned __float2uint_rn(float x) { return __ocu_sat_u32(__builtin_rintf(x)); }
OCU_D unsigned __float2uint_rz(float x) { return __ocu_sat_u32(__builtin_truncf(x)); }
OCU_D unsigned __float2uint_rd(float x) { return __ocu_sat_u32(__builtin_floorf(x)); }
OCU_D unsigned __float2uint_ru(float x) { return __ocu_sat_u32(__builtin_ceilf(x)); }
OCU_D float __int2float_rn(int x) { return (float)x; }
OCU_D float __uint2float_rn(unsigned x) { return (float)x; }
OCU_D int __double2int_rn(double x) { return __ocu_sat_i32(__builtin_rint(x)); }
OCU_D int __double2int_rz(double x) { return __ocu_sat_i32(__builtin_trunc(x)); }
OCU_D int __double2int_rd(double x) { return __ocu_sat_i32(__builtin_floor(x)); }
OCU_D int __double2int_ru(double x) { return __ocu_sat_i32(__builtin_ceil(x)); }

/* ---- min / max (host and device, as in the documented math API) ------------------------ */

OCU_HD int min(int a, int b) { return a < b ? a : b; }
OCU_HD int max(int a, int b) { return a > b ? a : b; }
OCU_HD unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
OCU_HD unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
OCU_HD unsigned min(int a, unsigned b) { return min(static_cast<unsigned>(a), b); }
OCU_HD unsigned min(unsigned a, int b) { return min(a, static_cast<unsigned>(b)); }
OCU_HD unsigned max(int a, unsigned b) { return max(static_cast<unsigned>(a), b); }
OCU_HD unsigned max(unsigned a, int b) { return max(a, static_cast<unsigned>(b)); }
OCU_HD long min(long a, long b) { return a < b ? a : b; }
OCU_HD long max(long a, long b) { return a > b ? a : b; }
OCU_HD unsigned long min(unsigned long a, unsigned long b) { return a < b ? a : b; }
OCU_HD unsigned long max(unsigned long a, unsigned long b) { return a > b ? a : b; }
OCU_HD long long min(long long a, long long b) { return a < b ? a : b; }
OCU_HD long long max(long long a, long long b) { return a > b ? a : b; }
OCU_HD unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
OCU_HD unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
OCU_HD float min(float a, float b) { return __builtin_fminf(a, b); }
OCU_HD float max(float a, float b) { return __builtin_fmaxf(a, b); }
OCU_HD double min(double a, double b) { return __builtin_fmin(a, b); }
OCU_HD double max(double a, double b) { return __builtin_fmax(a, b); }
OCU_HD double min(float a, double b) { return __builtin_fmin(a, b); }
OCU_HD double min(double a, float b) { return __builtin_fmin(a, b); }
OCU_HD double max(float a, double b) { return __builtin_fmax(a, b); }
OCU_HD double max(double a, float b) { return __builtin_fmax(a, b); }

/* ---- device math ---------------------------------------------------------------------- */

/* Functions computed exactly by hardware operations (IEEE sqrt: Clang's
   HIP default rounds float division and square root correctly). */
#define OCU_EXACT1(name, b)                                                           \
    OCU_D double name(double x) { return __builtin_##b(x); }                          \
    OCU_D float name##f(float x) { return __builtin_##b##f(x); }
OCU_EXACT1(sqrt, sqrt)
OCU_EXACT1(fabs, fabs)
OCU_EXACT1(floor, floor)
OCU_EXACT1(ceil, ceil)
OCU_EXACT1(trunc, trunc)
OCU_EXACT1(round, round)
OCU_EXACT1(rint, rint)
OCU_EXACT1(nearbyint, rint)
#undef OCU_EXACT1
OCU_D double fmin(double a, double b) { return __builtin_fmin(a, b); }
OCU_D float fminf(float a, float b) { return __builtin_fminf(a, b); }
OCU_D double fmax(double a, double b) { return __builtin_fmax(a, b); }
OCU_D float fmaxf(float a, float b) { return __builtin_fmaxf(a, b); }
OCU_D double fma(double a, double b, double c) { return __builtin_fma(a, b, c); }
OCU_D float fmaf(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
OCU_D double copysign(double a, double b) { return __builtin_copysign(a, b); }
OCU_D float copysignf(float a, float b) { return __builtin_copysignf(a, b); }
OCU_D double fdim(double a, double b) { return a > b ? a - b : 0.0; }
OCU_D float fdimf(float a, float b) { return a > b ? a - b : 0.0f; }
OCU_D double ldexp(double x, int n) { return ocu_dm::ldexp(x, n); }
OCU_D float ldexpf(float x, int n) { return (float)ocu_dm::ldexp(x, n); }
OCU_D double frexp(double x, int *e)
{
    if (x == 0.0 || !ocu_dm::isfinite(x)) { *e = 0; return x; }
    return ocu_dm::frexp(x, e);
}
OCU_D float frexpf(float x, int *e) { return (float)frexp((double)x, e); }
OCU_D double rsqrt(double x) { return 1.0 / __builtin_sqrt(x); }
OCU_D float rsqrtf(float x) { return 1.0f / __builtin_sqrtf(x); }
OCU_D long lround(double x) { return (long)__builtin_round(x); }
OCU_D long lroundf(float x) { return (long)__builtin_roundf(x); }
OCU_D long long llround(double x) { return (long long)__builtin_round(x); }
OCU_D long lrint(double x) { return (long)__builtin_rint(x); }
OCU_D int abs(int x) { return x < 0 ? -x : x; }
OCU_D long labs(long x) { return x < 0 ? -x : x; }
OCU_D long long llabs(long long x) { return x < 0 ? -x : x; }

#define OCU_LIB1(name)                                                                \
    OCU_D double name(double x) { return ocu_dm::name(x); }                           \
    OCU_D float name##f(float x) { return ocu_dm::name##f(x); }
OCU_LIB1(exp)
OCU_LIB1(expm1)
OCU_LIB1(exp2)
OCU_LIB1(exp10)
OCU_LIB1(log)
OCU_LIB1(log2)
OCU_LIB1(log10)
OCU_LIB1(log1p)
OCU_LIB1(sin)
OCU_LIB1(cos)
OCU_LIB1(tan)
OCU_LIB1(asin)
OCU_LIB1(acos)
OCU_LIB1(atan)
OCU_LIB1(sinh)
OCU_LIB1(cosh)
OCU_LIB1(tanh)
OCU_LIB1(cbrt)
#undef OCU_LIB1
OCU_D double pow(double x, double y) { return ocu_dm::pow(x, y); }
OCU_D double pow(double x, int y) { return ocu_dm::pow(x, (double)y); }
OCU_D float pow(float x, int y) { return ocu_dm::powf(x, (float)y); }
OCU_D double pow(int x, int y) { return ocu_dm::pow((double)x, (double)y); }
OCU_D double pow(int x, double y) { return ocu_dm::pow((double)x, y); }
OCU_D double pow(double x, float y) { return ocu_dm::pow(x, (double)y); }
OCU_D double pow(float x, double y) { return ocu_dm::pow((double)x, y); }
OCU_D float powf(float x, float y) { return ocu_dm::powf(x, y); }
OCU_D double atan2(double y, double x) { return ocu_dm::atan2(y, x); }
OCU_D float atan2f(float y, float x) { return ocu_dm::atan2f(y, x); }
OCU_D double hypot(double x, double y) { return ocu_dm::hypot(x, y); }
OCU_D float hypotf(float x, float y) { return ocu_dm::hypotf(x, y); }
OCU_D double fmod(double x, double y) { return ocu_dm::fmod(x, y); }
OCU_D float fmodf(float x, float y) { return ocu_dm::fmodf(x, y); }
OCU_D void sincos(double x, double *s, double *c) { *s = ocu_dm::sin(x); *c = ocu_dm::cos(x); }
OCU_D void sincosf(float x, float *s, float *c) { ocu_dm::sincosf(x, s, c); }
OCU_D float erff(float x) { return ocu_dm::erff(x); }
OCU_D float erfcf(float x) { return ocu_dm::erfcf(x); }

/* Fast approximate intrinsics (documented as approximate): the GPU's own
   approximate instructions. v_sin/v_cos take the angle in turns. */
#if defined(OCU_SPV_DEVICE)
OCU_D float __expf(float x) { return native_exp2(x * 1.44269504088896341f); }
OCU_D float __exp10f(float x) { return native_exp2(x * 3.32192809488736235f); }
OCU_D float __logf(float x) { return native_log2(x) * 0.693147180559945309f; }
OCU_D float __log2f(float x) { return native_log2(x); }
OCU_D float __log10f(float x) { return native_log2(x) * 0.301029995663981195f; }
OCU_D float __sinf(float x) { return native_sin(x); }
OCU_D float __cosf(float x) { return native_cos(x); }
OCU_D float __powf(float x, float y) { return native_exp2(y * native_log2(x)); }
OCU_D float __fdividef(float x, float y) { return x * native_recip(y); }
#else
OCU_D float __expf(float x) { return ocu_dm::expf(x); }
OCU_D float __exp10f(float x) { return ocu_dm::exp10f(x); }
OCU_D float __logf(float x) { return ocu_dm::logf(x); }
OCU_D float __log2f(float x) { return ocu_dm::log2f(x); }
OCU_D float __log10f(float x) { return ocu_dm::log10f(x); }
OCU_D float __sinf(float x) { return ocu_dm::sinf(x); }
OCU_D float __cosf(float x) { return ocu_dm::cosf(x); }
OCU_D float __powf(float x, float y) { return ocu_dm::powf(x, y); }
OCU_D float __fdividef(float x, float y) { return x / y; }
#endif
OCU_D float __tanf(float x) { return __sinf(x) / __cosf(x); }
OCU_D void __sincosf(float x, float *s, float *c) { *s = __sinf(x); *c = __cosf(x); }
/* Round-to-nearest forms: the default IEEE operations, never contracted. */
OCU_D float __frcp_rn(float x) { return 1.0f / x; }
OCU_D float __fsqrt_rn(float x) { return __builtin_sqrtf(x); }
OCU_D float __fadd_rn(float a, float b) { float r = a + b; return r; }
OCU_D float __fmul_rn(float a, float b) { float r = a * b; return r; }
OCU_D float __fmaf_rn(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
OCU_D float __fdiv_rn(float a, float b) { return a / b; }
OCU_D double __dadd_rn(double a, double b) { double r = a + b; return r; }
OCU_D double __dmul_rn(double a, double b) { double r = a * b; return r; }
OCU_D double __fma_rn(double a, double b, double c) { return __builtin_fma(a, b, c); }
OCU_D double __drcp_rn(double x) { return 1.0 / x; }
OCU_D double __dsqrt_rn(double x) { return __builtin_sqrt(x); }

#if defined(OCU_SPV_HAVE_CXXLIB)
namespace std {
using ::sqrt; using ::fabs; using ::floor; using ::ceil; using ::trunc; using ::round;
using ::rint; using ::exp; using ::expm1; using ::exp2; using ::log; using ::log2;
using ::log10; using ::log1p; using ::sin; using ::cos; using ::tan; using ::asin;
using ::acos; using ::atan; using ::sinh; using ::cosh; using ::tanh; using ::cbrt;
using ::pow; using ::atan2; using ::hypot; using ::fmod; using ::fmin; using ::fmax;
using ::fma; using ::ldexp; using ::frexp; using ::copysign;
} // namespace std
#endif
OCU_D bool isnan(double x) { return __builtin_isnan(x); }
OCU_D bool isinf(double x) { return __builtin_isinf(x); }
OCU_D bool isfinite(double x) { return __builtin_isfinite(x); }

/* cudaLaunchKernel(kernel, ...) with a kernel name, as in user code. */
template <typename... P>
static inline cudaError_t cudaLaunchKernel(void (*func)(P...), dim3 grid, dim3 block, void **args,
                                           size_t shared = 0, cudaStream_t stream = nullptr)
{
    return cudaLaunchKernel(reinterpret_cast<const void *>(func), grid, block, args, shared, stream);
}

#include "ocu_cuda_lang_scoped.h"

#if !defined(OCU_NO_TEXTURE_DECLS)
#define OCU_TEX_FN static __device__ inline
#define OCU_TEX_MFN __device__ inline
#include "ocu_texture_fetch.h"
#endif

/* more of the CUDA Math API (ocu_cuda_math_ext.h) */
#define OCU_MX_Q OCU_D
#define OCU_MX_STD 1
#include "ocu_cuda_math_ext.h"

#endif /* OCU_CUDA_LANG_SPIRV_H */

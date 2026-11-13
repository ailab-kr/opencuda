/*
 * ocu_cuda_lang_nvptx.h -- CUDA language support for Clang's CUDA mode
 * (host pass and NVPTX device pass): qualifiers, built-in variables,
 * synchronisation, atomics, warp functions, bit and conversion intrinsics,
 * and device math (ocu_device_math.h).
 *
 * Written for this project from the public CUDA programming guide and math
 * API descriptions, using only compiler builtins of the open-source Clang
 * NVPTX target. Not derived from any vendor header; no vendor device
 * library is used.
 *
 * Included by cuda_runtime.h when the compiler is in CUDA mode.
 */
#ifndef OCU_CUDA_LANG_NVPTX_H
#define OCU_CUDA_LANG_NVPTX_H

/*
 * Device math first. In the device pass the standard library's own math
 * functions (libstdc++'s float overloads and integer templates call the
 * compiler's __builtin_<name>) are pointed at it, so std::exp(1.0f) or
 * pow(x, 2) in device code use this library instead of a missing vendor
 * function. Float functions are computed in float (no double operations);
 * their log table lives in global memory (a const or constexpr table goes
 * to constant memory, which serialises different per-thread indices: powf
 * ran at less than half the speed).
 */
#define OCU_DM_FN static inline __attribute__((device))
#define OCU_DMF_TABLE static __attribute__((device))
#include "ocu_device_math.h"
#if defined(__CUDA_ARCH__)
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

/* ---- declaration specifiers -------------------------------------------- */

#define __host__        __attribute__((host))
#define __device__      __attribute__((device))
#define __global__      __attribute__((global))
#define __shared__      __attribute__((shared))
#define __constant__    __attribute__((constant))
#define __forceinline__ __inline__ __attribute__((always_inline))
/* A hint only; left empty because the standard library spells
   __attribute__((__noinline__)). */
#define __noinline__
#define __align__(n)    __attribute__((aligned(n)))
#define __restrict__    __restrict
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

#define OCU_D  static __device__ inline __attribute__((always_inline))
#define OCU_HD static __host__ __device__ inline

#define OCU_LANG_FN static __device__ inline __attribute__((always_inline))
#include "ocu_cuda_lang_common.h"

/* ---- launch ABI the compiler calls (see runtime/cudart/fatbin.c) ---------- */

extern "C" unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shared = 0,
                                                cudaStream_t stream = 0);
extern "C" int __cudaPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shared, void *stream);
/* The older spelling; the device pass looks it up for <<<>>> in host code. */
extern "C" int cudaConfigureCall(dim3 grid, dim3 block, size_t shared = 0, cudaStream_t stream = 0);

/* ---- built-in variables ----------------------------------------------------- */

#if defined(__CUDA_ARCH__)
#define OCU_SREG(kind, d) __nvvm_read_ptx_sreg_##kind##_##d()
#else
#define OCU_SREG(kind, d) 0u /* host pass: never executed */
#endif

#define OCU_BUILTIN_VAR(TYPE, VAR, KIND, CONV)                                     \
    struct TYPE {                                                                  \
        __declspec(property(get = get_x)) unsigned x;                              \
        __declspec(property(get = get_y)) unsigned y;                              \
        __declspec(property(get = get_z)) unsigned z;                              \
        static __device__ __forceinline__ unsigned get_x() { return OCU_SREG(KIND, x); } \
        static __device__ __forceinline__ unsigned get_y() { return OCU_SREG(KIND, y); } \
        static __device__ __forceinline__ unsigned get_z() { return OCU_SREG(KIND, z); } \
        __device__ __forceinline__ operator CONV() const { return CONV{get_x(), get_y(), get_z()}; } \
    };                                                                             \
    extern const __device__ TYPE VAR;

OCU_BUILTIN_VAR(ocu_thread_idx_t, threadIdx, tid, uint3)
OCU_BUILTIN_VAR(ocu_block_idx_t, blockIdx, ctaid, uint3)
OCU_BUILTIN_VAR(ocu_block_dim_t, blockDim, ntid, dim3)
OCU_BUILTIN_VAR(ocu_grid_dim_t, gridDim, nctaid, dim3)
#undef OCU_BUILTIN_VAR

static constexpr int warpSize = 32;

/* ---- synchronisation and fences ---------------------------------------------- */

/* __syncthreads is a compiler builtin in both passes. */
OCU_D int __syncthreads_count(int p) { return __nvvm_bar0_popc(p); }
OCU_D int __syncthreads_and(int p) { return __nvvm_bar0_and(p); }
OCU_D int __syncthreads_or(int p) { return __nvvm_bar0_or(p); }
OCU_D void __threadfence() { __nvvm_membar_gl(); }
[[noreturn]] OCU_D void __trap() { __builtin_trap(); }
OCU_D void __threadfence_block() { __nvvm_membar_cta(); }
OCU_D void __threadfence_system() { __nvvm_membar_sys(); }

/* ---- device library pieces the programs call ------------------------------------ */

extern "C" __device__ int vprintf(const char *format, void *args);
extern "C" __device__ int printf(const char *format, ...);
OCU_D void *memcpy(void *d, const void *s, size_t n) { return __builtin_memcpy(d, s, n); }
OCU_D void *memset(void *d, int v, size_t n) { return __builtin_memset(d, v, n); }

/* ---- atomics (device scope, relaxed, as documented) ------------------------------ */

#define OCU_ATOMIC_INT(T)                                                              \
    OCU_D T atomicAdd(T *a, T v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); } \
    OCU_D T atomicSub(T *a, T v) { return __atomic_fetch_sub(a, v, __ATOMIC_RELAXED); } \
    OCU_D T atomicExch(T *a, T v) { return __atomic_exchange_n(a, v, __ATOMIC_RELAXED); } \
    OCU_D T atomicAnd(T *a, T v) { return __atomic_fetch_and(a, v, __ATOMIC_RELAXED); } \
    OCU_D T atomicOr(T *a, T v) { return __atomic_fetch_or(a, v, __ATOMIC_RELAXED); }  \
    OCU_D T atomicXor(T *a, T v) { return __atomic_fetch_xor(a, v, __ATOMIC_RELAXED); } \
    OCU_D T atomicMin(T *a, T v) { return __atomic_fetch_min(a, v, __ATOMIC_RELAXED); } \
    OCU_D T atomicMax(T *a, T v) { return __atomic_fetch_max(a, v, __ATOMIC_RELAXED); } \
    OCU_D T atomicCAS(T *a, T c, T v)                                                   \
    {                                                                                   \
        __atomic_compare_exchange_n(a, &c, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED); \
        return c;                                                                       \
    }
OCU_ATOMIC_INT(int)
OCU_ATOMIC_INT(unsigned int)
OCU_ATOMIC_INT(unsigned long long)
OCU_ATOMIC_INT(long long)
#undef OCU_ATOMIC_INT
OCU_D float atomicAdd(float *a, float v) { return __nvvm_atom_add_gen_f(a, v); }
OCU_D double atomicAdd(double *a, double v) { return __nvvm_atom_add_gen_d(a, v); }
OCU_D float atomicExch(float *a, float v)
{
    return __builtin_bit_cast(float, atomicExch(reinterpret_cast<int *>(a), __builtin_bit_cast(int, v)));
}
OCU_D unsigned atomicInc(unsigned *a, unsigned v) { return __nvvm_atom_inc_gen_ui(a, v); }
OCU_D unsigned atomicDec(unsigned *a, unsigned v) { return __nvvm_atom_dec_gen_ui(a, v); }

/* ---- warp functions --------------------------------------------------------------- */

OCU_D unsigned __ocu_lane() { return __nvvm_read_ptx_sreg_laneid(); }
OCU_D unsigned __activemask()
{
    unsigned m;
    asm volatile("activemask.b32 %0;" : "=r"(m));
    return m;
}
OCU_D void __syncwarp(unsigned m = 0xffffffffu) { __nvvm_bar_warp_sync(m); }
OCU_D unsigned __ballot_sync(unsigned m, int p) { return __nvvm_vote_ballot_sync(m, p != 0); }
OCU_D int __any_sync(unsigned m, int p) { return __nvvm_vote_any_sync(m, p != 0); }
OCU_D int __all_sync(unsigned m, int p) { return __nvvm_vote_all_sync(m, p != 0); }
OCU_D unsigned __match_any_sync(unsigned m, unsigned v) { return __nvvm_match_any_sync_i32(m, v); }

/* Shuffles of values of at most 8 bytes; `c` is the documented
   clamp/segment word. A value narrower than 4 bytes (char, short, __half)
   moves in one 32-bit word, as the documented overloads take it after
   promotion. */
template <typename T>
OCU_D T __ocu_shfl(int kind, unsigned m, T v, int arg, int c)
{
    static_assert(sizeof(T) <= 4 || sizeof(T) == 8, "shuffles move values of at most 8 bytes");
    auto one = [&](int x) -> int {
        switch (kind) {
        case 0: return __nvvm_shfl_sync_idx_i32(m, x, arg, c);
        case 1: return __nvvm_shfl_sync_up_i32(m, x, arg, c);
        case 2: return __nvvm_shfl_sync_down_i32(m, x, arg, c);
        default: return __nvvm_shfl_sync_bfly_i32(m, x, arg, c);
        }
    };
    if constexpr (sizeof(T) == 4) {
        return __builtin_bit_cast(T, one(__builtin_bit_cast(int, v)));
    } else if constexpr (sizeof(T) < 4) {
        int w = 0;
        __builtin_memcpy(&w, &v, sizeof(T));
        w = one(w);
        T r;
        __builtin_memcpy(&r, &w, sizeof(T));
        return r;
    } else {
        long long w = __builtin_bit_cast(long long, v);
        int lo = one((int)(w & 0xffffffff)), hi = one((int)(w >> 32));
        return __builtin_bit_cast(T, ((long long)hi << 32) | (unsigned)lo);
    }
}
template <typename T>
OCU_D T __shfl_sync(unsigned m, T v, int src, int width = 32)
{ return __ocu_shfl(0, m, v, src, ((32 - width) << 8) | 0x1f); }
template <typename T>
OCU_D T __shfl_up_sync(unsigned m, T v, unsigned d, int width = 32)
{ return __ocu_shfl(1, m, v, (int)d, (32 - width) << 8); }
template <typename T>
OCU_D T __shfl_down_sync(unsigned m, T v, unsigned d, int width = 32)
{ return __ocu_shfl(2, m, v, (int)d, ((32 - width) << 8) | 0x1f); }
template <typename T>
OCU_D T __shfl_xor_sync(unsigned m, T v, int x, int width = 32)
{ return __ocu_shfl(3, m, v, x, ((32 - width) << 8) | 0x1f); }

/* ---- integer and bit intrinsics ----------------------------------------------------- */

OCU_D int __popc(unsigned x) { return __builtin_popcount(x); }
OCU_D int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
OCU_D int __clz(int x) { return x == 0 ? 32 : __builtin_clz((unsigned)x); }
OCU_D int __clzll(long long x) { return x == 0 ? 64 : __builtin_clzll((unsigned long long)x); }
OCU_D int __ffs(int x) { return __builtin_ffs(x); }
OCU_D int __ffsll(long long x) { return __builtin_ffsll(x); }
OCU_D unsigned __brev(unsigned x) { return __builtin_bitreverse32(x); }
OCU_D unsigned long long __brevll(unsigned long long x) { return __builtin_bitreverse64(x); }
OCU_D int __mul24(int a, int b) { return __nvvm_mul24_i(a, b); }
OCU_D unsigned __umul24(unsigned a, unsigned b) { return __nvvm_mul24_ui(a, b); }
OCU_D int __mulhi(int a, int b) { return __nvvm_mulhi_i(a, b); }
OCU_D unsigned __umulhi(unsigned a, unsigned b) { return __nvvm_mulhi_ui(a, b); }
OCU_D long long __mul64hi(long long a, long long b) { return __nvvm_mulhi_ll(a, b); }
OCU_D unsigned long long __umul64hi(unsigned long long a, unsigned long long b) { return __nvvm_mulhi_ull(a, b); }
/* 3-bit byte indices only: PTX prmt treats the fourth bit of a selector nibble as
   "replicate the sign", which __byte_perm does not do. */
OCU_D unsigned __byte_perm(unsigned a, unsigned b, unsigned s) { return __nvvm_prmt(a, b, s & 0x7777u); }
OCU_D int __float_as_int(float x) { return __builtin_bit_cast(int, x); }
OCU_D unsigned __float_as_uint(float x) { return __builtin_bit_cast(unsigned, x); }
OCU_D float __int_as_float(int x) { return __builtin_bit_cast(float, x); }
OCU_D float __uint_as_float(unsigned x) { return __builtin_bit_cast(float, x); }
OCU_D long long __double_as_longlong(double x) { return __builtin_bit_cast(long long, x); }
OCU_D double __longlong_as_double(long long x) { return __builtin_bit_cast(double, x); }
OCU_D float __saturatef(float x) { return __nvvm_saturate_f(x); }
OCU_D unsigned __sad(int a, int b, unsigned c) { return (a > b ? a - b : b - a) + c; }
OCU_D int __float2int_rn(float x) { return __nvvm_f2i_rn(x); }
OCU_D int __float2int_rz(float x) { return __nvvm_f2i_rz(x); }
OCU_D int __float2int_rd(float x) { return __nvvm_f2i_rm(x); }
OCU_D int __float2int_ru(float x) { return __nvvm_f2i_rp(x); }
OCU_D float __int2float_rn(int x) { return __nvvm_i2f_rn(x); }
/* cvt to an integer saturates (PTX ISA, cvt); NaN is forced to 0 explicitly because the
   GPU gives INT_MIN for a double NaN (observed), unlike the float forms. */
OCU_D int __double2int_rn(double x) { int r; asm("cvt.rni.s32.f64 %0, %1;" : "=r"(r) : "d"(x)); return x != x ? 0 : r; }
OCU_D int __double2int_rz(double x) { int r; asm("cvt.rzi.s32.f64 %0, %1;" : "=r"(r) : "d"(x)); return x != x ? 0 : r; }
OCU_D int __double2int_rd(double x) { int r; asm("cvt.rmi.s32.f64 %0, %1;" : "=r"(r) : "d"(x)); return x != x ? 0 : r; }
OCU_D int __double2int_ru(double x) { int r; asm("cvt.rpi.s32.f64 %0, %1;" : "=r"(r) : "d"(x)); return x != x ? 0 : r; }

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

/* Functions computed exactly by hardware operations. */
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

/* Functions from ocu_device_math.h. */
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
/* Mixed argument types, exact matches before the standard library's
   templates (which would call host functions). */
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
/* Float only (no double counterpart yet). */
OCU_D float erff(float x) { return ocu_dm::erff(x); }
OCU_D float erfcf(float x) { return ocu_dm::erfcf(x); }

/* Fast approximate intrinsics (documented as approximate). */
OCU_D float __expf(float x) { return __nvvm_ex2_approx_f(x * 1.44269504088896341f); }
OCU_D float __exp10f(float x) { return __nvvm_ex2_approx_f(x * 3.32192809488736235f); }
OCU_D float __logf(float x) { return __nvvm_lg2_approx_f(x) * 0.693147180559945309f; }
OCU_D float __log2f(float x) { return __nvvm_lg2_approx_f(x); }
OCU_D float __log10f(float x) { return __nvvm_lg2_approx_f(x) * 0.301029995663981195f; }
OCU_D float __sinf(float x) { return __nvvm_sin_approx_f(x); }
OCU_D float __cosf(float x) { return __nvvm_cos_approx_f(x); }
OCU_D float __tanf(float x) { return __nvvm_sin_approx_f(x) / __nvvm_cos_approx_f(x); }
OCU_D void __sincosf(float x, float *s, float *c) { *s = __sinf(x); *c = __cosf(x); }
OCU_D float __powf(float x, float y) { return __nvvm_ex2_approx_f(y * __nvvm_lg2_approx_f(x)); }
OCU_D float __fdividef(float x, float y) { return __nvvm_div_approx_f(x, y); }
OCU_D float __frcp_rn(float x) { return __nvvm_rcp_rn_f(x); }
OCU_D float __fsqrt_rn(float x) { return __nvvm_sqrt_rn_f(x); }
OCU_D float __fadd_rn(float a, float b) { return __nvvm_add_rn_f(a, b); }
OCU_D float __fmul_rn(float a, float b) { return __nvvm_mul_rn_f(a, b); }
OCU_D float __fmaf_rn(float a, float b, float c) { return __nvvm_fma_rn_f(a, b, c); }
OCU_D float __fmaf_rz(float a, float b, float c) { return __nvvm_fma_rz_f(a, b, c); }
OCU_D float __fmaf_ru(float a, float b, float c) { return __nvvm_fma_rp_f(a, b, c); }
OCU_D float __fmaf_rd(float a, float b, float c) { return __nvvm_fma_rm_f(a, b, c); }
OCU_D float __fdiv_rn(float a, float b) { return __nvvm_div_rn_f(a, b); }
OCU_D double __dadd_rn(double a, double b) { return __nvvm_add_rn_d(a, b); }
OCU_D double __dmul_rn(double a, double b) { return __nvvm_mul_rn_d(a, b); }
OCU_D double __fma_rn(double a, double b, double c) { return __nvvm_fma_rn_d(a, b, c); }
OCU_D double __fma_rz(double a, double b, double c) { return __nvvm_fma_rz_d(a, b, c); }
OCU_D double __fma_ru(double a, double b, double c) { return __nvvm_fma_rp_d(a, b, c); }
OCU_D double __fma_rd(double a, double b, double c) { return __nvvm_fma_rm_d(a, b, c); }
OCU_D double __drcp_rn(double x) { return __nvvm_rcp_rn_d(x); }
OCU_D double __dsqrt_rn(double x) { return __nvvm_sqrt_rn_d(x); }

/* std:: spellings: libstdc++'s float overloads are constexpr (host and
   device in Clang, and use the library above in device code); the double
   ones are the global functions, made visible in std here. */
namespace std {
using ::sqrt; using ::fabs; using ::floor; using ::ceil; using ::trunc; using ::round;
using ::rint; using ::exp; using ::expm1; using ::exp2; using ::log; using ::log2;
using ::log10; using ::log1p; using ::sin; using ::cos; using ::tan; using ::asin;
using ::acos; using ::atan; using ::sinh; using ::cosh; using ::tanh; using ::cbrt;
using ::pow; using ::atan2; using ::hypot; using ::fmod; using ::fmin; using ::fmax;
using ::fma; using ::ldexp; using ::frexp; using ::copysign;
} // namespace std
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

#endif /* OCU_CUDA_LANG_NVPTX_H */

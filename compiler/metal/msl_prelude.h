// msl_prelude.h -- put in front of every kernel cu2msl.py translates: the
// CUDA names device code uses, in Metal Shading Language. Written for this
// project from the public CUDA and Metal Shading Language documentation.
#include <metal_stdlib>
using namespace metal;

// ---- arguments and indices -----------------------------------------------------
struct ocu_context {
    uint3 tid, bid, bdim, gdim;
    threadgroup uchar *dyn; // the launch's dynamic threadgroup memory
    device uchar *glob;     // the module's __device__/__constant__ variables (buffer 2)
};
// Built-in index variables: every translated function has ocu_ctx in scope
// (kernels declare it, device functions take it as their first parameter).
#define threadIdx ocu_ctx.tid
#define blockIdx ocu_ctx.bid
#define blockDim ocu_ctx.bdim
#define gridDim ocu_ctx.gdim
#define warpSize 32
template <typename T> T ocu_arg(constant uchar *a, constant uint *o, uint i)
{
    return *(constant T *)(a + o[i]);
}
template <typename P> struct ocu_ptr_box { P p; };
template <typename P> P ocu_ptr(constant uchar *a, constant uint *o, uint i)
{
    return ((constant ocu_ptr_box<P> *)(a + o[i]))->p;
}

// ---- qualifiers that have no MSL spelling -----------------------------------------
#define __device__
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __restrict__
#define __launch_bounds__(...)

// ---- loads through the read-only path: plain loads on Apple GPUs ----------------------
template <typename T> T __ldg(const device T *p) { return *p; }

// ---- synchronisation ------------------------------------------------------------------
#define __syncthreads() threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define __syncwarp(...) simdgroup_barrier(mem_flags::mem_threadgroup)

// ---- atomics (relaxed, as documented for CUDA) -------------------------------------------
#define OCU_ATOMIC2(fn, op, T, AT, SPACE)                                                \
    inline T fn(SPACE T *p, T v)                                                        \
    { return atomic_##op##_explicit((SPACE AT *)p, v, memory_order_relaxed); }
#define OCU_ATOMIC_INT(T, AT, SPACE)                                                     \
    OCU_ATOMIC2(atomicAdd, fetch_add, T, AT, SPACE)                                      \
    OCU_ATOMIC2(atomicSub, fetch_sub, T, AT, SPACE)                                      \
    OCU_ATOMIC2(atomicMin, fetch_min, T, AT, SPACE)                                      \
    OCU_ATOMIC2(atomicMax, fetch_max, T, AT, SPACE)                                      \
    OCU_ATOMIC2(atomicAnd, fetch_and, T, AT, SPACE)                                      \
    OCU_ATOMIC2(atomicOr, fetch_or, T, AT, SPACE)                                        \
    OCU_ATOMIC2(atomicXor, fetch_xor, T, AT, SPACE)                                      \
    OCU_ATOMIC2(atomicExch, exchange, T, AT, SPACE)                                      \
    inline T atomicCAS(SPACE T *p, T c, T v)                                             \
    {                                                                                    \
        atomic_compare_exchange_weak_explicit((SPACE AT *)p, &c, v, memory_order_relaxed, \
                                              memory_order_relaxed);                     \
        return c;                                                                        \
    }
OCU_ATOMIC_INT(int, atomic_int, device)
OCU_ATOMIC_INT(uint, atomic_uint, device)
OCU_ATOMIC_INT(int, atomic_int, threadgroup)
OCU_ATOMIC_INT(uint, atomic_uint, threadgroup)
OCU_ATOMIC2(atomicAdd, fetch_add, float, atomic_float, device)
OCU_ATOMIC2(atomicExch, exchange, float, atomic_float, device)
#undef OCU_ATOMIC_INT
#undef OCU_ATOMIC2

// ---- math under the CUDA names (precise forms; fast forms for __ intrinsics) ---------------
#define OCU_M1(cu, msl) inline float cu(float x) { return precise::msl(x); }
OCU_M1(expf, exp) OCU_M1(exp2f, exp2) OCU_M1(exp10f, exp10) OCU_M1(logf, log)
OCU_M1(log2f, log2) OCU_M1(log10f, log10) OCU_M1(sinf, sin) OCU_M1(cosf, cos)
OCU_M1(tanf, tan) OCU_M1(asinf, asin) OCU_M1(acosf, acos) OCU_M1(atanf, atan)
OCU_M1(sinhf, sinh) OCU_M1(coshf, cosh) OCU_M1(tanhf, tanh) OCU_M1(sqrtf, sqrt)
OCU_M1(rsqrtf, rsqrt) OCU_M1(fabsf, fabs) OCU_M1(floorf, floor) OCU_M1(ceilf, ceil)
OCU_M1(truncf, trunc) OCU_M1(roundf, round) OCU_M1(rintf, rint)
#undef OCU_M1
inline float powf(float x, float y) { return precise::pow(x, y); }
inline float atan2f(float y, float x) { return precise::atan2(y, x); }
inline float fmodf(float x, float y) { return precise::fmod(x, y); }
inline float fminf(float a, float b) { return fmin(a, b); }
inline float fmaxf(float a, float b) { return fmax(a, b); }
inline float fmaf(float a, float b, float c) { return fma(a, b, c); }
inline float __expf(float x) { return fast::exp(x); }
inline float __logf(float x) { return fast::log(x); }
inline float __sinf(float x) { return fast::sin(x); }
inline float __cosf(float x) { return fast::cos(x); }
inline float __powf(float x, float y) { return fast::pow(x, y); }
inline float __fdividef(float x, float y) { return fast::divide(x, y); }
inline float __saturatef(float x) { return saturate(x); }
inline float __log2f(float x) { return fast::log2(x); }
inline float __log10f(float x) { return fast::log10(x); }
inline float __exp10f(float x) { return fast::exp10(x); }
inline float __tanf(float x) { return fast::tan(x); }
inline float __fmaf_rn(float a, float b, float c) { return fma(a, b, c); }
// 24-bit multiplies: the low 24 bits of each operand (sign-extended for __mul24)
inline int __mul24(int a, int b) { return ((a << 8) >> 8) * ((b << 8) >> 8); }
inline uint __umul24(uint a, uint b) { return (a & 0xffffffu) * (b & 0xffffffu); }
inline int __float2int_rn(float x) { return int(rint(x)); }
inline int __float2int_rz(float x) { return int(x); }
inline uint __float2uint_rz(float x) { return uint(x); }
inline int __float_as_int(float x) { return as_type<int>(x); }
inline float __int_as_float(int x) { return as_type<float>(x); }
inline uint __float_as_uint(float x) { return as_type<uint>(x); }
inline float __uint_as_float(uint x) { return as_type<float>(x); }
inline int __popc(uint x) { return popcount(x); }
inline int __clz(int x) { return clz(x); }
inline int __ffs(int x) { return x == 0 ? 0 : ctz(x) + 1; }
inline uint __brev(uint x) { return reverse_bits(x); }

// ---- warp functions (SIMD-group width 32; width must be 32) -----------------------------------
template <typename T> T __shfl_sync(uint, T v, int src, int = 32) { return simd_shuffle(v, ushort(src)); }
template <typename T> T __shfl_up_sync(uint, T v, uint d, int = 32) { return simd_shuffle_up(v, ushort(d)); }
template <typename T> T __shfl_down_sync(uint, T v, uint d, int = 32) { return simd_shuffle_down(v, ushort(d)); }
template <typename T> T __shfl_xor_sync(uint, T v, int x, int = 32) { return simd_shuffle_xor(v, ushort(x)); }
inline uint __ballot_sync(uint, int p) { return uint(uint64_t(simd_ballot(p != 0))); }
inline int __any_sync(uint, int p) { return simd_any(p != 0); }
inline int __all_sync(uint, int p) { return simd_all(p != 0); }

// ---- vector constructors -------------------------------------------------------------------------
inline float2 make_float2(float x, float y) { return float2(x, y); }
inline float3 make_float3(float x, float y, float z) { return float3(x, y, z); }
inline float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }
inline int2 make_int2(int x, int y) { return int2(x, y); }
inline int3 make_int3(int x, int y, int z) { return int3(x, y, z); }
inline int4 make_int4(int x, int y, int z, int w) { return int4(x, y, z, w); }
inline uint3 make_uint3(uint x, uint y, uint z) { return uint3(x, y, z); }

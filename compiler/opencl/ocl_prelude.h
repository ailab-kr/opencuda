// ocl_prelude.h -- put in front of every kernel cu2ocl.py translates: the
// CUDA names device code uses, in C++ for OpenCL (2021). Written for this
// project from the public CUDA documentation, the Khronos OpenCL C 3.0
// specification (built-in functions, section 6.15) and the C++ for OpenCL
// 2021 documentation.
//
// The translator shares its address-space analysis with cu2msl.py and
// writes address spaces with the MSL words; they are mapped here
// (device = __global, threadgroup = __local, constant = __constant,
// thread = __private). Generic pointers are not needed: every pointer the
// translation writes has its address space.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define device __global
#define threadgroup __local
#define constant __constant
#define thread __private
#define kernel __kernel
// OpenCL keywords that are ordinary names in CUDA code
#define global ocu_kw_global
#define local ocu_kw_local
#define generic ocu_kw_generic
#define pipe ocu_kw_pipe
#define read_only ocu_kw_read_only
#define write_only ocu_kw_write_only
#define read_write ocu_kw_read_write

typedef signed char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;
typedef long int64_t;
typedef ulong uint64_t;

// ---- arguments and indices -----------------------------------------------------
struct ocu_context {
    uint3 tid, bid, bdim, gdim;
    threadgroup uchar *dyn; // the launch's dynamic local memory
    device uchar *glob;     // the module's __device__/__constant__ variables
};
#define OCU_U3(f) ((uint3)((uint)f(0), (uint)f(1), (uint)f(2)))
#define threadIdx ocu_ctx.tid
#define blockIdx ocu_ctx.bid
#define blockDim ocu_ctx.bdim
#define gridDim ocu_ctx.gdim
// Argument values are copied byte by byte: no copy between address spaces
// through a constructor, no reliance on the alignment of the packing.
template <typename T> T ocu_arg(constant uchar *a, constant uint *o, uint i)
{
    T v;
    thread uchar *d = (thread uchar *)&v;
    constant uchar *s = a + o[i];
    for (uint k = 0; k < sizeof(T); ++k) d[k] = s[k];
    return v;
}
// CUDA packs a 3-component vector in three scalars; OpenCL's take four.
template <typename V, typename S> V ocu_arg3(constant uchar *a, constant uint *o, uint i)
{
    S s[3];
    thread uchar *d = (thread uchar *)s;
    constant uchar *p = a + o[i];
    for (uint k = 0; k < 3 * sizeof(S); ++k) d[k] = p[k];
    V v;
    v.x = s[0];
    v.y = s[1];
    v.z = s[2];
    return v;
}
template <typename P> P ocu_ptr(constant uchar *a, constant uint *o, uint i)
{
    return ocu_arg<P>(a, o, i);
}

// ---- qualifiers that have no OpenCL spelling -------------------------------------
#define __device__
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __restrict__
#define __launch_bounds__(...)

template <typename T> T __ldg(const device T *p) { return *p; }

// ---- synchronisation ----------------------------------------------------------------
#define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#define __threadfence_block() mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#define __threadfence() mem_fence(CLK_GLOBAL_MEM_FENCE)

// ---- atomics (OpenCL 1.x built-ins: relaxed, device scope) ---------------------------
#define OCU_ATOMIC_INT(T, SPACE)                                                         \
    inline T atomicAdd(SPACE T *p, T v) { return atomic_add((volatile SPACE T *)p, v); } \
    inline T atomicSub(SPACE T *p, T v) { return atomic_sub((volatile SPACE T *)p, v); } \
    inline T atomicMin(SPACE T *p, T v) { return atomic_min((volatile SPACE T *)p, v); } \
    inline T atomicMax(SPACE T *p, T v) { return atomic_max((volatile SPACE T *)p, v); } \
    inline T atomicAnd(SPACE T *p, T v) { return atomic_and((volatile SPACE T *)p, v); } \
    inline T atomicOr(SPACE T *p, T v) { return atomic_or((volatile SPACE T *)p, v); }   \
    inline T atomicXor(SPACE T *p, T v) { return atomic_xor((volatile SPACE T *)p, v); } \
    inline T atomicExch(SPACE T *p, T v) { return atomic_xchg((volatile SPACE T *)p, v); } \
    inline T atomicCAS(SPACE T *p, T c, T v) { return atomic_cmpxchg((volatile SPACE T *)p, c, v); }
OCU_ATOMIC_INT(int, device)
OCU_ATOMIC_INT(uint, device)
OCU_ATOMIC_INT(int, threadgroup)
OCU_ATOMIC_INT(uint, threadgroup)
#undef OCU_ATOMIC_INT
// wrap-around increment and decrement (CUDA atomicInc / atomicDec)
#define OCU_ATOMIC_WRAP(SPACE)                                                           \
    inline uint atomicInc(SPACE uint *p, uint m)                                         \
    {                                                                                    \
        uint old = *(volatile SPACE uint *)p, seen;                                      \
        while ((seen = atomic_cmpxchg((volatile SPACE uint *)p, old,                     \
                                      old >= m ? 0u : old + 1u)) != old) old = seen;     \
        return old;                                                                      \
    }                                                                                    \
    inline uint atomicDec(SPACE uint *p, uint m)                                         \
    {                                                                                    \
        uint old = *(volatile SPACE uint *)p, seen;                                      \
        while ((seen = atomic_cmpxchg((volatile SPACE uint *)p, old,                     \
                                      (old == 0u || old > m) ? m : old - 1u)) != old)    \
            old = seen;                                                                  \
        return old;                                                                      \
    }                                                                                    \
    inline float atomicAdd(SPACE float *p, float v)                                      \
    {                                                                                    \
        volatile SPACE uint *u = (volatile SPACE uint *)p;                               \
        uint old = *u, seen;                                                             \
        while ((seen = atomic_cmpxchg(u, old, as_uint(as_float(old) + v))) != old) old = seen; \
        return as_float(old);                                                            \
    }                                                                                    \
    inline float atomicExch(SPACE float *p, float v)                                     \
    {                                                                                    \
        return as_float(atomic_xchg((volatile SPACE uint *)p, as_uint(v)));              \
    }
OCU_ATOMIC_WRAP(device)
OCU_ATOMIC_WRAP(threadgroup)
#undef OCU_ATOMIC_WRAP
#if defined(cl_khr_int64_base_atomics)
inline ulong atomicAdd(device ulong *p, ulong v) { return atom_add((volatile device ulong *)p, v); }
inline ulong atomicExch(device ulong *p, ulong v) { return atom_xchg((volatile device ulong *)p, v); }
inline ulong atomicCAS(device ulong *p, ulong c, ulong v)
{
    return atom_cmpxchg((volatile device ulong *)p, c, v);
}
inline double atomicAdd(device double *p, double v)
{
    volatile device ulong *u = (volatile device ulong *)p;
    ulong old = *u, seen;
    while ((seen = atom_cmpxchg(u, old, as_ulong(as_double(old) + v))) != old) old = seen;
    return as_double(old);
}
#endif
#if defined(cl_khr_int64_extended_atomics)
inline ulong atomicMin(device ulong *p, ulong v) { return atom_min((volatile device ulong *)p, v); }
inline ulong atomicMax(device ulong *p, ulong v) { return atom_max((volatile device ulong *)p, v); }
#endif

// ---- math under the CUDA names ----------------------------------------------------------
// OpenCL's built-ins are overloaded for float and double; the CUDA float
// names map to them (precise forms), the __ intrinsics to the native forms.
#define OCU_M1(cu, cl) inline float cu(float x) { return cl(x); }
OCU_M1(expf, exp) OCU_M1(exp2f, exp2) OCU_M1(exp10f, exp10) OCU_M1(expm1f, expm1)
OCU_M1(logf, log) OCU_M1(log2f, log2) OCU_M1(log10f, log10) OCU_M1(log1pf, log1p)
OCU_M1(sinf, sin) OCU_M1(cosf, cos) OCU_M1(tanf, tan) OCU_M1(asinf, asin)
OCU_M1(acosf, acos) OCU_M1(atanf, atan) OCU_M1(sinhf, sinh) OCU_M1(coshf, cosh)
OCU_M1(tanhf, tanh) OCU_M1(asinhf, asinh) OCU_M1(acoshf, acosh) OCU_M1(atanhf, atanh)
OCU_M1(sqrtf, sqrt) OCU_M1(rsqrtf, rsqrt) OCU_M1(cbrtf, cbrt) OCU_M1(fabsf, fabs)
OCU_M1(floorf, floor) OCU_M1(ceilf, ceil) OCU_M1(truncf, trunc) OCU_M1(roundf, round)
OCU_M1(rintf, rint) OCU_M1(erff, erf) OCU_M1(erfcf, erfc) OCU_M1(tgammaf, tgamma)
OCU_M1(lgammaf, lgamma)
#undef OCU_M1
inline float powf(float x, float y) { return pow(x, y); }
inline float atan2f(float y, float x) { return atan2(y, x); }
inline float fmodf(float x, float y) { return fmod(x, y); }
inline float hypotf(float x, float y) { return hypot(x, y); }
inline float fminf(float a, float b) { return fmin(a, b); }
inline float fmaxf(float a, float b) { return fmax(a, b); }
inline float fdimf(float a, float b) { return fdim(a, b); }
inline float fmaf(float a, float b, float c) { return fma(a, b, c); }
inline float copysignf(float a, float b) { return copysign(a, b); }
inline float ldexpf(float a, int e) { return ldexp(a, e); }
inline void sincosf(float x, thread float *s, thread float *c) { *s = sin(x); *c = cos(x); }
inline float __expf(float x) { return native_exp(x); }
inline float __exp10f(float x) { return native_exp10(x); }
inline float __logf(float x) { return native_log(x); }
inline float __log2f(float x) { return native_log2(x); }
inline float __log10f(float x) { return native_log10(x); }
inline float __sinf(float x) { return native_sin(x); }
inline float __cosf(float x) { return native_cos(x); }
inline float __tanf(float x) { return native_tan(x); }
inline float __powf(float x, float y) { return native_powr(x, y); }
inline float __fdividef(float x, float y) { return native_divide(x, y); }
inline float __saturatef(float x) { return clamp(x, 0.0f, 1.0f); }
inline float __fmaf_rn(float a, float b, float c) { return fma(a, b, c); }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fmul_rn(float a, float b) { return a * b; }
// 24-bit multiplies: the low 24 bits of each operand (sign-extended for __mul24)
inline int __mul24(int a, int b) { return ((a << 8) >> 8) * ((b << 8) >> 8); }
inline uint __umul24(uint a, uint b) { return (a & 0xffffffu) * (b & 0xffffffu); }
inline int __float2int_rn(float x) { return convert_int_rte(x); }
inline int __float2int_rz(float x) { return convert_int_rtz(x); }
inline int __float2int_rd(float x) { return convert_int_rtn(x); }
inline int __float2int_ru(float x) { return convert_int_rtp(x); }
inline uint __float2uint_rn(float x) { return convert_uint_rte(x); }
inline uint __float2uint_rz(float x) { return convert_uint_rtz(x); }
inline int __float_as_int(float x) { return as_int(x); }
inline float __int_as_float(int x) { return as_float(x); }
inline uint __float_as_uint(float x) { return as_uint(x); }
inline float __uint_as_float(uint x) { return as_float(x); }
inline long __double_as_longlong(double x) { return as_long(x); }
inline double __longlong_as_double(long x) { return as_double(x); }
inline int __popc(uint x) { return (int)popcount(x); }
inline int __popcll(ulong x) { return (int)popcount(x); }
inline int __clz(int x) { return (int)clz(x); }
inline int __clzll(long x) { return (int)clz(x); }
inline int __ffs(int x) { return x == 0 ? 0 : (int)popcount((uint)((x & -x) - 1)) + 1; }
inline uint __brev(uint x)
{
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
}
inline uint __umulhi(uint a, uint b) { return mul_hi(a, b); }
inline int __mulhi(int a, int b) { return mul_hi(a, b); }
inline uint __usad(uint a, uint b, uint c) { return abs_diff(a, b) + c; }

// ---- warp functions: OpenCL sub-groups, only where they have the CUDA width ------------
// A device whose sub-groups are not 32 wide gets no warp functions: code
// that uses them fails to build (the backend reports "not supported"),
// rather than running with the wrong lane count.
#if defined(OCU_SUBGROUP_32) && defined(cl_khr_subgroup_shuffle) && \
    defined(cl_khr_subgroup_shuffle_relative)
#define warpSize 32
template <typename T> T __shfl_sync(uint, T v, int src, int w = 32)
{
    uint lane = get_sub_group_local_id();
    return sub_group_shuffle(v, (lane & ~(uint)(w - 1)) + ((uint)src & (uint)(w - 1)));
}
template <typename T> T __shfl_xor_sync(uint, T v, int x, int = 32)
{
    return sub_group_shuffle_xor(v, (uint)x);
}
template <typename T> T __shfl_up_sync(uint, T v, uint d, int = 32)
{
    return sub_group_shuffle_up(v, d);
}
template <typename T> T __shfl_down_sync(uint, T v, uint d, int = 32)
{
    return sub_group_shuffle_down(v, d);
}
#endif
#if defined(OCU_SUBGROUP_32) && defined(cl_khr_subgroup_ballot)
inline uint __ballot_sync(uint, int p) { return sub_group_ballot(p != 0).x; }
inline int __any_sync(uint, int p) { return sub_group_any(p != 0); }
inline int __all_sync(uint, int p) { return sub_group_all(p != 0); }
#define __syncwarp(...) sub_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#endif

// ---- vector constructors -------------------------------------------------------------------
inline float2 make_float2(float x, float y) { return (float2)(x, y); }
inline float3 make_float3(float x, float y, float z) { return (float3)(x, y, z); }
inline float4 make_float4(float x, float y, float z, float w) { return (float4)(x, y, z, w); }
inline int2 make_int2(int x, int y) { return (int2)(x, y); }
inline int3 make_int3(int x, int y, int z) { return (int3)(x, y, z); }
inline int4 make_int4(int x, int y, int z, int w) { return (int4)(x, y, z, w); }
inline uint2 make_uint2(uint x, uint y) { return (uint2)(x, y); }
inline uint3 make_uint3(uint x, uint y, uint z) { return (uint3)(x, y, z); }
inline uint4 make_uint4(uint x, uint y, uint z, uint w) { return (uint4)(x, y, z, w); }
inline double2 make_double2(double x, double y) { return (double2)(x, y); }

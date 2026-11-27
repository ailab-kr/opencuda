/*
 * ocu_half_atomics.h -- atomicAdd for the 16-bit float types, from the CUDA
 * C++ Programming Guide ("atomicAdd": __half, __half2, __nv_bfloat16,
 * __nv_bfloat162; the two-element forms are atomic for each element
 * separately, not as one 32-bit access).
 *
 * Written for this project from that public description; not derived from
 * any vendor header. Every path has a 32-bit atomicCAS, so each add is a
 * compare-and-swap on the aligned 32-bit word that holds the element (a
 * 16-bit element may sit at either half of it). The sum is the type's own
 * rounded add (__hadd): float sum rounded to nearest even, which is the
 * correctly rounded result (float's 24 bits >= 2 x 11 + 2, Figueroa).
 *
 * Included at the end of cuda_fp16.h / cuda_bf16.h and of cuda_runtime.h,
 * whichever comes last, so that both the type and atomicCAS are declared.
 */
#if defined(OCU_CUDA_LANG_READY) && defined(__cplusplus)

#include <stdint.h>

#ifndef OCU_HALF_ATOMICS_CORE
#define OCU_HALF_ATOMICS_CORE 1
namespace ocu_half_atomics_ {
/* Adds val to the 16-bit element at p with op(old, val); returns the old bits. */
template <class Op>
__device__ inline unsigned short add16(unsigned short *p, unsigned short val, Op op)
{
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    unsigned int *w = reinterpret_cast<unsigned int *>(a & ~static_cast<uintptr_t>(3));
    const unsigned int sh = (a & 2u) ? 16u : 0u; /* little-endian: the low half is the lower address */
    unsigned int old = *reinterpret_cast<volatile unsigned int *>(w), assumed;
    do {
        assumed = old;
        const unsigned short cur = static_cast<unsigned short>(assumed >> sh);
        const unsigned int nw = (assumed & ~(0xffffu << sh)) | (static_cast<unsigned int>(op(cur, val)) << sh);
        old = atomicCAS(w, assumed, nw);
    } while (old != assumed);
    return static_cast<unsigned short>(old >> sh);
}
} // namespace ocu_half_atomics_
#endif /* OCU_HALF_ATOMICS_CORE */

#if defined(OCU_CUDA_FP16_H) && !defined(OCU_HALF_ATOMICS_FP16)
#define OCU_HALF_ATOMICS_FP16 1
namespace ocu_half_atomics_ {
struct add_f16 {
    __device__ unsigned short operator()(unsigned short x, unsigned short y) const
    {
        return __half_as_ushort(__hadd(__ushort_as_half(x), __ushort_as_half(y)));
    }
};
} // namespace ocu_half_atomics_
__device__ inline __half atomicAdd(__half *address, __half val)
{
    return __ushort_as_half(ocu_half_atomics_::add16(reinterpret_cast<unsigned short *>(address),
                                                     __half_as_ushort(val), ocu_half_atomics_::add_f16()));
}
__device__ inline __half2 atomicAdd(__half2 *address, __half2 val)
{
    __half lo = atomicAdd(&address->x, val.x);
    __half hi = atomicAdd(&address->y, val.y);
    return __half2(lo, hi);
}
#endif

#if defined(OCU_CUDA_BF16_H) && !defined(OCU_HALF_ATOMICS_BF16)
#define OCU_HALF_ATOMICS_BF16 1
namespace ocu_half_atomics_ {
struct add_bf16 {
    __device__ unsigned short operator()(unsigned short x, unsigned short y) const
    {
        return __bfloat16_as_ushort(__hadd(__ushort_as_bfloat16(x), __ushort_as_bfloat16(y)));
    }
};
} // namespace ocu_half_atomics_
__device__ inline __nv_bfloat16 atomicAdd(__nv_bfloat16 *address, __nv_bfloat16 val)
{
    return __ushort_as_bfloat16(ocu_half_atomics_::add16(reinterpret_cast<unsigned short *>(address),
                                                         __bfloat16_as_ushort(val), ocu_half_atomics_::add_bf16()));
}
__device__ inline __nv_bfloat162 atomicAdd(__nv_bfloat162 *address, __nv_bfloat162 val)
{
    __nv_bfloat16 lo = atomicAdd(&address->x, val.x);
    __nv_bfloat16 hi = atomicAdd(&address->y, val.y);
    return __nv_bfloat162(lo, hi);
}
#endif

#endif /* OCU_CUDA_LANG_READY */

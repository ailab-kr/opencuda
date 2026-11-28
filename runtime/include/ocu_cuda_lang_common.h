/*
 * ocu_cuda_lang_common.h -- device intrinsics whose definition is plain
 * integer or floating-point code, shared by the CPU path and the NVPTX path.
 * Included by ocu_cuda_lang_cpu.h and ocu_cuda_lang_nvptx.h after they
 * define OCU_LANG_FN (the declaration specifiers of a device function).
 *
 * Written for this project from the public CUDA math API ("SIMD
 * intrinsics", "Integer intrinsics", "Type casting intrinsics") and
 * programming guide (read-only data cache loads). Not derived from any
 * vendor header.
 */
#ifndef OCU_CUDA_LANG_COMMON_H
#define OCU_CUDA_LANG_COMMON_H

#ifndef OCU_LANG_HD
#define OCU_LANG_HD OCU_LANG_FN
#endif
#ifndef OCU_LANG_FN
#error "define OCU_LANG_FN before including ocu_cuda_lang_common.h"
#endif

/*
 * Per-byte SIMD on the four bytes of an unsigned int (byte 0 = least
 * significant).
 *   __vsub4(a, b)    a - b in each byte, modulo 256, no borrow between bytes.
 *   __vcmpne4(a, b)  0xff in each byte where the bytes differ, 0x00 where
 *                    they are equal.
 */
OCU_LANG_FN unsigned int __vsub4(unsigned int a, unsigned int b)
{
    unsigned int r = 0;
    for (int i = 0; i < 4; ++i) {
        r |= (((a >> (8 * i)) - (b >> (8 * i))) & 0xffu) << (8 * i);
    }
    return r;
}

OCU_LANG_FN unsigned int __vcmpne4(unsigned int a, unsigned int b)
{
    unsigned int r = 0;
    for (int i = 0; i < 4; ++i) {
        if (((a >> (8 * i)) & 0xffu) != ((b >> (8 * i)) & 0xffu)) r |= 0xffu << (8 * i);
    }
    return r;
}

/*
 * The rest of the SIMD intrinsics, from the same documentation. Lanes are
 * the two halfwords (...2) or four bytes (...4) of an unsigned int, lane 0
 * least significant. In each lane: ua / ub are the unsigned lane values,
 * sa / sb the signed ones; "ss" saturates to the signed lane range, "us"
 * to the unsigned one; others wrap. Averages: signed rounded away from
 * zero (__vavgs), unsigned rounded up (__vavgu) or toward zero (__vhaddu).
 * cmp: all ones in a true lane; set: 1. __vsads / __vsadu: the sum over the
 * lanes of |a - b| (one integer). Written as one loop per function so every
 * path (CPU, SPIR-V, the OpenCL translation of library code) compiles it.
 */
#define OCU_SIMD_BIN(name, W, expr)                                                     \
    OCU_LANG_FN unsigned int name(unsigned int a, unsigned int b)                       \
    {                                                                                   \
        const int smin = -(1 << ((W) - 1)), smax = (1 << ((W) - 1)) - 1;                \
        const int umax = (1 << (W)) - 1;                                                \
        (void)smin; (void)smax; (void)umax;                                             \
        unsigned int r = 0;                                                             \
        for (int i = 0; i < 32; i += (W)) {                                             \
            const int ua = (int)((a >> i) & (unsigned)umax), ub = (int)((b >> i) & (unsigned)umax); \
            const int sa = ua > smax ? ua - umax - 1 : ua, sb = ub > smax ? ub - umax - 1 : ub; \
            (void)ua; (void)ub; (void)sa; (void)sb;                                     \
            r |= ((unsigned int)(expr) & (unsigned)umax) << i;                          \
        }                                                                               \
        return r;                                                                       \
    }
#define OCU_SIMD_UN(name, W, expr)                                                      \
    OCU_LANG_FN unsigned int name(unsigned int a)                                       \
    {                                                                                   \
        const int smin = -(1 << ((W) - 1)), smax = (1 << ((W) - 1)) - 1;                \
        const int umax = (1 << (W)) - 1;                                                \
        (void)smin; (void)smax;                                                         \
        unsigned int r = 0;                                                             \
        for (int i = 0; i < 32; i += (W)) {                                             \
            const int ua = (int)((a >> i) & (unsigned)umax);                            \
            const int sa = ua > smax ? ua - umax - 1 : ua;                              \
            r |= ((unsigned int)(expr) & (unsigned)umax) << i;                          \
        }                                                                               \
        return r;                                                                       \
    }
#define OCU_SIMD_BOTH(stem, expr) OCU_SIMD_BIN(stem##2, 16, expr) OCU_SIMD_BIN(stem##4, 8, expr)
#define OCU_SIMD_CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : (v) > (hi) ? (hi) : (v))
#define OCU_SIMD_ABS(v) ((v) < 0 ? -(v) : (v))

OCU_SIMD_UN(__vabs2, 16, OCU_SIMD_ABS(sa))
OCU_SIMD_UN(__vabs4, 8, OCU_SIMD_ABS(sa))
OCU_SIMD_UN(__vabsss2, 16, OCU_SIMD_CLAMP(OCU_SIMD_ABS(sa), smin, smax))
OCU_SIMD_UN(__vabsss4, 8, OCU_SIMD_CLAMP(OCU_SIMD_ABS(sa), smin, smax))
OCU_SIMD_UN(__vneg2, 16, -sa)
OCU_SIMD_UN(__vneg4, 8, -sa)
OCU_SIMD_UN(__vnegss2, 16, OCU_SIMD_CLAMP(-sa, smin, smax))
OCU_SIMD_UN(__vnegss4, 8, OCU_SIMD_CLAMP(-sa, smin, smax))
OCU_SIMD_BOTH(__vabsdiffs, OCU_SIMD_ABS(sa - sb))
OCU_SIMD_BOTH(__vabsdiffu, OCU_SIMD_ABS(ua - ub))
OCU_SIMD_BOTH(__vadd, ua + ub)
OCU_SIMD_BOTH(__vaddss, OCU_SIMD_CLAMP(sa + sb, smin, smax))
OCU_SIMD_BOTH(__vaddus, OCU_SIMD_CLAMP(ua + ub, 0, umax))
OCU_SIMD_BIN(__vsub2, 16, ua - ub)
OCU_SIMD_BOTH(__vsubss, OCU_SIMD_CLAMP(sa - sb, smin, smax))
OCU_SIMD_BOTH(__vsubus, OCU_SIMD_CLAMP(ua - ub, 0, umax))
OCU_SIMD_BOTH(__vavgs, (sa + sb >= 0 ? (sa + sb + 1) >> 1 : -((-(sa + sb) + 1) >> 1)))
OCU_SIMD_BOTH(__vavgu, (ua + ub + 1) >> 1)
OCU_SIMD_BOTH(__vhaddu, (ua + ub) >> 1)
OCU_SIMD_BOTH(__vcmpeq, ua == ub ? -1 : 0)
OCU_SIMD_BIN(__vcmpne2, 16, ua != ub ? -1 : 0)
OCU_SIMD_BOTH(__vcmpges, sa >= sb ? -1 : 0)
OCU_SIMD_BOTH(__vcmpgeu, ua >= ub ? -1 : 0)
OCU_SIMD_BOTH(__vcmpgts, sa > sb ? -1 : 0)
OCU_SIMD_BOTH(__vcmpgtu, ua > ub ? -1 : 0)
OCU_SIMD_BOTH(__vcmples, sa <= sb ? -1 : 0)
OCU_SIMD_BOTH(__vcmpleu, ua <= ub ? -1 : 0)
OCU_SIMD_BOTH(__vcmplts, sa < sb ? -1 : 0)
OCU_SIMD_BOTH(__vcmpltu, ua < ub ? -1 : 0)
OCU_SIMD_BOTH(__vseteq, ua == ub)
OCU_SIMD_BOTH(__vsetne, ua != ub)
OCU_SIMD_BOTH(__vsetges, sa >= sb)
OCU_SIMD_BOTH(__vsetgeu, ua >= ub)
OCU_SIMD_BOTH(__vsetgts, sa > sb)
OCU_SIMD_BOTH(__vsetgtu, ua > ub)
OCU_SIMD_BOTH(__vsetles, sa <= sb)
OCU_SIMD_BOTH(__vsetleu, ua <= ub)
OCU_SIMD_BOTH(__vsetlts, sa < sb)
OCU_SIMD_BOTH(__vsetltu, ua < ub)
OCU_SIMD_BOTH(__vmaxs, sa > sb ? sa : sb)
OCU_SIMD_BOTH(__vmaxu, ua > ub ? ua : ub)
OCU_SIMD_BOTH(__vmins, sa < sb ? sa : sb)
OCU_SIMD_BOTH(__vminu, ua < ub ? ua : ub)

#define OCU_SIMD_SAD(name, W, sgn)                                                      \
    OCU_LANG_FN unsigned int name(unsigned int a, unsigned int b)                       \
    {                                                                                   \
        const int smax = (1 << ((W) - 1)) - 1, umax = (1 << (W)) - 1;                   \
        unsigned int s = 0;                                                             \
        for (int i = 0; i < 32; i += (W)) {                                             \
            int x = (int)((a >> i) & (unsigned)umax), y = (int)((b >> i) & (unsigned)umax); \
            if (sgn) { x = x > smax ? x - umax - 1 : x; y = y > smax ? y - umax - 1 : y; } \
            s += (unsigned int)(x > y ? x - y : y - x);                                 \
        }                                                                               \
        return s;                                                                       \
    }
OCU_SIMD_SAD(__vsads2, 16, 1)
OCU_SIMD_SAD(__vsads4, 8, 1)
OCU_SIMD_SAD(__vsadu2, 16, 0)
OCU_SIMD_SAD(__vsadu4, 8, 0)

/*
 * 32-bit and 16x2 forms (__host__ __device__ in the documentation): three
 * operands, add-then-max/min (a + b wraps in the lane), max/min with a
 * predicate output, and _relu, which takes max(result, 0) in each lane.
 */
#define OCU_SIMD_MAX(x, y) ((x) > (y) ? (x) : (y))
#define OCU_SIMD_MIN(x, y) ((x) < (y) ? (x) : (y))
#define OCU_SIMD_H3(name, sgn, expr)                                                    \
    OCU_LANG_HD unsigned int name(const unsigned int a, const unsigned int b, const unsigned int c) \
    {                                                                                   \
        unsigned int r = 0;                                                             \
        for (int i = 0; i < 32; i += 16) {                                              \
            int x = (int)((a >> i) & 0xffffu), y = (int)((b >> i) & 0xffffu), z = (int)((c >> i) & 0xffffu); \
            if (sgn) { x = (short)x; y = (short)y; z = (short)z; }                      \
            const int sum = sgn ? (int)(short)(x + y) : ((x + y) & 0xffff);             \
            (void)sum;                                                                  \
            r |= ((unsigned int)(expr) & 0xffffu) << i;                                 \
        }                                                                               \
        return r;                                                                       \
    }
#define OCU_SIMD_H2(name, expr)                                                         \
    OCU_LANG_HD unsigned int name(const unsigned int a, const unsigned int b)          \
    {                                                                                   \
        unsigned int r = 0;                                                             \
        for (int i = 0; i < 32; i += 16) {                                              \
            const int x = (short)((a >> i) & 0xffffu), y = (short)((b >> i) & 0xffffu);  \
            r |= ((unsigned int)(expr) & 0xffffu) << i;                                 \
        }                                                                               \
        return r;                                                                       \
    }

OCU_SIMD_H3(__vimax3_s16x2, 1, OCU_SIMD_MAX(OCU_SIMD_MAX(x, y), z))
OCU_SIMD_H3(__vimax3_u16x2, 0, OCU_SIMD_MAX(OCU_SIMD_MAX(x, y), z))
OCU_SIMD_H3(__vimax3_s16x2_relu, 1, OCU_SIMD_MAX(OCU_SIMD_MAX(OCU_SIMD_MAX(x, y), z), 0))
OCU_SIMD_H3(__vimin3_s16x2, 1, OCU_SIMD_MIN(OCU_SIMD_MIN(x, y), z))
OCU_SIMD_H3(__vimin3_u16x2, 0, OCU_SIMD_MIN(OCU_SIMD_MIN(x, y), z))
OCU_SIMD_H3(__vimin3_s16x2_relu, 1, OCU_SIMD_MAX(OCU_SIMD_MIN(OCU_SIMD_MIN(x, y), z), 0))
OCU_SIMD_H3(__viaddmax_s16x2, 1, OCU_SIMD_MAX(sum, z))
OCU_SIMD_H3(__viaddmax_u16x2, 0, OCU_SIMD_MAX(sum, z))
OCU_SIMD_H3(__viaddmax_s16x2_relu, 1, OCU_SIMD_MAX(OCU_SIMD_MAX(sum, z), 0))
OCU_SIMD_H3(__viaddmin_s16x2, 1, OCU_SIMD_MIN(sum, z))
OCU_SIMD_H3(__viaddmin_u16x2, 0, OCU_SIMD_MIN(sum, z))
OCU_SIMD_H3(__viaddmin_s16x2_relu, 1, OCU_SIMD_MAX(OCU_SIMD_MIN(sum, z), 0))
OCU_SIMD_H2(__vimax_s16x2_relu, OCU_SIMD_MAX(OCU_SIMD_MAX(x, y), 0))
OCU_SIMD_H2(__vimin_s16x2_relu, OCU_SIMD_MAX(OCU_SIMD_MIN(x, y), 0))

OCU_LANG_HD int __vimax3_s32(const int a, const int b, const int c) { return OCU_SIMD_MAX(OCU_SIMD_MAX(a, b), c); }
OCU_LANG_HD unsigned int __vimax3_u32(const unsigned int a, const unsigned int b, const unsigned int c) { return OCU_SIMD_MAX(OCU_SIMD_MAX(a, b), c); }
OCU_LANG_HD int __vimax3_s32_relu(const int a, const int b, const int c) { return OCU_SIMD_MAX(OCU_SIMD_MAX(OCU_SIMD_MAX(a, b), c), 0); }
OCU_LANG_HD int __vimin3_s32(const int a, const int b, const int c) { return OCU_SIMD_MIN(OCU_SIMD_MIN(a, b), c); }
OCU_LANG_HD unsigned int __vimin3_u32(const unsigned int a, const unsigned int b, const unsigned int c) { return OCU_SIMD_MIN(OCU_SIMD_MIN(a, b), c); }
OCU_LANG_HD int __vimin3_s32_relu(const int a, const int b, const int c) { return OCU_SIMD_MAX(OCU_SIMD_MIN(OCU_SIMD_MIN(a, b), c), 0); }
OCU_LANG_HD int __vimax_s32_relu(const int a, const int b) { return OCU_SIMD_MAX(OCU_SIMD_MAX(a, b), 0); }
OCU_LANG_HD int __vimin_s32_relu(const int a, const int b) { return OCU_SIMD_MAX(OCU_SIMD_MIN(a, b), 0); }
/* a + b wraps (computed as unsigned). */
OCU_LANG_HD int __viaddmax_s32(const int a, const int b, const int c) { const int s = (int)((unsigned)a + (unsigned)b); return OCU_SIMD_MAX(s, c); }
OCU_LANG_HD unsigned int __viaddmax_u32(const unsigned int a, const unsigned int b, const unsigned int c) { const unsigned int s = a + b; return OCU_SIMD_MAX(s, c); }
OCU_LANG_HD int __viaddmax_s32_relu(const int a, const int b, const int c) { const int s = (int)((unsigned)a + (unsigned)b); return OCU_SIMD_MAX(OCU_SIMD_MAX(s, c), 0); }
OCU_LANG_HD int __viaddmin_s32(const int a, const int b, const int c) { const int s = (int)((unsigned)a + (unsigned)b); return OCU_SIMD_MIN(s, c); }
OCU_LANG_HD unsigned int __viaddmin_u32(const unsigned int a, const unsigned int b, const unsigned int c) { const unsigned int s = a + b; return OCU_SIMD_MIN(s, c); }
OCU_LANG_HD int __viaddmin_s32_relu(const int a, const int b, const int c) { const int s = (int)((unsigned)a + (unsigned)b); return OCU_SIMD_MAX(OCU_SIMD_MIN(s, c), 0); }

#ifdef __cplusplus
/* max / min and a predicate: *pred = (a >= b) for max, (a <= b) for min;
   in the 16x2 forms per halfword, pred_hi for the upper one. */
OCU_LANG_HD int __vibmax_s32(const int a, const int b, bool *const pred) { *pred = a >= b; return OCU_SIMD_MAX(a, b); }
OCU_LANG_HD unsigned int __vibmax_u32(const unsigned int a, const unsigned int b, bool *const pred) { *pred = a >= b; return OCU_SIMD_MAX(a, b); }
OCU_LANG_HD int __vibmin_s32(const int a, const int b, bool *const pred) { *pred = a <= b; return OCU_SIMD_MIN(a, b); }
OCU_LANG_HD unsigned int __vibmin_u32(const unsigned int a, const unsigned int b, bool *const pred) { *pred = a <= b; return OCU_SIMD_MIN(a, b); }
#define OCU_SIMD_PRED(name, sgn, ismax)                                                 \
    OCU_LANG_HD unsigned int name(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo) \
    {                                                                                   \
        unsigned int r = 0;                                                             \
        for (int i = 0; i < 32; i += 16) {                                              \
            int x = (int)((a >> i) & 0xffffu), y = (int)((b >> i) & 0xffffu);           \
            if (sgn) { x = (short)x; y = (short)y; }                                    \
            const bool p = ismax ? x >= y : x <= y;                                     \
            *(i ? pred_hi : pred_lo) = p;                                               \
            r |= ((unsigned int)(ismax ? OCU_SIMD_MAX(x, y) : OCU_SIMD_MIN(x, y)) & 0xffffu) << i; \
        }                                                                               \
        return r;                                                                       \
    }
OCU_SIMD_PRED(__vibmax_s16x2, 1, 1)
OCU_SIMD_PRED(__vibmax_u16x2, 0, 1)
OCU_SIMD_PRED(__vibmin_s16x2, 1, 0)
OCU_SIMD_PRED(__vibmin_u16x2, 0, 0)
#undef OCU_SIMD_PRED
#endif

#undef OCU_SIMD_BIN
#undef OCU_SIMD_UN
#undef OCU_SIMD_BOTH
#undef OCU_SIMD_SAD
#undef OCU_SIMD_H3
#undef OCU_SIMD_H2
#undef OCU_SIMD_CLAMP
#undef OCU_SIMD_ABS
#undef OCU_SIMD_MAX
#undef OCU_SIMD_MIN

/* Classification of a float: 1 for NaN (__isnanf) or infinity (__isinff), else 0. */
#if !defined(__MINGW32__) /* MinGW's math.h has its own __isnanf (nonzero for NaN) */
OCU_LANG_HD int __isnanf(float x) { return __builtin_isnan(x) ? 1 : 0; }
#endif
OCU_LANG_HD int __isinff(float x) { return __builtin_isinf(x) ? 1 : 0; }

/*
 * Load through the read-only data cache. The cache changes speed, not the
 * value; the data must not be written during the kernel (as documented),
 * so a plain load is a correct implementation on every path.
 */
template <typename T>
OCU_LANG_FN T __ldg(const T *p)
{
    return *p;
}

/*
 * Loads and stores with a cache hint (programming guide, "Load Functions
 * Using Cache Hints" and "Store Functions Using Cache Hints"):
 *   __ldcg  cache at the global level (L2 and below), not in L1
 *   __ldca  cache at all levels
 *   __ldcs  cache streaming, likely to be accessed once
 *   __ldlu  last use
 *   __ldcv  do not cache, fetch again (volatile)
 *   __stcg  cache at global level      __stcs  cache streaming
 *   __stwb  cache write-back, all coherent levels
 *   __stwt  cache write-through (system)
 * The hint changes where the data is kept, never the value read or written,
 * so a plain access is a correct implementation on every path. The
 * documented operand types are the integer, floating-point and vector types
 * of that size; here any trivially copyable type is accepted. __ldcv also
 * stops the compiler from reusing an earlier read.
 */
template <typename T> struct ocu_hint_nd { using type = T; }; /* the value converts to the pointee type */
template <typename T> OCU_LANG_FN T __ldcg(const T *p) { return *p; }
template <typename T> OCU_LANG_FN T __ldca(const T *p) { return *p; }
template <typename T> OCU_LANG_FN T __ldcs(const T *p) { return *p; }
template <typename T> OCU_LANG_FN T __ldlu(const T *p) { return *p; }
template <typename T>
OCU_LANG_FN T __ldcv(const T *p)
{
    T r;
    __asm__ __volatile__("" ::: "memory");
    __builtin_memcpy(&r, static_cast<const void *>(p), sizeof(T));
    __asm__ __volatile__("" ::: "memory");
    return r;
}
template <typename T> OCU_LANG_FN void __stcg(T *p, typename ocu_hint_nd<T>::type v) { *p = v; }
template <typename T> OCU_LANG_FN void __stcs(T *p, typename ocu_hint_nd<T>::type v) { *p = v; }
template <typename T> OCU_LANG_FN void __stwb(T *p, typename ocu_hint_nd<T>::type v) { *p = v; }
template <typename T> OCU_LANG_FN void __stwt(T *p, typename ocu_hint_nd<T>::type v) { *p = v; }

/*
 * Programmatic dependent launch, device side.
 *   cudaGridDependencySynchronize(): waits until the grids this grid
 *     depends on have completed and their memory operations are visible.
 *   cudaTriggerProgrammaticLaunchCompletion(): allows a dependent grid to
 *     start before this one has finished.
 * This layer never starts a kernel before the earlier work in its stream
 * has completed (cudaLaunchKernelEx keeps full stream order), so at the
 * first instruction of a kernel the dependency is already resolved: the
 * wait returns at once and the trigger has nothing to release.
 */
OCU_LANG_FN void cudaGridDependencySynchronize(void) {}
OCU_LANG_FN void cudaTriggerProgrammaticLaunchCompletion(void) {}

#endif /* OCU_CUDA_LANG_COMMON_H */

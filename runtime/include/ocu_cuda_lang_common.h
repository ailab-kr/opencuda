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

#endif /* OCU_CUDA_LANG_COMMON_H */

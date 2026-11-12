/*
 * ocu_vector_types.h -- the built-in vector types (char1 ... double4) and
 * their make_<type> constructors, as the public CUDA programming guide
 * describes them: members x, y, z, w and the documented alignment of each
 * type. Written for this project; not derived from any vendor header.
 *
 * Included by cuda_runtime_api.h. C and C++.
 */
#ifndef OCU_VECTOR_TYPES_H
#define OCU_VECTOR_TYPES_H

#if defined(__cplusplus) && defined(__CUDA__)
/* Clang's CUDA mode: the constructors are host and device functions. */
#define OCU_VT_ALIGN(n) alignas(n)
#define OCU_VT_INLINE static inline __attribute__((host)) __attribute__((device))
#elif defined(__cplusplus)
#define OCU_VT_ALIGN(n) alignas(n)
#define OCU_VT_INLINE static inline
#else
#define OCU_VT_ALIGN(n) _Alignas(n)
#define OCU_VT_INLINE static inline
#endif

/* One type: struct + typedef + make_<name>(). `a` is the alignment, set
   on the first member so it applies in C and C++ alike. */
#define OCU_VT1(name, T, a)                                                       \
    struct name { OCU_VT_ALIGN(a) T x; };                                         \
    typedef struct name name;                                                     \
    OCU_VT_INLINE name make_##name(T x) { name v; v.x = x; return v; }
#define OCU_VT2(name, T, a)                                                       \
    struct name { OCU_VT_ALIGN(a) T x; T y; };                                      \
    typedef struct name name;                                                     \
    OCU_VT_INLINE name make_##name(T x, T y) { name v; v.x = x; v.y = y; return v; }
#define OCU_VT3(name, T, a)                                                       \
    struct name { OCU_VT_ALIGN(a) T x; T y, z; };                                   \
    typedef struct name name;                                                     \
    OCU_VT_INLINE name make_##name(T x, T y, T z)                                 \
    { name v; v.x = x; v.y = y; v.z = z; return v; }
#define OCU_VT4(name, T, a)                                                       \
    struct name { OCU_VT_ALIGN(a) T x; T y, z, w; };                                \
    typedef struct name name;                                                     \
    OCU_VT_INLINE name make_##name(T x, T y, T z, T w)                            \
    { name v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }

/* Element type, and the alignment of the 1-, 2-, 3- and 4-element forms. */
#define OCU_VT_FAMILY(base, T, a1, a2, a3, a4)                                    \
    OCU_VT1(base##1, T, a1)                                                       \
    OCU_VT2(base##2, T, a2)                                                       \
    OCU_VT3(base##3, T, a3)                                                       \
    OCU_VT4(base##4, T, a4)

#ifdef __cplusplus
extern "C++" {
#endif

OCU_VT_FAMILY(char, signed char, 1, 2, 1, 4)
OCU_VT_FAMILY(uchar, unsigned char, 1, 2, 1, 4)
OCU_VT_FAMILY(short, short, 2, 4, 2, 8)
OCU_VT_FAMILY(ushort, unsigned short, 2, 4, 2, 8)
OCU_VT_FAMILY(int, int, 4, 8, 4, 16)
OCU_VT_FAMILY(uint, unsigned int, 4, 8, 4, 16)
#if defined(__LP64__) || defined(_LP64)
OCU_VT_FAMILY(long, long, 8, 16, 8, 16)
OCU_VT_FAMILY(ulong, unsigned long, 8, 16, 8, 16)
#else
OCU_VT_FAMILY(long, long, 4, 8, 4, 16)
OCU_VT_FAMILY(ulong, unsigned long, 4, 8, 4, 16)
#endif
OCU_VT_FAMILY(longlong, long long, 8, 16, 8, 16)
OCU_VT_FAMILY(ulonglong, unsigned long long, 8, 16, 8, 16)
OCU_VT_FAMILY(float, float, 4, 8, 4, 16)
OCU_VT_FAMILY(double, double, 8, 16, 8, 16)

#ifdef __cplusplus
}
#endif

#undef OCU_VT_FAMILY
#undef OCU_VT1
#undef OCU_VT2
#undef OCU_VT3
#undef OCU_VT4
#undef OCU_VT_INLINE
#undef OCU_VT_ALIGN

/* Grid and block sizes: unspecified components are 1. */
#ifndef OCU_DIM3_DEFINED
#define OCU_DIM3_DEFINED
struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
    constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    constexpr operator uint3() const { return uint3{x, y, z}; }
#endif
};
typedef struct dim3 dim3;
#endif

#endif /* OCU_VECTOR_TYPES_H */

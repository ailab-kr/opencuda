// msl_stub.h -- the Metal Shading Language spellings the translated kernels
// use, mapped to C++ for OpenCL, so that a host Clang can check generated
// MSL (tests/metal/msl_check.py) on machines without a Metal compiler.
// Written for this project from the public MSL specification: names and
// signatures only, declarations without bodies (the check is -fsyntax-only).
//
// C++ for OpenCL is used because, like MSL, it has address spaces that are
// part of the type (device = __global, threadgroup = __local, constant =
// __constant, thread = __private, no implicit conversions between them),
// overloading on them, and threadgroup variables only in kernel scope.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define device __global
#define threadgroup __local
#define constant __constant
#define thread __private
#define kernel __kernel
// OpenCL keywords that are ordinary names in MSL
#define global ocu_kw_global
#define local ocu_kw_local
#define generic ocu_kw_generic
#define pipe ocu_kw_pipe
#define read_only ocu_kw_read_only
#define write_only ocu_kw_write_only
#define read_write ocu_kw_read_write

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef signed char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;
typedef long int64_t;
typedef ulong uint64_t;

#define OCU_STUB_VEC(T)                                       \
    typedef T T##2 __attribute__((ext_vector_type(2)));       \
    typedef T T##3 __attribute__((ext_vector_type(3)));       \
    typedef T T##4 __attribute__((ext_vector_type(4)));       \
    struct packed_##T##3 {                                    \
        T x, y, z;                                            \
        operator T##3() const;                                \
    };
OCU_STUB_VEC(char)
OCU_STUB_VEC(uchar)
OCU_STUB_VEC(short)
OCU_STUB_VEC(ushort)
OCU_STUB_VEC(int)
OCU_STUB_VEC(uint)
OCU_STUB_VEC(long)
OCU_STUB_VEC(ulong)
OCU_STUB_VEC(float)
OCU_STUB_VEC(half)
#undef OCU_STUB_VEC

#define FLT_MAX 0x1.fffffep127f
#define FLT_MIN 0x1p-126f
#define FLT_EPSILON 0x1p-23f
#define INFINITY __builtin_inff()
#define NAN __builtin_nanf("")
#define M_PI_F 3.14159274f
#define INT_MAX 2147483647
#define INT_MIN (-INT_MAX - 1)
#define UINT_MAX 4294967295u

namespace metal {
// memory_order, atomic_int, atomic_uint and atomic_float: Clang's OpenCL base header has them.
struct mem_flags {
    enum { mem_none = 0, mem_device = 1, mem_threadgroup = 2, mem_texture = 4 };
};
void threadgroup_barrier(int flags);
void simdgroup_barrier(int flags);
#define OCU_STUB_ATOMIC(name) template <class P, class V> V name(P p, V v, memory_order o);
OCU_STUB_ATOMIC(atomic_fetch_add_explicit)
OCU_STUB_ATOMIC(atomic_fetch_sub_explicit)
OCU_STUB_ATOMIC(atomic_fetch_min_explicit)
OCU_STUB_ATOMIC(atomic_fetch_max_explicit)
OCU_STUB_ATOMIC(atomic_fetch_and_explicit)
OCU_STUB_ATOMIC(atomic_fetch_or_explicit)
OCU_STUB_ATOMIC(atomic_fetch_xor_explicit)
OCU_STUB_ATOMIC(atomic_exchange_explicit)
#undef OCU_STUB_ATOMIC
template <class P, class E, class V> bool atomic_compare_exchange_weak_explicit(P p, E expected, V desired,
                                                                     memory_order s, memory_order f);
template <class P> auto atomic_load_explicit(P p, memory_order o) -> int;
template <class P, class V> void atomic_store_explicit(P p, V v, memory_order o);

#define OCU_STUB_M1(name) template <class T> T name(T x);
#define OCU_STUB_M2(name) template <class T> T name(T x, T y);
#define OCU_STUB_MATH                                                                  \
    OCU_STUB_M1(exp) OCU_STUB_M1(exp2) OCU_STUB_M1(exp10) OCU_STUB_M1(log)             \
    OCU_STUB_M1(log2) OCU_STUB_M1(log10) OCU_STUB_M1(sin) OCU_STUB_M1(cos)             \
    OCU_STUB_M1(tan) OCU_STUB_M1(asin) OCU_STUB_M1(acos) OCU_STUB_M1(atan)             \
    OCU_STUB_M1(sinh) OCU_STUB_M1(cosh) OCU_STUB_M1(tanh) OCU_STUB_M1(sqrt)            \
    OCU_STUB_M1(rsqrt) OCU_STUB_M1(fabs) OCU_STUB_M1(floor) OCU_STUB_M1(ceil)          \
    OCU_STUB_M1(trunc) OCU_STUB_M1(round) OCU_STUB_M1(rint) OCU_STUB_M1(abs)           \
    OCU_STUB_M1(saturate) OCU_STUB_M1(sign) OCU_STUB_M1(fract) OCU_STUB_M1(asinh)      \
    OCU_STUB_M1(acosh) OCU_STUB_M1(atanh) OCU_STUB_M1(cbrt)                            \
    OCU_STUB_M2(pow) OCU_STUB_M2(powr) OCU_STUB_M2(atan2) OCU_STUB_M2(fmod)            \
    OCU_STUB_M2(fmin) OCU_STUB_M2(fmax) OCU_STUB_M2(min) OCU_STUB_M2(max)              \
    OCU_STUB_M2(step) OCU_STUB_M2(divide) OCU_STUB_M2(copysign) OCU_STUB_M2(fdim)      \
    template <class T> T fma(T a, T b, T c);                                           \
    template <class T> T clamp(T x, T lo, T hi);                                       \
    template <class T> T mix(T x, T y, T a);                                           \
    template <class T> bool isnan(T x);                                                \
    template <class T> bool isinf(T x);                                                \
    template <class T> bool isfinite(T x);                                             \
    template <class T> bool signbit(T x);
OCU_STUB_MATH
namespace precise { OCU_STUB_MATH }
namespace fast { OCU_STUB_MATH }
#undef OCU_STUB_MATH
#undef OCU_STUB_M1
#undef OCU_STUB_M2
template <class T, class U> T select(T a, T b, U c);
template <class T> T popcount(T x);
template <class T> T clz(T x);
template <class T> T ctz(T x);
template <class T> T reverse_bits(T x);
template <class T> T mulhi(T x, T y);
template <class T> T rotate(T x, T y);
template <class T, class U> T as_type(U x);
template <class T> float dot(T a, T b);
template <class T> float length(T a);
template <class T> T normalize(T a);
template <class T> T cross(T a, T b);
template <class T> float distance(T a, T b);

typedef ulong simd_vote; // a struct in MSL, explicitly convertible to uint64_t
template <class T> T simd_shuffle(T v, ushort lane);
template <class T> T simd_shuffle_up(T v, ushort d);
template <class T> T simd_shuffle_down(T v, ushort d);
template <class T> T simd_shuffle_xor(T v, ushort m);
template <class T> T simd_broadcast(T v, ushort lane);
template <class T> T simd_sum(T v);
template <class T> T simd_min(T v);
template <class T> T simd_max(T v);
template <class T> T simd_prefix_exclusive_sum(T v);
template <class T> T simd_prefix_inclusive_sum(T v);
simd_vote simd_ballot(bool p);
bool simd_any(bool p);
bool simd_all(bool p);
} // namespace metal

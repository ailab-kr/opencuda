/* Shift helpers with PTX semantics, for device code compiled with
 * ocu_cc.py --ptx-shift (compiler/probe/ptx_shift.py rewrites a << b into
 * __ocu_shift_lhs(a) << b and so on, see the end of this file; this header
 * is force-included then).
 *
 * Semantics (PTX ISA 8.5, 9.7.7.8 "Logic and Shift Instructions: shl" and
 * 9.7.7.9 "...: shr", 9.7.9.8/9.7.9.9 in ISA 9.5; 9.4: the amount operand
 * is always .u32): the shift
 * amount is taken as an unsigned number and amounts
 * larger than the operand width are clamped to the width. So
 *   shl:            amount >= width gives 0;
 *   shr, unsigned:  amount >= width gives 0;
 *   shr, signed:    arithmetic; amount >= width gives all sign bits
 *                   (0 or -1).
 * A negative amount is a large unsigned one, so it clamps too. The width
 * is that of the left operand after the usual integer promotion (32 or
 * 64 bits; 8- and 16-bit values are shifted as int), and the result has
 * that type, as for the built-in operator. A left shift of a signed value
 * shifts its bits (two's complement), as shl does, also for negative
 * values. Shifts by amounts within the width give what C++ gives.
 *
 * Amounts are compared as values: an amount of 2^32 or more clamps (PTX's
 * amount operand is 32 bits; C++ gives no rule for wider amounts).
 *
 * Every helper is constexpr, so a rewritten shift is still a constant
 * expression where the original was one. Operands that are not integers
 * after promotion (a class with its own operator<<, in a template) use
 * the operator unchanged.
 */
#ifndef OCU_PTX_SHIFT_H
#define OCU_PTX_SHIFT_H

#include "cuda_runtime.h" /* __host__ / __device__ */

#include <type_traits>
#include <utility>

namespace ocu_ptx_shift {

template <class T, class = void>
struct promoted {
    using type = void;
};
template <class T>
struct promoted<T, std::void_t<decltype(+std::declval<T>())>> {
    using type = decltype(+std::declval<T>());
};
template <class T>
using promoted_t = typename promoted<T>::type;

template <class A, class C>
inline constexpr bool integral_v =
    std::is_integral<promoted_t<A>>::value && std::is_integral<promoted_t<C>>::value;

/* The amount, clamped to `width`. */
template <class C>
__host__ __device__ constexpr unsigned amount(C c, unsigned width) {
    using U = std::make_unsigned_t<promoted_t<C>>;
    U u = static_cast<U>(c);
    return u >= width ? width : static_cast<unsigned>(u);
}

#if (defined(__NVPTX__) && defined(__CUDA_ARCH__))
/* NVIDIA device pass: the PTX instructions clamp by themselves, so a shift
 * by an amount the compiler cannot see is one instruction (the portable
 * form below costs a compare and a select). The amount operand is .u32. */
__device__ inline unsigned ptx_shl(unsigned a, unsigned n) {
    unsigned r;
    asm("shl.b32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(n));
    return r;
}
__device__ inline unsigned long long ptx_shl(unsigned long long a, unsigned n) {
    unsigned long long r;
    asm("shl.b64 %0, %1, %2;" : "=l"(r) : "l"(a), "r"(n));
    return r;
}
__device__ inline unsigned ptx_shr(unsigned a, unsigned n) {
    unsigned r;
    asm("shr.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(n));
    return r;
}
__device__ inline int ptx_shr(int a, unsigned n) {
    int r;
    asm("shr.s32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(n));
    return r;
}
__device__ inline unsigned long long ptx_shr(unsigned long long a, unsigned n) {
    unsigned long long r;
    asm("shr.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "r"(n));
    return r;
}
__device__ inline long long ptx_shr(long long a, unsigned n) {
    long long r;
    asm("shr.s64 %0, %1, %2;" : "=l"(r) : "l"(a), "r"(n));
    return r;
}

/* The instruction can be used: not a constant evaluation, an amount the
 * compiler cannot fold (constants keep the portable form, which folds),
 * of at most 32 bits (wider amounts clamp by value, see above), and a
 * 32- or 64-bit operand. */
template <class P, class C>
__device__ constexpr bool use_ptx(C c) {
    return !__builtin_is_constant_evaluated() && !__builtin_constant_p(c) &&
           sizeof(promoted_t<C>) <= 4 && (sizeof(P) == 4 || sizeof(P) == 8);
}
#endif

template <class P, class C>
__host__ __device__ constexpr P shl(P a, C c) {
    constexpr unsigned width = sizeof(P) * 8;
    using U = std::make_unsigned_t<P>;
#if (defined(__NVPTX__) && defined(__CUDA_ARCH__))
    if (use_ptx<P>(c)) {
        using W = std::conditional_t<sizeof(P) == 4, unsigned, unsigned long long>;
        return static_cast<P>(ptx_shl(static_cast<W>(static_cast<U>(a)), static_cast<unsigned>(c)));
    }
#endif
    unsigned n = amount(c, width);
    return n >= width ? P(0) : static_cast<P>(static_cast<U>(static_cast<U>(a) << n));
}

template <class P, class C>
__host__ __device__ constexpr P shr(P a, C c) {
    constexpr unsigned width = sizeof(P) * 8;
#if (defined(__NVPTX__) && defined(__CUDA_ARCH__))
    if (use_ptx<P>(c)) {
        using W = std::conditional_t<
            std::is_signed<P>::value, std::conditional_t<sizeof(P) == 4, int, long long>,
            std::conditional_t<sizeof(P) == 4, unsigned, unsigned long long>>;
        return static_cast<P>(ptx_shr(static_cast<W>(a), static_cast<unsigned>(c)));
    }
#endif
    unsigned n = amount(c, width);
    if constexpr (std::is_signed<P>::value) {
        return static_cast<P>(a >> (n >= width ? width - 1 : n));
    } else {
        return n >= width ? P(0) : static_cast<P>(a >> n);
    }
}

} // namespace ocu_ptx_shift

template <class A, class C>
__host__ __device__ constexpr decltype(auto) __ocu_shl(A a, C c) {
    if constexpr (ocu_ptx_shift::integral_v<A, C>) {
        using P = ocu_ptx_shift::promoted_t<A>;
        return ocu_ptx_shift::shl<P>(static_cast<P>(a), +c);
    } else {
        return a << c;
    }
}

template <class A, class C>
__host__ __device__ constexpr decltype(auto) __ocu_shr(A a, C c) {
    if constexpr (ocu_ptx_shift::integral_v<A, C>) {
        using P = ocu_ptx_shift::promoted_t<A>;
        return ocu_ptx_shift::shr<P>(static_cast<P>(a), +c);
    } else {
        return a >> c;
    }
}

/* a <<= c and a >>= c: `a` is evaluated once, as in the built-in form. */
template <class A, class C>
__host__ __device__ constexpr decltype(auto) __ocu_shl_assign(A &a, C c) {
    if constexpr (ocu_ptx_shift::integral_v<A &, C>) {
        a = static_cast<std::remove_volatile_t<A>>(__ocu_shl(a, c));
        return (a);
    } else {
        return (a <<= c);
    }
}

template <class A, class C>
__host__ __device__ constexpr decltype(auto) __ocu_shr_assign(A &a, C c) {
    if constexpr (ocu_ptx_shift::integral_v<A &, C>) {
        a = static_cast<std::remove_volatile_t<A>>(__ocu_shr(a, c));
        return (a);
    } else {
        return (a >>= c);
    }
}

/* What the rewrite produces: the left operand wrapped, the operator kept.
 *   a << b   ->  __ocu_shift_lhs(a) << b     (also >>)
 *   a <<= b  ->  __ocu_shift_ref(a) <<= b    (also >>=)
 * An overloaded operator written as an operator is sequenced like the
 * built-in one (ISO C++17 [over.match.oper]/2), so the left operand of <<
 * and >> is still evaluated before the right one ([expr.shift]/4) and the
 * right operand of <<= and >>= before the left one ([expr.ass]/1). Calls
 * such as __ocu_shl(a, b) would leave the order unspecified. */
namespace ocu_ptx_shift {

template <class A>
struct lhs {
    A a;
    template <class C>
    __host__ __device__ constexpr decltype(auto) operator<<(C c) {
        return __ocu_shl(a, c);
    }
    template <class C>
    __host__ __device__ constexpr decltype(auto) operator>>(C c) {
        return __ocu_shr(a, c);
    }
};

template <class A>
struct ref {
    A *a;
    template <class C>
    __host__ __device__ constexpr decltype(auto) operator<<=(C c) {
        return __ocu_shl_assign(*a, c);
    }
    template <class C>
    __host__ __device__ constexpr decltype(auto) operator>>=(C c) {
        return __ocu_shr_assign(*a, c);
    }
};

} // namespace ocu_ptx_shift

template <class A>
__host__ __device__ constexpr ocu_ptx_shift::lhs<A> __ocu_shift_lhs(A a) {
    return {a};
}

template <class A>
__host__ __device__ constexpr ocu_ptx_shift::ref<A> __ocu_shift_ref(A &a) {
    return {&a};
}

#endif /* OCU_PTX_SHIFT_H */

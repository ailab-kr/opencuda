/* Shift helpers with PTX semantics, for device code compiled with
 * ocu_cc.py --ptx-shift (compiler/probe/ptx_shift.py rewrites a << b into
 * __ocu_shl(a , b) and so on; this header is force-included then).
 *
 * Semantics (PTX ISA 9.5, 9.7.9.8 "Logic and Shift Instructions: shl" and
 * 9.7.9.9 "...: shr"; 9.4: the amount operand is always .u32): the shift
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

template <class P>
__host__ __device__ constexpr P shl(P a, unsigned n) {
    constexpr unsigned width = sizeof(P) * 8;
    using U = std::make_unsigned_t<P>;
    return n >= width ? P(0) : static_cast<P>(static_cast<U>(static_cast<U>(a) << n));
}

template <class P>
__host__ __device__ constexpr P shr(P a, unsigned n) {
    constexpr unsigned width = sizeof(P) * 8;
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
        return ocu_ptx_shift::shl<P>(static_cast<P>(a), ocu_ptx_shift::amount(c, sizeof(P) * 8));
    } else {
        return a << c;
    }
}

template <class A, class C>
__host__ __device__ constexpr decltype(auto) __ocu_shr(A a, C c) {
    if constexpr (ocu_ptx_shift::integral_v<A, C>) {
        using P = ocu_ptx_shift::promoted_t<A>;
        return ocu_ptx_shift::shr<P>(static_cast<P>(a), ocu_ptx_shift::amount(c, sizeof(P) * 8));
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

#endif /* OCU_PTX_SHIFT_H */

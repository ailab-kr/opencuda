/*
 * ocu_cuda_lang_scoped.h -- scoped atomics and the removed non-sync warp
 * intrinsics. Included at the end of ocu_cuda_lang_cpu.h and
 * ocu_cuda_lang_nvptx.h, after their atomicAdd/atomicExch/... exist and
 * after ocu_cuda_lang_common.h defined ocu_hint_nd.
 *
 * Written for this project from the public CUDA C++ Programming Guide
 * ("Atomic Functions": the _block and _system variants, available from
 * compute capability 6.0; "Warp Shuffle Functions" and "Warp Vote
 * Functions": the variants without _sync are deprecated since CUDA 9.0 and
 * not supported on compute capability 7.0 and above). Not derived from any
 * vendor header.
 *
 * Scopes:
 *   _block   the atomic is atomic with respect to threads of the same block.
 *            An atomic that is atomic for the whole device satisfies it, so
 *            each _block function does what the plain one does.
 *   _system  atomic with respect to every thread of the system, host
 *            threads included. On the CPU path every thread is a host
 *            thread, so the plain atomic satisfies it. On the NVIDIA path
 *            the plain atomic is device-wide only; a system-wide atomic is
 *            not provided there and the function fails to compile with a
 *            message (never silently weaker).
 */
#ifndef OCU_CUDA_LANG_SCOPED_H
#define OCU_CUDA_LANG_SCOPED_H

#if defined(OCU_SCOPED_SYSTEM_ATOMICS)
#define OCU_SCOPED_SYSTEM_BODY(call) return call;
#else
#define OCU_SCOPED_SYSTEM_BODY(call) \
    static_assert(sizeof(T) == 0, "system-scope atomics are not supported on this target"); \
    return T();
#endif

#define OCU_SCOPED_RMW(op)                                                                  \
    template <typename T>                                                                   \
    OCU_LANG_FN T atomic##op##_block(T *a, typename ocu_hint_nd<T>::type v)                 \
    { return atomic##op(a, v); }                                                            \
    template <typename T>                                                                   \
    OCU_LANG_FN T atomic##op##_system(T *a, typename ocu_hint_nd<T>::type v)                \
    { OCU_SCOPED_SYSTEM_BODY(atomic##op(a, v)) }

OCU_SCOPED_RMW(Add)
OCU_SCOPED_RMW(Sub)
OCU_SCOPED_RMW(Exch)
OCU_SCOPED_RMW(Min)
OCU_SCOPED_RMW(Max)
OCU_SCOPED_RMW(And)
OCU_SCOPED_RMW(Or)
OCU_SCOPED_RMW(Xor)
#undef OCU_SCOPED_RMW

/* atomicInc / atomicDec: unsigned int, wrapping at the limit. */
OCU_LANG_FN unsigned atomicInc_block(unsigned *a, unsigned limit) { return atomicInc(a, limit); }
OCU_LANG_FN unsigned atomicDec_block(unsigned *a, unsigned limit) { return atomicDec(a, limit); }
#if defined(OCU_SCOPED_SYSTEM_ATOMICS)
OCU_LANG_FN unsigned atomicInc_system(unsigned *a, unsigned limit) { return atomicInc(a, limit); }
OCU_LANG_FN unsigned atomicDec_system(unsigned *a, unsigned limit) { return atomicDec(a, limit); }
#else
template <typename T = unsigned>
OCU_LANG_FN unsigned atomicInc_system(unsigned *, unsigned)
{ static_assert(sizeof(T) == 0, "system-scope atomics are not supported on this target"); return 0; }
template <typename T = unsigned>
OCU_LANG_FN unsigned atomicDec_system(unsigned *, unsigned)
{ static_assert(sizeof(T) == 0, "system-scope atomics are not supported on this target"); return 0; }
#endif

template <typename T>
OCU_LANG_FN T atomicCAS_block(T *a, typename ocu_hint_nd<T>::type c, typename ocu_hint_nd<T>::type v)
{ return atomicCAS(a, c, v); }
template <typename T>
OCU_LANG_FN T atomicCAS_system(T *a, typename ocu_hint_nd<T>::type c, typename ocu_hint_nd<T>::type v)
{ OCU_SCOPED_SYSTEM_BODY(atomicCAS(a, c, v)) }
#undef OCU_SCOPED_SYSTEM_BODY

/*
 * __shfl, __shfl_up, __shfl_down, __shfl_xor, __ballot, __any and __all
 * (no _sync, no mask): removed for compute capability 7.0 and above, which
 * is every architecture this project reports. Their meaning there is not
 * defined, so a call stops the build and names the replacement. (The
 * templates only fail when instantiated, so code behind a guard that is
 * never taken still compiles.)
 */
#define OCU_REMOVED_WARP(name, sync)                                                       \
    template <typename T, typename... A>                                                   \
    OCU_LANG_FN T name(T, A...)                                                            \
    {                                                                                      \
        static_assert(sizeof(T) == 0, #name " was removed for compute capability 7.0 and " \
                      "above and is not supported; use " #sync " with a lane mask");       \
        return T();                                                                        \
    }
OCU_REMOVED_WARP(__shfl, __shfl_sync)
OCU_REMOVED_WARP(__shfl_up, __shfl_up_sync)
OCU_REMOVED_WARP(__shfl_down, __shfl_down_sync)
OCU_REMOVED_WARP(__shfl_xor, __shfl_xor_sync)
OCU_REMOVED_WARP(__ballot, __ballot_sync)
OCU_REMOVED_WARP(__any, __any_sync)
OCU_REMOVED_WARP(__all, __all_sync)
#undef OCU_REMOVED_WARP

#endif /* OCU_CUDA_LANG_SCOPED_H */

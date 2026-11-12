/*
 * ocu_cuda_lang_cpu.h -- the CUDA language names a kernel body uses, for
 * the CPU path: .cu files rewritten by the CPU rewriter and compiled as
 * plain C++17. cuda_runtime.h includes it when OCU_CPU_PATH is defined.
 *
 * Written for this project from the public description of the CUDA C++
 * language (built-in variables, __syncthreads, warp functions, atomics,
 * intrinsics). Not derived from any vendor header.
 *
 * Launches go through the runtime (ocu_rt_launch): the stream argument of
 * K<<<grid, block, shared, stream>>> is honoured and launch errors become
 * the thread's last error, as with any CUDA launch.
 */
#ifndef OCU_CUDA_LANG_CPU_H
#define OCU_CUDA_LANG_CPU_H

#ifndef __cplusplus
#error "the CPU path compiles .cu files as C++17"
#endif

#include "cuda_runtime_api.h"
#include "ocu_rt.h"
#include "ocu_backend.h"
#include "ocu_cpu_kernel.h"
#include "ocu_fma_exact.h"

/* C headers whose names CUDA programs use without including them: the
   vendor toolchain's own headers bring them in. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

/* ---- declaration specifiers -------------------------------------------- */

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
/* An optimisation hint only. Left empty: the standard library spells the
   GCC attribute __attribute__((__noinline__)), which a non-empty macro
   would break. */
#define __noinline__
/* ocu_cc preprocesses first and rewrites after: there __shared__ must stay
   recognisable (extern __shared__ arrays are rewritten). */
#ifdef OCU_CPU_PREPROCESS
#define __shared__ __ocu_shared__
#else
#define __shared__ static thread_local
#endif
#define __constant__
#define __managed__
#define __align__(n) alignas(n)
#define __restrict__ __restrict
#define __launch_bounds__(...)

namespace ocu_cpu {

/* A literal 0 passed to a kernel: converts to any pointer or number, as
   the literal would in a direct call. */
struct literal_zero {
    template <class T, class = std::enable_if_t<std::is_pointer_v<T> || std::is_arithmetic_v<T> ||
                                                std::is_enum_v<T> || std::is_member_pointer_v<T>>>
    constexpr operator T() const { return T(0); }
};


inline void *dynamic_shared() { return ocu_cpu_dynamic_shared(); }

/*
 * `extern __shared__ T name[];` becomes `static dyn_shared<T> name;`: the
 * address is looked up at each use, so it works at file scope and inside
 * functions, and every block sees its own memory.
 */
template <typename T>
struct dyn_shared {
    operator T *() const { return static_cast<T *>(ocu_cpu_dynamic_shared()); }
    /* (U *)name and static_cast<U *>(name), as for the array. */
    template <typename U>
    explicit operator U *() const { return static_cast<U *>(ocu_cpu_dynamic_shared()); }
    T &operator[](std::size_t i) const { return static_cast<T *>(*this)[i]; }
};

/* ---- launches ------------------------------------------------------------ */

struct config {
    dim3 grid, block;
    std::size_t shared;
    cudaStream_t stream;
};

inline config cfg(dim3 grid, dim3 block, std::size_t shared = 0,
                  cudaStream_t stream = nullptr)
{
    return config{grid, block, shared, stream};
}

/* K<<<g, b, s, 0>>>: a literal 0 for the stream. */
inline config cfg(dim3 grid, dim3 block, std::size_t shared, std::nullptr_t)
{
    return config{grid, block, shared, nullptr};
}
/* K<<<g, b, s, NULL>>>: NULL is __null (a long) after GCC preprocessing. */
inline config cfg(dim3 grid, dim3 block, std::size_t shared, long stream)
{
    return config{grid, block, shared,
                  stream == 0 ? nullptr
                              : reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream))};
}
inline config cfg(dim3 grid, dim3 block, std::size_t shared, int stream)
{
    return config{grid, block, shared,
                  stream == 0 ? nullptr
                              : reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream))};
}

/*
 * One module per (launch site, argument types): the rewriter turns
 * K<<<cfg>>>(args) into launch(cfg, [](auto &&...a) { K(a...); }, args), so
 * the kernel call inside the lambda is an ordinary C++ call: template
 * argument deduction, overloads and conversions work as written.
 */
template <typename F, typename... A>
struct thunk {
    template <std::size_t... I>
    static void call(void **a, std::index_sequence<I...>)
    {
        F &f = *static_cast<F *>(a[0]);
        f(*static_cast<A *>(a[I + 1])...);
    }
    static void trampoline(void **a)
    {
        call(a, std::index_sequence_for<A...>{});
    }
    static ocu_kernel *kernel()
    {
        static ocu_kernel *k = [] {
            static const ocu_cpu_kernel_entry entry = {"trampoline", &trampoline};
            static const ocu_cpu_image image = {OCU_CPU_IMAGE_MAGIC, OCU_CPU_IMAGE_VERSION,
                                                1, &entry};
            const ocu_backend *b = ocu_rt_backend();
            ocu_module *m = nullptr;
            ocu_kernel *out = nullptr;
            if (b == nullptr ||
                b->module_load(ocu_rt_device(), &image, sizeof(image), &m) != OCU_OK ||
                b->module_get_kernel(m, "trampoline", &out) != OCU_OK) {
                return static_cast<ocu_kernel *>(nullptr); /* reported by the launch */
            }
            return out;
        }();
        return k;
    }
};

/* A literal 0 argument is passed as int when the kernel accepts an int
   there (a template parameter deduces int, as in a direct call), and as
   literal_zero (which converts to a pointer) otherwise. */
template <typename A>
using zero_as_int = std::conditional_t<std::is_same_v<std::decay_t<A>, literal_zero>, int,
                                       std::decay_t<A>>;

template <typename F, typename... V, typename... A>
cudaError_t launch_values(const config &c, F f, A &&...a);

template <typename F, typename... A>
cudaError_t launch(const config &c, F f, A &&...a)
{
    static_assert(std::is_empty_v<F>, "launch lambda must not capture");
    if constexpr ((std::is_same_v<std::decay_t<A>, literal_zero> || ...) &&
                  std::is_invocable_v<F &, zero_as_int<A> &...>) {
        return launch_values<F, zero_as_int<A>...>(c, f, std::forward<A>(a)...);
    } else {
        return launch_values<F, std::decay_t<A>...>(c, f, std::forward<A>(a)...);
    }
}

template <typename F, typename... V, typename... A>
cudaError_t launch_values(const config &c, F f, A &&...a)
{
    std::tuple<V...> values(static_cast<V>(std::forward<A>(a))...);
    constexpr std::size_t n_args = sizeof...(V);
    void *ptrs[n_args + 1];
    std::size_t sizes[n_args + 1];
    ptrs[0] = &f;
    sizes[0] = sizeof(f);
    std::apply(
        [&](auto &...v) {
            std::size_t i = 1;
            ((ptrs[i] = static_cast<void *>(&v), sizes[i] = sizeof(v), ++i), ...);
        },
        values);
    ocu_launch_params p{};
    p.grid[0] = c.grid.x; p.grid[1] = c.grid.y; p.grid[2] = c.grid.z;
    p.block[0] = c.block.x; p.block[1] = c.block.y; p.block[2] = c.block.z;
    p.shared_memory_bytes = c.shared;
    p.args = ptrs;
    p.arg_sizes = sizes;
    p.num_args = n_args + 1;
    return ocu_rt_launch(thunk<F, V...>::kernel(), &p, c.stream);
}

/* Calls the kernel through a pointer carried as the first argument. */
struct call_through {
    template <typename K, typename... A>
    auto operator()(K k, A &...a) const -> decltype(k(a...))
    {
        k(a...);
    }
};

template <typename... P, std::size_t... I>
cudaError_t launch_from_array(const config &c, void (*k)(P...), void **args,
                              std::index_sequence<I...>)
{
    return launch(c, call_through{}, k,
                  *static_cast<std::remove_cv_t<std::remove_reference_t<P>> *>(args[I])...);
}

} // namespace ocu_cpu

/*
 * cudaLaunchKernel(func, grid, block, args, shared, stream): args[i] points
 * at the value of the kernel's i-th parameter. On the CPU path the kernel's
 * parameter types come from its function type, so the values are copied
 * with their real types. (A plain `const void *` kernel cannot be launched
 * here: there is no table from addresses to kernels yet.)
 */
template <typename... P>
inline cudaError_t cudaLaunchKernel(void (*func)(P...), dim3 grid, dim3 block, void **args,
                                    size_t shared = 0, cudaStream_t stream = nullptr)
{
    if (func == nullptr || (sizeof...(P) > 0 && args == nullptr)) {
        return ocu_rt_launch(nullptr, nullptr, stream); /* records the error */
    }
    return ::ocu_cpu::launch_from_array(::ocu_cpu::cfg(grid, block, shared, stream), func, args,
                                        std::index_sequence_for<P...>{});
}

namespace ocu_cpu {

/* ---- atomics: blocks run on different OS threads ------------------------ */

template <typename T, typename Op>
inline T atomic_update(T *addr, Op op)
{
    T old = __atomic_load_n(addr, __ATOMIC_RELAXED);
    while (!__atomic_compare_exchange_n(addr, &old, op(old), true, __ATOMIC_RELAXED,
                                        __ATOMIC_RELAXED)) {
    }
    return old;
}

/* Floating-point values go through their bit pattern. */
template <typename T, typename Op>
inline T atomic_update_fp(T *addr, Op op)
{
    using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    U *ua = reinterpret_cast<U *>(addr);
    U old = __atomic_load_n(ua, __ATOMIC_RELAXED);
    for (;;) {
        T cur;
        std::memcpy(&cur, &old, sizeof(T));
        T next = op(cur);
        U nu;
        std::memcpy(&nu, &next, sizeof(T));
        if (__atomic_compare_exchange_n(ua, &old, nu, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            return cur;
        }
    }
}

} // namespace ocu_cpu

/* ---- built-in variables --------------------------------------------------- */

/*
 * Real variables, as in CUDA (so host code may declare a local named
 * gridDim): per-thread variables of the CPU backend under their CUDA
 * names. Each GPU thread sees its own values; they never change while it
 * runs, so `const` is accurate. dim3 and ocu_cpu_dim3 have the same layout.
 */
#define OCU_STR2_(x) #x
#define OCU_STR_(x) OCU_STR2_(x)
#define OCU_CPU_TLS_(name) __asm__(OCU_STR_(__USER_LABEL_PREFIX__) #name)
/* The definitions (C, zero-initialised) need no dynamic initialisation:
   saying so lets the compiler read them directly instead of through a C++
   thread-local wrapper function, which macOS would otherwise require. */
#if defined(__clang__)
#define OCU_CPU_CONSTINIT [[clang::require_constant_initialization]]
#elif defined(__GNUC__) && __GNUC__ >= 10
#define OCU_CPU_CONSTINIT __constinit
#else
#define OCU_CPU_CONSTINIT
#endif
static_assert(sizeof(dim3) == sizeof(ocu_cpu_dim3) && sizeof(uint3) == sizeof(ocu_cpu_dim3),
              "built-in variable layout");
#if defined(_WIN32)
/* Windows (MinGW): thread-local storage is emulated, and assembler names
   cannot be given to emulated variables. Per-thread references to the
   backend's variables keep the CUDA names as real names. */
#define OCU_CPU_BUILTIN_REF(T, name, var)                                              \
    static thread_local const T &name = *reinterpret_cast<const T *>(&var);
OCU_CPU_BUILTIN_REF(uint3, threadIdx, ocu_cpu_tls_thread_idx)
OCU_CPU_BUILTIN_REF(uint3, blockIdx, ocu_cpu_tls_block_idx)
OCU_CPU_BUILTIN_REF(dim3, blockDim, ocu_cpu_tls_block_dim)
OCU_CPU_BUILTIN_REF(dim3, gridDim, ocu_cpu_tls_grid_dim)
OCU_CPU_BUILTIN_REF(int, warpSize, ocu_cpu_tls_warp_size)
#undef OCU_CPU_BUILTIN_REF
#else
extern "C" {
OCU_CPU_CONSTINIT extern thread_local const uint3 threadIdx OCU_CPU_TLS_(ocu_cpu_tls_thread_idx);
OCU_CPU_CONSTINIT extern thread_local const uint3 blockIdx OCU_CPU_TLS_(ocu_cpu_tls_block_idx);
OCU_CPU_CONSTINIT extern thread_local const dim3 blockDim OCU_CPU_TLS_(ocu_cpu_tls_block_dim);
OCU_CPU_CONSTINIT extern thread_local const dim3 gridDim OCU_CPU_TLS_(ocu_cpu_tls_grid_dim);
OCU_CPU_CONSTINIT extern thread_local const int warpSize OCU_CPU_TLS_(ocu_cpu_tls_warp_size);
}
#endif
#undef OCU_CPU_TLS_

/* ---- synchronisation and fences ------------------------------------------- */

inline void __syncthreads() { ocu_cpu_barrier(); }
inline void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void __threadfence_block() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void __threadfence_system() { std::atomic_thread_fence(std::memory_order_seq_cst); }

/* ---- atomics ---------------------------------------------------------------- */

/* The operand types follow the address type only (CUDA declares one
   function per type, so atomicAdd(unsigned *, 1) converts the 1). */
namespace ocu_cpu {
template <typename T> struct nd_ { using type = T; };
template <typename T> using nd = typename nd_<T>::type;
} // namespace ocu_cpu

template <typename T>
inline T atomicAdd(T *addr, ::ocu_cpu::nd<T> v)
{
    if constexpr (std::is_floating_point_v<T>) {
        return ::ocu_cpu::atomic_update_fp(addr, [v](T c) { return c + v; });
    } else {
        return __atomic_fetch_add(addr, v, __ATOMIC_RELAXED);
    }
}
template <typename T>
inline T atomicSub(T *addr, ::ocu_cpu::nd<T> v) { return __atomic_fetch_sub(addr, v, __ATOMIC_RELAXED); }
template <typename T>
inline T atomicExch(T *addr, ::ocu_cpu::nd<T> v)
{
    if constexpr (std::is_floating_point_v<T>) {
        return ::ocu_cpu::atomic_update_fp(addr, [v](T) { return v; });
    } else {
        return __atomic_exchange_n(addr, v, __ATOMIC_RELAXED);
    }
}
template <typename T>
inline T atomicMin(T *addr, ::ocu_cpu::nd<T> v)
{
    return ::ocu_cpu::atomic_update(addr, [v](T c) { return v < c ? v : c; });
}
template <typename T>
inline T atomicMax(T *addr, ::ocu_cpu::nd<T> v)
{
    return ::ocu_cpu::atomic_update(addr, [v](T c) { return v > c ? v : c; });
}
template <typename T>
inline T atomicAnd(T *addr, ::ocu_cpu::nd<T> v) { return __atomic_fetch_and(addr, v, __ATOMIC_RELAXED); }
template <typename T>
inline T atomicOr(T *addr, ::ocu_cpu::nd<T> v) { return __atomic_fetch_or(addr, v, __ATOMIC_RELAXED); }
template <typename T>
inline T atomicXor(T *addr, ::ocu_cpu::nd<T> v) { return __atomic_fetch_xor(addr, v, __ATOMIC_RELAXED); }
template <typename T>
inline T atomicCAS(T *addr, ::ocu_cpu::nd<T> compare, ::ocu_cpu::nd<T> v)
{
    T expected = compare;
    __atomic_compare_exchange_n(addr, &expected, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected;
}
inline unsigned atomicInc(unsigned *addr, unsigned limit)
{
    return ::ocu_cpu::atomic_update(addr, [limit](unsigned c) { return c >= limit ? 0u : c + 1u; });
}
inline unsigned atomicDec(unsigned *addr, unsigned limit)
{
    return ::ocu_cpu::atomic_update(
        addr, [limit](unsigned c) { return (c == 0u || c > limit) ? limit : c - 1u; });
}

/* ---- math: device spellings and intrinsics --------------------------------- */

inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
inline long long min(long long a, long long b) { return a < b ? a : b; }
inline long long max(long long a, long long b) { return a > b ? a : b; }
inline float min(float a, float b) { return std::fmin(a, b); }
inline float max(float a, float b) { return std::fmax(a, b); }
inline double min(double a, double b) { return std::fmin(a, b); }
inline double max(double a, double b) { return std::fmax(a, b); }
inline unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
inline unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
inline long min(long a, long b) { return a < b ? a : b; }
inline long max(long a, long b) { return a > b ? a : b; }
inline unsigned long min(unsigned long a, unsigned long b) { return a < b ? a : b; }
inline unsigned long max(unsigned long a, unsigned long b) { return a > b ? a : b; }
/* Mixed signedness and precision: the wider or unsigned type wins, as in
   the documented overload set. */
inline unsigned min(int a, unsigned b) { return min(static_cast<unsigned>(a), b); }
inline unsigned min(unsigned a, int b) { return min(a, static_cast<unsigned>(b)); }
inline unsigned max(int a, unsigned b) { return max(static_cast<unsigned>(a), b); }
inline unsigned max(unsigned a, int b) { return max(a, static_cast<unsigned>(b)); }
inline unsigned long long min(long long a, unsigned long long b)
{ return min(static_cast<unsigned long long>(a), b); }
inline unsigned long long min(unsigned long long a, long long b)
{ return min(a, static_cast<unsigned long long>(b)); }
inline unsigned long long max(long long a, unsigned long long b)
{ return max(static_cast<unsigned long long>(a), b); }
inline unsigned long long max(unsigned long long a, long long b)
{ return max(a, static_cast<unsigned long long>(b)); }
inline double min(float a, double b) { return std::fmin(static_cast<double>(a), b); }
inline double min(double a, float b) { return std::fmin(a, static_cast<double>(b)); }
inline double max(float a, double b) { return std::fmax(static_cast<double>(a), b); }
inline double max(double a, float b) { return std::fmax(a, static_cast<double>(b)); }

inline float __expf(float x) { return std::exp(x); }
inline float __logf(float x) { return std::log(x); }
inline float __log2f(float x) { return std::log2(x); }
inline float __sinf(float x) { return std::sin(x); }
inline float __cosf(float x) { return std::cos(x); }
inline float __tanf(float x) { return std::tan(x); }
inline float __powf(float x, float y) { return std::pow(x, y); }
inline float __fdividef(float x, float y) { return x / y; }
inline float __saturatef(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }
inline float __fsqrt_rn(float x) { return std::sqrt(x); }
/* Fused multiply-add, one rounding in the named mode (ocu_fma_exact.h:
   hardware or exact software, never a C library fma that may round
   twice). Plain fma/fmaf come from <math.h>; where that library's are not
   correctly rounded the runtime library replaces them (cudart/libm_fma.c). */
inline float __fmaf_rn(float a, float b, float c) { return ::ocu_fmaf_rn(a, b, c); }
inline double __fma_rn(double a, double b, double c) { return ::ocu_fma_rn(a, b, c); }
#if defined(OCU_FMA_HAVE_SOFT_DOUBLE)
inline float __fmaf_rz(float a, float b, float c) { return ::ocu_fmaf_soft(a, b, c, OCU_FMA_RZ); }
inline float __fmaf_ru(float a, float b, float c) { return ::ocu_fmaf_soft(a, b, c, OCU_FMA_RU); }
inline float __fmaf_rd(float a, float b, float c) { return ::ocu_fmaf_soft(a, b, c, OCU_FMA_RD); }
inline double __fma_rz(double a, double b, double c) { return ::ocu_fma_soft(a, b, c, OCU_FMA_RZ); }
inline double __fma_ru(double a, double b, double c) { return ::ocu_fma_soft(a, b, c, OCU_FMA_RU); }
inline double __fma_rd(double a, double b, double c) { return ::ocu_fma_soft(a, b, c, OCU_FMA_RD); }
#endif
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __int_as_float(int v) { float f; std::memcpy(&f, &v, 4); return f; }
inline int __float_as_int(float f) { int v; std::memcpy(&v, &f, 4); return v; }
inline unsigned __float_as_uint(float f) { unsigned v; std::memcpy(&v, &f, 4); return v; }
inline float __uint_as_float(unsigned v) { float f; std::memcpy(&f, &v, 4); return f; }
inline int __popc(unsigned v) { return __builtin_popcount(v); }
inline int __popcll(unsigned long long v) { return __builtin_popcountll(v); }
inline int __clz(int v) { return v == 0 ? 32 : __builtin_clz(static_cast<unsigned>(v)); }
inline int __clzll(long long v)
{
    return v == 0 ? 64 : __builtin_clzll(static_cast<unsigned long long>(v));
}
inline int __ffs(int v) { return __builtin_ffs(v); }
inline int __ffsll(long long v) { return __builtin_ffsll(v); }
inline unsigned __brev(unsigned v)
{
    unsigned r = 0;
    for (int i = 0; i < 32; ++i) { r = (r << 1) | (v & 1u); v >>= 1; }
    return r;
}
inline int __mul24(int a, int b) { return a * b; }
inline unsigned __umul24(unsigned a, unsigned b) { return a * b; }

/* ---- device intrinsics shared with the NVPTX path ---------------------------- */

#define OCU_LANG_FN inline
#include "ocu_cuda_lang_common.h"
#undef OCU_LANG_FN

/*
 * __trap: the kernel hit something it must never do (for example a
 * NO_DEVICE_CODE body for an architecture that was not compiled). On a GPU
 * it aborts the kernel and every later call fails; here it stops the process
 * with a message naming the thread, so a wrongly selected kernel is never
 * silent. (A program that gets __CUDA_ARCH__ defined for the CPU path, for
 * example with -D__CUDA_ARCH__=500, reaches its own NO_DEVICE_CODE handler.)
 */
[[noreturn]] inline void __trap()
{
    std::fprintf(stderr,
                 "ocu: __trap() in kernel, block (%u,%u,%u) thread (%u,%u,%u): "
                 "aborting (a kernel with no device code for this device was probably launched)\n",
                 blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    std::fflush(stdout); /* the kernel's own printf output comes first */
    std::fflush(stderr);
    std::abort();
}

/* ---- block-wide votes ---------------------------------------------------------
 * Every thread of the block that has not returned calls it; each gets the
 * same answer. Threads of a block run as fibers of one OS thread, so a
 * thread-local counter is the block's counter; three barriers keep the
 * reset, the additions and the reads apart.
 */
inline int __syncthreads_count(int pred)
{
    static thread_local int acc;
    ocu_cpu_barrier();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) acc = 0;
    ocu_cpu_barrier();
    if (pred != 0) ++acc;
    ocu_cpu_barrier();
    return acc;
}
inline int __syncthreads_or(int pred) { return __syncthreads_count(pred) != 0; }
inline int __syncthreads_and(int pred)
{
    int n = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
    return __syncthreads_count(pred) == n;
}

/* ---- integer and conversion intrinsics ---------------------------------------- */

inline unsigned __umulhi(unsigned a, unsigned b)
{
    return static_cast<unsigned>((static_cast<unsigned long long>(a) * b) >> 32);
}
inline int __mulhi(int a, int b)
{
    return static_cast<int>((static_cast<long long>(a) * b) >> 32);
}

/*
 * __byte_perm(x, y, s): the eight bytes {y,x} (x = bytes 0..3, y = 4..7);
 * result byte i is the byte named by the low 3 bits of selector nibble i.
 * When nibble i's top bit is set, that byte's sign bit is replicated instead.
 */
inline unsigned __byte_perm(unsigned x, unsigned y, unsigned s)
{
    unsigned long long v = (static_cast<unsigned long long>(y) << 32) | x;
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned sel = (s >> (4 * i)) & 0xfu;
        unsigned b = static_cast<unsigned>((v >> (8 * (sel & 7u))) & 0xffu);
        if (sel & 8u) b = (b & 0x80u) ? 0xffu : 0u;
        r |= b << (8 * i);
    }
    return r;
}

namespace ocu_cpu {
/* float or double to int: NaN gives 0, out-of-range values saturate. */
template <typename F>
inline int to_int_sat(F r)
{
    if (r != r) return 0;
    if (r >= static_cast<F>(2147483648.0)) return 2147483647;
    if (r <= static_cast<F>(-2147483648.0)) return -2147483647 - 1;
    return static_cast<int>(r);
}
} // namespace ocu_cpu
/* Rounding: _rn nearest even, _rz toward zero, _rd down, _ru up. */
inline int __float2int_rn(float x) { return ::ocu_cpu::to_int_sat(__builtin_rintf(x)); }
inline int __float2int_rz(float x) { return ::ocu_cpu::to_int_sat(__builtin_truncf(x)); }
inline int __float2int_rd(float x) { return ::ocu_cpu::to_int_sat(__builtin_floorf(x)); }
inline int __float2int_ru(float x) { return ::ocu_cpu::to_int_sat(__builtin_ceilf(x)); }
inline int __double2int_rn(double x) { return ::ocu_cpu::to_int_sat(__builtin_rint(x)); }
inline int __double2int_rz(double x) { return ::ocu_cpu::to_int_sat(__builtin_trunc(x)); }
inline int __double2int_rd(double x) { return ::ocu_cpu::to_int_sat(__builtin_floor(x)); }
inline int __double2int_ru(double x) { return ::ocu_cpu::to_int_sat(__builtin_ceil(x)); }

/* ---- warp-level functions ---------------------------------------------------
 * Built on the backend's masked warp gather: a lane waits only for the
 * lanes its mask names (see ocu_cpu_kernel.h).
 */
namespace ocu_cpu {

template <typename T>
inline std::uint64_t to_bits(T v)
{
    static_assert(sizeof(T) <= 8 && std::is_trivially_copyable_v<T>,
                  "warp shuffles move values of at most 8 bytes");
    std::uint64_t b = 0;
    std::memcpy(&b, &v, sizeof(T));
    return b;
}

template <typename T>
inline T from_bits(std::uint64_t b)
{
    T v;
    std::memcpy(&v, &b, sizeof(T));
    return v;
}

/* Tag for a warp call: operation, width, mask. Lanes of one call match. */
inline std::uint64_t warp_tag(unsigned op, unsigned mask, int width)
{
    return (std::uint64_t(op) << 56) ^ (std::uint64_t(unsigned(width)) << 32) ^ mask;
}

/*
 * Value of lane `src`; our own value when `src` lies in a later group of
 * `width` lanes or past the end of the warp. `earlier_ok`: whether an
 * earlier group may be read (xor may; up, down and idx stay in their group).
 */
template <typename T>
inline T shuffle_from(T v, int src, int width, std::uint64_t tag, unsigned mask,
                      bool earlier_ok = false)
{
    std::uint64_t all[OCU_CPU_MAX_WARP];
    unsigned count = ocu_cpu_warp_gather_masked(to_bits(v), tag, mask, all);
    int lane = static_cast<int>(ocu_cpu_lane_id());
    int w = width > 0 ? width : static_cast<int>(ocu_cpu_warp_size());
    int base = lane - lane % w;
    bool before = src < base;
    if ((before && !earlier_ok) || src < 0 || src >= base + w ||
        src >= static_cast<int>(count)) {
        src = lane;
    }
    return from_bits<T>(all[src]);
}

/* Predicates of the lanes the mask names, as bits of the caller's group. */
inline unsigned vote_bits(unsigned op, unsigned mask, int pred)
{
    std::uint64_t all[OCU_CPU_MAX_WARP];
    unsigned count = ocu_cpu_warp_gather_masked(pred != 0, warp_tag(op, mask, 0), mask, all);
    unsigned group = ocu_cpu_lane_id() & ~31u;
    unsigned bits = 0;
    for (unsigned i = 0; i < 32 && group + i < count; ++i) {
        if ((mask >> i & 1u) && all[group + i] != 0) {
            bits |= 1u << i;
        }
    }
    return bits;
}

} // namespace ocu_cpu

template <typename T>
inline T __shfl_sync(unsigned m, T v, int src, int width = warpSize)
{
    int lane = static_cast<int>(ocu_cpu_lane_id());
    int base = lane - lane % width;
    return ::ocu_cpu::shuffle_from(v, base + ((src % width) + width) % width, width,
                                   ::ocu_cpu::warp_tag(1, m, width), m);
}

template <typename T>
inline T __shfl_up_sync(unsigned m, T v, unsigned delta, int width = warpSize)
{
    int lane = static_cast<int>(ocu_cpu_lane_id());
    return ::ocu_cpu::shuffle_from(v, lane - static_cast<int>(delta), width,
                                   ::ocu_cpu::warp_tag(2, m, width), m);
}

template <typename T>
inline T __shfl_down_sync(unsigned m, T v, unsigned delta, int width = warpSize)
{
    int lane = static_cast<int>(ocu_cpu_lane_id());
    return ::ocu_cpu::shuffle_from(v, lane + static_cast<int>(delta), width,
                                   ::ocu_cpu::warp_tag(3, m, width), m);
}

template <typename T>
inline T __shfl_xor_sync(unsigned m, T v, int lane_mask, int width = warpSize)
{
    int lane = static_cast<int>(ocu_cpu_lane_id());
    return ::ocu_cpu::shuffle_from(v, lane ^ lane_mask, width,
                                   ::ocu_cpu::warp_tag(4, m, width), m, true);
}

/* Bits of the caller's 32-lane group (CUDA's result has 32 bits). */
inline unsigned __ballot_sync(unsigned m, int pred) { return ::ocu_cpu::vote_bits(5, m, pred); }
inline int __any_sync(unsigned m, int pred) { return ::ocu_cpu::vote_bits(6, m, pred) != 0; }
inline int __all_sync(unsigned m, int pred)
{
    std::uint64_t all[OCU_CPU_MAX_WARP];
    unsigned count = ocu_cpu_warp_gather_masked(pred != 0, ::ocu_cpu::warp_tag(7, m, 0), m, all);
    unsigned group = ocu_cpu_lane_id() & ~31u;
    for (unsigned i = 0; i < 32 && group + i < count; ++i) {
        if ((m >> i & 1u) && all[group + i] == 0) return 0;
    }
    return 1;
}
/* Lanes of the caller's 32-lane group that exist in this block. It does
   not synchronise, as in CUDA; lanes that returned are still counted. */
inline unsigned __activemask()
{
    ocu_cpu_dim3 t = ocu_cpu_thread_idx(), d = ocu_cpu_block_dim();
    unsigned n = d.x * d.y * d.z;
    unsigned me = t.x + d.x * (t.y + d.y * t.z);
    unsigned start = me - me % 32u;
    unsigned have = n - start < 32u ? n - start : 32u;
    return have == 32u ? 0xffffffffu : ((1u << have) - 1u);
}
inline void __syncwarp(unsigned m = 0xffffffffu) { ocu_cpu_warp_sync_masked(m); }

#endif /* OCU_CUDA_LANG_CPU_H */

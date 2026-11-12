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
#define __noinline__ __attribute__((noinline))
#define __shared__ static thread_local
#define __constant__
#define __managed__
#define __align__(n) alignas(n)
#define __restrict__ __restrict
#define __launch_bounds__(...)

namespace ocu_cpu {

struct idx3 {
    unsigned x, y, z;
};
inline idx3 to_idx3(ocu_cpu_dim3 d) { return idx3{d.x, d.y, d.z}; }

inline void *dynamic_shared() { return ocu_cpu_dynamic_shared(); }

/*
 * `extern __shared__ T name[];` becomes `static dyn_shared<T> name;`: the
 * address is looked up at each use, so it works at file scope and inside
 * functions, and every block sees its own memory.
 */
template <typename T>
struct dyn_shared {
    operator T *() const { return static_cast<T *>(ocu_cpu_dynamic_shared()); }
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

template <typename F, typename... A>
void launch(const config &c, F f, A &&...a)
{
    static_assert(std::is_empty_v<F>, "launch lambda must not capture");
    std::tuple<std::decay_t<A>...> values(std::forward<A>(a)...);
    void *ptrs[sizeof...(A) + 1];
    std::size_t sizes[sizeof...(A) + 1];
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
    p.num_args = sizeof...(A) + 1;
    (void)ocu_rt_launch(thunk<F, std::decay_t<A>...>::kernel(), &p, c.stream);
}

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

#define threadIdx (::ocu_cpu::to_idx3(ocu_cpu_thread_idx()))
#define blockIdx  (::ocu_cpu::to_idx3(ocu_cpu_block_idx()))
#define blockDim  (::ocu_cpu::to_idx3(ocu_cpu_block_dim()))
#define gridDim   (::ocu_cpu::to_idx3(ocu_cpu_grid_dim()))
/* warpSize is a variable, not a macro: host code reads the field
   cudaDeviceProp::warpSize under the same name. */
namespace ocu_cpu {
struct warp_size_t {
    operator int() const { return static_cast<int>(ocu_cpu_warp_size()); }
};
} // namespace ocu_cpu
static constexpr ::ocu_cpu::warp_size_t warpSize{};

/* ---- synchronisation and fences ------------------------------------------- */

inline void __syncthreads() { ocu_cpu_barrier(); }
inline void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void __threadfence_block() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void __threadfence_system() { std::atomic_thread_fence(std::memory_order_seq_cst); }

/* ---- atomics ---------------------------------------------------------------- */

template <typename T>
inline T atomicAdd(T *addr, T v)
{
    if constexpr (std::is_floating_point_v<T>) {
        return ::ocu_cpu::atomic_update_fp(addr, [v](T c) { return c + v; });
    } else {
        return __atomic_fetch_add(addr, v, __ATOMIC_RELAXED);
    }
}
template <typename T>
inline T atomicSub(T *addr, T v) { return __atomic_fetch_sub(addr, v, __ATOMIC_RELAXED); }
template <typename T>
inline T atomicExch(T *addr, T v)
{
    if constexpr (std::is_floating_point_v<T>) {
        return ::ocu_cpu::atomic_update_fp(addr, [v](T) { return v; });
    } else {
        return __atomic_exchange_n(addr, v, __ATOMIC_RELAXED);
    }
}
template <typename T>
inline T atomicMin(T *addr, T v)
{
    return ::ocu_cpu::atomic_update(addr, [v](T c) { return v < c ? v : c; });
}
template <typename T>
inline T atomicMax(T *addr, T v)
{
    return ::ocu_cpu::atomic_update(addr, [v](T c) { return v > c ? v : c; });
}
template <typename T>
inline T atomicAnd(T *addr, T v) { return __atomic_fetch_and(addr, v, __ATOMIC_RELAXED); }
template <typename T>
inline T atomicOr(T *addr, T v) { return __atomic_fetch_or(addr, v, __ATOMIC_RELAXED); }
template <typename T>
inline T atomicXor(T *addr, T v) { return __atomic_fetch_xor(addr, v, __ATOMIC_RELAXED); }
template <typename T>
inline T atomicCAS(T *addr, T compare, T v)
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
inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
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

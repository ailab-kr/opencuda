/*
 * PROBE ONLY (PLAN P0-W1-11): CPU-path definitions of the CUDA language
 * names a kernel body uses, for .cu files rewritten by cpu_rewrite.py and
 * compiled as plain C++17. Not a shipped header; see docs/compiler-probe.md.
 *
 * Launches go through the backend interface only: one module per kernel
 * signature, holding a trampoline that unpacks the copied arguments.
 */
#pragma once

#include "ocu_backend.h"
#include "ocu_backend_cpu.h"
#include "ocu_cpu_kernel.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__   static thread_local
#define __constant__
#define __align__(n) alignas(n)
#define __restrict__ __restrict

struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

namespace ocu_cpu {

struct idx3 {
    unsigned x, y, z;
};
inline idx3 to_idx3(ocu_cpu_dim3 d) { return idx3{d.x, d.y, d.z}; }

inline void *dynamic_shared() { return ocu_cpu_dynamic_shared(); }

/*
 * `extern __shared__ T name[];` becomes `static dyn_shared<T> name;`. The
 * address is looked up at each use, so this works at file scope and inside
 * functions, and every block sees its own memory.
 */
template <typename T>
struct dyn_shared {
    operator T *() const { return static_cast<T *>(ocu_cpu_dynamic_shared()); }
    T &operator[](std::size_t i) const { return static_cast<T *>(*this)[i]; }
};

/* The calling program supplies the device to launch on. */
const ocu_backend *backend();
ocu_device *device();

struct config {
    dim3 grid, block;
    std::size_t shared;
    void *stream; /* ignored by the probe: launches run synchronously */
};
inline config cfg(dim3 grid, dim3 block, std::size_t shared = 0,
                  void *stream = nullptr)
{
    return config{grid, block, shared, stream};
}

/* Last launch status, so a probe test can check it. */
inline ocu_status &last_status()
{
    static thread_local ocu_status s = OCU_OK;
    return s;
}

/*
 * One module per (launch site, argument types): the rewriter turns
 * K<<<cfg>>>(args) into launch(cfg, [](auto &&...a) { K(a...); }, args), so
 * the kernel call inside the lambda is an ordinary C++ call and the compiler
 * deduces template arguments, picks overloads and converts arguments the
 * same way it would for any call.
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
            static const ocu_cpu_image image = {OCU_CPU_IMAGE_MAGIC,
                                                OCU_CPU_IMAGE_VERSION, 1, &entry};
            ocu_module *m = nullptr;
            ocu_kernel *out = nullptr;
            if (backend()->module_load(device(), &image, sizeof(image), &m) != OCU_OK ||
                backend()->module_get_kernel(m, "trampoline", &out) != OCU_OK) {
                std::fprintf(stderr, "probe: cannot load CPU trampoline\n");
                std::abort();
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
    last_status() = backend()->kernel_launch(
        nullptr, thunk<F, std::decay_t<A>...>::kernel(), &p);
}

/* ---- atomics: blocks run on different OS threads ----------------------- */

template <typename T>
inline T atomic_add(T *addr, T v)
{
    if constexpr (std::is_integral_v<T>) {
        return __atomic_fetch_add(addr, v, __ATOMIC_RELAXED);
    } else {
        T old = __atomic_load_n(addr, __ATOMIC_RELAXED);
        while (!__atomic_compare_exchange_n(addr, &old, old + v, true,
                                            __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        }
        return old;
    }
}

} // namespace ocu_cpu

#define threadIdx (::ocu_cpu::to_idx3(ocu_cpu_thread_idx()))
#define blockIdx  (::ocu_cpu::to_idx3(ocu_cpu_block_idx()))
#define blockDim  (::ocu_cpu::to_idx3(ocu_cpu_block_dim()))
#define gridDim   (::ocu_cpu::to_idx3(ocu_cpu_grid_dim()))
#define warpSize  (static_cast<int>(ocu_cpu_warp_size()))

inline void __syncthreads() { ocu_cpu_barrier(); }

template <typename T>
inline T atomicAdd(T *addr, T v) { return ::ocu_cpu::atomic_add(addr, v); }

/* ---- warp-level intrinsics (CPU path) ----------------------------------
 * Built on ocu_cpu_warp_gather. The mask argument is accepted and not used:
 * every live lane of the warp takes part, which is what code written with
 * a full mask (0xffffffff) expects.
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

/* Tag for a warp call: operation, mask, width. Lanes of one call match. */
inline std::uint64_t warp_tag(unsigned op, unsigned mask, int width)
{
    return (std::uint64_t(op) << 56) ^ (std::uint64_t(unsigned(width)) << 32) ^ mask;
}

/*
 * Value of lane `src`. Our own value when `src` lies in a later group of
 * `width` lanes, or past the end of the warp. `earlier_ok`: whether an
 * earlier group may be read (true for xor, as the CUDA C++ Programming
 * Guide describes; up/down/idx never leave their group).
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

/*
 * One bit per lane. CUDA's result is 32 bits, so with a 64-lane warp each
 * lane gets the bits of its own 32-lane half (as if each half were a CUDA
 * warp). Votes (any/all) cover the whole warp.
 */
inline unsigned __ballot_sync(unsigned m, int pred)
{
    std::uint64_t all[OCU_CPU_MAX_WARP];
    unsigned count = ocu_cpu_warp_gather_masked(pred != 0, ::ocu_cpu::warp_tag(5, m, 0), m, all);
    unsigned half = ocu_cpu_lane_id() & ~31u;
    unsigned bits = 0;
    for (unsigned i = 0; i < 32 && half + i < count; ++i) {
        bits |= static_cast<unsigned>(all[half + i] != 0) << i;
    }
    return bits;
}

inline int __any_sync(unsigned m, int pred)
{
    std::uint64_t all[OCU_CPU_MAX_WARP];
    unsigned count = ocu_cpu_warp_gather_masked(pred != 0, ::ocu_cpu::warp_tag(6, m, 0), m, all);
    for (unsigned i = 0; i < count; ++i) {
        if (all[i]) return 1;
    }
    return 0;
}

inline int __all_sync(unsigned m, int pred)
{
    std::uint64_t all[OCU_CPU_MAX_WARP];
    unsigned count = ocu_cpu_warp_gather_masked(pred != 0, ::ocu_cpu::warp_tag(7, m, 0), m, all);
    for (unsigned i = 0; i < count; ++i) {
        if (!all[i]) return 0;
    }
    return 1;
}

inline void __syncwarp(unsigned m = 0xffffffffu) { ocu_cpu_warp_sync_masked(m); }

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

template <typename... P>
struct signature {
    using fn = void (*)(P...);

    template <std::size_t... I>
    static void call(void **a, std::index_sequence<I...>)
    {
        fn k = *static_cast<fn *>(a[0]);
        k(*static_cast<std::decay_t<P> *>(a[I + 1])...);
    }
    static void trampoline(void **a)
    {
        call(a, std::index_sequence_for<P...>{});
    }

    /* One module per signature, loaded on first use. */
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

/* Last launch status, so a probe test can check it. */
inline ocu_status &last_status()
{
    static thread_local ocu_status s = OCU_OK;
    return s;
}

template <typename... P, typename... A>
void launch(void (*k)(P...), const config &c, A &&...a)
{
    static_assert(sizeof...(P) == sizeof...(A), "kernel argument count");
    std::tuple<std::decay_t<P>...> values(std::forward<A>(a)...);
    void *ptrs[sizeof...(P) + 1];
    std::size_t sizes[sizeof...(P) + 1];
    ptrs[0] = &k;
    sizes[0] = sizeof(k);
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
    p.num_args = sizeof...(P) + 1;
    last_status() = backend()->kernel_launch(nullptr, signature<P...>::kernel(), &p);
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

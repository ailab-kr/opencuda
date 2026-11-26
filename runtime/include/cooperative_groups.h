/*
 * cooperative_groups.h -- a minimal subset of the Cooperative Groups API:
 * thread_block, this_thread_block(), thread_block_tile<N> /
 * tiled_partition<N>() with the warp-level members, and grid_group /
 * this_grid() whose sync() fails loudly.
 *
 * Written for this project from the public CUDA C++ Programming Guide
 * chapter "Cooperative Groups". Not derived from any vendor header.
 *
 * What is not here is missing on purpose: using it is a compile error
 * (no such name), never a silent wrong answer. In particular
 *   - tiled_partition with a run-time size, coalesced_threads(),
 *     labeled_partition, multi_grid_group, cluster groups, memcpy_async,
 *     wait, and thread_block::barrier_*: not provided.
 *   - grid_group::sync() needs a cooperative launch, which no backend
 *     supports (cudaLaunchCooperativeKernel returns cudaErrorNotSupported
 *     and cudaDevAttrCooperativeLaunch is 0). Reaching it prints a message
 *     and traps (see __trap); grid_group::is_valid() is false.
 * Tile sizes: a power of two up to 32 (a tile never spans warps).
 */
#ifndef OCU_COOPERATIVE_GROUPS_H
#define OCU_COOPERATIVE_GROUPS_H

#ifndef __cplusplus
#error "cooperative_groups.h needs C++"
#endif

#include "cuda_runtime.h"

#include <type_traits>

#if !defined(OCU_CPU_PATH) && !defined(OCU_SPIRV_PATH) && !(defined(__CUDA__) && !defined(__HIP__))
#error "cooperative_groups.h is for device code (CPU path or Clang CUDA mode)"
#endif

namespace cooperative_groups {

namespace detail {
__device__ inline unsigned linear_tid()
{
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}
__device__ inline unsigned block_threads() { return blockDim.x * blockDim.y * blockDim.z; }
} // namespace detail

class thread_block {
public:
    __device__ void sync() const { __syncthreads(); }
    __device__ unsigned long long size() const { return detail::block_threads(); }
    __device__ unsigned long long num_threads() const { return detail::block_threads(); }
    __device__ unsigned long long thread_rank() const { return detail::linear_tid(); }
    __device__ dim3 group_index() const { return dim3(blockIdx.x, blockIdx.y, blockIdx.z); }
    __device__ dim3 thread_index() const { return dim3(threadIdx.x, threadIdx.y, threadIdx.z); }
    __device__ dim3 dim_threads() const { return dim3(blockDim.x, blockDim.y, blockDim.z); }
    __device__ static bool is_valid() { return true; }
};

__device__ inline thread_block this_thread_block() { return thread_block(); }
__device__ inline void sync(const thread_block &g) { g.sync(); }

/* The whole grid. Valid only under a cooperative launch, which does not exist
   here, so sync() cannot succeed. */
class grid_group {
public:
    __device__ static bool is_valid() { return false; }
    __device__ void sync() const
    {
        printf("ocu: cooperative_groups::grid_group::sync() needs a cooperative launch, "
               "which this device does not support\n");
        __trap();
    }
    __device__ unsigned long long size() const
    {
        return static_cast<unsigned long long>(gridDim.x) * gridDim.y * gridDim.z *
               detail::block_threads();
    }
    __device__ unsigned long long thread_rank() const
    {
        unsigned long long block =
            blockIdx.x + static_cast<unsigned long long>(gridDim.x) *
                             (blockIdx.y + static_cast<unsigned long long>(gridDim.y) * blockIdx.z);
        return block * detail::block_threads() + detail::linear_tid();
    }
};

__device__ inline grid_group this_grid() { return grid_group(); }
__device__ inline void sync(const grid_group &g) { g.sync(); }

template <unsigned N, typename ParentT = void>
class thread_block_tile;

namespace detail {
/* Shape of the group a tile was cut from: how many tiles it holds and the
   rank of the caller's tile in it. */
template <unsigned N, typename ParentT>
struct tile_parent;
template <unsigned N>
struct tile_parent<N, thread_block> {
    __device__ static unsigned size() { return block_threads() / N; }
    __device__ static unsigned rank() { return linear_tid() / N; }
};
template <unsigned N, unsigned P, typename Q>
struct tile_parent<N, thread_block_tile<P, Q>> {
    __device__ static unsigned size() { return P / N; }
    __device__ static unsigned rank() { return (linear_tid() & 31u) % P / N; }
};
} // namespace detail

/* A tile of N consecutive threads of one warp (by linear thread index in
   the block). ParentT is the group it was partitioned from. */
template <unsigned N, typename ParentT>
class thread_block_tile {
    static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0,
                  "tile size must be a power of two up to 32");

    __device__ static unsigned lane() { return detail::linear_tid() & 31u; }
    __device__ static unsigned tile_bits() { return 0xffffffffu >> (32 - N); }
    /* Bits of the lanes of this thread's tile. */
    __device__ static unsigned mask() { return tile_bits() << (lane() / N * N); }

    /* Meta group of a tile typed without its parent: captured when the
       tile with a parent converts to it (as the documented API allows). */
    unsigned meta_size_ = 1;
    unsigned meta_rank_ = 0;

public:
    __device__ thread_block_tile() {}
    template <typename P, typename = typename std::enable_if<std::is_void<ParentT>::value &&
                                                             !std::is_void<P>::value>::type>
    __device__ thread_block_tile(const thread_block_tile<N, P> &t)
        : meta_size_(t.meta_group_size()), meta_rank_(t.meta_group_rank()) {}

    /* Number of tiles in the parent group, and this tile's rank among them. */
    __device__ unsigned meta_group_size() const
    {
        if constexpr (std::is_void<ParentT>::value) return meta_size_;
        else return detail::tile_parent<N, ParentT>::size();
    }
    __device__ unsigned meta_group_rank() const
    {
        if constexpr (std::is_void<ParentT>::value) return meta_rank_;
        else return detail::tile_parent<N, ParentT>::rank();
    }

    __device__ void sync() const { __syncwarp(mask()); }
    __device__ unsigned long long size() const { return N; }
    __device__ unsigned long long num_threads() const { return N; }
    __device__ unsigned long long thread_rank() const { return lane() % N; }
    __device__ static bool is_valid() { return true; }

    /* Lane arguments are ranks within the tile, as documented. */
    template <typename T>
    __device__ T shfl(T v, int src) const { return __shfl_sync(mask(), v, src, N); }
    template <typename T>
    __device__ T shfl_up(T v, unsigned delta) const { return __shfl_up_sync(mask(), v, delta, N); }
    template <typename T>
    __device__ T shfl_down(T v, unsigned delta) const { return __shfl_down_sync(mask(), v, delta, N); }
    template <typename T>
    __device__ T shfl_xor(T v, unsigned lane_mask) const
    {
        return __shfl_xor_sync(mask(), v, static_cast<int>(lane_mask), N);
    }
    /* Bit i of the result is the predicate of rank i of the tile. */
    __device__ unsigned ballot(int pred) const
    {
        unsigned bits = __ballot_sync(mask(), pred);
        return (bits >> (lane() / N * N)) & tile_bits();
    }
    __device__ int any(int pred) const { return __any_sync(mask(), pred); }
    __device__ int all(int pred) const { return __all_sync(mask(), pred); }
};

template <unsigned N>
__device__ inline thread_block_tile<N, thread_block> tiled_partition(const thread_block &)
{
    return thread_block_tile<N, thread_block>();
}
/* Tiles of a tile: N must not exceed the parent's size. */
template <unsigned N, unsigned P, typename Q>
__device__ inline thread_block_tile<N, thread_block_tile<P, Q>>
tiled_partition(const thread_block_tile<P, Q> &)
{
    static_assert(N <= P, "a sub-tile cannot be larger than its parent");
    return thread_block_tile<N, thread_block_tile<P, Q>>();
}
template <unsigned N, typename Q>
__device__ inline void sync(const thread_block_tile<N, Q> &g) { g.sync(); }

} // namespace cooperative_groups


#endif /* OCU_COOPERATIVE_GROUPS_H */

/*
 * cooperative_groups/reduce.h -- cooperative_groups::reduce over a
 * thread_block_tile, with the documented operator objects (plus, less,
 * greater, bit_and, bit_or, bit_xor).
 *
 * Written for this project from the public CUDA C++ Programming Guide
 * ("Cooperative Groups", collectives). Not derived from any vendor header.
 *
 * Every thread of the tile gets the result. The order in which values are
 * combined is a butterfly over the tile; for floating-point sums this may
 * differ in the last bits from another implementation's order, as the
 * guide allows. Groups other than tiles (thread_block, grid_group) are not
 * supported: a compile error.
 */
#ifndef OCU_COOPERATIVE_GROUPS_REDUCE_H
#define OCU_COOPERATIVE_GROUPS_REDUCE_H

#include "../cooperative_groups.h"

namespace cooperative_groups {

template <typename T>
struct plus {
    __device__ T operator()(const T &a, const T &b) const { return a + b; }
};
template <typename T>
struct less {
    __device__ T operator()(const T &a, const T &b) const { return b < a ? b : a; }
};
template <typename T>
struct greater {
    __device__ T operator()(const T &a, const T &b) const { return a < b ? b : a; }
};
template <typename T>
struct bit_and {
    __device__ T operator()(const T &a, const T &b) const { return a & b; }
};
template <typename T>
struct bit_or {
    __device__ T operator()(const T &a, const T &b) const { return a | b; }
};
template <typename T>
struct bit_xor {
    __device__ T operator()(const T &a, const T &b) const { return a ^ b; }
};

template <unsigned N, typename T, typename Op>
__device__ inline T reduce(const thread_block_tile<N> &g, T v, Op op)
{
    for (unsigned off = N / 2; off > 0; off >>= 1) {
        v = op(v, g.shfl_xor(v, off));
    }
    return v;
}

template <typename G, typename T, typename Op>
__device__ inline T reduce(const G &, T, Op)
{
    static_assert(sizeof(G) == 0, "reduce is provided for thread_block_tile only");
    return T();
}

} // namespace cooperative_groups

#endif /* OCU_COOPERATIVE_GROUPS_REDUCE_H */

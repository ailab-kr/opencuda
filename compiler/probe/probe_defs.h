/*
 * Minimal definitions written for the compiler probe only (PLAN P0-W1-11).
 * Nothing here comes from a vendor SDK. The point is to learn what the
 * compiler itself requires from a header set, per target.
 *
 * The launch-ABI declarations below use the names clang looks up when it
 * lowers the <<<...>>> syntax. Whether and how our real headers may use
 * those names is outside the scope of this probe.
 */
#pragma once

#if defined(OCU_CUDA_RUNTIME_H)
/* Built through ocu_cc.py: the product headers are already included. */
#elif defined(OCU_CPU_PATH)
/* CPU path, probe headers (tests/probe): rewritten, plain C++. */
#include "probe_cpu.h"
#else

#include <stddef.h>

#define __global__   __attribute__((global))
#define __device__   __attribute__((device))
#define __host__     __attribute__((host))
#define __shared__   __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __align__(n) __attribute__((aligned(n)))

struct dim3 {
    unsigned x, y, z;
    __host__ __device__ constexpr dim3(unsigned x_ = 1, unsigned y_ = 1,
                                       unsigned z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

/* ---- thread / block index, one implementation per device target ---- */

#if defined(__NVPTX__)
#define PROBE_TID(d)  __nvvm_read_ptx_sreg_tid_##d()
#define PROBE_BID(d)  __nvvm_read_ptx_sreg_ctaid_##d()
#define PROBE_BDIM(d) __nvvm_read_ptx_sreg_ntid_##d()
#define PROBE_BARRIER() __nvvm_bar_sync(0)
#elif defined(__AMDGPU__)
#define PROBE_TID(d)  __builtin_amdgcn_workitem_id_##d()
#define PROBE_BID(d)  __builtin_amdgcn_workgroup_id_##d()
#define PROBE_BDIM(d) __builtin_amdgcn_workgroup_size_##d()
#define PROBE_BARRIER() \
    (__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"), \
     __builtin_amdgcn_s_barrier(), \
     __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"))
#else /* host pass: declarations only, never executed */
#define PROBE_TID(d)  0u
#define PROBE_BID(d)  0u
#define PROBE_BDIM(d) 0u
#define PROBE_BARRIER() ((void)0)
#endif

#define PROBE_INDEX_VAR(TYPE, VAR, GET)                                   \
    struct TYPE {                                                         \
        __declspec(property(get = get_x)) unsigned x;                     \
        __declspec(property(get = get_y)) unsigned y;                     \
        __declspec(property(get = get_z)) unsigned z;                     \
        static inline __device__ unsigned get_x() { return GET(x); }      \
        static inline __device__ unsigned get_y() { return GET(y); }      \
        static inline __device__ unsigned get_z() { return GET(z); }      \
    };                                                                    \
    extern const __device__ TYPE VAR;

PROBE_INDEX_VAR(probe_tid_t, threadIdx, PROBE_TID)
PROBE_INDEX_VAR(probe_bid_t, blockIdx, PROBE_BID)
PROBE_INDEX_VAR(probe_bdim_t, blockDim, PROBE_BDIM)

#if !defined(__NVPTX__) /* on NVPTX, __syncthreads is a compiler builtin */
static inline __device__ void __syncthreads() { PROBE_BARRIER(); }
#endif

/* ---- atomics -------------------------------------------------------- */

template <typename T>
static inline __device__ T atomicAdd(T *addr, T v)
{
    return __atomic_fetch_add(addr, v, __ATOMIC_RELAXED);
}

/* ---- warp-level intrinsics (4-byte values) ------------------------- */

template <typename T>
static inline __device__ int probe_as_int(T v)
{
    static_assert(sizeof(T) == 4, "probe shuffles move 4-byte values");
    return __builtin_bit_cast(int, v);
}

template <typename T>
static inline __device__ T probe_from_int(int v) { return __builtin_bit_cast(T, v); }

#if defined(__NVPTX__)
#define warpSize 32
static inline __device__ unsigned probe_lane() { return __nvvm_read_ptx_sreg_laneid(); }
#define PROBE_SHFL(kind, mask, v, arg, c)     probe_from_int<T>(__nvvm_shfl_sync_##kind##_i32(mask, probe_as_int(v), arg, c))
template <typename T>
static inline __device__ T __shfl_sync(unsigned m, T v, int src, int width = 32)
{ return PROBE_SHFL(idx, m, v, src, ((32 - width) << 8) | 0x1f); }
template <typename T>
static inline __device__ T __shfl_up_sync(unsigned m, T v, unsigned d, int width = 32)
{ return PROBE_SHFL(up, m, v, (int)d, (32 - width) << 8); }
template <typename T>
static inline __device__ T __shfl_down_sync(unsigned m, T v, unsigned d, int width = 32)
{ return PROBE_SHFL(down, m, v, (int)d, ((32 - width) << 8) | 0x1f); }
template <typename T>
static inline __device__ T __shfl_xor_sync(unsigned m, T v, int x, int width = 32)
{ return PROBE_SHFL(bfly, m, v, x, ((32 - width) << 8) | 0x1f); }
static inline __device__ unsigned __ballot_sync(unsigned m, int p)
{ return __nvvm_vote_ballot_sync(m, p != 0); }
static inline __device__ int __any_sync(unsigned m, int p) { return __nvvm_vote_any_sync(m, p != 0); }
static inline __device__ int __all_sync(unsigned m, int p) { return __nvvm_vote_all_sync(m, p != 0); }
static inline __device__ void __syncwarp(unsigned m = 0xffffffffu) { __nvvm_bar_warp_sync(m); }
#elif defined(__AMDGPU__)
#define warpSize __AMDGCN_WAVEFRONT_SIZE
static inline __device__ unsigned probe_lane()
{
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
template <typename T>
static inline __device__ T probe_from_lane(T v, int src, int width, bool earlier_ok = false)
{
    int lane = (int)probe_lane();
    int base = lane - lane % width;
    if ((src < base && !earlier_ok) || src < 0 || src >= base + width) src = lane;
    return probe_from_int<T>(__builtin_amdgcn_ds_bpermute(src << 2, probe_as_int(v)));
}
template <typename T>
static inline __device__ T __shfl_sync(unsigned, T v, int src, int width = warpSize)
{
    int lane = (int)probe_lane();
    return probe_from_lane(v, lane - lane % width + ((src % width) + width) % width, width);
}
template <typename T>
static inline __device__ T __shfl_up_sync(unsigned, T v, unsigned d, int width = warpSize)
{ return probe_from_lane(v, (int)probe_lane() - (int)d, width); }
template <typename T>
static inline __device__ T __shfl_down_sync(unsigned, T v, unsigned d, int width = warpSize)
{ return probe_from_lane(v, (int)probe_lane() + (int)d, width); }
template <typename T>
static inline __device__ T __shfl_xor_sync(unsigned, T v, int x, int width = warpSize)
{ return probe_from_lane(v, (int)probe_lane() ^ x, width, true); }
static inline __device__ unsigned __ballot_sync(unsigned, int p)
{ return __builtin_amdgcn_ballot_w32(p != 0); }
static inline __device__ int __any_sync(unsigned m, int p) { return __ballot_sync(m, p) != 0; }
static inline __device__ int __all_sync(unsigned m, int p)
{ return __ballot_sync(m, 1) == __ballot_sync(m, p); }
static inline __device__ void __syncwarp(unsigned = 0xffffffffu) { __builtin_amdgcn_wave_barrier(); }
#else /* host pass: declarations only */
#define warpSize 32
template <typename T> T __shfl_sync(unsigned, T, int, int = 32);
template <typename T> T __shfl_up_sync(unsigned, T, unsigned, int = 32);
template <typename T> T __shfl_down_sync(unsigned, T, unsigned, int = 32);
template <typename T> T __shfl_xor_sync(unsigned, T, int, int = 32);
unsigned __ballot_sync(unsigned, int);
int __any_sync(unsigned, int);
int __all_sync(unsigned, int);
void __syncwarp(unsigned = 0xffffffffu);
#endif

/* ---- launch ABI the compiler calls into ---------------------------- */

typedef struct probe_stream *probe_stream_t;

#if defined(__HIP__)
extern "C" int hipConfigureCall(dim3 grid, dim3 block, size_t shm = 0,
                                probe_stream_t stream = 0);
extern "C" int __hipPushCallConfiguration(dim3 grid, dim3 block,
                                          size_t shm = 0,
                                          probe_stream_t stream = 0);
extern "C" int hipLaunchKernel(const void *func, dim3 grid, dim3 block,
                               void **args, size_t shm,
                               probe_stream_t stream);
#else
extern "C" int cudaConfigureCall(dim3 grid, dim3 block, size_t shm = 0,
                                 probe_stream_t stream = 0);
extern "C" unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block,
                                                size_t shm = 0,
                                                probe_stream_t stream = 0);
extern "C" int __cudaPopCallConfiguration(dim3 *grid, dim3 *block,
                                          size_t *shm, void *stream);
extern "C" int cudaLaunchKernel(const void *func, dim3 grid, dim3 block,
                                void **args, size_t shm,
                                probe_stream_t stream);
#endif

#endif /* OCU_CPU_PATH */

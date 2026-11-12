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
#include <stddef.h>

#define __global__   __attribute__((global))
#define __device__   __attribute__((device))
#define __host__     __attribute__((host))
#define __shared__   __attribute__((shared))
#define __constant__ __attribute__((constant))

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
extern "C" int cudaLaunchKernel(const void *func, dim3 grid, dim3 block,
                                void **args, size_t shm,
                                probe_stream_t stream);
#endif

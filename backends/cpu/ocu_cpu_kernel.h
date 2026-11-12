/*
 * ocu_cpu_kernel.h -- what device code compiled for the CPU backend sees.
 *
 * Two parts:
 *   1. The module image format: a table of kernel entry points that
 *      module_load accepts.
 *   2. The device-side API a kernel body calls while it runs: its thread
 *      and block indices, the block barrier, dynamic shared memory.
 *
 * Kernel bodies run on lightweight fibers. All threads of one block run on
 * one OS thread, one after another, switching only at the block barrier
 * and at warp operations, so a
 * block never needs locks for its own shared memory. Different blocks run
 * on different OS threads at the same time, so device memory shared across
 * blocks needs real atomics, as on a GPU.
 */
#ifndef OCU_CPU_KERNEL_H
#define OCU_CPU_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- module image ---------------------------------------------------- */

#define OCU_CPU_IMAGE_MAGIC   0x4f435543u /* "OCUC" */
#define OCU_CPU_IMAGE_VERSION 1u

/*
 * One kernel entry point. `args[i]` points at the i-th argument value, laid
 * out exactly as the caller passed it to kernel_launch.
 */
typedef void (*ocu_cpu_kernel_fn)(void **args);

typedef struct ocu_cpu_kernel_entry {
    const char       *name;
    ocu_cpu_kernel_fn fn;
} ocu_cpu_kernel_entry;

/* Pass a pointer to this as `image`, and sizeof(ocu_cpu_image) as `bytes`. */
typedef struct ocu_cpu_image {
    uint32_t                    magic;   /* OCU_CPU_IMAGE_MAGIC */
    uint32_t                    version; /* OCU_CPU_IMAGE_VERSION */
    uint32_t                    count;
    const ocu_cpu_kernel_entry *entries; /* must outlive the module */
} ocu_cpu_image;

/* ---- device-side API (valid only inside a running kernel) ------------- */

typedef struct ocu_cpu_dim3 {
    uint32_t x, y, z;
} ocu_cpu_dim3;

ocu_cpu_dim3 ocu_cpu_thread_idx(void);
ocu_cpu_dim3 ocu_cpu_block_idx(void);
ocu_cpu_dim3 ocu_cpu_block_dim(void);
ocu_cpu_dim3 ocu_cpu_grid_dim(void);
uint32_t     ocu_cpu_warp_size(void);

/*
 * The same values as per-thread variables, for language layers that need
 * real variables (CUDA's threadIdx and friends are variables: host code may
 * declare locals of the same name). Inside a kernel each GPU thread sees
 * its own values, which do not change while it runs: the backend updates
 * them whenever it switches to a thread. Outside a kernel they are zero.
 */
#ifdef __cplusplus
#define OCU_CPU_THREAD_LOCAL thread_local
#else
#define OCU_CPU_THREAD_LOCAL _Thread_local
#endif
extern OCU_CPU_THREAD_LOCAL ocu_cpu_dim3 ocu_cpu_tls_thread_idx;
extern OCU_CPU_THREAD_LOCAL ocu_cpu_dim3 ocu_cpu_tls_block_idx;
extern OCU_CPU_THREAD_LOCAL ocu_cpu_dim3 ocu_cpu_tls_block_dim;
extern OCU_CPU_THREAD_LOCAL ocu_cpu_dim3 ocu_cpu_tls_grid_dim;
extern OCU_CPU_THREAD_LOCAL int          ocu_cpu_tls_warp_size;

/* Block-wide barrier. Threads that already returned count as arrived. */
void ocu_cpu_barrier(void);

/* Start of this block's dynamic shared memory (NULL when none). */
void *ocu_cpu_dynamic_shared(void);

/* ---- warp-level operations ---------------------------------------------
 * A warp is warp_size consecutive threads of a block (by linear thread
 * index). Every live lane of the warp must reach the same warp operation;
 * lanes that already returned do not take part. A warp whose lanes wait on
 * each other in incompatible ways (one at a warp operation, another at the
 * block barrier) fails the launch with OCU_ERR_LAUNCH_FAILED instead of
 * hanging.
 */

/* This thread's lane within its warp. */
uint32_t ocu_cpu_lane_id(void);

/* Warp-wide barrier. */
void ocu_cpu_warp_sync(void);

/*
 * Every lane contributes `value`; `all[i]` receives lane i's value. Returns
 * the number of lanes in this warp (smaller than warp_size only for the
 * last warp of a block whose size is not a multiple of it). `all` must
 * hold OCU_CPU_MAX_WARP entries. Shuffles and votes are built on this.
 */
#define OCU_CPU_MAX_WARP 64
uint32_t ocu_cpu_warp_gather(uint64_t value, uint64_t *all);

/*
 * The same, with a tag naming the call (for example the operation, mask
 * and width). Every live lane meeting at this gather must pass the same
 * tag; otherwise the lanes came from different calls (as with partial-mask
 * code) and the launch fails with OCU_ERR_LAUNCH_FAILED instead of pairing
 * them silently. ocu_cpu_warp_gather uses tag 0.
 */
uint32_t ocu_cpu_warp_gather_tagged(uint64_t value, uint64_t tag, uint64_t *all);

/*
 * The same, with the participation mask of a CUDA-style *_sync call: bit b
 * names lane b of the caller's 32-lane group (the caller always takes
 * part). A lane is released once every lane its mask names waits at a warp
 * operation or has returned; lanes the mask does not name may do anything
 * else meanwhile, including waiting at the block barrier. The named lanes
 * must be at the same call, the same occurrence of it and with the same
 * mask, else the launch fails with OCU_ERR_LAUNCH_FAILED. So disjoint
 * masks (0x0000ffff in one half, 0xffff0000 in the other) work as in CUDA.
 * A named lane that has already returned is undefined in CUDA: tolerated by
 * default, an error with OCU_CPU_STRICT=1 (this also catches a lane that
 * skipped an occurrence and finished early). Mask 0 names the whole warp.
 */
uint32_t ocu_cpu_warp_gather_masked(uint64_t value, uint64_t tag, uint32_t mask,
                                    uint64_t *all);

/* Warp barrier with a participation mask (see above). */
void ocu_cpu_warp_sync_masked(uint32_t mask);

/*
 * The same meeting as ocu_cpu_warp_gather_masked, returning one value
 * only: lane `src`'s (warp-relative), or the caller's own when `src` is not
 * a lane of this warp. Cheaper than a gather for shuffles.
 */
uint64_t ocu_cpu_warp_shuffle_masked(uint64_t value, uint64_t tag, uint32_t mask,
                                     uint32_t src);

/*
 * The same meeting, for votes: bit i of the result is set when lane i
 * (warp-relative) passed a nonzero `pred`. Lanes the warp does not have
 * read as 0.
 */
uint64_t ocu_cpu_warp_vote_masked(int pred, uint64_t tag, uint32_t mask);

#ifdef __cplusplus
}
#endif

#endif /* OCU_CPU_KERNEL_H */

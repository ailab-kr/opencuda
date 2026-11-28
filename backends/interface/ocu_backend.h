/*
 * ocu_backend.h -- device backend interface (the only chip-specific boundary).
 *
 * Everything above this header (compiler glue, runtime, libraries) is
 * chip-neutral. Everything a chip needs is implemented behind this table.
 * Adding a chip means adding one implementation of `ocu_backend`; nothing
 * above this header may change.
 *
 * Rules for this file:
 *   - Self-contained C11. No includes other than the three standard headers.
 *   - Keep it narrow. A new entry needs a design decision record.
 *   - Any incompatible change bumps OCU_BACKEND_ABI.
 *
 * "ocu" is a working prefix; the public name is not decided yet.
 */
#ifndef OCU_BACKEND_H
#define OCU_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OCU_BACKEND_ABI 2u

/* Largest total size of one launch's argument values, in bytes. */
#define OCU_MAX_ARG_BYTES 32768u

/* ---- status --------------------------------------------------------- */

typedef enum ocu_status {
    OCU_OK = 0,
    OCU_ERR_INVALID_VALUE,
    OCU_ERR_INVALID_DEVICE,
    OCU_ERR_INVALID_HANDLE,
    OCU_ERR_OUT_OF_MEMORY,
    OCU_ERR_NOT_READY,        /* async work still running (query calls only) */
    OCU_ERR_LAUNCH_FAILED,
    OCU_ERR_NOT_SUPPORTED,    /* the chip cannot do this */
    OCU_ERR_NOT_IMPLEMENTED,  /* the backend does not do this yet */
    OCU_ERR_UNKNOWN,
    OCU_ERR_NOT_FOUND         /* e.g. no kernel with that name in a module */
} ocu_status;

/* ---- opaque handles (owned by the backend) -------------------------- */

typedef struct ocu_device ocu_device;
typedef struct ocu_queue  ocu_queue;   /* ordered async work; maps to a stream */
typedef struct ocu_module ocu_module;  /* loaded device code */
typedef struct ocu_kernel ocu_kernel;  /* one entry point inside a module */
typedef struct ocu_marker ocu_marker;  /* point in a queue; maps to an event */

/* ---- device properties ---------------------------------------------- */

typedef struct ocu_dev_info {
    char     name[256];
    uint64_t total_memory_bytes;
    uint32_t warp_size;              /* real value for the chip: 32 or 64 */
    uint32_t max_threads_per_block;
    uint32_t max_block_dim[3];
    uint32_t max_grid_dim[3];
    uint64_t max_shared_memory_per_block;
    uint32_t supports_unified_memory; /* 0 or 1 */
    /* Units the device runs blocks on, as a CUDA multiprocessor count
       (an NVIDIA SM, an Intel Xe core = sub-slice, ...); 0 = not known.
       Programs size grids by it (llama.cpp's FlashAttention KV split,
       stream-k matrix products). */
    uint32_t multiprocessor_count;
    /* 1: memory from host_alloc_pinned is device memory too, at the same
       address (kernels read and write it directly, as CUDA's mapped
       page-locked memory); 0: copies only. */
    uint32_t pinned_is_mapped;
} ocu_dev_info;

typedef enum ocu_copy_dir {
    OCU_COPY_HOST_TO_DEVICE = 0,
    OCU_COPY_DEVICE_TO_HOST,
    OCU_COPY_DEVICE_TO_DEVICE
} ocu_copy_dir;

typedef struct ocu_launch_params {
    uint32_t      grid[3];
    uint32_t      block[3];
    size_t        shared_memory_bytes;
    void        **args;       /* args[i] points at the i-th argument value */
    const size_t *arg_sizes;  /* size of each argument value in bytes */
    uint32_t      num_args;
} ocu_launch_params;

/* ---- the table ------------------------------------------------------ */

typedef struct ocu_backend {
    uint32_t    abi;   /* must equal OCU_BACKEND_ABI */
    const char *name;  /* "cpu", "amd", ... (lower case, stable) */

    /* device */
    ocu_status (*dev_count)(uint32_t *count);
    ocu_status (*dev_info)(uint32_t index, ocu_dev_info *info);
    ocu_status (*dev_open)(uint32_t index, ocu_device **dev);
    ocu_status (*dev_close)(ocu_device *dev);

    /*
     * memory
     * Device pointers must be plain addresses: `ptr + n` has to be a valid
     * device pointer inside the same allocation, because user code does
     * pointer arithmetic on them.
     * `queue == NULL` means: run synchronously and return when done.
     * mem_free and host_free_pinned first wait for all queued work on the
     * device, so memory still used by queued work is never released early
     * (the same promise as an implicit device synchronize).
     */
    ocu_status (*mem_alloc)(ocu_device *dev, size_t bytes, void **ptr);
    ocu_status (*mem_free)(ocu_device *dev, void *ptr);
    ocu_status (*mem_copy)(ocu_device *dev, ocu_queue *queue, void *dst,
                           const void *src, size_t bytes, ocu_copy_dir dir);
    ocu_status (*mem_set)(ocu_device *dev, ocu_queue *queue, void *dst,
                          int value, size_t bytes);
    ocu_status (*mem_info)(ocu_device *dev, uint64_t *free_bytes,
                           uint64_t *total_bytes);
    ocu_status (*host_alloc_pinned)(ocu_device *dev, size_t bytes, void **ptr);
    ocu_status (*host_free_pinned)(ocu_device *dev, void *ptr);

    /*
     * queues: work submitted to one queue runs in submission order.
     * An error raised by queued work is reported by the next queue_sync or
     * queue_query on that queue, then cleared.
     */
    ocu_status (*queue_create)(ocu_device *dev, ocu_queue **queue);
    ocu_status (*queue_destroy)(ocu_queue *queue);
    ocu_status (*queue_sync)(ocu_queue *queue);
    ocu_status (*queue_query)(ocu_queue *queue); /* OCU_OK or OCU_ERR_NOT_READY */

    /*
     * code
     * `image` is device code produced by the compiler glue for this backend.
     * Its format is private to the backend (see docs in each backend folder).
     */
    ocu_status (*module_load)(ocu_device *dev, const void *image, size_t bytes,
                              ocu_module **module);
    ocu_status (*module_unload)(ocu_module *module);
    ocu_status (*module_get_kernel)(ocu_module *module, const char *name,
                                    ocu_kernel **kernel);

    /*
     * execution
     * Argument values are copied before kernel_launch returns, so the caller
     * may reuse `params->args` storage right away. `queue == NULL` runs the
     * kernel to completion before returning. The argument values may total
     * at most OCU_MAX_ARG_BYTES. Calling kernel_launch, queue_sync,
     * queue_destroy or marker_wait from inside a running kernel returns
     * OCU_ERR_NOT_SUPPORTED.
     */
    ocu_status (*kernel_launch)(ocu_queue *queue, ocu_kernel *kernel,
                                const ocu_launch_params *params);

    /*
     * markers
     * A marker that was never recorded is complete: query returns OCU_OK and
     * wait returns at once. wait waits for the record that was current when
     * wait was called; recording again meanwhile does not extend the wait. elapsed_ms needs two recorded, completed markers
     * (OCU_ERR_INVALID_HANDLE if either was never recorded, OCU_ERR_NOT_READY
     * if either is still pending).
     */
    ocu_status (*marker_create)(ocu_device *dev, ocu_marker **marker);
    ocu_status (*marker_destroy)(ocu_marker *marker);
    ocu_status (*marker_record)(ocu_marker *marker, ocu_queue *queue);
    ocu_status (*marker_wait)(ocu_marker *marker);
    ocu_status (*marker_query)(ocu_marker *marker); /* OK or NOT_READY */
    ocu_status (*marker_elapsed_ms)(ocu_marker *start, ocu_marker *end,
                                    float *milliseconds);

    /*
     * Optional library providers.
     * Lets a backend expose a tuned vendor library (e.g. "blas", "fft")
     * without the library layer knowing which chip it runs on.
     * Returns a provider table defined by the library layer, or NULL when
     * the backend has none; the library layer then uses its portable path.
     * May itself be NULL.
     */
    const void *(*get_provider)(const char *library);

    /*
     * Optional (ABI 2). May be NULL.
     *
     * module_get_global: device address and size of a variable defined in
     * the module's device code (`__device__` / `__constant__` variables), by
     * its device-code name. The address is a device pointer accepted by
     * mem_copy and mem_set for `bytes` bytes. OCU_ERR_NOT_FOUND if absent.
     *
     * kernel_arg_sizes: the kernel's parameter count and the size in bytes
     * of each parameter, as the device code declares them, for callers that
     * hold only an array of argument pointers (an API launch). At most
     * `max` sizes are written; `*count` is the real count. Backends whose
     * images carry no parameter list return OCU_ERR_NOT_SUPPORTED.
     */
    ocu_status (*module_get_global)(ocu_module *module, const char *name,
                                    void **ptr, size_t *bytes);
    ocu_status (*kernel_arg_sizes)(ocu_kernel *kernel, uint32_t *count,
                                   size_t *sizes, uint32_t max);

    /*
     * Optional (ABI 2). May be NULL; the runtime then offers these only
     * where supports_unified_memory is set.
     *
     * mem_alloc_shared: memory the host and the device both use directly
     * at the same address (freed with mem_free).
     * host_register / host_unregister: make existing host memory usable by
     * the device; *dev_ptr receives the address device code uses for it.
     */
    ocu_status (*mem_alloc_shared)(ocu_device *dev, size_t bytes, void **ptr);
    ocu_status (*host_register)(ocu_device *dev, void *ptr, size_t bytes, void **dev_ptr);
    ocu_status (*host_unregister)(ocu_device *dev, void *ptr);
} ocu_backend;

/* Human-readable name of a status value. Never returns NULL. */
static inline const char *ocu_status_name(ocu_status s)
{
    switch (s) {
    case OCU_OK:                  return "OCU_OK";
    case OCU_ERR_INVALID_VALUE:   return "OCU_ERR_INVALID_VALUE";
    case OCU_ERR_INVALID_DEVICE:  return "OCU_ERR_INVALID_DEVICE";
    case OCU_ERR_INVALID_HANDLE:  return "OCU_ERR_INVALID_HANDLE";
    case OCU_ERR_OUT_OF_MEMORY:   return "OCU_ERR_OUT_OF_MEMORY";
    case OCU_ERR_NOT_READY:       return "OCU_ERR_NOT_READY";
    case OCU_ERR_LAUNCH_FAILED:   return "OCU_ERR_LAUNCH_FAILED";
    case OCU_ERR_NOT_SUPPORTED:   return "OCU_ERR_NOT_SUPPORTED";
    case OCU_ERR_NOT_IMPLEMENTED: return "OCU_ERR_NOT_IMPLEMENTED";
    case OCU_ERR_UNKNOWN:         return "OCU_ERR_UNKNOWN";
    case OCU_ERR_NOT_FOUND:       return "OCU_ERR_NOT_FOUND";
    }
    return "OCU_ERR_UNKNOWN";
}

#ifdef __cplusplus
}
#endif

#endif /* OCU_BACKEND_H */

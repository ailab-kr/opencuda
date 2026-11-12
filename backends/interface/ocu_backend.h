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

#define OCU_BACKEND_ABI 1u

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
    OCU_ERR_UNKNOWN
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

    /* queues: work submitted to one queue runs in submission order */
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

    /* execution */
    ocu_status (*kernel_launch)(ocu_queue *queue, ocu_kernel *kernel,
                                const ocu_launch_params *params);

    /* markers */
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
    }
    return "OCU_ERR_UNKNOWN";
}

#ifdef __cplusplus
}
#endif

#endif /* OCU_BACKEND_H */

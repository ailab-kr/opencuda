/*
 * CPU reference backend -- skeleton.
 *
 * Purpose: run device code on the CPU so correctness tests need no GPU.
 * Only device enumeration is implemented. Every other entry returns
 * OCU_ERR_NOT_IMPLEMENTED until the real implementation lands
 * (see PLAN.md P0-W3-01, P0-W3-02 and ARCHITECTURE.md "CPU reference backend").
 */
#include "ocu_backend.h"
#include "ocu_backend_cpu.h"

#include <stdlib.h>
#include <string.h>

/* OCU_CPU_WARP=32|64 lets tests run with either warp size without a GPU. */
static uint32_t cpu_warp_size(void)
{
    const char *v = getenv("OCU_CPU_WARP");
    if (v != NULL && strcmp(v, "64") == 0) {
        return 64u;
    }
    return 32u;
}

static ocu_status cpu_dev_count(uint32_t *count)
{
    if (count == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    *count = 1u;
    return OCU_OK;
}

static ocu_status cpu_dev_info(uint32_t index, ocu_dev_info *info)
{
    if (info == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    if (index != 0u) {
        return OCU_ERR_INVALID_DEVICE;
    }
    memset(info, 0, sizeof(*info));
    strncpy(info->name, "OCU CPU reference device", sizeof(info->name) - 1);
    info->total_memory_bytes = 0; /* filled in with the memory implementation */
    info->warp_size = cpu_warp_size();
    info->max_threads_per_block = 1024u;
    info->max_block_dim[0] = 1024u;
    info->max_block_dim[1] = 1024u;
    info->max_block_dim[2] = 64u;
    info->max_grid_dim[0] = 2147483647u;
    info->max_grid_dim[1] = 65535u;
    info->max_grid_dim[2] = 65535u;
    info->max_shared_memory_per_block = 48u * 1024u;
    info->supports_unified_memory = 1u;
    return OCU_OK;
}

/* ---- not implemented yet -------------------------------------------- */

static ocu_status cpu_dev_open(uint32_t index, ocu_device **dev)
{ (void)index; (void)dev; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_dev_close(ocu_device *dev)
{ (void)dev; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_mem_alloc(ocu_device *dev, size_t bytes, void **ptr)
{ (void)dev; (void)bytes; (void)ptr; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_mem_free(ocu_device *dev, void *ptr)
{ (void)dev; (void)ptr; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_mem_copy(ocu_device *dev, ocu_queue *queue, void *dst,
                               const void *src, size_t bytes, ocu_copy_dir dir)
{
    (void)dev; (void)queue; (void)dst; (void)src; (void)bytes; (void)dir;
    return OCU_ERR_NOT_IMPLEMENTED;
}

static ocu_status cpu_mem_set(ocu_device *dev, ocu_queue *queue, void *dst,
                              int value, size_t bytes)
{
    (void)dev; (void)queue; (void)dst; (void)value; (void)bytes;
    return OCU_ERR_NOT_IMPLEMENTED;
}

static ocu_status cpu_mem_info(ocu_device *dev, uint64_t *free_bytes,
                               uint64_t *total_bytes)
{ (void)dev; (void)free_bytes; (void)total_bytes; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_host_alloc_pinned(ocu_device *dev, size_t bytes, void **ptr)
{ (void)dev; (void)bytes; (void)ptr; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_host_free_pinned(ocu_device *dev, void *ptr)
{ (void)dev; (void)ptr; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_queue_create(ocu_device *dev, ocu_queue **queue)
{ (void)dev; (void)queue; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_queue_destroy(ocu_queue *queue)
{ (void)queue; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_queue_sync(ocu_queue *queue)
{ (void)queue; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_queue_query(ocu_queue *queue)
{ (void)queue; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_module_load(ocu_device *dev, const void *image,
                                  size_t bytes, ocu_module **module)
{ (void)dev; (void)image; (void)bytes; (void)module; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_module_unload(ocu_module *module)
{ (void)module; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_module_get_kernel(ocu_module *module, const char *name,
                                        ocu_kernel **kernel)
{ (void)module; (void)name; (void)kernel; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_kernel_launch(ocu_queue *queue, ocu_kernel *kernel,
                                    const ocu_launch_params *params)
{ (void)queue; (void)kernel; (void)params; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_marker_create(ocu_device *dev, ocu_marker **marker)
{ (void)dev; (void)marker; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_marker_destroy(ocu_marker *marker)
{ (void)marker; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_marker_record(ocu_marker *marker, ocu_queue *queue)
{ (void)marker; (void)queue; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_marker_wait(ocu_marker *marker)
{ (void)marker; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_marker_query(ocu_marker *marker)
{ (void)marker; return OCU_ERR_NOT_IMPLEMENTED; }

static ocu_status cpu_marker_elapsed_ms(ocu_marker *start, ocu_marker *end,
                                        float *milliseconds)
{ (void)start; (void)end; (void)milliseconds; return OCU_ERR_NOT_IMPLEMENTED; }

static const ocu_backend cpu_backend = {
    .abi = OCU_BACKEND_ABI,
    .name = "cpu",
    .dev_count = cpu_dev_count,
    .dev_info = cpu_dev_info,
    .dev_open = cpu_dev_open,
    .dev_close = cpu_dev_close,
    .mem_alloc = cpu_mem_alloc,
    .mem_free = cpu_mem_free,
    .mem_copy = cpu_mem_copy,
    .mem_set = cpu_mem_set,
    .mem_info = cpu_mem_info,
    .host_alloc_pinned = cpu_host_alloc_pinned,
    .host_free_pinned = cpu_host_free_pinned,
    .queue_create = cpu_queue_create,
    .queue_destroy = cpu_queue_destroy,
    .queue_sync = cpu_queue_sync,
    .queue_query = cpu_queue_query,
    .module_load = cpu_module_load,
    .module_unload = cpu_module_unload,
    .module_get_kernel = cpu_module_get_kernel,
    .kernel_launch = cpu_kernel_launch,
    .marker_create = cpu_marker_create,
    .marker_destroy = cpu_marker_destroy,
    .marker_record = cpu_marker_record,
    .marker_wait = cpu_marker_wait,
    .marker_query = cpu_marker_query,
    .marker_elapsed_ms = cpu_marker_elapsed_ms,
    .get_provider = NULL,
};

const ocu_backend *ocu_backend_cpu(void)
{
    return &cpu_backend;
}

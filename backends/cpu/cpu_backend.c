/*
 * CPU reference backend: devices, modules and the backend table.
 *
 * Purpose: run device code on the CPU so correctness tests need no GPU.
 * Memory lives in cpu_memory.c, queues and markers in cpu_queue.c, kernel
 * execution in cpu_exec.c. See ARCHITECTURE.md, "CPU reference backend".
 */
#include "ocu_backend.h"
#include "ocu_backend_cpu.h"
#include "cpu_internal.h"

#include <stdlib.h>
#include <string.h>

static ocu_device g_device; /* the one CPU device */

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
    info->total_memory_bytes = cpu_total_memory();
    info->warp_size = cpu_get_config()->warp_size;
    info->max_threads_per_block = CPU_MAX_THREADS_PER_BLOCK;
    info->max_block_dim[0] = CPU_MAX_BLOCK_X;
    info->max_block_dim[1] = CPU_MAX_BLOCK_Y;
    info->max_block_dim[2] = CPU_MAX_BLOCK_Z;
    info->max_grid_dim[0] = CPU_MAX_GRID_X;
    info->max_grid_dim[1] = CPU_MAX_GRID_Y;
    info->max_grid_dim[2] = CPU_MAX_GRID_Z;
    info->max_shared_memory_per_block = CPU_MAX_SHARED_BYTES;
    info->supports_unified_memory = 1u;
    return OCU_OK;
}

static ocu_status cpu_dev_open(uint32_t index, ocu_device **dev)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    if (index != 0u) {
        return OCU_ERR_INVALID_DEVICE;
    }
    *dev = &g_device;
    return OCU_OK;
}

static ocu_status cpu_dev_close(ocu_device *dev)
{
    return dev == &g_device ? OCU_OK : OCU_ERR_INVALID_DEVICE;
}

/* ---- modules ----------------------------------------------------------- */

typedef struct module_box {
    ocu_module module;   /* first member: the handle points here */
    ocu_kernel *kernels; /* one per image entry */
} module_box;

static ocu_status cpu_module_load(ocu_device *dev, const void *image,
                                  size_t bytes, ocu_module **module)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_DEVICE;
    }
    if (image == NULL || module == NULL || bytes != sizeof(ocu_cpu_image)) {
        return OCU_ERR_INVALID_VALUE;
    }
    const ocu_cpu_image *img = image;
    if (img->magic != OCU_CPU_IMAGE_MAGIC ||
        img->version != OCU_CPU_IMAGE_VERSION) {
        return OCU_ERR_INVALID_VALUE;
    }
    if (img->count > 0 && img->entries == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    for (uint32_t i = 0; i < img->count; ++i) {
        if (img->entries[i].name == NULL || img->entries[i].fn == NULL) {
            return OCU_ERR_INVALID_VALUE;
        }
    }
    module_box *box = calloc(1, sizeof(*box));
    if (box == NULL) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    if (img->count > 0) {
        box->kernels = calloc(img->count, sizeof(*box->kernels));
        if (box->kernels == NULL) {
            free(box);
            return OCU_ERR_OUT_OF_MEMORY;
        }
    }
    box->module.image = img;
    for (uint32_t i = 0; i < img->count; ++i) {
        box->kernels[i].module = &box->module;
        box->kernels[i].fn = img->entries[i].fn;
    }
    *module = &box->module;
    return OCU_OK;
}

static ocu_status cpu_module_unload(ocu_module *module)
{
    if (module == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    module_box *box = (module_box *)module;
    free(box->kernels);
    free(box);
    return OCU_OK;
}

static ocu_status cpu_module_get_kernel(ocu_module *module, const char *name,
                                        ocu_kernel **kernel)
{
    if (module == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    if (name == NULL || kernel == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    module_box *box = (module_box *)module;
    const ocu_cpu_image *img = module->image;
    for (uint32_t i = 0; i < img->count; ++i) {
        if (strcmp(img->entries[i].name, name) == 0) {
            *kernel = &box->kernels[i];
            return OCU_OK;
        }
    }
    return OCU_ERR_NOT_FOUND;
}

/* ---- table --------------------------------------------------------------- */

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
    .get_provider = cpu_get_provider,
    .module_get_global = NULL, /* host addresses are device addresses */
    .kernel_arg_sizes = NULL,  /* CPU images are called with typed thunks */
    .mem_alloc_shared = NULL,  /* all memory is shared: the runtime uses mem_alloc */
    .host_register = NULL,
    .host_unregister = NULL,
};

const ocu_backend *ocu_backend_cpu(void)
{
    return &cpu_backend;
}

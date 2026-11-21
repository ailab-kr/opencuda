/*
 * amd_backend.c -- the device backend interface on AMD GPUs, through the
 * installed HIP runtime's public API (see amd_hip.h), following the mapping
 * of backends/amd/README.md section 2.
 *
 * Device code images are amdgcn code objects (ELF) made by Clang in HIP
 * mode with OpenCuda's headers (runtime/include/ocu_cuda_lang_amdgcn.h):
 * tools/amdgpu/ocu_amd_cc.py, or `ocu_cc.py --target amd`. The runtime
 * loads them as they are (hipModuleLoadData). Every call selects the
 * device on the calling thread first (hipSetDevice), the HIP counterpart
 * of making a context current.
 */
#include "ocu_backend.h"
#include "ocu_backend_amd.h"
#include "amd_hip.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- objects ----------------------------------------------------------- */

typedef struct alloc_range {
    uintptr_t base;
    size_t    size;
} alloc_range;

struct ocu_device {
    const amd_api  *hip;
    int             ordinal;
    hip_stream      sync_stream; /* for queue == NULL */
    pthread_mutex_t lock;        /* guards the allocation table */
    alloc_range    *allocs;
    size_t          alloc_count, alloc_cap;
};

struct ocu_queue {
    ocu_device *dev;
    hip_stream  stream;
};

#define AMD_MAX_PARAMS 256

struct ocu_kernel {
    struct ocu_module *module;
    hip_function       fn;
    char              *name;
    int32_t            param_count; /* -1: not in the code object's metadata */
    size_t            *param_sizes;
    struct ocu_kernel *next;
};

struct ocu_module {
    ocu_device        *dev;
    hip_module         mod;
    void              *image; /* kept for the parameter lists */
    size_t             image_bytes;
    pthread_mutex_t    lock;  /* guards the kernel list */
    struct ocu_kernel *kernels;
};

struct ocu_marker {
    ocu_device *dev;
    hip_event   event;
    bool        recorded;
};

/* Result mapping. Configuration errors at launch (block too large, too
   much shared memory) come back as INVALID_VALUE and are not sticky
   (observed: the next good launch succeeds). */
static ocu_status map(hip_result r)
{
    switch (r) {
    case HIP_SUCCESS:                       return OCU_OK;
    case HIP_ERROR_INVALID_VALUE:           return OCU_ERR_INVALID_VALUE;
    case HIP_ERROR_OUT_OF_MEMORY:           return OCU_ERR_OUT_OF_MEMORY;
    case HIP_ERROR_NO_DEVICE:
    case HIP_ERROR_INVALID_DEVICE:          return OCU_ERR_INVALID_DEVICE;
    case HIP_ERROR_INVALID_IMAGE:           return OCU_ERR_INVALID_VALUE;
    case HIP_ERROR_NO_BINARY_FOR_GPU:       return OCU_ERR_NOT_SUPPORTED;
    case HIP_ERROR_INVALID_HANDLE:
    case HIP_ERROR_INVALID_CONTEXT:         return OCU_ERR_INVALID_HANDLE;
    case HIP_ERROR_NOT_FOUND:               return OCU_ERR_NOT_FOUND;
    case HIP_ERROR_NOT_READY:               return OCU_ERR_NOT_READY;
    case HIP_ERROR_LAUNCH_OUT_OF_RESOURCES: return OCU_ERR_INVALID_VALUE; /* configuration */
    case HIP_ERROR_NOT_SUPPORTED:           return OCU_ERR_NOT_SUPPORTED;
    default:
        if (r >= 700 && r < 800) return OCU_ERR_LAUNCH_FAILED; /* faults while running */
        return OCU_ERR_UNKNOWN;
    }
}

static const amd_api *api(void) { return amd_load(); }

static char *dup_string(const char *s)
{
    size_t n = strlen(s) + 1;
    char *c = malloc(n);
    if (c != NULL) memcpy(c, s, n);
    return c;
}

static ocu_status enter(ocu_device *d) { return map(d->hip->SetDevice(d->ordinal)); }

/* ---- allocation table (device pointers must be checkable) ------------- */

static bool table_add(ocu_device *d, uintptr_t base, size_t size)
{
    pthread_mutex_lock(&d->lock);
    bool ok = true;
    if (d->alloc_count == d->alloc_cap) {
        size_t cap = d->alloc_cap ? d->alloc_cap * 2 : 64;
        alloc_range *n = realloc(d->allocs, cap * sizeof(*n));
        if (n == NULL) {
            ok = false;
        } else {
            d->allocs = n;
            d->alloc_cap = cap;
        }
    }
    if (ok) {
        d->allocs[d->alloc_count].base = base;
        d->allocs[d->alloc_count].size = size;
        d->alloc_count++;
    }
    pthread_mutex_unlock(&d->lock);
    return ok;
}

static bool table_remove(ocu_device *d, uintptr_t base)
{
    bool found = false;
    pthread_mutex_lock(&d->lock);
    for (size_t i = 0; i < d->alloc_count; ++i) {
        if (d->allocs[i].base == base) {
            d->allocs[i] = d->allocs[--d->alloc_count];
            found = true;
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
    return found;
}

static bool table_contains(ocu_device *d, uintptr_t p, size_t n)
{
    bool found = false;
    pthread_mutex_lock(&d->lock);
    for (size_t i = 0; i < d->alloc_count && !found; ++i) {
        const alloc_range *r = &d->allocs[i];
        found = p >= r->base && p - r->base <= r->size && n <= r->size - (p - r->base);
    }
    pthread_mutex_unlock(&d->lock);
    return found;
}

/* ---- device -------------------------------------------------------------- */

static ocu_status amd_dev_count(uint32_t *count)
{
    if (count == NULL) return OCU_ERR_INVALID_VALUE;
    *count = 0;
    const amd_api *a = api();
    if (a == NULL) return OCU_OK; /* no runtime: no devices */
    int n = 0;
    if (a->GetDeviceCount(&n) != HIP_SUCCESS || n < 0) return OCU_OK;
    *count = (uint32_t)n;
    return OCU_OK;
}

static ocu_status amd_dev_info(uint32_t index, ocu_dev_info *info)
{
    if (info == NULL) return OCU_ERR_INVALID_VALUE;
    const amd_api *a = api();
    uint32_t n = 0;
    amd_dev_count(&n);
    if (a == NULL || index >= n) return OCU_ERR_INVALID_DEVICE;
    hip_device dev = 0;
    ocu_status st = map(a->DeviceGet(&dev, (int)index));
    if (st != OCU_OK) return st;
    memset(info, 0, sizeof(*info));
    if (a->DeviceGetName(info->name, (int)sizeof(info->name), dev) != HIP_SUCCESS) {
        snprintf(info->name, sizeof(info->name), "AMD GPU %u", index);
    }
    info->name[sizeof(info->name) - 1] = '\0';
    size_t total = 0;
    (void)a->DeviceTotalMem(&total, dev);
    info->total_memory_bytes = total;
    int v = 0;
#define AMD_ATTR(field, attr) (v = 0, a->DeviceGetAttribute(&v, attr, (int)index), info->field = (uint32_t)v)
    AMD_ATTR(warp_size, HIP_ATTR_WARP_SIZE);
    AMD_ATTR(max_threads_per_block, HIP_ATTR_MAX_THREADS_PER_BLOCK);
    AMD_ATTR(max_block_dim[0], HIP_ATTR_MAX_BLOCK_DIM_X);
    AMD_ATTR(max_block_dim[1], HIP_ATTR_MAX_BLOCK_DIM_Y);
    AMD_ATTR(max_block_dim[2], HIP_ATTR_MAX_BLOCK_DIM_Z);
    AMD_ATTR(max_grid_dim[0], HIP_ATTR_MAX_GRID_DIM_X);
    AMD_ATTR(max_grid_dim[1], HIP_ATTR_MAX_GRID_DIM_Y);
    AMD_ATTR(max_grid_dim[2], HIP_ATTR_MAX_GRID_DIM_Z);
#undef AMD_ATTR
    v = 0;
    a->DeviceGetAttribute(&v, HIP_ATTR_MAX_SHARED_MEMORY_PER_BLOCK, (int)index);
    info->max_shared_memory_per_block = (uint64_t)v;
    /* Device memory is reached through its own allocations; host memory is
       not directly addressable by kernels (even on an integrated GPU, the
       runtime keeps the two apart unless registered). */
    info->supports_unified_memory = 0;
    return OCU_OK;
}

static ocu_status amd_dev_open(uint32_t index, ocu_device **out)
{
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    const amd_api *a = api();
    uint32_t n = 0;
    amd_dev_count(&n);
    if (a == NULL || index >= n) return OCU_ERR_INVALID_DEVICE;
    ocu_device *d = calloc(1, sizeof(*d));
    if (d == NULL) return OCU_ERR_OUT_OF_MEMORY;
    d->hip = a;
    d->ordinal = (int)index;
    pthread_mutex_init(&d->lock, NULL);
    ocu_status st = enter(d);
    if (st == OCU_OK) st = map(a->StreamCreateWithFlags(&d->sync_stream, HIP_STREAM_NON_BLOCKING));
    if (st != OCU_OK) {
        pthread_mutex_destroy(&d->lock);
        free(d);
        return st;
    }
    *out = d;
    return OCU_OK;
}

static ocu_status amd_dev_close(ocu_device *d)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(d);
    if (st == OCU_OK) st = map(d->hip->DeviceSynchronize());
    d->hip->StreamDestroy(d->sync_stream);
    pthread_mutex_destroy(&d->lock);
    free(d->allocs);
    free(d);
    return st;
}

/* ---- memory -------------------------------------------------------------- */

static ocu_status amd_mem_alloc(ocu_device *d, size_t bytes, void **ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    if (bytes == 0) {
        *ptr = NULL;
        return OCU_OK;
    }
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    void *p = NULL;
    st = map(d->hip->Malloc(&p, bytes));
    if (st != OCU_OK) return st;
    if (!table_add(d, (uintptr_t)p, bytes)) {
        d->hip->Free(p);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    *ptr = p;
    return OCU_OK;
}

static ocu_status amd_mem_alloc_shared(ocu_device *d, size_t bytes, void **ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    if (bytes == 0) {
        *ptr = NULL;
        return OCU_OK;
    }
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    int managed = 0;
    d->hip->DeviceGetAttribute(&managed, HIP_ATTR_MANAGED_MEMORY, d->ordinal);
    if (!managed) return OCU_ERR_NOT_SUPPORTED; /* e.g. Windows (observed: 0 on the 780M) */
    void *p = NULL;
    st = map(d->hip->MallocManaged(&p, bytes, HIP_MEM_ATTACH_GLOBAL));
    if (st != OCU_OK) return st;
    if (!table_add(d, (uintptr_t)p, bytes)) {
        d->hip->Free(p);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    *ptr = p;
    return OCU_OK;
}

static ocu_status amd_host_register(ocu_device *d, void *ptr, size_t bytes, void **dev_ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL || bytes == 0 || dev_ptr == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    st = map(d->hip->HostRegister(ptr, bytes, HIP_HOST_REGISTER_PORTABLE | HIP_HOST_REGISTER_MAPPED));
    if (st != OCU_OK) return st;
    void *p = NULL;
    st = map(d->hip->HostGetDevicePointer(&p, ptr, 0));
    if (st == OCU_OK && !table_add(d, (uintptr_t)p, bytes)) st = OCU_ERR_OUT_OF_MEMORY;
    if (st != OCU_OK) {
        d->hip->HostUnregister(ptr);
        return st;
    }
    *dev_ptr = p;
    return OCU_OK;
}

static ocu_status amd_host_unregister(ocu_device *d, void *ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    void *p = NULL;
    if (d->hip->HostGetDevicePointer(&p, ptr, 0) == HIP_SUCCESS) (void)table_remove(d, (uintptr_t)p);
    st = map(d->hip->DeviceSynchronize()); /* work may still read it */
    ocu_status u = map(d->hip->HostUnregister(ptr));
    return st != OCU_OK ? st : u;
}

static ocu_status amd_mem_free(ocu_device *d, void *ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_OK;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    uintptr_t p = (uintptr_t)ptr;
    if (!table_contains(d, p, 0)) return OCU_ERR_INVALID_VALUE;
    /* Whether hipFree waits for queued work is not documented; wait here. */
    st = map(d->hip->DeviceSynchronize());
    if (!table_remove(d, p)) return OCU_ERR_INVALID_VALUE; /* not the start */
    ocu_status f = map(d->hip->Free(ptr));
    return st != OCU_OK ? st : f;
}

static hip_stream stream_of(ocu_device *d, ocu_queue *q) { return q ? q->stream : d->sync_stream; }

static ocu_status finish(ocu_device *d, ocu_queue *q, hip_result r)
{
    ocu_status st = map(r);
    if (st == OCU_OK && q == NULL) st = map(d->hip->StreamSynchronize(d->sync_stream));
    return st;
}

static ocu_status amd_mem_copy(ocu_device *d, ocu_queue *q, void *dst, const void *src, size_t bytes,
                               ocu_copy_dir dir)
{
    if (d == NULL || (q != NULL && q->dev != d)) return OCU_ERR_INVALID_HANDLE;
    if (bytes == 0) return OCU_OK;
    if (dst == NULL || src == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    hip_stream s = stream_of(d, q);
    switch (dir) {
    case OCU_COPY_HOST_TO_DEVICE:
        if (!table_contains(d, (uintptr_t)dst, bytes)) return OCU_ERR_INVALID_VALUE;
        return finish(d, q, d->hip->MemcpyAsync(dst, src, bytes, HIP_MEMCPY_HOST_TO_DEVICE, s));
    case OCU_COPY_DEVICE_TO_HOST:
        if (!table_contains(d, (uintptr_t)src, bytes)) return OCU_ERR_INVALID_VALUE;
        return finish(d, q, d->hip->MemcpyAsync(dst, src, bytes, HIP_MEMCPY_DEVICE_TO_HOST, s));
    case OCU_COPY_DEVICE_TO_DEVICE:
        if (!table_contains(d, (uintptr_t)dst, bytes) || !table_contains(d, (uintptr_t)src, bytes)) {
            return OCU_ERR_INVALID_VALUE;
        }
        return finish(d, q, d->hip->MemcpyAsync(dst, src, bytes, HIP_MEMCPY_DEVICE_TO_DEVICE, s));
    }
    return OCU_ERR_INVALID_VALUE;
}

static ocu_status amd_mem_set(ocu_device *d, ocu_queue *q, void *dst, int value, size_t bytes)
{
    if (d == NULL || (q != NULL && q->dev != d)) return OCU_ERR_INVALID_HANDLE;
    if (bytes == 0) return OCU_OK;
    if (!table_contains(d, (uintptr_t)dst, bytes)) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    return finish(d, q, d->hip->MemsetAsync(dst, (unsigned char)value, bytes, stream_of(d, q)));
}

static ocu_status amd_mem_info(ocu_device *d, uint64_t *free_bytes, uint64_t *total_bytes)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (free_bytes == NULL || total_bytes == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    size_t f = 0, t = 0;
    st = map(d->hip->MemGetInfo(&f, &t));
    *free_bytes = f;
    *total_bytes = t;
    return st;
}

static ocu_status amd_host_alloc_pinned(ocu_device *d, size_t bytes, void **ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    if (bytes == 0) {
        *ptr = NULL;
        return OCU_OK;
    }
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    return map(d->hip->HostMalloc(ptr, bytes, HIP_HOST_MALLOC_PORTABLE));
}

static ocu_status amd_host_free_pinned(ocu_device *d, void *ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_OK;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    st = map(d->hip->DeviceSynchronize());
    ocu_status f = map(d->hip->HostFree(ptr));
    return st != OCU_OK ? st : f;
}

/* ---- queues -------------------------------------------------------------- */

static ocu_status amd_queue_create(ocu_device *d, ocu_queue **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    ocu_queue *q = calloc(1, sizeof(*q));
    if (q == NULL) return OCU_ERR_OUT_OF_MEMORY;
    q->dev = d;
    st = map(d->hip->StreamCreateWithFlags(&q->stream, HIP_STREAM_NON_BLOCKING));
    if (st != OCU_OK) {
        free(q);
        return st;
    }
    *out = q;
    return OCU_OK;
}

static ocu_status amd_queue_destroy(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(q->dev);
    if (st == OCU_OK) st = map(q->dev->hip->StreamSynchronize(q->stream));
    q->dev->hip->StreamDestroy(q->stream);
    free(q);
    return st;
}

static ocu_status amd_queue_sync(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(q->dev);
    if (st != OCU_OK) return st;
    return map(q->dev->hip->StreamSynchronize(q->stream));
}

static ocu_status amd_queue_query(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(q->dev);
    if (st != OCU_OK) return st;
    return map(q->dev->hip->StreamQuery(q->stream));
}

/* ---- code ---------------------------------------------------------------- */

static ocu_status amd_module_load(ocu_device *d, const void *image, size_t bytes, ocu_module **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (image == NULL || out == NULL || bytes < 64) return OCU_ERR_INVALID_VALUE;
    if (memcmp(image, "\x7f" "ELF", 4) != 0) return OCU_ERR_INVALID_VALUE; /* an amdgcn code object */
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    ocu_module *m = calloc(1, sizeof(*m));
    void *copy = malloc(bytes);
    if (m == NULL || copy == NULL) {
        free(m);
        free(copy);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    memcpy(copy, image, bytes);
    st = map(d->hip->ModuleLoadData(&m->mod, copy));
    if (st != OCU_OK) {
        if (getenv("OCU_AMD_VERBOSE") != NULL) {
            fprintf(stderr, "ocu amd: hipModuleLoadData failed (%s): built for another GPU?\n", ocu_status_name(st));
        }
        free(copy);
        free(m);
        return st;
    }
    m->dev = d;
    m->image = copy;
    m->image_bytes = bytes;
    pthread_mutex_init(&m->lock, NULL);
    *out = m;
    return OCU_OK;
}

static ocu_status amd_module_unload(ocu_module *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(m->dev);
    if (st == OCU_OK) st = map(m->dev->hip->DeviceSynchronize());
    ocu_status u = map(m->dev->hip->ModuleUnload(m->mod));
    for (struct ocu_kernel *k = m->kernels; k != NULL;) {
        struct ocu_kernel *next = k->next;
        free(k->name);
        free(k->param_sizes);
        free(k);
        k = next;
    }
    free(m->image);
    pthread_mutex_destroy(&m->lock);
    free(m);
    return st != OCU_OK ? st : u;
}

static void read_params(struct ocu_kernel *k, const ocu_module *m)
{
    k->param_count = -1;
    size_t *sizes = calloc(AMD_MAX_PARAMS, sizeof(size_t));
    if (sizes == NULL) return;
    int n = amd_kernel_params(m->image, m->image_bytes, k->name, sizes, AMD_MAX_PARAMS);
    if (n < 0 || n > AMD_MAX_PARAMS) {
        free(sizes);
        return;
    }
    k->param_sizes = sizes;
    k->param_count = n;
}

static ocu_status amd_module_get_kernel(ocu_module *m, const char *name, ocu_kernel **out)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (name == NULL || out == NULL) return OCU_ERR_INVALID_VALUE;
    pthread_mutex_lock(&m->lock);
    for (struct ocu_kernel *k = m->kernels; k != NULL; k = k->next) {
        if (strcmp(k->name, name) == 0) {
            *out = k;
            pthread_mutex_unlock(&m->lock);
            return OCU_OK;
        }
    }
    ocu_status st = enter(m->dev);
    hip_function fn = NULL;
    if (st == OCU_OK) st = map(m->dev->hip->ModuleGetFunction(&fn, m->mod, name));
    if (st == OCU_OK) {
        struct ocu_kernel *k = calloc(1, sizeof(*k));
        char *copy = dup_string(name);
        if (k == NULL || copy == NULL) {
            free(k);
            free(copy);
            st = OCU_ERR_OUT_OF_MEMORY;
        } else {
            k->module = m;
            k->fn = fn;
            k->name = copy;
            read_params(k, m);
            k->next = m->kernels;
            m->kernels = k;
            *out = k;
        }
    }
    pthread_mutex_unlock(&m->lock);
    return st;
}

/* ---- execution ----------------------------------------------------------- */

static ocu_status amd_kernel_launch(ocu_queue *q, ocu_kernel *k, const ocu_launch_params *p)
{
    if (k == NULL) return OCU_ERR_INVALID_HANDLE;
    if (p == NULL || (p->num_args > 0 && (p->args == NULL || p->arg_sizes == NULL))) {
        return OCU_ERR_INVALID_VALUE;
    }
    ocu_device *d = k->module->dev;
    if (q != NULL && q->dev != d) return OCU_ERR_INVALID_HANDLE;
    size_t total = 0;
    for (uint32_t i = 0; i < p->num_args; ++i) {
        if (p->args[i] == NULL || p->arg_sizes[i] > OCU_MAX_ARG_BYTES - total) return OCU_ERR_INVALID_VALUE;
        total += p->arg_sizes[i];
    }
    if (k->param_count >= 0 && (uint32_t)k->param_count != p->num_args) return OCU_ERR_INVALID_VALUE;
    for (int i = 0; i < 3; ++i) {
        if (p->grid[i] == 0 || p->block[i] == 0) return OCU_ERR_INVALID_VALUE;
    }
    if (p->shared_memory_bytes > UINT32_MAX) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    /* The runtime copies the argument values before returning (observed:
       backends/amd/README.md section 7), as the interface requires. */
    return finish(d, q, d->hip->ModuleLaunchKernel(k->fn, p->grid[0], p->grid[1], p->grid[2], p->block[0],
                                                   p->block[1], p->block[2], (unsigned)p->shared_memory_bytes,
                                                   stream_of(d, q), p->args, NULL));
}

/* ---- markers ------------------------------------------------------------- */

static ocu_status amd_marker_create(ocu_device *d, ocu_marker **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    ocu_marker *m = calloc(1, sizeof(*m));
    if (m == NULL) return OCU_ERR_OUT_OF_MEMORY;
    m->dev = d;
    st = map(d->hip->EventCreateWithFlags(&m->event, HIP_EVENT_DEFAULT));
    if (st != OCU_OK) {
        free(m);
        return st;
    }
    *out = m;
    return OCU_OK;
}

static ocu_status amd_marker_destroy(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(m->dev);
    if (st == OCU_OK) st = map(m->dev->hip->EventDestroy(m->event));
    free(m);
    return st;
}

static ocu_status amd_marker_record(ocu_marker *m, ocu_queue *q)
{
    if (m == NULL || (q != NULL && q->dev != m->dev)) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(m->dev);
    if (st != OCU_OK) return st;
    st = map(m->dev->hip->EventRecord(m->event, stream_of(m->dev, q)));
    if (st == OCU_OK) m->recorded = true;
    return st;
}

/* A never-recorded marker is complete (the interface rule; the runtime
   agrees: query and synchronize return success, observed). */
static ocu_status amd_marker_wait(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (!m->recorded) return OCU_OK;
    ocu_status st = enter(m->dev);
    if (st != OCU_OK) return st;
    return map(m->dev->hip->EventSynchronize(m->event));
}

static ocu_status amd_marker_query(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (!m->recorded) return OCU_OK;
    ocu_status st = enter(m->dev);
    if (st != OCU_OK) return st;
    return map(m->dev->hip->EventQuery(m->event));
}

static ocu_status amd_marker_elapsed_ms(ocu_marker *a, ocu_marker *b, float *ms)
{
    if (a == NULL || b == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ms == NULL) return OCU_ERR_INVALID_VALUE;
    if (!a->recorded || !b->recorded) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(a->dev);
    if (st != OCU_OK) return st;
    return map(a->dev->hip->EventElapsedTime(ms, a->event, b->event));
}

static ocu_status amd_module_get_global(ocu_module *m, const char *name, void **ptr, size_t *bytes)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (name == NULL || ptr == NULL || bytes == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(m->dev);
    if (st != OCU_OK) return st;
    void *p = NULL;
    size_t n = 0;
    st = map(m->dev->hip->ModuleGetGlobal(&p, &n, m->mod, name));
    if (st != OCU_OK) return st;
    /* Make the variable a valid copy target for mem_copy / mem_set. */
    if (!table_contains(m->dev, (uintptr_t)p, n) && !table_add(m->dev, (uintptr_t)p, n)) return OCU_ERR_OUT_OF_MEMORY;
    *ptr = p;
    *bytes = n;
    return OCU_OK;
}

static ocu_status amd_kernel_arg_sizes(ocu_kernel *k, uint32_t *count, size_t *sizes, uint32_t max)
{
    if (k == NULL) return OCU_ERR_INVALID_HANDLE;
    if (count == NULL || (max > 0 && sizes == NULL)) return OCU_ERR_INVALID_VALUE;
    if (k->param_count < 0) return OCU_ERR_NOT_SUPPORTED;
    *count = (uint32_t)k->param_count;
    for (uint32_t i = 0; i < *count && i < max; ++i) sizes[i] = k->param_sizes[i];
    return OCU_OK;
}

static const void *amd_get_provider(const char *library)
{
    (void)library; /* rocBLAS behind the BLAS provider: README section 4, not done yet */
    return NULL;
}

static const ocu_backend g_amd = {
    .abi = OCU_BACKEND_ABI,
    .name = "amd",
    .dev_count = amd_dev_count,
    .dev_info = amd_dev_info,
    .dev_open = amd_dev_open,
    .dev_close = amd_dev_close,
    .mem_alloc = amd_mem_alloc,
    .mem_free = amd_mem_free,
    .mem_copy = amd_mem_copy,
    .mem_set = amd_mem_set,
    .mem_info = amd_mem_info,
    .host_alloc_pinned = amd_host_alloc_pinned,
    .host_free_pinned = amd_host_free_pinned,
    .queue_create = amd_queue_create,
    .queue_destroy = amd_queue_destroy,
    .queue_sync = amd_queue_sync,
    .queue_query = amd_queue_query,
    .module_load = amd_module_load,
    .module_unload = amd_module_unload,
    .module_get_kernel = amd_module_get_kernel,
    .kernel_launch = amd_kernel_launch,
    .marker_create = amd_marker_create,
    .marker_destroy = amd_marker_destroy,
    .marker_record = amd_marker_record,
    .marker_wait = amd_marker_wait,
    .marker_query = amd_marker_query,
    .marker_elapsed_ms = amd_marker_elapsed_ms,
    .get_provider = amd_get_provider,
    .module_get_global = amd_module_get_global,
    .kernel_arg_sizes = amd_kernel_arg_sizes,
    .mem_alloc_shared = amd_mem_alloc_shared,
    .host_register = amd_host_register,
    .host_unregister = amd_host_unregister,
};

const ocu_backend *ocu_backend_amd(void) { return &g_amd; }

/*
 * nv_backend.c -- the device backend interface on NVIDIA GPUs, through the
 * installed driver's public API (see nv_driver.h).
 *
 * Device code images are PTX text (from the compiler glue, Clang's NVPTX
 * target); the driver compiles PTX for the installed GPU when a module is
 * loaded. Every call makes the device's primary context current on the
 * calling thread first.
 */
#include "ocu_backend.h"
#include "ocu_backend_nvidia.h"
#include "nv_driver.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- objects ----------------------------------------------------------- */

typedef struct alloc_range {
    nv_dptr base;
    size_t  size;
} alloc_range;

struct ocu_device {
    const nv_api   *nv;
    nv_device       dev;
    nv_ctx          ctx;
    nv_stream       sync_stream; /* for queue == NULL */
    pthread_mutex_t lock;        /* guards the allocation table */
    alloc_range    *allocs;
    size_t          alloc_count, alloc_cap;
};

struct ocu_queue {
    ocu_device *dev;
    nv_stream   stream;
};

#define NV_MAX_PARAMS 256

struct ocu_kernel {
    struct ocu_module *module;
    nv_function        fn;
    char              *name;
    int32_t            param_count; /* -1: not found in the PTX */
    size_t            *param_sizes;
    struct ocu_kernel *next;
};

struct ocu_module {
    ocu_device        *dev;
    nv_module          mod;
    char              *ptx;  /* kept for the parameter lists */
    pthread_mutex_t    lock; /* guards the kernel list */
    struct ocu_kernel *kernels;
};

struct ocu_marker {
    ocu_device *dev;
    nv_event    event;
    bool        recorded;
};

static ocu_status map(nv_result r)
{
    switch (r) {
    case NV_SUCCESS:                       return OCU_OK;
    case NV_ERROR_INVALID_VALUE:           return OCU_ERR_INVALID_VALUE;
    case NV_ERROR_OUT_OF_MEMORY:           return OCU_ERR_OUT_OF_MEMORY;
    case NV_ERROR_NO_DEVICE:
    case NV_ERROR_INVALID_DEVICE:          return OCU_ERR_INVALID_DEVICE;
    case NV_ERROR_INVALID_IMAGE:
    case NV_ERROR_INVALID_PTX:             return OCU_ERR_INVALID_VALUE;
    case NV_ERROR_NO_BINARY_FOR_GPU:       return OCU_ERR_NOT_SUPPORTED;
    case NV_ERROR_INVALID_HANDLE:
    case NV_ERROR_INVALID_CONTEXT:         return OCU_ERR_INVALID_HANDLE;
    case NV_ERROR_NOT_FOUND:               return OCU_ERR_NOT_FOUND;
    case NV_ERROR_NOT_READY:               return OCU_ERR_NOT_READY;
    case NV_ERROR_LAUNCH_OUT_OF_RESOURCES: return OCU_ERR_INVALID_VALUE; /* configuration */
    case NV_ERROR_NOT_SUPPORTED:           return OCU_ERR_NOT_SUPPORTED;
    default:
        if (r >= 700 && r < 800) return OCU_ERR_LAUNCH_FAILED; /* faults while running */
        return OCU_ERR_UNKNOWN;
    }
}

static const nv_api *api(void) { return nv_load(); }

static char *dup_string(const char *s)
{
    size_t n = strlen(s) + 1;
    char *c = malloc(n);
    if (c != NULL) memcpy(c, s, n);
    return c;
}

static ocu_status enter(ocu_device *d)
{
    return map(d->nv->CtxSetCurrent(d->ctx));
}

/* ---- allocation table (device pointers must be checkable) ------------- */

static bool table_add(ocu_device *d, nv_dptr base, size_t size)
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

/* Removes the allocation starting exactly at base; false if none. */
static bool table_remove(ocu_device *d, nv_dptr base)
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

/* True when [p, p + n) lies inside one allocation. */
static bool table_contains(ocu_device *d, nv_dptr p, size_t n)
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

static ocu_status nv_dev_count(uint32_t *count)
{
    if (count == NULL) return OCU_ERR_INVALID_VALUE;
    *count = 0;
    const nv_api *a = api();
    if (a == NULL) return OCU_OK; /* no driver: no devices */
    int n = 0;
    if (a->DeviceGetCount(&n) != NV_SUCCESS || n < 0) return OCU_OK;
    *count = (uint32_t)n;
    return OCU_OK;
}

static ocu_status nv_dev_info(uint32_t index, ocu_dev_info *info)
{
    if (info == NULL) return OCU_ERR_INVALID_VALUE;
    const nv_api *a = api();
    uint32_t n = 0;
    nv_dev_count(&n);
    if (a == NULL || index >= n) return OCU_ERR_INVALID_DEVICE;
    nv_device dev;
    ocu_status st = map(a->DeviceGet(&dev, (int)index));
    if (st != OCU_OK) return st;
    memset(info, 0, sizeof(*info));
    if (a->DeviceGetName(info->name, (int)sizeof(info->name), dev) != NV_SUCCESS) {
        snprintf(info->name, sizeof(info->name), "NVIDIA GPU %u", index);
    }
    info->name[sizeof(info->name) - 1] = '\0';
    size_t total = 0;
    (void)a->DeviceTotalMem(&total, dev);
    info->total_memory_bytes = total;
    int v = 0;
#define NV_ATTR(field, attr) (v = 0, a->DeviceGetAttribute(&v, attr, dev), info->field = (uint32_t)v)
    NV_ATTR(warp_size, NV_ATTR_WARP_SIZE);
    NV_ATTR(max_threads_per_block, NV_ATTR_MAX_THREADS_PER_BLOCK);
    NV_ATTR(max_block_dim[0], NV_ATTR_MAX_BLOCK_DIM_X);
    NV_ATTR(max_block_dim[1], NV_ATTR_MAX_BLOCK_DIM_Y);
    NV_ATTR(max_block_dim[2], NV_ATTR_MAX_BLOCK_DIM_Z);
    NV_ATTR(max_grid_dim[0], NV_ATTR_MAX_GRID_DIM_X);
    NV_ATTR(max_grid_dim[1], NV_ATTR_MAX_GRID_DIM_Y);
    NV_ATTR(max_grid_dim[2], NV_ATTR_MAX_GRID_DIM_Z);
#undef NV_ATTR
    v = 0;
    a->DeviceGetAttribute(&v, NV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK, dev);
    info->max_shared_memory_per_block = (uint64_t)v;
    info->supports_unified_memory = 0; /* device memory is separate */
    return OCU_OK;
}

static ocu_status nv_dev_open(uint32_t index, ocu_device **out)
{
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    const nv_api *a = api();
    uint32_t n = 0;
    nv_dev_count(&n);
    if (a == NULL || index >= n) return OCU_ERR_INVALID_DEVICE;
    ocu_device *d = calloc(1, sizeof(*d));
    if (d == NULL) return OCU_ERR_OUT_OF_MEMORY;
    d->nv = a;
    pthread_mutex_init(&d->lock, NULL);
    ocu_status st = map(a->DeviceGet(&d->dev, (int)index));
    if (st == OCU_OK) st = map(a->DevicePrimaryCtxRetain(&d->ctx, d->dev));
    if (st == OCU_OK) st = enter(d);
    if (st == OCU_OK) st = map(a->StreamCreate(&d->sync_stream, NV_STREAM_NON_BLOCKING));
    if (st != OCU_OK) {
        if (d->ctx != NULL) a->DevicePrimaryCtxRelease(d->dev);
        pthread_mutex_destroy(&d->lock);
        free(d);
        return st;
    }
    *out = d;
    return OCU_OK;
}

static ocu_status nv_dev_close(ocu_device *d)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(d);
    if (st == OCU_OK) st = map(d->nv->CtxSynchronize());
    d->nv->StreamDestroy(d->sync_stream);
    d->nv->DevicePrimaryCtxRelease(d->dev);
    pthread_mutex_destroy(&d->lock);
    free(d->allocs);
    free(d);
    return st;
}

/* ---- memory -------------------------------------------------------------- */

static ocu_status nv_mem_alloc(ocu_device *d, size_t bytes, void **ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    if (bytes == 0) {
        *ptr = NULL;
        return OCU_OK;
    }
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    nv_dptr p = 0;
    st = map(d->nv->MemAlloc(&p, bytes));
    if (st != OCU_OK) return st;
    if (!table_add(d, p, bytes)) {
        d->nv->MemFree(p);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    *ptr = (void *)(uintptr_t)p;
    return OCU_OK;
}

static ocu_status nv_mem_free(ocu_device *d, void *ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_OK;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    nv_dptr p = (nv_dptr)(uintptr_t)ptr;
    if (!table_contains(d, p, 0)) return OCU_ERR_INVALID_VALUE;
    st = map(d->nv->CtxSynchronize()); /* queued work may still use it */
    if (!table_remove(d, p)) return OCU_ERR_INVALID_VALUE; /* not the start */
    ocu_status f = map(d->nv->MemFree(p));
    return st != OCU_OK ? st : f;
}

static ocu_status finish(ocu_device *d, ocu_queue *q, nv_result r)
{
    ocu_status st = map(r);
    if (st == OCU_OK && q == NULL) st = map(d->nv->StreamSynchronize(d->sync_stream));
    return st;
}

static nv_stream stream_of(ocu_device *d, ocu_queue *q) { return q ? q->stream : d->sync_stream; }

static ocu_status nv_mem_copy(ocu_device *d, ocu_queue *q, void *dst, const void *src,
                              size_t bytes, ocu_copy_dir dir)
{
    if (d == NULL || (q != NULL && q->dev != d)) return OCU_ERR_INVALID_HANDLE;
    if (bytes == 0) return OCU_OK;
    if (dst == NULL || src == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    nv_stream s = stream_of(d, q);
    nv_dptr ddst = (nv_dptr)(uintptr_t)dst, dsrc = (nv_dptr)(uintptr_t)src;
    switch (dir) {
    case OCU_COPY_HOST_TO_DEVICE:
        if (!table_contains(d, ddst, bytes)) return OCU_ERR_INVALID_VALUE;
        return finish(d, q, d->nv->MemcpyHtoDAsync(ddst, src, bytes, s));
    case OCU_COPY_DEVICE_TO_HOST:
        if (!table_contains(d, dsrc, bytes)) return OCU_ERR_INVALID_VALUE;
        return finish(d, q, d->nv->MemcpyDtoHAsync(dst, dsrc, bytes, s));
    case OCU_COPY_DEVICE_TO_DEVICE:
        if (!table_contains(d, ddst, bytes) || !table_contains(d, dsrc, bytes)) {
            return OCU_ERR_INVALID_VALUE;
        }
        return finish(d, q, d->nv->MemcpyDtoDAsync(ddst, dsrc, bytes, s));
    }
    return OCU_ERR_INVALID_VALUE;
}

static ocu_status nv_mem_set(ocu_device *d, ocu_queue *q, void *dst, int value, size_t bytes)
{
    if (d == NULL || (q != NULL && q->dev != d)) return OCU_ERR_INVALID_HANDLE;
    if (bytes == 0) return OCU_OK;
    nv_dptr p = (nv_dptr)(uintptr_t)dst;
    if (!table_contains(d, p, bytes)) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    return finish(d, q, d->nv->MemsetD8Async(p, (unsigned char)value, bytes, stream_of(d, q)));
}

static ocu_status nv_mem_info(ocu_device *d, uint64_t *free_bytes, uint64_t *total_bytes)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (free_bytes == NULL || total_bytes == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    size_t f = 0, t = 0;
    st = map(d->nv->MemGetInfo(&f, &t));
    *free_bytes = f;
    *total_bytes = t;
    return st;
}

static ocu_status nv_host_alloc_pinned(ocu_device *d, size_t bytes, void **ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    if (bytes == 0) {
        *ptr = NULL;
        return OCU_OK;
    }
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    return map(d->nv->MemHostAlloc(ptr, bytes, NV_MEMHOSTALLOC_PORTABLE));
}

static ocu_status nv_host_free_pinned(ocu_device *d, void *ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_OK;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    st = map(d->nv->CtxSynchronize());
    ocu_status f = map(d->nv->MemFreeHost(ptr));
    return st != OCU_OK ? st : f;
}

/* ---- queues -------------------------------------------------------------- */

static ocu_status nv_queue_create(ocu_device *d, ocu_queue **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    ocu_queue *q = calloc(1, sizeof(*q));
    if (q == NULL) return OCU_ERR_OUT_OF_MEMORY;
    q->dev = d;
    st = map(d->nv->StreamCreate(&q->stream, NV_STREAM_NON_BLOCKING));
    if (st != OCU_OK) {
        free(q);
        return st;
    }
    *out = q;
    return OCU_OK;
}

static ocu_status nv_queue_destroy(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(q->dev);
    if (st == OCU_OK) st = map(q->dev->nv->StreamSynchronize(q->stream));
    q->dev->nv->StreamDestroy(q->stream);
    free(q);
    return st;
}

static ocu_status nv_queue_sync(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(q->dev);
    if (st != OCU_OK) return st;
    return map(q->dev->nv->StreamSynchronize(q->stream));
}

static ocu_status nv_queue_query(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(q->dev);
    if (st != OCU_OK) return st;
    return map(q->dev->nv->StreamQuery(q->stream));
}

/* ---- code ---------------------------------------------------------------- */

static ocu_status nv_module_load(ocu_device *d, const void *image, size_t bytes,
                                 ocu_module **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (image == NULL || out == NULL || bytes == 0) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    /* The loader wants NUL-terminated PTX text. */
    char *text = malloc(bytes + 1);
    if (text == NULL) return OCU_ERR_OUT_OF_MEMORY;
    memcpy(text, image, bytes);
    text[bytes] = '\0';
    char log[4096];
    log[0] = '\0';
    int opts[2] = {NV_JIT_ERROR_LOG_BUFFER, NV_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *vals[2] = {log, (void *)(uintptr_t)sizeof(log)};
    ocu_module *m = calloc(1, sizeof(*m));
    if (m == NULL) {
        free(text);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    st = map(d->nv->ModuleLoadDataEx(&m->mod, text, 2, opts, vals));
    if (st == OCU_OK) {
        m->ptx = text;
    } else {
        free(text);
    }
    if (st != OCU_OK) {
        if (log[0] != '\0' && getenv("OCU_NV_VERBOSE") != NULL) {
            fprintf(stderr, "ocu nvidia: PTX load failed: %s\n", log);
        }
        free(m);
        return st;
    }
    m->dev = d;
    pthread_mutex_init(&m->lock, NULL);
    *out = m;
    return OCU_OK;
}

static ocu_status nv_module_unload(ocu_module *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(m->dev);
    if (st == OCU_OK) st = map(m->dev->nv->CtxSynchronize());
    ocu_status u = map(m->dev->nv->ModuleUnload(m->mod));
    for (struct ocu_kernel *k = m->kernels; k != NULL;) {
        struct ocu_kernel *next = k->next;
        free(k->name);
        free(k->param_sizes);
        free(k);
        k = next;
    }
    free(m->ptx);
    pthread_mutex_destroy(&m->lock);
    free(m);
    return st != OCU_OK ? st : u;
}

/* Size in bytes of one PTX parameter declaration such as
   ".param .u64 name" or ".param .align 8 .b8 name[24]"; 0 if unknown. */
static size_t param_size(const char *decl, const char *end)
{
    size_t elem = 0;
    const char *p = decl;
    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
        const char *t = p;
        while (p < end && !(*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
        size_t n = (size_t)(p - t);
        if (n == 0) break;
        if (n >= 3 && t[0] == '.' && (t[1] == 'u' || t[1] == 's' || t[1] == 'b' || t[1] == 'f')) {
            int bits = atoi(t + 2);
            if (bits == 8 || bits == 16 || bits == 32 || bits == 64) elem = (size_t)bits / 8;
        }
        const char *br = memchr(t, '[', n);
        if (br != NULL && elem != 0) {
            return elem * (size_t)strtoul(br + 1, NULL, 10);
        }
    }
    return elem;
}

/* Fills k->param_count / param_sizes from ".entry <name>( ... )". */
static void parse_params(struct ocu_kernel *k, const char *ptx)
{
    k->param_count = -1;
    if (ptx == NULL) return;
    size_t len = strlen(k->name);
    for (const char *p = strstr(ptx, ".entry "); p != NULL; p = strstr(p + 1, ".entry ")) {
        const char *n = p + 7;
        while (*n == ' ') ++n;
        if (strncmp(n, k->name, len) != 0 || (n[len] != '(' && n[len] != ' ' && n[len] != '\n')) {
            continue;
        }
        const char *open = strchr(n + len, '(');
        const char *close = open ? strchr(open, ')') : NULL;
        const char *brace = strchr(n + len, '{');
        if (open == NULL || (brace != NULL && open > brace)) {
            k->param_count = 0; /* no parameter list */
            return;
        }
        if (close == NULL) return;
        size_t *sizes = calloc(NV_MAX_PARAMS, sizeof(size_t));
        if (sizes == NULL) return;
        int32_t count = 0;
        for (const char *q = strstr(open, ".param"); q != NULL && q < close && count < NV_MAX_PARAMS;
             q = strstr(q + 6, ".param")) {
            const char *stop = strchr(q, ',');
            if (stop == NULL || stop > close) stop = close;
            size_t sz = param_size(q + 6, stop);
            if (sz == 0) {
                free(sizes);
                return;
            }
            sizes[count++] = sz;
        }
        k->param_sizes = sizes;
        k->param_count = count;
        return;
    }
}

static ocu_status nv_module_get_kernel(ocu_module *m, const char *name, ocu_kernel **out)
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
    nv_function fn = NULL;
    if (st == OCU_OK) st = map(m->dev->nv->ModuleGetFunction(&fn, m->mod, name));
    struct ocu_kernel *k = NULL;
    if (st == OCU_OK) {
        k = calloc(1, sizeof(*k));
        char *copy = dup_string(name);
        if (k == NULL || copy == NULL) {
            free(k);
            free(copy);
            st = OCU_ERR_OUT_OF_MEMORY;
        } else {
            k->module = m;
            k->fn = fn;
            k->name = copy;
            parse_params(k, m->ptx);
            k->next = m->kernels;
            m->kernels = k;
            *out = k;
        }
    }
    pthread_mutex_unlock(&m->lock);
    return st;
}

/* ---- execution ----------------------------------------------------------- */

static ocu_status nv_kernel_launch(ocu_queue *q, ocu_kernel *k, const ocu_launch_params *p)
{
    if (k == NULL) return OCU_ERR_INVALID_HANDLE;
    if (p == NULL || (p->num_args > 0 && (p->args == NULL || p->arg_sizes == NULL))) {
        return OCU_ERR_INVALID_VALUE;
    }
    ocu_device *d = k->module->dev;
    if (q != NULL && q->dev != d) return OCU_ERR_INVALID_HANDLE;
    size_t total = 0;
    for (uint32_t i = 0; i < p->num_args; ++i) {
        if (p->args[i] == NULL || p->arg_sizes[i] > OCU_MAX_ARG_BYTES - total) {
            return OCU_ERR_INVALID_VALUE;
        }
        total += p->arg_sizes[i];
    }
    for (int i = 0; i < 3; ++i) {
        if (p->grid[i] == 0 || p->block[i] == 0) return OCU_ERR_INVALID_VALUE;
    }
    if (p->shared_memory_bytes > UINT32_MAX) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    /* The driver copies the argument values before returning. */
    return finish(d, q, d->nv->LaunchKernel(k->fn, p->grid[0], p->grid[1], p->grid[2],
                                            p->block[0], p->block[1], p->block[2],
                                            (unsigned)p->shared_memory_bytes,
                                            stream_of(d, q), p->args, NULL));
}

/* ---- markers ------------------------------------------------------------- */

static ocu_status nv_marker_create(ocu_device *d, ocu_marker **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(d);
    if (st != OCU_OK) return st;
    ocu_marker *m = calloc(1, sizeof(*m));
    if (m == NULL) return OCU_ERR_OUT_OF_MEMORY;
    m->dev = d;
    st = map(d->nv->EventCreate(&m->event, NV_EVENT_DEFAULT));
    if (st != OCU_OK) {
        free(m);
        return st;
    }
    *out = m;
    return OCU_OK;
}

static ocu_status nv_marker_destroy(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(m->dev);
    if (st == OCU_OK) st = map(m->dev->nv->EventDestroy(m->event));
    free(m);
    return st;
}

static ocu_status nv_marker_record(ocu_marker *m, ocu_queue *q)
{
    if (m == NULL || (q != NULL && q->dev != m->dev)) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(m->dev);
    if (st != OCU_OK) return st;
    st = map(m->dev->nv->EventRecord(m->event, stream_of(m->dev, q)));
    if (st == OCU_OK) m->recorded = true;
    return st;
}

static ocu_status nv_marker_wait(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (!m->recorded) return OCU_OK;
    ocu_status st = enter(m->dev);
    if (st != OCU_OK) return st;
    return map(m->dev->nv->EventSynchronize(m->event));
}

static ocu_status nv_marker_query(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (!m->recorded) return OCU_OK;
    ocu_status st = enter(m->dev);
    if (st != OCU_OK) return st;
    return map(m->dev->nv->EventQuery(m->event));
}

static ocu_status nv_marker_elapsed_ms(ocu_marker *a, ocu_marker *b, float *ms)
{
    if (a == NULL || b == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ms == NULL) return OCU_ERR_INVALID_VALUE;
    if (!a->recorded || !b->recorded) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = enter(a->dev);
    if (st != OCU_OK) return st;
    return map(a->dev->nv->EventElapsedTime(ms, a->event, b->event));
}

static ocu_status nv_module_get_global(ocu_module *m, const char *name, void **ptr,
                                       size_t *bytes)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (name == NULL || ptr == NULL || bytes == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = enter(m->dev);
    if (st != OCU_OK) return st;
    nv_dptr p = 0;
    size_t n = 0;
    st = map(m->dev->nv->ModuleGetGlobal(&p, &n, m->mod, name));
    if (st != OCU_OK) return st;
    /* Make the variable a valid copy target for mem_copy / mem_set. */
    if (!table_contains(m->dev, p, n) && !table_add(m->dev, p, n)) return OCU_ERR_OUT_OF_MEMORY;
    *ptr = (void *)(uintptr_t)p;
    *bytes = n;
    return OCU_OK;
}

static ocu_status nv_kernel_arg_sizes(ocu_kernel *k, uint32_t *count, size_t *sizes, uint32_t max)
{
    if (k == NULL) return OCU_ERR_INVALID_HANDLE;
    if (count == NULL || (max > 0 && sizes == NULL)) return OCU_ERR_INVALID_VALUE;
    if (k->param_count < 0) return OCU_ERR_NOT_SUPPORTED;
    *count = (uint32_t)k->param_count;
    for (uint32_t i = 0; i < *count && i < max; ++i) sizes[i] = k->param_sizes[i];
    return OCU_OK;
}

static const void *nv_get_provider(const char *library)
{
    (void)library;
    return NULL;
}

static const ocu_backend g_nvidia = {
    .abi = OCU_BACKEND_ABI,
    .name = "nvidia",
    .dev_count = nv_dev_count,
    .dev_info = nv_dev_info,
    .dev_open = nv_dev_open,
    .dev_close = nv_dev_close,
    .mem_alloc = nv_mem_alloc,
    .mem_free = nv_mem_free,
    .mem_copy = nv_mem_copy,
    .mem_set = nv_mem_set,
    .mem_info = nv_mem_info,
    .host_alloc_pinned = nv_host_alloc_pinned,
    .host_free_pinned = nv_host_free_pinned,
    .queue_create = nv_queue_create,
    .queue_destroy = nv_queue_destroy,
    .queue_sync = nv_queue_sync,
    .queue_query = nv_queue_query,
    .module_load = nv_module_load,
    .module_unload = nv_module_unload,
    .module_get_kernel = nv_module_get_kernel,
    .kernel_launch = nv_kernel_launch,
    .marker_create = nv_marker_create,
    .marker_destroy = nv_marker_destroy,
    .marker_record = nv_marker_record,
    .marker_wait = nv_marker_wait,
    .marker_query = nv_marker_query,
    .marker_elapsed_ms = nv_marker_elapsed_ms,
    .get_provider = nv_get_provider,
    .module_get_global = nv_module_get_global,
    .kernel_arg_sizes = nv_kernel_arg_sizes,
};

const ocu_backend *ocu_backend_nvidia(void) { return &g_nvidia; }

/*
 * mt_backend.m -- the device backend interface on Apple GPUs through the
 * public Metal API (see README.md). Manual reference counting; every entry
 * point runs inside an autorelease pool.
 *
 * Device code images are Metal Shading Language source with a small
 * header (see mt_module_load). Kernels read their arguments from buffer 0
 * at the offsets in buffer 1 (README.md, section 5), and the module's
 * __device__ / __constant__ variables from buffer 2.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ocu_backend.h"
#include "ocu_backend_metal.h"

#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- objects ---------------------------------------------------------------- */

typedef struct mt_alloc {
    uint64_t        gpu;     /* address device code uses */
    size_t          size;
    char           *cpu;     /* the same memory, for the host */
    id<MTLBuffer>   buffer;  /* retained */
} mt_alloc;

struct ocu_device {
    id<MTLDevice>       dev;    /* retained */
    struct ocu_queue   *sync_q; /* for queue == NULL */
    pthread_mutex_t     lock;   /* guards the allocation table */
    mt_alloc           *allocs;
    size_t              count, cap;
};

struct ocu_queue {
    ocu_device           *dev;
    id<MTLCommandQueue>   queue;  /* retained */
    pthread_mutex_t       lock;
    id<MTLCommandBuffer>  last;   /* retained; the newest committed */
    ocu_status            sticky; /* a failure seen by sync/query */
};

struct ocu_kernel {
    struct ocu_module            *module;
    id<MTLComputePipelineState>   pipeline; /* retained */
    char                         *name;
    uint32_t                      nparams;
    size_t                       *param_sizes;
    struct ocu_kernel            *next;
};

struct ocu_module {
    ocu_device        *dev;
    id<MTLLibrary>     library;   /* retained */
    char              *meta;      /* the image's header lines */
    pthread_mutex_t    lock;
    struct ocu_kernel *kernels;
    void              *globals;   /* device address of the variables, or NULL */
    size_t             globals_size;
    id<MTLBuffer>      globals_buf; /* retained; nil without variables */
};

struct ocu_marker {
    ocu_device           *dev;
    pthread_mutex_t       lock;
    id<MTLCommandBuffer>  cb;      /* retained; nil when never recorded */
    _Atomic uint64_t      done_ns; /* host time at completion, 0 while pending */
};

static uint64_t now_ns(void)
{
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
}

/* ---- allocation table ---------------------------------------------------------- */

static bool table_add(ocu_device *d, mt_alloc a)
{
    pthread_mutex_lock(&d->lock);
    bool ok = true;
    if (d->count == d->cap) {
        size_t cap = d->cap ? d->cap * 2 : 64;
        mt_alloc *n = realloc(d->allocs, cap * sizeof(*n));
        if (n == NULL) {
            ok = false;
        } else {
            d->allocs = n;
            d->cap = cap;
        }
    }
    if (ok) d->allocs[d->count++] = a;
    pthread_mutex_unlock(&d->lock);
    return ok;
}

/* CPU address for [gpu, gpu + n) inside one allocation, or NULL. */
static char *table_cpu(ocu_device *d, uint64_t gpu, size_t n)
{
    char *cpu = NULL;
    pthread_mutex_lock(&d->lock);
    for (size_t i = 0; i < d->count; ++i) {
        const mt_alloc *a = &d->allocs[i];
        if (gpu >= a->gpu && gpu - a->gpu <= a->size && n <= a->size - (gpu - a->gpu)) {
            cpu = a->cpu + (gpu - a->gpu);
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
    return cpu;
}

/* The buffer (retained) of the allocation that starts at `gpu`, or nil. */
static id<MTLBuffer> table_buffer(ocu_device *d, uint64_t gpu)
{
    id<MTLBuffer> b = nil;
    pthread_mutex_lock(&d->lock);
    for (size_t i = 0; i < d->count; ++i) {
        if (d->allocs[i].gpu == gpu) {
            b = [d->allocs[i].buffer retain];
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
    return b;
}

/* ---- device ------------------------------------------------------------------------ */

static NSArray<id<MTLDevice>> *all_devices(void)
{
    static NSArray<id<MTLDevice>> *devices; /* +1 from MTLCopyAllDevices, kept */
    static dispatch_once_t once;
    dispatch_once(&once, ^{ devices = MTLCopyAllDevices(); });
    return devices;
}

static ocu_status mt_dev_count(uint32_t *count)
{
    if (count == NULL) return OCU_ERR_INVALID_VALUE;
    @autoreleasepool {
        if (@available(macOS 13.0, *)) {
            *count = (uint32_t)all_devices().count;
        } else {
            *count = 0; /* GPU addresses (Metal 3) are required */
        }
    }
    return OCU_OK;
}

static ocu_status mt_dev_info(uint32_t index, ocu_dev_info *info)
{
    if (info == NULL) return OCU_ERR_INVALID_VALUE;
    uint32_t n = 0;
    mt_dev_count(&n);
    if (index >= n) return OCU_ERR_INVALID_DEVICE;
    @autoreleasepool {
        id<MTLDevice> dev = all_devices()[index];
        memset(info, 0, sizeof(*info));
        const char *name = dev.name.UTF8String;
        strncpy(info->name, name ? name : "Apple GPU", sizeof(info->name) - 1);
        info->total_memory_bytes = dev.recommendedMaxWorkingSetSize;
        info->warp_size = 32; /* SIMD-group width of Apple GPUs */
        MTLSize t = dev.maxThreadsPerThreadgroup;
        info->max_threads_per_block = (uint32_t)(t.width < 1024 ? t.width : 1024);
        info->max_block_dim[0] = (uint32_t)t.width;
        info->max_block_dim[1] = (uint32_t)t.height;
        info->max_block_dim[2] = (uint32_t)t.depth;
        info->max_grid_dim[0] = 0x7fffffffu;
        info->max_grid_dim[1] = 65535u;
        info->max_grid_dim[2] = 65535u;
        info->max_shared_memory_per_block = dev.maxThreadgroupMemoryLength;
        info->supports_unified_memory = 0; /* CPU and GPU addresses differ */
    }
    return OCU_OK;
}

static ocu_status mt_queue_create(ocu_device *d, ocu_queue **out);
static ocu_status mt_queue_destroy(ocu_queue *q);
static ocu_status mt_queue_sync(ocu_queue *q);

static ocu_status mt_dev_open(uint32_t index, ocu_device **out)
{
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    uint32_t n = 0;
    mt_dev_count(&n);
    if (index >= n) return OCU_ERR_INVALID_DEVICE;
    ocu_device *d = calloc(1, sizeof(*d));
    if (d == NULL) return OCU_ERR_OUT_OF_MEMORY;
    @autoreleasepool {
        d->dev = [all_devices()[index] retain];
    }
    pthread_mutex_init(&d->lock, NULL);
    ocu_status st = mt_queue_create(d, &d->sync_q);
    if (st != OCU_OK) {
        [d->dev release];
        pthread_mutex_destroy(&d->lock);
        free(d);
        return st;
    }
    *out = d;
    return OCU_OK;
}

static ocu_status mt_dev_close(ocu_device *d)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = mt_queue_destroy(d->sync_q);
    for (size_t i = 0; i < d->count; ++i) [d->allocs[i].buffer release];
    free(d->allocs);
    [d->dev release];
    pthread_mutex_destroy(&d->lock);
    free(d);
    return st;
}

/* ---- queues ---------------------------------------------------------------------------- */

static ocu_status cb_status(id<MTLCommandBuffer> cb)
{
    if (cb == nil) return OCU_OK;
    return cb.status == MTLCommandBufferStatusError ? OCU_ERR_LAUNCH_FAILED : OCU_OK;
}

static ocu_status mt_queue_create(ocu_device *d, ocu_queue **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_queue *q = calloc(1, sizeof(*q));
    if (q == NULL) return OCU_ERR_OUT_OF_MEMORY;
    q->dev = d;
    q->queue = [d->dev newCommandQueue]; /* +1 */
    if (q->queue == nil) {
        free(q);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    pthread_mutex_init(&q->lock, NULL);
    *out = q;
    return OCU_OK;
}

/* The newest committed command buffer of q, retained (or nil). */
static id<MTLCommandBuffer> last_of(ocu_queue *q)
{
    pthread_mutex_lock(&q->lock);
    id<MTLCommandBuffer> cb = [q->last retain];
    pthread_mutex_unlock(&q->lock);
    return cb;
}

static void set_last(ocu_queue *q, id<MTLCommandBuffer> cb)
{
    pthread_mutex_lock(&q->lock);
    [q->last release];
    q->last = [cb retain];
    pthread_mutex_unlock(&q->lock);
}

static ocu_status mt_queue_sync(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = last_of(q);
        [cb waitUntilCompleted];
        ocu_status st = cb_status(cb);
        [cb release];
        /* Report a failure once, then clear it (interface rule). */
        pthread_mutex_lock(&q->lock);
        if (st == OCU_OK) st = q->sticky;
        q->sticky = OCU_OK;
        pthread_mutex_unlock(&q->lock);
        return st;
    }
}

static ocu_status mt_queue_query(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = last_of(q);
        ocu_status st = OCU_OK;
        if (cb != nil) {
            MTLCommandBufferStatus s = cb.status;
            if (s == MTLCommandBufferStatusError) st = OCU_ERR_LAUNCH_FAILED;
            else if (s != MTLCommandBufferStatusCompleted) st = OCU_ERR_NOT_READY;
        }
        [cb release];
        return st;
    }
}

static ocu_status mt_queue_destroy(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = mt_queue_sync(q);
    [q->last release];
    [q->queue release];
    pthread_mutex_destroy(&q->lock);
    free(q);
    return st;
}

/* Wait for every queue that might use memory: the device's sync queue and the
   given one. (The runtime orders other queues itself.) */
static ocu_status order_host_access(ocu_device *d, ocu_queue *q)
{
    ocu_status st = mt_queue_sync(q != NULL ? q : d->sync_q);
    return st;
}

/* ---- memory ------------------------------------------------------------------------------- */

static ocu_status mt_mem_alloc(ocu_device *d, size_t bytes, void **ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    if (bytes == 0) {
        *ptr = NULL;
        return OCU_OK;
    }
    @autoreleasepool {
        id<MTLBuffer> b = [d->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (b == nil) return OCU_ERR_OUT_OF_MEMORY;
        mt_alloc a;
        if (@available(macOS 13.0, *)) {
            a.gpu = b.gpuAddress;
        } else {
            [b release];
            return OCU_ERR_NOT_SUPPORTED;
        }
        a.size = bytes;
        a.cpu = (char *)b.contents;
        a.buffer = b; /* the +1 from new... */
        if (!table_add(d, a)) {
            [b release];
            return OCU_ERR_OUT_OF_MEMORY;
        }
        *ptr = (void *)(uintptr_t)a.gpu;
    }
    return OCU_OK;
}

static ocu_status mt_mem_free(ocu_device *d, void *ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_OK;
    (void)mt_queue_sync(d->sync_q);
    uint64_t gpu = (uint64_t)(uintptr_t)ptr;
    id<MTLBuffer> found = nil;
    pthread_mutex_lock(&d->lock);
    for (size_t i = 0; i < d->count; ++i) {
        if (d->allocs[i].gpu == gpu) {
            found = d->allocs[i].buffer;
            d->allocs[i] = d->allocs[--d->count];
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
    if (found == nil) return OCU_ERR_INVALID_VALUE;
    [found release]; /* Metal keeps it alive while committed work still uses it */
    return OCU_OK;
}

static ocu_status mt_mem_copy(ocu_device *d, ocu_queue *q, void *dst, const void *src, size_t bytes,
                              ocu_copy_dir dir)
{
    if (d == NULL || (q != NULL && q->dev != d)) return OCU_ERR_INVALID_HANDLE;
    if (bytes == 0) return OCU_OK;
    if (dst == NULL || src == NULL) return OCU_ERR_INVALID_VALUE;
    char *cd = (char *)dst;
    const char *cs = (const char *)src;
    if (dir == OCU_COPY_HOST_TO_DEVICE || dir == OCU_COPY_DEVICE_TO_DEVICE) {
        cd = table_cpu(d, (uint64_t)(uintptr_t)dst, bytes);
        if (cd == NULL) return OCU_ERR_INVALID_VALUE;
    }
    if (dir == OCU_COPY_DEVICE_TO_HOST || dir == OCU_COPY_DEVICE_TO_DEVICE) {
        cs = table_cpu(d, (uint64_t)(uintptr_t)src, bytes);
        if (cs == NULL) return OCU_ERR_INVALID_VALUE;
    }
    if (dir != OCU_COPY_HOST_TO_DEVICE && dir != OCU_COPY_DEVICE_TO_HOST &&
        dir != OCU_COPY_DEVICE_TO_DEVICE) {
        return OCU_ERR_INVALID_VALUE;
    }
    /* Same memory: keep the order by finishing the queue's earlier work. */
    ocu_status st = order_host_access(d, q);
    if (st != OCU_OK) return st;
    memmove(cd, cs, bytes);
    return OCU_OK;
}

static ocu_status mt_mem_set(ocu_device *d, ocu_queue *q, void *dst, int value, size_t bytes)
{
    if (d == NULL || (q != NULL && q->dev != d)) return OCU_ERR_INVALID_HANDLE;
    if (bytes == 0) return OCU_OK;
    char *cd = table_cpu(d, (uint64_t)(uintptr_t)dst, bytes);
    if (cd == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = order_host_access(d, q);
    if (st != OCU_OK) return st;
    memset(cd, value, bytes);
    return OCU_OK;
}

static ocu_status mt_mem_info(ocu_device *d, uint64_t *free_bytes, uint64_t *total_bytes)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (free_bytes == NULL || total_bytes == NULL) return OCU_ERR_INVALID_VALUE;
    @autoreleasepool {
        uint64_t total = d->dev.recommendedMaxWorkingSetSize;
        uint64_t used = d->dev.currentAllocatedSize;
        *total_bytes = total;
        *free_bytes = used < total ? total - used : 0;
    }
    return OCU_OK;
}

/* Host memory is ordinary memory here: page-locking has no meaning. */
static ocu_status mt_host_alloc_pinned(ocu_device *d, size_t bytes, void **ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    if (bytes == 0) {
        *ptr = NULL;
        return OCU_OK;
    }
    void *p = NULL;
    if (posix_memalign(&p, 16384, bytes) != 0) return OCU_ERR_OUT_OF_MEMORY;
    *ptr = p;
    return OCU_OK;
}

static ocu_status mt_host_free_pinned(ocu_device *d, void *ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    (void)mt_queue_sync(d->sync_q);
    free(ptr);
    return OCU_OK;
}

/* ---- code ------------------------------------------------------------------------------------- */

/*
 * Image: "OCUMSL1\n", then header lines, then a line "---\n", then the MSL
 * source. Header lines (compiler/metal/cu2msl.py writes them):
 *   param <kernel> <size> <size> ...   argument sizes of a kernel
 *   global <name> <offset> <size>      a variable, in the module's buffer
 *   init <offset> <hex bytes>          initial bytes there (zero elsewhere)
 */

/* Next header line after `line`, or NULL. */
static const char *next_line(const char *line)
{
    const char *nl = strchr(line, '\n');
    return nl != NULL && nl[1] != '\0' ? nl + 1 : NULL;
}

/* "global <name> <offset> <size>": the name (not terminated) and numbers. */
static bool parse_global(const char *line, const char **name, size_t *name_len, size_t *offset,
                         size_t *size)
{
    if (strncmp(line, "global ", 7) != 0) return false;
    const char *p = line + 7;
    const char *e = p;
    while (*e != '\0' && *e != ' ' && *e != '\n') ++e;
    if (e == p || *e != ' ') return false;
    char *end1, *end2;
    unsigned long long off = strtoull(e + 1, &end1, 10);
    if (end1 == e + 1 || *end1 != ' ') return false;
    unsigned long long sz = strtoull(end1 + 1, &end2, 10);
    if (end2 == end1 + 1 || (*end2 != '\n' && *end2 != '\0')) return false;
    *name = p;
    *name_len = (size_t)(e - p);
    *offset = (size_t)off;
    *size = (size_t)sz;
    return true;
}

static int hex_digit(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Allocate the module's variables (one buffer, zero-filled) and write
   the initial values. */
static ocu_status load_globals(ocu_module *m)
{
    size_t total = 0;
    for (const char *line = m->meta; line != NULL && *line; line = next_line(line)) {
        const char *name;
        size_t len, off, size;
        if (strncmp(line, "global ", 7) != 0) continue;
        if (!parse_global(line, &name, &len, &off, &size) || off + size < off) return OCU_ERR_INVALID_VALUE;
        if (off + size > total) total = off + size;
    }
    if (total == 0) return OCU_OK;
    void *gpu = NULL;
    ocu_status st = mt_mem_alloc(m->dev, total, &gpu);
    if (st != OCU_OK) return st;
    char *cpu = table_cpu(m->dev, (uint64_t)(uintptr_t)gpu, total);
    id<MTLBuffer> buf = table_buffer(m->dev, (uint64_t)(uintptr_t)gpu);
    if (cpu == NULL || buf == nil) {
        [buf release];
        mt_mem_free(m->dev, gpu);
        return OCU_ERR_INVALID_VALUE;
    }
    memset(cpu, 0, total);
    for (const char *line = m->meta; line != NULL && *line; line = next_line(line)) {
        if (strncmp(line, "init ", 5) != 0) continue;
        char *end;
        unsigned long long off = strtoull(line + 5, &end, 10);
        if (end == line + 5 || *end != ' ') st = OCU_ERR_INVALID_VALUE;
        const char *h = end + 1;
        for (size_t at = (size_t)off; st == OCU_OK && *h != '\n' && *h != '\0'; h += 2, ++at) {
            int hi = hex_digit(h[0]), lo = hi >= 0 ? hex_digit(h[1]) : -1;
            if (hi < 0 || lo < 0 || at >= total) {
                st = OCU_ERR_INVALID_VALUE;
            } else {
                cpu[at] = (char)(hi * 16 + lo);
            }
        }
        if (st != OCU_OK) break;
    }
    if (st != OCU_OK) {
        [buf release];
        mt_mem_free(m->dev, gpu);
        return st;
    }
    m->globals = gpu;
    m->globals_size = total;
    m->globals_buf = buf;
    return OCU_OK;
}

static ocu_status mt_module_load(ocu_device *d, const void *image, size_t bytes, ocu_module **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (image == NULL || out == NULL || bytes < 8) return OCU_ERR_INVALID_VALUE;
    const char *img = image;
    if (memcmp(img, "OCUMSL1\n", 8) != 0) return OCU_ERR_INVALID_VALUE;
    const char *sep = NULL;
    for (size_t i = 8; i + 4 <= bytes; ++i) {
        if (memcmp(img + i, "---\n", 4) == 0 && (img[i - 1] == '\n')) {
            sep = img + i;
            break;
        }
    }
    if (sep == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_module *m = calloc(1, sizeof(*m));
    if (m == NULL) return OCU_ERR_OUT_OF_MEMORY;
    size_t meta_len = (size_t)(sep - (img + 8));
    m->meta = malloc(meta_len + 1);
    if (m->meta == NULL) {
        free(m);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    memcpy(m->meta, img + 8, meta_len);
    m->meta[meta_len] = '\0';
    @autoreleasepool {
        const char *src = sep + 4;
        NSString *text = [[[NSString alloc] initWithBytes:src
                                                   length:bytes - (size_t)(src - img)
                                                 encoding:NSUTF8StringEncoding] autorelease];
        MTLCompileOptions *opts = [[[MTLCompileOptions alloc] init] autorelease];
        if (@available(macOS 13.0, *)) opts.languageVersion = MTLLanguageVersion3_0;
        NSError *err = nil;
        id<MTLLibrary> lib = text ? [d->dev newLibraryWithSource:text options:opts error:&err] : nil;
        if (lib == nil) {
            if (getenv("OCU_METAL_VERBOSE") != NULL && err != nil) {
                fprintf(stderr, "ocu metal: MSL compile failed: %s\n",
                        err.localizedDescription.UTF8String);
            }
            free(m->meta);
            free(m);
            return OCU_ERR_INVALID_VALUE;
        }
        m->library = lib;
    }
    m->dev = d;
    ocu_status gst = load_globals(m);
    if (gst != OCU_OK) {
        [m->library release];
        free(m->meta);
        free(m);
        return gst;
    }
    pthread_mutex_init(&m->lock, NULL);
    *out = m;
    return OCU_OK;
}

static ocu_status mt_module_unload(ocu_module *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    ocu_status st = mt_queue_sync(m->dev->sync_q);
    for (struct ocu_kernel *k = m->kernels; k != NULL;) {
        struct ocu_kernel *next = k->next;
        [k->pipeline release];
        free(k->name);
        free(k->param_sizes);
        free(k);
        k = next;
    }
    [m->library release];
    if (m->globals != NULL) {
        [m->globals_buf release];
        ocu_status fst = mt_mem_free(m->dev, m->globals);
        if (st == OCU_OK) st = fst;
    }
    free(m->meta);
    pthread_mutex_destroy(&m->lock);
    free(m);
    return st;
}

/* A variable of the module, by the name the image gives (the device-code
   name the host side registers). */
static ocu_status mt_module_get_global(ocu_module *m, const char *name, void **ptr, size_t *bytes)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (name == NULL || ptr == NULL || bytes == NULL) return OCU_ERR_INVALID_VALUE;
    size_t want = strlen(name);
    for (const char *line = m->meta; line != NULL && *line; line = next_line(line)) {
        const char *n;
        size_t len, off, size;
        if (!parse_global(line, &n, &len, &off, &size)) continue;
        if (len != want || memcmp(n, name, len) != 0) continue;
        if (m->globals == NULL || off + size > m->globals_size) return OCU_ERR_INVALID_VALUE;
        *ptr = (char *)m->globals + off;
        *bytes = size;
        return OCU_OK;
    }
    return OCU_ERR_NOT_FOUND;
}

static void parse_params(struct ocu_kernel *k, const char *meta)
{
    size_t len = strlen(k->name);
    for (const char *line = meta; line != NULL && *line; line = strchr(line, '\n') ? strchr(line, '\n') + 1 : NULL) {
        if (strncmp(line, "param ", 6) != 0) continue;
        const char *p = line + 6;
        if (strncmp(p, k->name, len) != 0 || (p[len] != ' ' && p[len] != '\n' && p[len] != '\0')) continue;
        p += len;
        size_t sizes[256];
        uint32_t n = 0;
        while (*p == ' ' && n < 256) {
            char *end;
            unsigned long v = strtoul(p + 1, &end, 10);
            if (end == p + 1) break;
            sizes[n++] = v;
            p = end;
        }
        k->param_sizes = n ? malloc(n * sizeof(size_t)) : NULL;
        if (n && k->param_sizes == NULL) return;
        memcpy(k->param_sizes, sizes, n * sizeof(size_t));
        k->nparams = n;
        return;
    }
}

static ocu_status mt_module_get_kernel(ocu_module *m, const char *name, ocu_kernel **out)
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
    ocu_status st = OCU_OK;
    @autoreleasepool {
        id<MTLFunction> fn = [m->library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (fn == nil) {
            st = OCU_ERR_NOT_FOUND;
        } else {
            NSError *err = nil;
            id<MTLComputePipelineState> ps = [m->dev->dev newComputePipelineStateWithFunction:fn error:&err];
            [fn release];
            struct ocu_kernel *k = ps ? calloc(1, sizeof(*k)) : NULL;
            char *copy = k ? strdup(name) : NULL;
            if (ps == nil) {
                st = OCU_ERR_INVALID_VALUE;
            } else if (k == NULL || copy == NULL) {
                [ps release];
                free(k);
                st = OCU_ERR_OUT_OF_MEMORY;
            } else {
                k->module = m;
                k->pipeline = ps;
                k->name = copy;
                parse_params(k, m->meta);
                k->next = m->kernels;
                m->kernels = k;
                *out = k;
            }
        }
    }
    pthread_mutex_unlock(&m->lock);
    return st;
}

static ocu_status mt_kernel_arg_sizes(ocu_kernel *k, uint32_t *count, size_t *sizes, uint32_t max)
{
    if (k == NULL) return OCU_ERR_INVALID_HANDLE;
    if (count == NULL || (max > 0 && sizes == NULL)) return OCU_ERR_INVALID_VALUE;
    *count = k->nparams;
    for (uint32_t i = 0; i < k->nparams && i < max; ++i) sizes[i] = k->param_sizes[i];
    return OCU_OK;
}

/* ---- execution ------------------------------------------------------------------------------------ */

/* Alignment of an argument value of `size` bytes: its largest power-of-two
   divisor, at most 16 (README.md, section 5). */
static size_t arg_align(size_t size)
{
    size_t a = 1;
    while (a < 16 && size % (a * 2) == 0) a *= 2;
    return a;
}

static ocu_status mt_kernel_launch(ocu_queue *q, ocu_kernel *k, const ocu_launch_params *p)
{
    if (k == NULL) return OCU_ERR_INVALID_HANDLE;
    if (p == NULL || (p->num_args > 0 && (p->args == NULL || p->arg_sizes == NULL))) {
        return OCU_ERR_INVALID_VALUE;
    }
    ocu_device *d = k->module->dev;
    if (q != NULL && q->dev != d) return OCU_ERR_INVALID_HANDLE;
    ocu_queue *queue = q != NULL ? q : d->sync_q;
    uint64_t threads = (uint64_t)p->block[0] * p->block[1] * p->block[2];
    for (int i = 0; i < 3; ++i) {
        if (p->grid[i] == 0 || p->block[i] == 0) return OCU_ERR_INVALID_VALUE;
    }
    if (threads > k->pipeline.maxTotalThreadsPerThreadgroup) return OCU_ERR_INVALID_VALUE;
    if (p->shared_memory_bytes > d->dev.maxThreadgroupMemoryLength) return OCU_ERR_INVALID_VALUE;
    /* Pack the argument values and their offsets. */
    size_t total = 0;
    for (uint32_t i = 0; i < p->num_args; ++i) {
        size_t a = arg_align(p->arg_sizes[i]);
        total = (total + a - 1) / a * a + p->arg_sizes[i];
        if (total > OCU_MAX_ARG_BYTES) return OCU_ERR_INVALID_VALUE;
    }
    unsigned char *packed = calloc(1, total + 16);
    uint32_t *offsets = calloc(p->num_args + 1, sizeof(uint32_t));
    if (packed == NULL || offsets == NULL) {
        free(packed);
        free(offsets);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    size_t at = 0;
    for (uint32_t i = 0; i < p->num_args; ++i) {
        size_t a = arg_align(p->arg_sizes[i]);
        at = (at + a - 1) / a * a;
        memcpy(packed + at, p->args[i], p->arg_sizes[i]);
        offsets[i] = (uint32_t)at;
        at += p->arg_sizes[i];
    }
    ocu_status st = OCU_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:k->pipeline];
        id<MTLBuffer> argbuf = [d->dev newBufferWithBytes:packed length:total + 16
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> offbuf = [d->dev newBufferWithBytes:offsets
                                                  length:(p->num_args + 1) * sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];
        [enc setBuffer:argbuf offset:0 atIndex:0];
        [enc setBuffer:offbuf offset:0 atIndex:1];
        /* The module's variables (translated kernels of a module with
           variables declare buffer 2). */
        if (k->module->globals_buf != nil) [enc setBuffer:k->module->globals_buf offset:0 atIndex:2];
        /* Translated kernels always declare threadgroup memory 0; give it
           at least the minimum length even when the launch asks for none. */
        NSUInteger len = (p->shared_memory_bytes + 15) / 16 * 16;
        [enc setThreadgroupMemoryLength:(len < 16 ? 16 : len) atIndex:0];
        /* Every allocation may be reached through a pointer argument. */
        pthread_mutex_lock(&d->lock);
        size_t n = d->count;
        id<MTLResource> *res = n ? malloc(n * sizeof(id<MTLResource>)) : NULL;
        for (size_t i = 0; i < n && res != NULL; ++i) res[i] = d->allocs[i].buffer;
        if (res != NULL) {
            [enc useResources:res count:n usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
        pthread_mutex_unlock(&d->lock);
        free(res);
        [enc dispatchThreadgroups:MTLSizeMake(p->grid[0], p->grid[1], p->grid[2])
            threadsPerThreadgroup:MTLSizeMake(p->block[0], p->block[1], p->block[2])];
        [enc endEncoding];
        [cb commit];
        set_last(queue, cb);
        [argbuf release];
        [offbuf release];
        if (q == NULL) {
            [cb waitUntilCompleted];
            st = cb_status(cb);
        }
    }
    free(packed);
    free(offsets);
    return st;
}

/* ---- markers ---------------------------------------------------------------------------------------- */

static ocu_status mt_marker_create(ocu_device *d, ocu_marker **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_marker *m = calloc(1, sizeof(*m));
    if (m == NULL) return OCU_ERR_OUT_OF_MEMORY;
    m->dev = d;
    pthread_mutex_init(&m->lock, NULL);
    *out = m;
    return OCU_OK;
}

static ocu_status mt_marker_destroy(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    [m->cb release];
    pthread_mutex_destroy(&m->lock);
    free(m);
    return OCU_OK;
}

/* Recording commits an empty command buffer: it completes after the
   queue's earlier work, and its completion time is the marker's time. */
static ocu_status mt_marker_record(ocu_marker *m, ocu_queue *q)
{
    if (m == NULL || (q != NULL && q->dev != m->dev)) return OCU_ERR_INVALID_HANDLE;
    ocu_queue *queue = q != NULL ? q : m->dev->sync_q;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue->queue commandBuffer];
        atomic_store(&m->done_ns, 0);
        ocu_marker *mm = m;
        [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
            (void)b;
            atomic_store(&mm->done_ns, now_ns());
        }];
        [cb commit];
        set_last(queue, cb);
        pthread_mutex_lock(&m->lock);
        [m->cb release];
        m->cb = [cb retain];
        pthread_mutex_unlock(&m->lock);
    }
    return OCU_OK;
}

static id<MTLCommandBuffer> marker_cb(ocu_marker *m)
{
    pthread_mutex_lock(&m->lock);
    id<MTLCommandBuffer> cb = [m->cb retain];
    pthread_mutex_unlock(&m->lock);
    return cb;
}

static ocu_status mt_marker_wait(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = marker_cb(m);
        [cb waitUntilCompleted];
        /* the completion handler may run just after the wait returns */
        while (cb != nil && atomic_load(&m->done_ns) == 0) {
        }
        [cb release];
    }
    return OCU_OK;
}

static ocu_status mt_marker_query(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = marker_cb(m);
        ocu_status st = (cb == nil || atomic_load(&m->done_ns) != 0) ? OCU_OK : OCU_ERR_NOT_READY;
        [cb release];
        return st;
    }
}

static ocu_status mt_marker_elapsed_ms(ocu_marker *a, ocu_marker *b, float *ms)
{
    if (a == NULL || b == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ms == NULL) return OCU_ERR_INVALID_VALUE;
    @autoreleasepool {
        id<MTLCommandBuffer> ca = marker_cb(a), cbb = marker_cb(b);
        bool recorded = ca != nil && cbb != nil;
        [ca release];
        [cbb release];
        if (!recorded) return OCU_ERR_INVALID_HANDLE;
    }
    uint64_t ta = atomic_load(&a->done_ns), tb = atomic_load(&b->done_ns);
    if (ta == 0 || tb == 0) return OCU_ERR_NOT_READY;
    *ms = (float)((double)((int64_t)(tb - ta)) / 1e6);
    return OCU_OK;
}

static const void *mt_get_provider(const char *library)
{
    (void)library;
    return NULL;
}

static const ocu_backend g_metal = {
    .abi = OCU_BACKEND_ABI,
    .name = "metal",
    .dev_count = mt_dev_count,
    .dev_info = mt_dev_info,
    .dev_open = mt_dev_open,
    .dev_close = mt_dev_close,
    .mem_alloc = mt_mem_alloc,
    .mem_free = mt_mem_free,
    .mem_copy = mt_mem_copy,
    .mem_set = mt_mem_set,
    .mem_info = mt_mem_info,
    .host_alloc_pinned = mt_host_alloc_pinned,
    .host_free_pinned = mt_host_free_pinned,
    .queue_create = mt_queue_create,
    .queue_destroy = mt_queue_destroy,
    .queue_sync = mt_queue_sync,
    .queue_query = mt_queue_query,
    .module_load = mt_module_load,
    .module_unload = mt_module_unload,
    .module_get_kernel = mt_module_get_kernel,
    .kernel_launch = mt_kernel_launch,
    .marker_create = mt_marker_create,
    .marker_destroy = mt_marker_destroy,
    .marker_record = mt_marker_record,
    .marker_wait = mt_marker_wait,
    .marker_query = mt_marker_query,
    .marker_elapsed_ms = mt_marker_elapsed_ms,
    .get_provider = mt_get_provider,
    .module_get_global = mt_module_get_global,
    .kernel_arg_sizes = mt_kernel_arg_sizes,
    .mem_alloc_shared = NULL,
    .host_register = NULL,
    .host_unregister = NULL,
};

const ocu_backend *ocu_backend_metal(void) { return &g_metal; }

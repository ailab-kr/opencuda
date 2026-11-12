/*
 * CPU backend: device and pinned-host memory.
 *
 * "Device" memory is ordinary heap memory, aligned to 256 bytes like GPU
 * allocations. Every allocation is tracked so that frees of foreign or
 * interior pointers, and copies that run past the end of an allocation,
 * are reported instead of corrupting memory silently.
 */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE /* _SC_PHYS_PAGES is outside POSIX */
#endif

#include "cpu_internal.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct alloc_rec {
    uintptr_t base;
    size_t    bytes;
} alloc_rec;

typedef struct alloc_table {
    alloc_rec *recs;     /* sorted by base */
    size_t     count;
    size_t     capacity;
    uint64_t   in_use;   /* sum of bytes */
} alloc_table;

static pthread_mutex_t g_mem_lock = PTHREAD_MUTEX_INITIALIZER;
static alloc_table     g_device;
static alloc_table     g_pinned;

uint64_t cpu_total_memory(void)
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) {
        return 0;
    }
    return (uint64_t)pages * (uint64_t)page;
}

/* Index of the first record with base >= key. */
static size_t lower_bound(const alloc_table *t, uintptr_t key)
{
    size_t lo = 0, hi = t->count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (t->recs[mid].base < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static ocu_status table_insert(alloc_table *t, uintptr_t base, size_t bytes)
{
    if (t->count == t->capacity) {
        size_t cap = t->capacity ? t->capacity * 2 : 64;
        alloc_rec *grown = realloc(t->recs, cap * sizeof(*grown));
        if (grown == NULL) {
            return OCU_ERR_OUT_OF_MEMORY;
        }
        t->recs = grown;
        t->capacity = cap;
    }
    size_t at = lower_bound(t, base);
    memmove(&t->recs[at + 1], &t->recs[at],
            (t->count - at) * sizeof(*t->recs));
    t->recs[at].base = base;
    t->recs[at].bytes = bytes;
    t->count += 1;
    t->in_use += bytes;
    return OCU_OK;
}

/* Remove the record whose base equals `base` exactly. */
static bool table_remove(alloc_table *t, uintptr_t base)
{
    size_t at = lower_bound(t, base);
    if (at == t->count || t->recs[at].base != base) {
        return false;
    }
    t->in_use -= t->recs[at].bytes;
    memmove(&t->recs[at], &t->recs[at + 1],
            (t->count - at - 1) * sizeof(*t->recs));
    t->count -= 1;
    return true;
}

static ocu_status tracked_alloc(alloc_table *t, size_t bytes, void **ptr)
{
    if (ptr == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    if (bytes == 0) {
        *ptr = NULL;
        return OCU_OK;
    }
    /* Round up so the size is a multiple of the alignment. */
    if (bytes > SIZE_MAX - CPU_ALLOC_ALIGN) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    size_t rounded = (bytes + CPU_ALLOC_ALIGN - 1) & ~(size_t)(CPU_ALLOC_ALIGN - 1);
    void *mem = NULL;
    if (posix_memalign(&mem, CPU_ALLOC_ALIGN, rounded) != 0 || mem == NULL) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    pthread_mutex_lock(&g_mem_lock);
    ocu_status st = table_insert(t, (uintptr_t)mem, bytes);
    pthread_mutex_unlock(&g_mem_lock);
    if (st != OCU_OK) {
        free(mem);
        return st;
    }
    *ptr = mem;
    return OCU_OK;
}

static ocu_status tracked_free(alloc_table *t, void *ptr)
{
    if (ptr == NULL) {
        return OCU_OK;
    }
    if (cpu_in_kernel()) {
        return OCU_ERR_NOT_SUPPORTED;
    }
    /* Queued work may still read or write this memory. */
    cpu_queues_drain();
    pthread_mutex_lock(&g_mem_lock);
    bool found = table_remove(t, (uintptr_t)ptr);
    pthread_mutex_unlock(&g_mem_lock);
    if (!found) {
        return OCU_ERR_INVALID_VALUE;
    }
    free(ptr);
    return OCU_OK;
}

ocu_status cpu_mem_alloc(ocu_device *dev, size_t bytes, void **ptr)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_DEVICE;
    }
    return tracked_alloc(&g_device, bytes, ptr);
}

ocu_status cpu_mem_free(ocu_device *dev, void *ptr)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_DEVICE;
    }
    return tracked_free(&g_device, ptr);
}

ocu_status cpu_host_alloc_pinned(ocu_device *dev, size_t bytes, void **ptr)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_DEVICE;
    }
    return tracked_alloc(&g_pinned, bytes, ptr);
}

ocu_status cpu_host_free_pinned(ocu_device *dev, void *ptr)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_DEVICE;
    }
    return tracked_free(&g_pinned, ptr);
}

ocu_status cpu_mem_info(ocu_device *dev, uint64_t *free_bytes,
                        uint64_t *total_bytes)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_DEVICE;
    }
    if (free_bytes == NULL || total_bytes == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    uint64_t total = cpu_total_memory();
    pthread_mutex_lock(&g_mem_lock);
    uint64_t used = g_device.in_use + g_pinned.in_use;
    pthread_mutex_unlock(&g_mem_lock);
    *total_bytes = total;
    *free_bytes = used < total ? total - used : 0;
    return OCU_OK;
}

ocu_status cpu_check_device_range(const void *ptr, size_t bytes)
{
    if (bytes == 0) {
        return OCU_OK;
    }
    if (ptr == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    uintptr_t p = (uintptr_t)ptr;
    ocu_status st = OCU_ERR_INVALID_VALUE;
    pthread_mutex_lock(&g_mem_lock);
    /* The candidate is the last allocation whose base is <= p. */
    size_t at = lower_bound(&g_device, p + 1);
    if (at > 0) {
        const alloc_rec *r = &g_device.recs[at - 1];
        if (p >= r->base && p - r->base <= r->bytes &&
            bytes <= r->bytes - (p - r->base)) {
            st = OCU_OK;
        }
    }
    pthread_mutex_unlock(&g_mem_lock);
    return st;
}

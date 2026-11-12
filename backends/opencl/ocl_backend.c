/*
 * ocl_backend.c -- the device backend interface on OpenCL devices (see
 * README.md). Memory is coarse-grained shared virtual memory (SVM), so a
 * device pointer is a plain address that kernels may offset and store;
 * device code comes as SPIR-V or C++ for OpenCL source made by
 * compiler/opencl/cu2ocl.py.
 */
#define _GNU_SOURCE
#include "ocu_backend.h"
#include "ocu_backend_opencl.h"
#include "ocl_api.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- objects ---------------------------------------------------------------- */

typedef struct ocl_alloc {
    char  *ptr;   /* SVM address: the same for host-side calls and kernels */
    size_t size;
    char  *base;  /* what clSVMAlloc returned (ptr is aligned up from it) */
} ocl_alloc;

typedef struct ocl_dev_entry {
    cl_platform_id platform;
    cl_device_id   device;
    ocu_dev_info   info;
    bool           il;          /* takes SPIR-V (clCreateProgramWithIL) */
    bool           cxx;         /* compiles C++ for OpenCL source */
    bool           fine_buffer; /* fine-grained SVM buffers (mem_alloc_shared) */
    pthread_once_t warp_once;
} ocl_dev_entry;

struct ocu_device {
    ocl_dev_entry    *entry;
    cl_context        ctx;
    struct ocu_queue *sync_q;   /* for queue == NULL */
    pthread_mutex_t   lock;     /* guards allocs and queues */
    ocl_alloc        *allocs;
    size_t            count, cap;
    uint64_t          allocated;
    struct ocu_queue *queues;   /* every queue of the device (mem_free waits for them) */
};

struct ocu_queue {
    ocu_device       *dev;
    cl_command_queue  q;
    pthread_mutex_t   lock;
    cl_event          last;    /* newest enqueued command (retained) or NULL */
    ocu_status        sticky;  /* a failure seen by launch, reported by sync/query */
    struct ocu_queue *next;
};

struct ocu_kernel {
    struct ocu_module *module;
    cl_kernel          k;
    char              *name;
    uint32_t           nparams;
    size_t            *param_sizes;
    size_t             max_threads;  /* CL_KERNEL_WORK_GROUP_SIZE */
    uint64_t           static_local; /* CL_KERNEL_LOCAL_MEM_SIZE */
    bool               takes_glob;   /* 4 parameters: the variables block is the third */
    pthread_mutex_t    lock;         /* argument setting + enqueue is one step */
    struct ocu_kernel *next;
};

typedef struct ocl_global {
    char  *name;
    size_t offset, size;
} ocl_global;

struct ocu_module {
    ocu_device        *dev;
    cl_program         program;
    char              *meta;
    char              *glob;      /* SVM block of the module's variables, or NULL */
    size_t             glob_size;
    ocl_global        *globals;
    size_t             nglobals;
    pthread_mutex_t    lock;
    struct ocu_kernel *kernels;
};

struct ocu_marker {
    ocu_device     *dev;
    pthread_mutex_t lock;
    cl_event        ev;  /* retained; NULL when never recorded */
};

static bool verbose(void)
{
    const char *v = getenv("OCU_OPENCL_VERBOSE");
    return v != NULL && v[0] != '\0' && v[0] != '0';
}

/* ---- device list ---------------------------------------------------------------- */

#define OCL_MAX_DEVICES 16
static ocl_dev_entry g_devs[OCL_MAX_DEVICES];
static uint32_t g_ndevs;
static pthread_once_t g_devs_once = PTHREAD_ONCE_INIT;

static bool has_word(const char *list, const char *word)
{
    size_t n = strlen(word);
    for (const char *p = list; (p = strstr(p, word)) != NULL; p += n) {
        if ((p == list || p[-1] == ' ') && (p[n] == ' ' || p[n] == '\0')) return true;
    }
    return false;
}

/* "OpenCL <major>.<minor> ..." */
static int cl_version(const char *s)
{
    int major = 0, minor = 0;
    if (sscanf(s, "OpenCL %d.%d", &major, &minor) != 2) return 0;
    return major * 100 + minor * 10;
}

static void enumerate_once(void)
{
    const ocl_api *a = ocl_load();
    if (a == NULL) return;
    cl_platform_id plats[16];
    cl_uint np = 0;
    if (a->GetPlatformIDs(16, plats, &np) != CL_SUCCESS) return;
    if (np > 16) np = 16;
    for (cl_uint p = 0; p < np && g_ndevs < OCL_MAX_DEVICES; ++p) {
        cl_device_id devs[OCL_MAX_DEVICES];
        cl_uint nd = 0;
        if (a->GetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, OCL_MAX_DEVICES, devs, &nd) != CL_SUCCESS) {
            continue;
        }
        if (nd > OCL_MAX_DEVICES) nd = OCL_MAX_DEVICES;
        for (cl_uint i = 0; i < nd && g_ndevs < OCL_MAX_DEVICES; ++i) {
            cl_device_id d = devs[i];
            char name[256] = "", ver[256] = "", il[256] = "";
            static char ext[16384];
            a->GetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name) - 1, name, NULL);
            a->GetDeviceInfo(d, CL_DEVICE_VERSION, sizeof(ver) - 1, ver, NULL);
            ext[0] = '\0';
            a->GetDeviceInfo(d, CL_DEVICE_EXTENSIONS, sizeof(ext) - 1, ext, NULL);
            ext[sizeof(ext) - 1] = '\0';
            cl_device_svm_capabilities svm = 0;
            int v = cl_version(ver);
            if (v >= 200) a->GetDeviceInfo(d, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm), &svm, NULL);
            if (v >= 210) a->GetDeviceInfo(d, CL_DEVICE_IL_VERSION, sizeof(il) - 1, il, NULL);
            bool can_il = (strstr(il, "SPIR-V") != NULL || has_word(ext, "cl_khr_il_program")) &&
                          a->CreateProgramWithIL != NULL;
            bool can_cxx = has_word(ext, "cl_ext_cxx_for_opencl");
            const char *why = NULL;
            if (v < 200) why = "OpenCL 2.0 or later is needed (shared virtual memory)";
            else if (!(svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) why = "no shared virtual memory";
            else if (!can_il && !can_cxx) why = "takes neither SPIR-V nor C++ for OpenCL";
            if (why != NULL) {
                if (verbose()) fprintf(stderr, "ocu opencl: skipping %s (%s): %s\n", name, ver, why);
                continue;
            }
            ocl_dev_entry *e = &g_devs[g_ndevs++];
            memset(e, 0, sizeof(*e));
            e->platform = plats[p];
            e->device = d;
            e->il = can_il;
            e->cxx = can_cxx;
            e->fine_buffer = (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0;
            pthread_once_t once = PTHREAD_ONCE_INIT;
            e->warp_once = once;
            ocu_dev_info *info = &e->info;
            snprintf(info->name, sizeof(info->name), "%s", name); /* truncates, always terminated */
            cl_ulong gm = 0, lm = 0;
            a->GetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gm), &gm, NULL);
            a->GetDeviceInfo(d, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lm), &lm, NULL);
            size_t wg = 0, wi[3] = {0, 0, 0};
            a->GetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, NULL);
            a->GetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(wi), wi, NULL);
            info->total_memory_bytes = gm;
            info->max_threads_per_block = (uint32_t)(wg < 1024 ? wg : 1024);
            for (int k = 0; k < 3; ++k) {
                size_t w = wi[k] < wg ? wi[k] : wg;
                info->max_block_dim[k] = (uint32_t)(w < 1024 ? w : (k < 2 ? 1024 : 64));
            }
            info->max_grid_dim[0] = 0x7fffffffu;
            info->max_grid_dim[1] = 65535u;
            info->max_grid_dim[2] = 65535u;
            info->max_shared_memory_per_block = lm;
            /* The host may not touch coarse-grained SVM directly. */
            info->supports_unified_memory = 0;
            info->warp_size = 0; /* measured on first dev_info (probe_warp) */
        }
    }
}

static const ocl_api *api(void)
{
    pthread_once(&g_devs_once, enumerate_once);
    return ocl_load();
}

/*
 * The warp size: OpenCL has no fixed one. With sub-groups, a kernel
 * reports the largest sub-group size it runs with; that is the device's
 * SIMD width the way CUDA code means warpSize. Without sub-groups the
 * device has no lanes that work together, and the size is 1.
 */
static const char k_warp_probe[] =
    "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
    "__kernel void ocu_warp(__global uint *o)\n"
    "{\n"
    "#if defined(cl_khr_subgroups) || defined(__opencl_c_subgroups)\n"
    "    if (get_global_id(0) == 0) o[0] = get_max_sub_group_size();\n"
    "#else\n"
    "    if (get_global_id(0) == 0) o[0] = 1;\n"
    "#endif\n"
    "}\n";

static ocl_dev_entry *g_probe_entry;
static pthread_mutex_t g_probe_lock = PTHREAD_MUTEX_INITIALIZER;

static void probe_warp_once(void)
{
    ocl_dev_entry *e = g_probe_entry;
    const ocl_api *a = ocl_load();
    uint32_t warp = 1;
    cl_int err = CL_SUCCESS;
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)e->platform, 0};
    cl_context ctx = a->CreateContext(props, 1, &e->device, NULL, NULL, &err);
    if (ctx == NULL) goto done;
    cl_command_queue q = a->CreateCommandQueueWithProperties(ctx, e->device, NULL, &err);
    const char *src = k_warp_probe;
    cl_program prog = q ? a->CreateProgramWithSource(ctx, 1, &src, NULL, &err) : NULL;
    if (prog != NULL && a->BuildProgram(prog, 1, &e->device, "", NULL, NULL) == CL_SUCCESS) {
        cl_kernel k = a->CreateKernel(prog, "ocu_warp", &err);
        uint32_t *out = a->SVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), 0);
        if (k != NULL && out != NULL && a->SetKernelArgSVMPointer(k, 0, out) == CL_SUCCESS) {
            size_t local = e->info.max_threads_per_block >= 256 ? 256 : e->info.max_threads_per_block;
            size_t global = local;
            uint32_t got = 0;
            if (a->EnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL) == CL_SUCCESS &&
                a->EnqueueSVMMemcpy(q, CL_TRUE, &got, out, sizeof(got), 0, NULL, NULL) == CL_SUCCESS &&
                got >= 1 && got <= 1024) {
                warp = got;
            }
        }
        if (out != NULL) {
            a->Finish(q);
            a->SVMFree(ctx, out);
        }
        if (k != NULL) a->ReleaseKernel(k);
    }
    if (prog != NULL) a->ReleaseProgram(prog);
    if (q != NULL) a->ReleaseCommandQueue(q);
    a->ReleaseContext(ctx);
done:
    e->info.warp_size = warp;
}

static void probe_warp(ocl_dev_entry *e)
{
    pthread_mutex_lock(&g_probe_lock);
    g_probe_entry = e;
    pthread_once(&e->warp_once, probe_warp_once);
    pthread_mutex_unlock(&g_probe_lock);
}

static ocu_status ocl_dev_count(uint32_t *count)
{
    if (count == NULL) return OCU_ERR_INVALID_VALUE;
    api();
    *count = g_ndevs;
    return OCU_OK;
}

static ocu_status ocl_dev_info(uint32_t index, ocu_dev_info *info)
{
    if (info == NULL) return OCU_ERR_INVALID_VALUE;
    api();
    if (index >= g_ndevs) return OCU_ERR_INVALID_DEVICE;
    probe_warp(&g_devs[index]);
    *info = g_devs[index].info;
    return OCU_OK;
}

static ocu_status ocl_queue_create(ocu_device *d, ocu_queue **out);
static ocu_status ocl_queue_destroy(ocu_queue *q);
static ocu_status ocl_queue_sync(ocu_queue *q);

static ocu_status map_err(cl_int e)
{
    switch (e) {
    case CL_SUCCESS: return OCU_OK;
    case CL_OUT_OF_HOST_MEMORY:
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return OCU_ERR_OUT_OF_MEMORY;
    case CL_OUT_OF_RESOURCES: return OCU_ERR_LAUNCH_FAILED;
    case CL_INVALID_WORK_GROUP_SIZE:
    case CL_INVALID_VALUE: return OCU_ERR_INVALID_VALUE;
    default: return OCU_ERR_UNKNOWN;
    }
}

static ocu_status ocl_dev_open(uint32_t index, ocu_device **out)
{
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    const ocl_api *a = api();
    if (index >= g_ndevs) return OCU_ERR_INVALID_DEVICE;
    ocu_device *d = calloc(1, sizeof(*d));
    if (d == NULL) return OCU_ERR_OUT_OF_MEMORY;
    d->entry = &g_devs[index];
    cl_int err = CL_SUCCESS;
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)d->entry->platform, 0};
    d->ctx = a->CreateContext(props, 1, &d->entry->device, NULL, NULL, &err);
    if (d->ctx == NULL) {
        free(d);
        return err == CL_OUT_OF_HOST_MEMORY ? OCU_ERR_OUT_OF_MEMORY : OCU_ERR_INVALID_DEVICE;
    }
    pthread_mutex_init(&d->lock, NULL);
    ocu_status st = ocl_queue_create(d, &d->sync_q);
    if (st != OCU_OK) {
        a->ReleaseContext(d->ctx);
        pthread_mutex_destroy(&d->lock);
        free(d);
        return st;
    }
    *out = d;
    return OCU_OK;
}

/* Wait for the work of every queue of the device. */
static void finish_all(ocu_device *d)
{
    const ocl_api *a = ocl_load();
    pthread_mutex_lock(&d->lock);
    for (ocu_queue *q = d->queues; q != NULL; q = q->next) a->Finish(q->q);
    pthread_mutex_unlock(&d->lock);
}

static ocu_status ocl_dev_close(ocu_device *d)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    const ocl_api *a = ocl_load();
    finish_all(d);
    ocu_status st = ocl_queue_destroy(d->sync_q);
    for (size_t i = 0; i < d->count; ++i) a->SVMFree(d->ctx, d->allocs[i].base);
    free(d->allocs);
    a->ReleaseContext(d->ctx);
    pthread_mutex_destroy(&d->lock);
    free(d);
    return st;
}

/* ---- queues -------------------------------------------------------------------- */

static ocu_status ocl_queue_create(ocu_device *d, ocu_queue **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    const ocl_api *a = ocl_load();
    ocu_queue *q = calloc(1, sizeof(*q));
    if (q == NULL) return OCU_ERR_OUT_OF_MEMORY;
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_int err = CL_SUCCESS;
    q->q = a->CreateCommandQueueWithProperties(d->ctx, d->entry->device, props, &err);
    if (q->q == NULL) {
        free(q);
        return err == CL_OUT_OF_HOST_MEMORY ? OCU_ERR_OUT_OF_MEMORY : OCU_ERR_UNKNOWN;
    }
    q->dev = d;
    pthread_mutex_init(&q->lock, NULL);
    pthread_mutex_lock(&d->lock);
    q->next = d->queues;
    d->queues = q;
    pthread_mutex_unlock(&d->lock);
    *out = q;
    return OCU_OK;
}

/* Keep `ev` (already retained by the enqueue) as the queue's newest command. */
static void set_last(ocu_queue *q, cl_event ev)
{
    const ocl_api *a = ocl_load();
    pthread_mutex_lock(&q->lock);
    cl_event old = q->last;
    q->last = ev;
    pthread_mutex_unlock(&q->lock);
    if (old != NULL) a->ReleaseEvent(old);
}

static cl_event last_of(ocu_queue *q)
{
    const ocl_api *a = ocl_load();
    pthread_mutex_lock(&q->lock);
    cl_event ev = q->last;
    if (ev != NULL) a->RetainEvent(ev);
    pthread_mutex_unlock(&q->lock);
    return ev;
}

static cl_int event_status(cl_event ev)
{
    cl_int s = CL_COMPLETE;
    if (ocl_load()->GetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(s), &s, NULL) != CL_SUCCESS) {
        return -1;
    }
    return s;
}

static ocu_status ocl_queue_sync(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    const ocl_api *a = ocl_load();
    cl_int err = a->Finish(q->q);
    cl_event ev = last_of(q);
    ocu_status st = err == CL_SUCCESS ? OCU_OK : OCU_ERR_LAUNCH_FAILED;
    if (ev != NULL) {
        if (event_status(ev) < 0) st = OCU_ERR_LAUNCH_FAILED;
        a->ReleaseEvent(ev);
    }
    /* Report a failure once, then clear it (interface rule). */
    pthread_mutex_lock(&q->lock);
    if (st == OCU_OK) st = q->sticky;
    q->sticky = OCU_OK;
    pthread_mutex_unlock(&q->lock);
    return st;
}

static ocu_status ocl_queue_query(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    const ocl_api *a = ocl_load();
    a->Flush(q->q);
    cl_event ev = last_of(q);
    ocu_status st = OCU_OK;
    if (ev != NULL) {
        cl_int s = event_status(ev);
        if (s < 0) st = OCU_ERR_LAUNCH_FAILED;
        else if (s != CL_COMPLETE) st = OCU_ERR_NOT_READY;
        a->ReleaseEvent(ev);
    }
    if (st == OCU_OK) {
        pthread_mutex_lock(&q->lock);
        st = q->sticky;
        q->sticky = OCU_OK;
        pthread_mutex_unlock(&q->lock);
    }
    return st;
}

static ocu_status ocl_queue_destroy(ocu_queue *q)
{
    if (q == NULL) return OCU_ERR_INVALID_HANDLE;
    const ocl_api *a = ocl_load();
    ocu_status st = ocl_queue_sync(q);
    ocu_device *d = q->dev;
    pthread_mutex_lock(&d->lock);
    for (ocu_queue **pp = &d->queues; *pp != NULL; pp = &(*pp)->next) {
        if (*pp == q) {
            *pp = q->next;
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
    if (q->last != NULL) a->ReleaseEvent(q->last);
    a->ReleaseCommandQueue(q->q);
    pthread_mutex_destroy(&q->lock);
    free(q);
    return st;
}

/* ---- allocation table ------------------------------------------------------------ */

static bool table_add(ocu_device *d, char *ptr, size_t size, char *base)
{
    pthread_mutex_lock(&d->lock);
    bool ok = true;
    if (d->count == d->cap) {
        size_t cap = d->cap ? d->cap * 2 : 64;
        ocl_alloc *n = realloc(d->allocs, cap * sizeof(*n));
        if (n == NULL) {
            ok = false;
        } else {
            d->allocs = n;
            d->cap = cap;
        }
    }
    if (ok) {
        d->allocs[d->count].ptr = ptr;
        d->allocs[d->count].size = size;
        d->allocs[d->count].base = base;
        d->count++;
        d->allocated += size;
    }
    pthread_mutex_unlock(&d->lock);
    return ok;
}

/* Remove the allocation that starts at ptr; its SVM base, or NULL if there
   is none. */
static char *table_remove(ocu_device *d, const char *ptr)
{
    char *found = NULL;
    pthread_mutex_lock(&d->lock);
    for (size_t i = 0; i < d->count; ++i) {
        if (d->allocs[i].ptr == ptr) {
            d->allocated -= d->allocs[i].size;
            found = d->allocs[i].base;
            d->allocs[i] = d->allocs[--d->count];
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
    return found;
}

/* Is [p, p + n) inside one allocation? */
static bool table_has(ocu_device *d, const void *p, size_t n)
{
    const char *c = p;
    bool ok = false;
    pthread_mutex_lock(&d->lock);
    for (size_t i = 0; i < d->count; ++i) {
        const ocl_alloc *a = &d->allocs[i];
        if (c >= a->ptr && (size_t)(c - a->ptr) <= a->size && n <= a->size - (size_t)(c - a->ptr)) {
            ok = true;
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
    return ok;
}

/* ---- memory ------------------------------------------------------------------------ */

static ocu_status alloc_svm(ocu_device *d, size_t bytes, cl_svm_mem_flags flags, void **ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    if (bytes == 0) {
        *ptr = NULL;
        return OCU_OK;
    }
    const ocl_api *a = ocl_load();
    /* 256-byte alignment, as CUDA allocations have. Drivers may refuse an
       alignment beyond their largest data type (0 = that one), so the
       block is made larger and the address aligned up inside it. */
    if (bytes > SIZE_MAX - 256) return OCU_ERR_OUT_OF_MEMORY;
    char *base = a->SVMAlloc(d->ctx, CL_MEM_READ_WRITE | flags, bytes + 256, 0);
    if (base == NULL) return OCU_ERR_OUT_OF_MEMORY;
    char *p = base + ((256 - ((uintptr_t)base & 255)) & 255);
    if (!table_add(d, p, bytes, base)) {
        a->SVMFree(d->ctx, base);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    *ptr = p;
    return OCU_OK;
}

static ocu_status ocl_mem_alloc(ocu_device *d, size_t bytes, void **ptr)
{
    return alloc_svm(d, bytes, 0, ptr);
}

/* Host and device at the same address: fine-grained SVM buffers. */
static ocu_status ocl_mem_alloc_shared(ocu_device *d, size_t bytes, void **ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (!d->entry->fine_buffer) return OCU_ERR_NOT_SUPPORTED;
    return alloc_svm(d, bytes, CL_MEM_SVM_FINE_GRAIN_BUFFER, ptr);
}

static ocu_status ocl_mem_free(ocu_device *d, void *ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_OK;
    finish_all(d);
    char *base = table_remove(d, ptr);
    if (base == NULL) return OCU_ERR_INVALID_VALUE;
    ocl_load()->SVMFree(d->ctx, base);
    return OCU_OK;
}

/* Run `ev`'s command to the end when the caller asked for synchronous work. */
static ocu_status finish_enqueue(ocu_queue *queue, bool sync, cl_int err, cl_event ev)
{
    const ocl_api *a = ocl_load();
    if (err != CL_SUCCESS) return map_err(err);
    ocu_status st = OCU_OK;
    if (sync && ev != NULL) {
        if (a->WaitForEvents(1, &ev) != CL_SUCCESS || event_status(ev) < 0) st = OCU_ERR_LAUNCH_FAILED;
    }
    set_last(queue, ev);
    return st;
}

static ocu_status ocl_mem_copy(ocu_device *d, ocu_queue *q, void *dst, const void *src, size_t bytes,
                               ocu_copy_dir dir)
{
    if (d == NULL || (q != NULL && q->dev != d)) return OCU_ERR_INVALID_HANDLE;
    if (bytes == 0) return OCU_OK;
    if (dst == NULL || src == NULL) return OCU_ERR_INVALID_VALUE;
    if (dir != OCU_COPY_HOST_TO_DEVICE && dir != OCU_COPY_DEVICE_TO_HOST &&
        dir != OCU_COPY_DEVICE_TO_DEVICE) {
        return OCU_ERR_INVALID_VALUE;
    }
    if ((dir == OCU_COPY_HOST_TO_DEVICE || dir == OCU_COPY_DEVICE_TO_DEVICE) && !table_has(d, dst, bytes)) {
        return OCU_ERR_INVALID_VALUE;
    }
    if ((dir == OCU_COPY_DEVICE_TO_HOST || dir == OCU_COPY_DEVICE_TO_DEVICE) && !table_has(d, src, bytes)) {
        return OCU_ERR_INVALID_VALUE;
    }
    ocu_queue *queue = q != NULL ? q : d->sync_q;
    cl_event ev = NULL;
    cl_int err = ocl_load()->EnqueueSVMMemcpy(queue->q, q == NULL ? CL_TRUE : CL_FALSE, dst, src, bytes,
                                              0, NULL, &ev);
    return finish_enqueue(queue, q == NULL, err, ev);
}

static ocu_status ocl_mem_set(ocu_device *d, ocu_queue *q, void *dst, int value, size_t bytes)
{
    if (d == NULL || (q != NULL && q->dev != d)) return OCU_ERR_INVALID_HANDLE;
    if (bytes == 0) return OCU_OK;
    if (dst == NULL || !table_has(d, dst, bytes)) return OCU_ERR_INVALID_VALUE;
    ocu_queue *queue = q != NULL ? q : d->sync_q;
    unsigned char pattern = (unsigned char)value;
    cl_event ev = NULL;
    cl_int err = ocl_load()->EnqueueSVMMemFill(queue->q, dst, &pattern, 1, bytes, 0, NULL, &ev);
    return finish_enqueue(queue, q == NULL, err, ev);
}

/* OpenCL has no free-memory query: total minus what this device handle holds. */
static ocu_status ocl_mem_info(ocu_device *d, uint64_t *free_bytes, uint64_t *total_bytes)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (free_bytes == NULL || total_bytes == NULL) return OCU_ERR_INVALID_VALUE;
    uint64_t total = d->entry->info.total_memory_bytes;
    pthread_mutex_lock(&d->lock);
    uint64_t used = d->allocated;
    pthread_mutex_unlock(&d->lock);
    *total_bytes = total;
    *free_bytes = used < total ? total - used : 0;
    return OCU_OK;
}

/* Copies read host memory through the driver; page-locking is the driver's
   business. Page-aligned memory is what drivers copy from fastest. */
static ocu_status ocl_host_alloc_pinned(ocu_device *d, size_t bytes, void **ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    if (bytes == 0) {
        *ptr = NULL;
        return OCU_OK;
    }
    void *p = NULL;
    if (posix_memalign(&p, 4096, bytes) != 0) return OCU_ERR_OUT_OF_MEMORY;
    *ptr = p;
    return OCU_OK;
}

static ocu_status ocl_host_free_pinned(ocu_device *d, void *ptr)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    finish_all(d);
    free(ptr);
    return OCU_OK;
}

/* ---- code -------------------------------------------------------------------------- */

/*
 * Image (compiler/opencl/cu2ocl.py): "OCUOCL1\n", lines
 *   param <kernel> <size>...   global <name> <offset> <size>   init <offset> <hex>
 *   source <bytes>             spirv <bytes>             clc <bytes>
 * then "---\n", the C++ for OpenCL source, the SPIR-V module and OpenCL C
 * source, each there when its length line is. The first form the device
 * takes is used: SPIR-V, C++ for OpenCL, then OpenCL C (hand-written
 * kernels; every driver compiles it).
 */
static int hexval(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static void free_module(ocu_module *m)
{
    for (size_t i = 0; i < m->nglobals; ++i) free(m->globals[i].name);
    free(m->globals);
    free(m->meta);
    free(m);
}

static ocu_status parse_globals(ocu_module *m)
{
    for (const char *l = m->meta; *l; l = strchr(l, '\n') ? strchr(l, '\n') + 1 : l + strlen(l)) {
        if (strncmp(l, "global ", 7) != 0) continue;
        char name[512];
        unsigned long long off = 0, size = 0;
        if (sscanf(l + 7, "%511s %llu %llu", name, &off, &size) != 3) return OCU_ERR_INVALID_VALUE;
        ocl_global *g = realloc(m->globals, (m->nglobals + 1) * sizeof(*g));
        if (g == NULL) return OCU_ERR_OUT_OF_MEMORY;
        m->globals = g;
        g[m->nglobals].name = strdup(name);
        if (g[m->nglobals].name == NULL) return OCU_ERR_OUT_OF_MEMORY;
        g[m->nglobals].offset = off;
        g[m->nglobals].size = size;
        m->nglobals++;
        if (off + size > m->glob_size) m->glob_size = off + size;
    }
    return OCU_OK;
}

/* The module's variables: one SVM block, zeroed, then the init lines. */
static ocu_status load_globals(ocu_module *m)
{
    if (m->nglobals == 0) return OCU_OK;
    ocu_device *d = m->dev;
    void *p = NULL;
    ocu_status st = ocl_mem_alloc(d, m->glob_size < 4 ? 4 : m->glob_size, &p);
    if (st != OCU_OK) return st;
    m->glob = p;
    st = ocl_mem_set(d, NULL, p, 0, m->glob_size < 4 ? 4 : m->glob_size);
    for (const char *l = m->meta; st == OCU_OK && *l; l = strchr(l, '\n') ? strchr(l, '\n') + 1 : l + strlen(l)) {
        if (strncmp(l, "init ", 5) != 0) continue;
        char *end = NULL;
        unsigned long long off = strtoull(l + 5, &end, 10);
        if (end == NULL || *end != ' ') return OCU_ERR_INVALID_VALUE;
        const char *h = end + 1;
        size_t n = 0;
        while (hexval(h[n]) >= 0) ++n;
        if (n % 2 || off + n / 2 > m->glob_size) return OCU_ERR_INVALID_VALUE;
        unsigned char *bytes = malloc(n / 2 + 1);
        if (bytes == NULL) return OCU_ERR_OUT_OF_MEMORY;
        for (size_t i = 0; i < n / 2; ++i) bytes[i] = (unsigned char)(hexval(h[2 * i]) * 16 + hexval(h[2 * i + 1]));
        st = ocl_mem_copy(d, NULL, m->glob + off, bytes, n / 2, OCU_COPY_HOST_TO_DEVICE);
        free(bytes);
    }
    return st;
}

static ocu_status ocl_module_load(ocu_device *d, const void *image, size_t bytes, ocu_module **out)
{
    if (d == NULL) return OCU_ERR_INVALID_HANDLE;
    if (image == NULL || out == NULL || bytes < 8) return OCU_ERR_INVALID_VALUE;
    const char *img = image;
    if (memcmp(img, "OCUOCL1\n", 8) != 0) return OCU_ERR_INVALID_VALUE;
    const char *sep = NULL;
    for (size_t i = 8; i + 4 <= bytes; ++i) {
        if (img[i - 1] == '\n' && memcmp(img + i, "---\n", 4) == 0) {
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
    m->dev = d;
    unsigned long long src_len = 0, spv_len = 0, clc_len = 0;
    for (const char *l = m->meta; *l; l = strchr(l, '\n') ? strchr(l, '\n') + 1 : l + strlen(l)) {
        if (strncmp(l, "source ", 7) == 0) src_len = strtoull(l + 7, NULL, 10);
        if (strncmp(l, "spirv ", 6) == 0) spv_len = strtoull(l + 6, NULL, 10);
        if (strncmp(l, "clc ", 4) == 0) clc_len = strtoull(l + 4, NULL, 10);
    }
    const char *body = sep + 4;
    size_t body_len = bytes - (size_t)(body - img);
    if (src_len > body_len || spv_len > body_len - src_len || clc_len > body_len - src_len - spv_len) {
        free_module(m);
        return OCU_ERR_INVALID_VALUE;
    }
    ocu_status st = parse_globals(m);
    if (st != OCU_OK) {
        free_module(m);
        return st;
    }
    const ocl_api *a = ocl_load();
    cl_int err = CL_SUCCESS;
    const char *options = "";
    if (d->entry->il && spv_len > 0) {
        m->program = a->CreateProgramWithIL(d->ctx, body + src_len, spv_len, &err);
    } else if (d->entry->cxx && src_len > 0) {
        const char *src = body;
        size_t len = src_len;
        m->program = a->CreateProgramWithSource(d->ctx, 1, &src, &len, &err);
        options = "-cl-std=CLC++2021";
    } else if (clc_len > 0) {
        const char *src = body + src_len + spv_len;
        size_t len = clc_len;
        m->program = a->CreateProgramWithSource(d->ctx, 1, &src, &len, &err);
    } else {
        if (verbose()) {
            fprintf(stderr, "ocu opencl: the image has no form this device takes (%s)\n",
                    spv_len ? "SPIR-V, but the device has no SPIR-V input" : "no SPIR-V in the image");
        }
        free_module(m);
        return OCU_ERR_NOT_SUPPORTED;
    }
    if (m->program == NULL) {
        free_module(m);
        return map_err(err) == OCU_OK ? OCU_ERR_INVALID_VALUE : map_err(err);
    }
    err = a->BuildProgram(m->program, 1, &d->entry->device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        if (verbose()) {
            size_t n = 0;
            a->GetProgramBuildInfo(m->program, d->entry->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &n);
            char *log = malloc(n + 1);
            if (log != NULL) {
                log[0] = '\0';
                a->GetProgramBuildInfo(m->program, d->entry->device, CL_PROGRAM_BUILD_LOG, n, log, NULL);
                log[n] = '\0';
                fprintf(stderr, "ocu opencl: build failed (%d):\n%s\n", (int)err, log);
                free(log);
            }
        }
        a->ReleaseProgram(m->program);
        free_module(m);
        return OCU_ERR_INVALID_VALUE;
    }
    st = load_globals(m);
    if (st != OCU_OK) {
        if (m->glob != NULL) ocl_mem_free(d, m->glob);
        a->ReleaseProgram(m->program);
        free_module(m);
        return st;
    }
    pthread_mutex_init(&m->lock, NULL);
    *out = m;
    return OCU_OK;
}

static ocu_status ocl_module_unload(ocu_module *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    const ocl_api *a = ocl_load();
    finish_all(m->dev);
    ocu_status st = ocl_queue_sync(m->dev->sync_q);
    for (struct ocu_kernel *k = m->kernels; k != NULL;) {
        struct ocu_kernel *next = k->next;
        a->ReleaseKernel(k->k);
        pthread_mutex_destroy(&k->lock);
        free(k->name);
        free(k->param_sizes);
        free(k);
        k = next;
    }
    if (m->glob != NULL) ocl_mem_free(m->dev, m->glob);
    a->ReleaseProgram(m->program);
    pthread_mutex_destroy(&m->lock);
    free_module(m);
    return st;
}

static void parse_params(struct ocu_kernel *k, const char *meta)
{
    size_t len = strlen(k->name);
    for (const char *line = meta; line != NULL && *line;
         line = strchr(line, '\n') ? strchr(line, '\n') + 1 : NULL) {
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
        if (n) memcpy(k->param_sizes, sizes, n * sizeof(size_t));
        k->nparams = n;
        return;
    }
}

static ocu_status ocl_module_get_kernel(ocu_module *m, const char *name, ocu_kernel **out)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (name == NULL || out == NULL) return OCU_ERR_INVALID_VALUE;
    const ocl_api *a = ocl_load();
    pthread_mutex_lock(&m->lock);
    for (struct ocu_kernel *k = m->kernels; k != NULL; k = k->next) {
        if (strcmp(k->name, name) == 0) {
            *out = k;
            pthread_mutex_unlock(&m->lock);
            return OCU_OK;
        }
    }
    ocu_status st = OCU_OK;
    cl_int err = CL_SUCCESS;
    cl_kernel ck = a->CreateKernel(m->program, name, &err);
    struct ocu_kernel *k = ck ? calloc(1, sizeof(*k)) : NULL;
    char *copy = k ? strdup(name) : NULL;
    if (ck == NULL) {
        st = err == CL_INVALID_KERNEL_NAME ? OCU_ERR_NOT_FOUND : map_err(err);
        if (st == OCU_OK) st = OCU_ERR_NOT_FOUND;
    } else if (k == NULL || copy == NULL) {
        a->ReleaseKernel(ck);
        free(k);
        st = OCU_ERR_OUT_OF_MEMORY;
    } else {
        k->module = m;
        k->k = ck;
        k->name = copy;
        a->GetKernelWorkGroupInfo(ck, m->dev->entry->device, CL_KERNEL_WORK_GROUP_SIZE,
                                  sizeof(k->max_threads), &k->max_threads, NULL);
        cl_ulong lm = 0;
        a->GetKernelWorkGroupInfo(ck, m->dev->entry->device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lm), &lm, NULL);
        k->static_local = lm;
        cl_uint nargs = 0;
        a->GetKernelInfo(ck, CL_KERNEL_NUM_ARGS, sizeof(nargs), &nargs, NULL);
        k->takes_glob = nargs == 4;
        pthread_mutex_init(&k->lock, NULL);
        parse_params(k, m->meta);
        k->next = m->kernels;
        m->kernels = k;
        *out = k;
    }
    pthread_mutex_unlock(&m->lock);
    return st;
}

static ocu_status ocl_kernel_arg_sizes(ocu_kernel *k, uint32_t *count, size_t *sizes, uint32_t max)
{
    if (k == NULL) return OCU_ERR_INVALID_HANDLE;
    if (count == NULL || (max > 0 && sizes == NULL)) return OCU_ERR_INVALID_VALUE;
    *count = k->nparams;
    for (uint32_t i = 0; i < k->nparams && i < max; ++i) sizes[i] = k->param_sizes[i];
    return OCU_OK;
}

static ocu_status ocl_module_get_global(ocu_module *m, const char *name, void **ptr, size_t *bytes)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (name == NULL || ptr == NULL) return OCU_ERR_INVALID_VALUE;
    for (size_t i = 0; i < m->nglobals; ++i) {
        if (strcmp(m->globals[i].name, name) == 0) {
            *ptr = m->glob + m->globals[i].offset;
            if (bytes != NULL) *bytes = m->globals[i].size;
            return OCU_OK;
        }
    }
    return OCU_ERR_NOT_FOUND;
}

/* ---- execution ------------------------------------------------------------------------ */

/* Alignment of an argument value of `size` bytes: its largest power-of-two
   divisor, at most 16 (README.md, section 4). */
static size_t arg_align(size_t size)
{
    size_t a = 1;
    while (a < 16 && size % (a * 2) == 0) a *= 2;
    return a;
}

static ocu_status ocl_kernel_launch(ocu_queue *q, ocu_kernel *k, const ocu_launch_params *p)
{
    if (k == NULL) return OCU_ERR_INVALID_HANDLE;
    if (p == NULL || (p->num_args > 0 && (p->args == NULL || p->arg_sizes == NULL))) {
        return OCU_ERR_INVALID_VALUE;
    }
    ocu_module *m = k->module;
    ocu_device *d = m->dev;
    if (q != NULL && q->dev != d) return OCU_ERR_INVALID_HANDLE;
    ocu_queue *queue = q != NULL ? q : d->sync_q;
    for (int i = 0; i < 3; ++i) {
        if (p->grid[i] == 0 || p->block[i] == 0) return OCU_ERR_INVALID_VALUE;
        if (p->block[i] > d->entry->info.max_block_dim[i]) return OCU_ERR_INVALID_VALUE;
    }
    uint64_t threads = (uint64_t)p->block[0] * p->block[1] * p->block[2];
    if (threads > k->max_threads || threads > d->entry->info.max_threads_per_block) return OCU_ERR_INVALID_VALUE;
    if (p->shared_memory_bytes + k->static_local > d->entry->info.max_shared_memory_per_block) {
        return OCU_ERR_INVALID_VALUE;
    }
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
    const ocl_api *a = ocl_load();
    cl_int err = CL_SUCCESS, e2 = CL_SUCCESS;
    /* The driver keeps both buffers alive until the kernel is done. */
    cl_mem argbuf = a->CreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, total + 16, packed, &err);
    cl_mem offbuf = a->CreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    (p->num_args + 1) * sizeof(uint32_t), offsets, &e2);
    free(packed);
    free(offsets);
    if (argbuf == NULL || offbuf == NULL) {
        if (argbuf != NULL) a->ReleaseMemObject(argbuf);
        if (offbuf != NULL) a->ReleaseMemObject(offbuf);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    size_t local[3] = {p->block[0], p->block[1], p->block[2]};
    size_t global[3] = {(size_t)p->grid[0] * p->block[0], (size_t)p->grid[1] * p->block[1],
                        (size_t)p->grid[2] * p->block[2]};
    size_t dyn = (p->shared_memory_bytes + 15) / 16 * 16;
    cl_event ev = NULL;
    pthread_mutex_lock(&k->lock);
    cl_uint idx = 0;
    err = a->SetKernelArg(k->k, idx++, sizeof(cl_mem), &argbuf);
    if (err == CL_SUCCESS) err = a->SetKernelArg(k->k, idx++, sizeof(cl_mem), &offbuf);
    if (err == CL_SUCCESS && k->takes_glob) err = a->SetKernelArgSVMPointer(k->k, idx++, m->glob);
    if (err == CL_SUCCESS) err = a->SetKernelArg(k->k, idx++, dyn < 16 ? 16 : dyn, NULL);
    /* Every allocation may be reached through a pointer the kernel reads
       from its arguments or from memory. */
    if (err == CL_SUCCESS) {
        pthread_mutex_lock(&d->lock);
        size_t n = d->count;
        void **ptrs = malloc((n ? n : 1) * sizeof(void *));
        if (ptrs == NULL) {
            err = CL_OUT_OF_HOST_MEMORY;
        } else {
            for (size_t i = 0; i < n; ++i) ptrs[i] = d->allocs[i].ptr;
            if (n > 0) err = a->SetKernelExecInfo(k->k, CL_KERNEL_EXEC_INFO_SVM_PTRS, n * sizeof(void *), ptrs);
            free(ptrs);
        }
        pthread_mutex_unlock(&d->lock);
    }
    if (err == CL_SUCCESS) {
        err = a->EnqueueNDRangeKernel(queue->q, k->k, 3, NULL, global, local, 0, NULL, &ev);
    }
    pthread_mutex_unlock(&k->lock);
    a->ReleaseMemObject(argbuf);
    a->ReleaseMemObject(offbuf);
    if (err != CL_SUCCESS) {
        if (verbose()) fprintf(stderr, "ocu opencl: launch of %s failed (%d)\n", k->name, (int)err);
        return err == CL_INVALID_WORK_GROUP_SIZE ? OCU_ERR_INVALID_VALUE
               : (map_err(err) == OCU_ERR_UNKNOWN ? OCU_ERR_LAUNCH_FAILED : map_err(err));
    }
    ocu_status st = finish_enqueue(queue, q == NULL, CL_SUCCESS, ev);
    if (q != NULL && st != OCU_OK) {
        pthread_mutex_lock(&q->lock);
        q->sticky = st;
        pthread_mutex_unlock(&q->lock);
    }
    return st;
}

/* ---- markers ---------------------------------------------------------------------------- */

static ocu_status ocl_marker_create(ocu_device *d, ocu_marker **out)
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

static ocu_status ocl_marker_destroy(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    if (m->ev != NULL) ocl_load()->ReleaseEvent(m->ev);
    pthread_mutex_destroy(&m->lock);
    free(m);
    return OCU_OK;
}

/* A marker command completes after the queue's earlier work. */
static ocu_status ocl_marker_record(ocu_marker *m, ocu_queue *q)
{
    if (m == NULL || (q != NULL && q->dev != m->dev)) return OCU_ERR_INVALID_HANDLE;
    const ocl_api *a = ocl_load();
    ocu_queue *queue = q != NULL ? q : m->dev->sync_q;
    cl_event ev = NULL;
    cl_int err = a->EnqueueMarkerWithWaitList(queue->q, 0, NULL, &ev);
    if (err != CL_SUCCESS) return map_err(err);
    a->Flush(queue->q);
    a->RetainEvent(ev);
    set_last(queue, ev);
    pthread_mutex_lock(&m->lock);
    cl_event old = m->ev;
    m->ev = ev;
    pthread_mutex_unlock(&m->lock);
    if (old != NULL) a->ReleaseEvent(old);
    return OCU_OK;
}

static cl_event marker_event(ocu_marker *m)
{
    pthread_mutex_lock(&m->lock);
    cl_event ev = m->ev;
    if (ev != NULL) ocl_load()->RetainEvent(ev);
    pthread_mutex_unlock(&m->lock);
    return ev;
}

static ocu_status ocl_marker_wait(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    cl_event ev = marker_event(m);
    if (ev == NULL) return OCU_OK;
    cl_int err = ocl_load()->WaitForEvents(1, &ev);
    ocl_load()->ReleaseEvent(ev);
    return err == CL_SUCCESS ? OCU_OK : OCU_ERR_LAUNCH_FAILED;
}

static ocu_status ocl_marker_query(ocu_marker *m)
{
    if (m == NULL) return OCU_ERR_INVALID_HANDLE;
    cl_event ev = marker_event(m);
    if (ev == NULL) return OCU_OK;
    cl_int s = event_status(ev);
    ocl_load()->ReleaseEvent(ev);
    if (s < 0) return OCU_ERR_LAUNCH_FAILED;
    return s == CL_COMPLETE ? OCU_OK : OCU_ERR_NOT_READY;
}

static ocu_status ocl_marker_elapsed_ms(ocu_marker *a, ocu_marker *b, float *ms)
{
    if (a == NULL || b == NULL) return OCU_ERR_INVALID_HANDLE;
    if (ms == NULL) return OCU_ERR_INVALID_VALUE;
    const ocl_api *api_ = ocl_load();
    cl_event ea = marker_event(a), eb = marker_event(b);
    ocu_status st = OCU_OK;
    if (ea == NULL || eb == NULL) {
        st = OCU_ERR_INVALID_HANDLE;
    } else if (event_status(ea) != CL_COMPLETE || event_status(eb) != CL_COMPLETE) {
        st = OCU_ERR_NOT_READY;
    } else {
        cl_ulong ta = 0, tb = 0;
        cl_int e1 = api_->GetEventProfilingInfo(ea, CL_PROFILING_COMMAND_END, sizeof(ta), &ta, NULL);
        cl_int e2 = api_->GetEventProfilingInfo(eb, CL_PROFILING_COMMAND_END, sizeof(tb), &tb, NULL);
        if (e1 != CL_SUCCESS || e2 != CL_SUCCESS) {
            st = OCU_ERR_NOT_SUPPORTED; /* the driver keeps no times for markers */
        } else {
            *ms = (float)((double)(int64_t)(tb - ta) / 1e6);
        }
    }
    if (ea != NULL) api_->ReleaseEvent(ea);
    if (eb != NULL) api_->ReleaseEvent(eb);
    return st;
}

static const void *ocl_get_provider(const char *library)
{
    (void)library;
    return NULL;
}

static const ocu_backend g_opencl = {
    .abi = OCU_BACKEND_ABI,
    .name = "opencl",
    .dev_count = ocl_dev_count,
    .dev_info = ocl_dev_info,
    .dev_open = ocl_dev_open,
    .dev_close = ocl_dev_close,
    .mem_alloc = ocl_mem_alloc,
    .mem_free = ocl_mem_free,
    .mem_copy = ocl_mem_copy,
    .mem_set = ocl_mem_set,
    .mem_info = ocl_mem_info,
    .host_alloc_pinned = ocl_host_alloc_pinned,
    .host_free_pinned = ocl_host_free_pinned,
    .queue_create = ocl_queue_create,
    .queue_destroy = ocl_queue_destroy,
    .queue_sync = ocl_queue_sync,
    .queue_query = ocl_queue_query,
    .module_load = ocl_module_load,
    .module_unload = ocl_module_unload,
    .module_get_kernel = ocl_module_get_kernel,
    .kernel_launch = ocl_kernel_launch,
    .marker_create = ocl_marker_create,
    .marker_destroy = ocl_marker_destroy,
    .marker_record = ocl_marker_record,
    .marker_wait = ocl_marker_wait,
    .marker_query = ocl_marker_query,
    .marker_elapsed_ms = ocl_marker_elapsed_ms,
    .get_provider = ocl_get_provider,
    .module_get_global = ocl_module_get_global,
    .kernel_arg_sizes = ocl_kernel_arg_sizes,
    .mem_alloc_shared = ocl_mem_alloc_shared,
    .host_register = NULL,
    .host_unregister = NULL,
};

const ocu_backend *ocu_backend_opencl(void) { return &g_opencl; }

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
#include "ocu_ops.h"
#include "ocl_api.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- objects ---------------------------------------------------------------- */

/* OCU_OPENCL_HOSTPROF=1: host time spent in each backend entry, printed at exit. */
#include <time.h>
enum { HP_LAUNCH, HP_ARGS, HP_SVM, HP_ENQ, HP_FIN, HP_COPY, HP_SET, HP_SYNC, HP_MARK, HP_ALLOC, HP_FREE, HP_N };
static const char *hp_names[HP_N] = {"launch(total)", " .args", " .svm_ptrs", " .enqueue", " .finish", "mem_copy", "mem_set", "queue_sync", "marker", "alloc", "free"};
static double hp_t[HP_N]; static unsigned long hp_c[HP_N]; static double hp_first, hp_last;
static int hp_on_v = -1;
static inline double hp_now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e6+ts.tv_nsec*1e-3; }
static void hp_report(void){ fprintf(stderr,"ocu opencl hostprof: span %.1f ms\n",(hp_last-hp_first)/1e3);
  for(int i=0;i<HP_N;i++) if(hp_c[i]) fprintf(stderr,"  %-14s %8lu x %8.2f us = %9.1f ms\n",hp_names[i],hp_c[i],hp_t[i]/hp_c[i],hp_t[i]/1e3); }
static inline int hp_on(void){ if(hp_on_v<0){ const char*e=getenv("OCU_OPENCL_HOSTPROF"); hp_on_v=e&&*e=='1'; if(hp_on_v) atexit(hp_report);} return hp_on_v; }
#define HP_B(id) double hp_s_##id = hp_on() ? hp_now() : 0
#define HP_E(id) do{ if(hp_on_v>0){ double e_=hp_now(); hp_t[id]+=e_-hp_s_##id; hp_c[id]++; if(!hp_first) hp_first=hp_s_##id; hp_last=e_; } }while(0)

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
    uint64_t          gen;      /* bumped when the allocation table changes */
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
    bool               takes_glob;   /* the variables block is a parameter */
    size_t             byval;        /* >0: packed arguments passed by value, this many bytes */
    uint64_t           svm_gen;      /* allocation table generation last given to the kernel */
    int                indirect;     /* 1: Intel indirect-access flags set (no SVM list), -1: not available */
    bool               direct;       /* compiled SPIR-V: one OpenCL argument per CUDA parameter */
    bool               dyn;          /* direct: one more argument, the launch's dynamic shared memory */
    char              *ptr_param;    /* direct: per parameter, 1 = global pointer (set as SVM pointer) */
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
    bool               direct;    /* "OCUSPV1": compiled SPIR-V (compiler/spirv), CUDA parameters as they are */
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
static ocu_status ocl_kernel_launch(ocu_queue *q, ocu_kernel *k, const ocu_launch_params *p);
static ocu_status ocl_mem_copy(ocu_device *d, ocu_queue *q, void *dst, const void *src, size_t bytes, ocu_copy_dir dir);
static ocu_status ocl_mem_set(ocu_device *d, ocu_queue *q, void *dst, int value, size_t bytes);
static ocu_status ocl_queue_sync(ocu_queue *q);
static ocu_status ocl_marker_record(ocu_marker *m, ocu_queue *q);
static ocu_status ocl_mem_alloc(ocu_device *d, size_t bytes, void **ptr);
static ocu_status ocl_mem_free(ocu_device *d, void *ptr);
static int trace_on(void);
static void trace_add_name(const char *name, cl_event ev);
static ocu_status ocl_queue_sync_impl(ocu_queue *q);

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

static ocu_status ocl_queue_sync_impl(ocu_queue *q)
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
        d->gen++;
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
            d->gen++;
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

static ocu_status ocl_mem_alloc_impl(ocu_device *d, size_t bytes, void **ptr)
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

static ocu_status ocl_mem_free_impl(ocu_device *d, void *ptr)
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
static int ops_enabled(ocu_device *d);
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

static ocu_status ocl_mem_copy_impl(ocu_device *d, ocu_queue *q, void *dst, const void *src, size_t bytes,
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
    if (err == CL_SUCCESS && trace_on()) trace_add_name(dir == OCU_COPY_HOST_TO_DEVICE ? (bytes < 4096 ? "[memcpy H2D <4K]" : bytes < 65536 ? "[memcpy H2D <64K]" : bytes < (1 << 20) ? "[memcpy H2D <1M]" : bytes < (16 << 20) ? "[memcpy H2D <16M]" : "[memcpy H2D >=16M]") : dir == OCU_COPY_DEVICE_TO_HOST ? "[memcpy D2H]" : "[memcpy D2D]", ev);
    return finish_enqueue(queue, q == NULL, err, ev);
}

static ocu_status ocl_mem_set_impl(ocu_device *d, ocu_queue *q, void *dst, int value, size_t bytes)
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
    const bool direct = memcmp(img, "OCUSPV1\n", 8) == 0;
    if (!direct && memcmp(img, "OCUOCL1\n", 8) != 0) return OCU_ERR_INVALID_VALUE;
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
    m->direct = direct;
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
    /* OCU_OPENCL_BUILD_OPTIONS: extra options for the driver's compiler (for
       example -cl-intel-enable-auto-large-GRF-mode on Intel GPUs). */
    char opts[1024];
    const char *extra = getenv("OCU_OPENCL_BUILD_OPTIONS");
    if (extra == NULL && ops_enabled(d)) extra = "-cl-intel-enable-auto-large-GRF-mode"; /* Intel GPUs */
    snprintf(opts, sizeof(opts), "%s%s%s", options ? options : "", extra && *extra ? " " : "", extra ? extra : "");
    err = a->BuildProgram(m->program, 1, &d->entry->device, opts, NULL, NULL);
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

/* "sparam <name> <p8|vN>... [dyn]" of a compiled SPIR-V module (compiler/spirv/cu2spv.py) */
static void parse_sparams(struct ocu_kernel *k, const char *meta)
{
    size_t len = strlen(k->name);
    for (const char *line = meta; line != NULL && *line;
         line = strchr(line, '\n') ? strchr(line, '\n') + 1 : NULL) {
        if (strncmp(line, "sparam ", 7) != 0 || strncmp(line + 7, k->name, len) != 0 ||
            (line[7 + len] != ' ' && line[7 + len] != '\n' && line[7 + len] != '\0'))
            continue;
        const char *p = line + 7 + len;
        size_t sizes[256];
        char ptr[256];
        uint32_t n = 0;
        while (*p == ' ' && n < 256) {
            ++p;
            if (strncmp(p, "dyn", 3) == 0) { k->dyn = true; p += 3; continue; }
            if (*p != 'p' && *p != 'v') break;
            ptr[n] = *p == 'p';
            char *end;
            sizes[n] = strtoul(p + 1, &end, 10);
            p = end;
            ++n;
        }
        k->param_sizes = n ? malloc(n * sizeof(size_t)) : NULL;
        k->ptr_param = n ? malloc(n) : NULL;
        if (n && (k->param_sizes == NULL || k->ptr_param == NULL)) return;
        if (n) { memcpy(k->param_sizes, sizes, n * sizeof(size_t)); memcpy(k->ptr_param, ptr, n); }
        k->nparams = n;
        k->direct = true;
        return;
    }
}

static void parse_params(struct ocu_kernel *k, const char *meta)
{
    size_t len = strlen(k->name);
    for (const char *line = meta; line != NULL && *line;
         line = strchr(line, '\n') ? strchr(line, '\n') + 1 : NULL) {
        if (strncmp(line, "byval ", 6) == 0 && strncmp(line + 6, k->name, len) == 0 && line[6 + len] == ' ') {
            k->byval = strtoul(line + 7 + len, NULL, 10);
            continue;
        }
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
        pthread_mutex_init(&k->lock, NULL);
        if (m->direct) parse_sparams(k, m->meta); else parse_params(k, m->meta);
        k->takes_glob = !k->direct && (k->byval ? nargs == 3 : nargs == 4);
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

/* Every allocation may be reached through a pointer the kernel reads from
   its arguments or from memory: the kernel gets the whole table, again only
   when it changed since this kernel last got it. Called with k->lock held. */
/* Intel (cl_intel_unified_shared_memory): the kernel may reach any allocation
   through pointers in memory. One flag per kernel instead of the list of every
   allocation, which the driver otherwise makes resident on each launch. */
static int indirect_flags(cl_kernel k)
{
    static int mode = -1;
    if (mode < 0) { const char *e = getenv("OCU_OPENCL_INDIRECT"); mode = e == NULL || *e != '0'; }
    if (!mode) return 0;
    const cl_bool yes = CL_TRUE;
    const ocl_api *a = ocl_load();
    return a->SetKernelExecInfo(k, 0x4200 /* INDIRECT_HOST_ACCESS_INTEL */, sizeof yes, &yes) == CL_SUCCESS &&
           a->SetKernelExecInfo(k, 0x4201 /* INDIRECT_DEVICE_ACCESS_INTEL */, sizeof yes, &yes) == CL_SUCCESS &&
           a->SetKernelExecInfo(k, 0x4202 /* INDIRECT_SHARED_ACCESS_INTEL */, sizeof yes, &yes) == CL_SUCCESS;
}

static cl_int set_svm_ptrs(ocu_kernel *k, ocu_device *d)
{
    const ocl_api *a = ocl_load();
    cl_int err = CL_SUCCESS;
    if (k->indirect == 0) k->indirect = indirect_flags(k->k) ? 1 : -1;
    if (k->indirect > 0) return CL_SUCCESS;
    pthread_mutex_lock(&d->lock);
    if (k->svm_gen != d->gen || d->gen == 0) {
        size_t n = d->count;
        void **ptrs = malloc((n ? n : 1) * sizeof(void *));
        if (ptrs == NULL) {
            err = CL_OUT_OF_HOST_MEMORY;
        } else {
            for (size_t i = 0; i < n; ++i) ptrs[i] = d->allocs[i].ptr;
            if (n > 0) err = a->SetKernelExecInfo(k->k, CL_KERNEL_EXEC_INFO_SVM_PTRS, n * sizeof(void *), ptrs);
            free(ptrs);
            if (err == CL_SUCCESS) k->svm_gen = d->gen;
        }
    }
    pthread_mutex_unlock(&d->lock);
    return err;
}

/* OCU_OPENCL_TRACE=1: GPU time per kernel name, printed at exit (the queues
   keep profiling on for markers already). */
typedef struct { const char *name; cl_event ev; } trace_rec;
static trace_rec *trace_v;
static size_t trace_n, trace_cap;
static pthread_mutex_t trace_lock = PTHREAD_MUTEX_INITIALIZER;
static int trace_on(void)
{
    static int on = -1;
    if (on < 0) { const char *e = getenv("OCU_OPENCL_TRACE"); on = e != NULL && *e == '1'; }
    return on;
}
static void trace_report(void)
{
    const ocl_api *a = ocl_load();
    typedef struct { const char *name; double us; size_t n; } agg;
    agg *g = calloc(trace_n + 1, sizeof(agg));
    size_t ng = 0;
    double total = 0, gap_small = 0, gap_mid = 0, gap_big = 0;
    size_t n_small = 0, n_mid = 0, n_big = 0;
    cl_ulong prev_end = 0, first_start = 0;
    for (size_t i = 0; i < trace_n; ++i) {
        cl_ulong t0 = 0, t1 = 0;
        a->WaitForEvents(1, &trace_v[i].ev);
        a->GetEventProfilingInfo(trace_v[i].ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
        a->GetEventProfilingInfo(trace_v[i].ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
        double us = (t1 - t0) / 1e3;
        total += us;
        if (i > 0 && prev_end && t0 >= prev_end) {
            double gap = (t0 - prev_end) / 1e3;
            if (gap < 50) { gap_small += gap; n_small++; } else if (gap < 1000) { gap_mid += gap; n_mid++; } else { gap_big += gap; n_big++; }
        }
        if (t1 > prev_end) prev_end = t1;
        if (i == 0) first_start = t0;
        size_t j = 0;
        while (j < ng && strcmp(g[j].name, trace_v[i].name) != 0) ++j;
        if (j == ng) { g[ng].name = trace_v[i].name; ++ng; }
        g[j].us += us;
        g[j].n++;
    }
    for (size_t i = 0; i < ng; ++i)
        for (size_t j = i + 1; j < ng; ++j)
            if (g[j].us > g[i].us) { agg t = g[i]; g[i] = g[j]; g[j] = t; }
    fprintf(stderr, "ocu opencl trace: %zu launches, %.1f ms of kernel time\n", trace_n, total / 1e3);
    fprintf(stderr, "ocu opencl trace: GPU span %.1f ms; gaps <50us: %zu x %.2f us = %.1f ms, 50us-1ms: %zu = %.1f ms, >1ms: %zu = %.1f ms\n",
            (prev_end - first_start) / 1e6, n_small, n_small ? gap_small / n_small : 0, gap_small / 1e3, n_mid, gap_mid / 1e3, n_big, gap_big / 1e3);
    for (size_t i = 0; i < ng && i < 25; ++i)
        fprintf(stderr, "  %6.2f%%  %9.1f us  %6zu x  %8.1f us/launch  %s\n", 100 * g[i].us / total, g[i].us, g[i].n,
                g[i].us / g[i].n, g[i].name);
    free(g);
}
static void trace_add_name(const char *name, cl_event ev);
static void trace_add(ocu_kernel *k, cl_event ev) { trace_add_name(k->name, ev); }
static void trace_add_name(const char *name, cl_event ev)
{
    if (ev == NULL) return;
    pthread_mutex_lock(&trace_lock);
    if (trace_v == NULL) atexit(trace_report);
    if (trace_n == trace_cap) {
        trace_cap = trace_cap ? trace_cap * 2 : 4096;
        trace_v = realloc(trace_v, trace_cap * sizeof(trace_rec));
    }
    ocl_load()->RetainEvent(ev);
    trace_v[trace_n].name = name;
    trace_v[trace_n].ev = ev;
    trace_n++;
    pthread_mutex_unlock(&trace_lock);
}

static ocu_status finish_launch(ocu_queue *queue, ocu_queue *q, ocu_kernel *k, cl_int err, cl_event ev)
{
    if (err == CL_SUCCESS && trace_on()) trace_add(k, ev);
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

/* Compiled SPIR-V: each CUDA parameter is an OpenCL kernel argument; global
   pointers go as SVM pointers, everything else by value; the launch's dynamic
   shared memory is one more (local) argument. */
static ocu_status launch_direct(ocu_queue *queue, ocu_queue *q, ocu_kernel *k, const ocu_launch_params *p)
{
    ocu_device *d = k->module->dev;
    if (p->num_args != k->nparams) return OCU_ERR_INVALID_VALUE;
    const ocl_api *a = ocl_load();
    size_t global[3] = {(size_t)p->grid[0] * p->block[0], (size_t)p->grid[1] * p->block[1],
                        (size_t)p->grid[2] * p->block[2]};
    size_t local[3] = {p->block[0], p->block[1], p->block[2]};
    size_t dyn = (p->shared_memory_bytes + 15) / 16 * 16;
    cl_event ev = NULL;
    cl_int err = CL_SUCCESS;
    pthread_mutex_lock(&k->lock);
    HP_B(HP_ARGS);
    for (uint32_t i = 0; i < p->num_args && err == CL_SUCCESS; ++i) {
        if (k->ptr_param[i]) {
            void *ptr = NULL;
            memcpy(&ptr, p->args[i], sizeof ptr);
            err = a->SetKernelArgSVMPointer(k->k, i, ptr);
        } else {
            err = a->SetKernelArg(k->k, i, k->param_sizes[i], p->args[i]);
        }
    }
    if (err == CL_SUCCESS && k->dyn) err = a->SetKernelArg(k->k, p->num_args, dyn < 16 ? 16 : dyn, NULL);
    HP_E(HP_ARGS); HP_B(HP_SVM);
    if (err == CL_SUCCESS) err = set_svm_ptrs(k, d);
    HP_E(HP_SVM); HP_B(HP_ENQ);
    if (err == CL_SUCCESS) err = a->EnqueueNDRangeKernel(queue->q, k->k, 3, NULL, global, local, 0, NULL, &ev);
    HP_E(HP_ENQ);
    pthread_mutex_unlock(&k->lock);
    HP_B(HP_FIN);
    ocu_status fst = finish_launch(queue, q, k, err, ev);
    HP_E(HP_FIN);
    return fst;
}

/* Arguments by value: the packed values are one kernel argument (the
   translator fixed the offsets), no buffer per launch. */
static ocu_status launch_byval(ocu_queue *queue, ocu_queue *q, ocu_kernel *k, const ocu_launch_params *p,
                               size_t total)
{
    ocu_module *m = k->module;
    ocu_device *d = m->dev;
    if (total > k->byval) return OCU_ERR_INVALID_VALUE;
    unsigned char packed[1024 + 16] __attribute__((aligned(16)));
    if (k->byval > sizeof(packed)) return OCU_ERR_INVALID_VALUE;
    memset(packed, 0, k->byval);
    size_t at = 0;
    for (uint32_t i = 0; i < p->num_args; ++i) {
        size_t a = arg_align(p->arg_sizes[i]);
        at = (at + a - 1) / a * a;
        memcpy(packed + at, p->args[i], p->arg_sizes[i]);
        at += p->arg_sizes[i];
    }
    const ocl_api *a = ocl_load();
    size_t local[3] = {p->block[0], p->block[1], p->block[2]};
    size_t global[3] = {(size_t)p->grid[0] * p->block[0], (size_t)p->grid[1] * p->block[1],
                        (size_t)p->grid[2] * p->block[2]};
    size_t dyn = (p->shared_memory_bytes + 15) / 16 * 16;
    cl_event ev = NULL;
    pthread_mutex_lock(&k->lock);
    cl_uint idx = 0;
    cl_int err = a->SetKernelArg(k->k, idx++, k->byval, packed);
    if (err == CL_SUCCESS && k->takes_glob) err = a->SetKernelArgSVMPointer(k->k, idx++, m->glob);
    if (err == CL_SUCCESS) err = a->SetKernelArg(k->k, idx++, dyn < 16 ? 16 : dyn, NULL);
    if (err == CL_SUCCESS) err = set_svm_ptrs(k, d);
    if (err == CL_SUCCESS) err = a->EnqueueNDRangeKernel(queue->q, k->k, 3, NULL, global, local, 0, NULL, &ev);
    pthread_mutex_unlock(&k->lock);
    return finish_launch(queue, q, k, err, ev);
}

static ocu_status ocl_kernel_launch_impl(ocu_queue *q, ocu_kernel *k, const ocu_launch_params *p)
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
    if (k->direct) return launch_direct(queue, q, k, p);
    if (k->byval) return launch_byval(queue, q, k, p, total);
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
        err = set_svm_ptrs(k, d);
    }
    if (0) {
        size_t n = 0;
        void **ptrs = NULL;
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
    return finish_launch(queue, q, k, err, ev);
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
static ocu_status ocl_marker_record_impl(ocu_marker *m, ocu_queue *q)
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

/* ---- operator table: GEMM on Intel GPUs ----------------------------------------------
 * libs/cublas sends GEMMs here first (gemm_by_op). F32 GEMM runs a kernel
 * written for Intel's sub-group extensions (block reads from local memory,
 * sub-group size 16, 256 registers per thread on Xe2): 128 x 128 tiles of C
 * per work-group of 256, 8 x 8 results per work-item, the next tile read
 * into registers while the current one is used. Devices without those
 * extensions (or OCU_OPENCL_OPS=0) get no table, and the CUDA kernels run.
 * Measured on an Arc Pro B70 (sgemm 4096^3): 10.2 to 11.8 TFLOP/s for the
 * four transpose cases, against 3.4 for the translated CUDA kernel. */
static const char k_sgemm_src[] =
    "// v2: SLM tiles + sub-group block reads from SLM + register prefetch of the next tile. Sub-group 16.\n"
    "#ifndef BM\n"
    "#define BM 128\n"
    "#endif\n"
    "#ifndef BN\n"
    "#define BN 128\n"
    "#endif\n"
    "#ifndef BK\n"
    "#define BK 32\n"
    "#endif\n"
    "#define WX 16\n"
    "#define WY 16\n"
    "#define NT 256\n"
    "#define TM (BM / WY)\n"
    "#define TN (BN / WX)\n"
    "#define LA (BM * BK / NT)\n"
    "#define LB (BN * BK / NT)\n"
    "#define A_AT(m, k) (TA ? A[(long)(k) * lda + (m)] : A[(long)(m) * lda + (k)])\n"
    "#define B_AT(k, n) (TB ? B[(long)(n) * ldb + (k)] : B[(long)(k) * ldb + (n)])\n"
    "#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n"
    "#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable\n"
    "\n"
    "__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(WX, WY, 1)))\n"
    "kernel void KNAME(int M, int N, int K, float alpha, float beta,\n"
    "                  global const float *A0, long lda, long sa,\n"
    "                  global const float *B0, long ldb, long sb,\n"
    "                  global float *C0, long ldc, long sc, int S, int kc)\n"
    "{\n"
    "    local float As[BK][BM];\n"
    "    local float Bs[BK][BN];\n"
    "    const int lx = get_local_id(0), ly = get_local_id(1), tid = ly * WX + lx;\n"
    "    const int n0 = get_group_id(0) * BN, m0 = get_group_id(1) * BM;\n"
    "    // z = batch * S + split: split s covers k in [s*kc, min(K, (s+1)*kc)); with S > 1\n"
    "    // the caller passes the partial-sum buffer as C (alpha 1, beta 0).\n"
    "    const long bz = get_group_id(2) / S;\n"
    "    const int ks = (int)(get_group_id(2) % S);\n"
    "    const int kbeg = ks * kc, kend = min(K, kbeg + kc);\n"
    "    const long cz = get_group_id(2);\n"
    "#ifndef PARR\n"
    "#define PARR 0\n"
    "#endif\n"
    "#if PARR\n"
    "    global const float *A = ((global const float *global const *)A0)[bz];\n"
    "    global const float *B = ((global const float *global const *)B0)[bz];\n"
    "    global float *C = ((global float *global const *)C0)[bz];\n"
    "#else\n"
    "    global const float *A = A0 + bz * sa;\n"
    "    global const float *B = B0 + bz * sb;\n"
    "    global float *C = C0 + (S > 1 ? cz : bz) * sc;\n"
    "#endif\n"
    "    float acc[TM][TN];\n"
    "    for (int i = 0; i < TM; ++i)\n"
    "        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;\n"
    "    float ra[LA], rb[LB];\n"
    "#define FETCH(k0)                                                                         \\\n"
    "    for (int r = 0; r < LA; ++r) {                                                        \\\n"
    "        const int idx = tid + r * NT;                                                     \\\n"
    "        int mm, kk;                                                                       \\\n"
    "        if (TA) { mm = idx % BM; kk = idx / BM; } else { kk = idx % BK; mm = idx / BK; } \\\n"
    "        const int gm = m0 + mm, gk = (k0) + kk;                                           \\\n"
    "        ra[r] = (gm < M && gk < kend) ? A_AT(gm, gk) : 0.0f;                                 \\\n"
    "    }                                                                                     \\\n"
    "    for (int r = 0; r < LB; ++r) {                                                        \\\n"
    "        const int idx = tid + r * NT;                                                     \\\n"
    "        int nn, kk;                                                                       \\\n"
    "        if (TB) { kk = idx % BK; nn = idx / BK; } else { nn = idx % BN; kk = idx / BN; } \\\n"
    "        const int gn = n0 + nn, gk = (k0) + kk;                                           \\\n"
    "        rb[r] = (gn < N && gk < kend) ? B_AT(gk, gn) : 0.0f;                                 \\\n"
    "    }\n"
    "    FETCH(kbeg)\n"
    "    for (int k0 = kbeg; k0 < kend; k0 += BK) {\n"
    "        for (int r = 0; r < LA; ++r) {\n"
    "            const int idx = tid + r * NT;\n"
    "            int mm, kk;\n"
    "            if (TA) { mm = idx % BM; kk = idx / BM; } else { kk = idx % BK; mm = idx / BK; }\n"
    "            As[kk][mm] = ra[r];\n"
    "        }\n"
    "        for (int r = 0; r < LB; ++r) {\n"
    "            const int idx = tid + r * NT;\n"
    "            int nn, kk;\n"
    "            if (TB) { kk = idx % BK; nn = idx / BK; } else { nn = idx % BN; kk = idx / BN; }\n"
    "            Bs[kk][nn] = rb[r];\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        if (k0 + BK < kend) { FETCH(k0 + BK) }\n"
    "        #pragma unroll\n"
    "        for (int kk = 0; kk < BK; ++kk) {\n"
    "            // b[j] = Bs[kk][lx + 16 j]: one block read gives lane lx its 8 values\n"
    "#if TN == 8\n"
    "            const float8 b = as_float8(intel_sub_group_block_read8((local const uint *)&Bs[kk][0]));\n"
    "            const float bv[8] = {b.s0, b.s1, b.s2, b.s3, b.s4, b.s5, b.s6, b.s7};\n"
    "#elif TN == 4\n"
    "            const float4 b = as_float4(intel_sub_group_block_read4((local const uint *)&Bs[kk][0]));\n"
    "            const float bv[4] = {b.s0, b.s1, b.s2, b.s3};\n"
    "#elif TN == 2\n"
    "            const float2 b = as_float2(intel_sub_group_block_read2((local const uint *)&Bs[kk][0]));\n"
    "            const float bv[2] = {b.s0, b.s1};\n"
    "#else\n"
    "            const float bv[1] = {as_float(intel_sub_group_block_read((local const uint *)&Bs[kk][0]))};\n"
    "#endif\n"
    "#if TM == 8\n"
    "            const float8 a = vload8(0, &As[kk][ly * TM]);  // same values in the whole sub-group\n"
    "            const float av[8] = {a.s0, a.s1, a.s2, a.s3, a.s4, a.s5, a.s6, a.s7};\n"
    "#elif TM == 4\n"
    "            const float4 a = vload4(0, &As[kk][ly * TM]);\n"
    "            const float av[4] = {a.s0, a.s1, a.s2, a.s3};\n"
    "#elif TM == 2\n"
    "            const float2 a = vload2(0, &As[kk][ly * TM]);\n"
    "            const float av[2] = {a.s0, a.s1};\n"
    "#else\n"
    "            const float av[1] = {As[kk][ly]};\n"
    "#endif\n"
    "            #pragma unroll\n"
    "            for (int i = 0; i < TM; ++i)\n"
    "                #pragma unroll\n"
    "                for (int j = 0; j < TN; ++j) acc[i][j] = fma(av[i], bv[j], acc[i][j]);\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    for (int i = 0; i < TM; ++i) {\n"
    "        const int gm = m0 + ly * TM + i;\n"
    "        if (gm >= M) break;\n"
    "        for (int j = 0; j < TN; ++j) {\n"
    "            const int gn = n0 + lx + j * WX;\n"
    "            if (gn < N) {\n"
    "                global float *o = C + (long)gm * ldc + gn;\n"
    "                float v = alpha * acc[i][j];\n"
    "                if (beta != 0.0f) v = fma(beta, *o, v);\n"
    "                *o = v;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "// Split-K: C = alpha * sum_s P[s] + beta * C, the sum in split order (deterministic).\n"
    "kernel void ocu_sgemm_reduce(int M, int N, int S, float alpha, float beta, global const float *P,\n"
    "                         global float *C, long ldc)\n"
    "{\n"
    "    const int n = get_global_id(0), m = get_global_id(1);\n"
    "    if (m >= M || n >= N) return;\n"
    "    const long mn = (long)M * N, at = (long)m * N + n;\n"
    "    float v = 0.0f;\n"
    "    for (int s = 0; s < S; ++s) v += P[s * mn + at];\n"
    "    global float *o = C + (long)m * ldc + n;\n"
    "    v *= alpha;\n"
    "    if (beta != 0.0f) v = fma(beta, *o, v);\n"
    "    *o = v;\n"
    "}\n";

/* F16 / BF16 inputs: the XMX engines through cl_intel_subgroup_matrix_multiply_accumulate
   (8x16x16 per instruction, sub-group 16), f32 accumulation, output f32 / f16 / bf16.
   Measured on an Arc Pro B70 at 4096^3: 18 to 38 TFLOP/s by transpose case, against
   about 3 for the translated CUDA kernel. */
static const char k_hgemm_src[] =
    "// XMX (DPAS) GEMM: row-major C[MxN] (f32) = alpha * op(A)[MxK] * op(B)[KxN] + beta * C, A/B f16 or bf16.\n"
    "// TA=0: A stored [M x K]; TA=1: [K x M].  TB=0: B stored [K x N]; TB=1: [N x K].  BF16=1: bfloat16 inputs.\n"
    "#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n"
    "#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable\n"
    "#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable\n"
    "#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable\n"
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "#ifndef OUT\n"
    "#define OUT 0\n"
    "#endif\n"
    "#define BM 128\n"
    "#define BN 128\n"
    "#define BK 32\n"
    "#define SGM 4                 // sub-groups along M (each 32 rows)\n"
    "#define SGN 2                 // sub-groups along N (each 64 cols)\n"
    "#define NT (16 * SGM * SGN)   // 128 work-items\n"
    "#if BF16\n"
    "#define MAD intel_sub_group_bf16_bf16_matrix_mad_k16\n"
    "#else\n"
    "#define MAD intel_sub_group_f16_f16_matrix_mad_k16\n"
    "#endif\n"
    "#define A_AT(m, k) (TA ? A[(long)(k) * lda + (m)] : A[(long)(m) * lda + (k)])\n"
    "#define B_AT(k, n) (TB ? B[(long)(n) * ldb + (k)] : B[(long)(k) * ldb + (n)])\n"
    "\n"
    "__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(NT, 1, 1)))\n"
    "kernel void KNAME(int M, int N, int K, float alpha, float beta,\n"
    "                  global const ushort *A0, long lda, long sa,\n"
    "                  global const ushort *B0, long ldb, long sb,\n"
    "                  global void *C0v, long ldc, long sc)\n"
    "{\n"
    "    // As: [BK/16][BM][16] halves (8 rows of one k-block are one 8x16 block read)\n"
    "    // Bv: [BN/16][BK/2][16] ints, VNNI: (B[2p][n], B[2p+1][n]) in one int\n"
    "    local ushort As[BK / 16][BM][16];\n"
    "    local uint Bv[BN / 16][BK / 2][16];\n"
    "    const int tid = get_local_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();\n"
    "    const int sgm = sg % SGM, sgn = sg / SGM;\n"
    "    const int n0 = get_group_id(0) * BN, m0 = get_group_id(1) * BM;\n"
    "    const long bz = get_group_id(2);\n"
    "#if PARR\n"
    "    global const ushort *A = ((global const ushort *global const *)A0)[bz];\n"
    "    global const ushort *B = ((global const ushort *global const *)B0)[bz];\n"
    "#else\n"
    "    global const ushort *A = A0 + bz * sa;\n"
    "    global const ushort *B = B0 + bz * sb;\n"
    "#endif\n"
    "    // OUT: 0 f32, 1 f16, 2 bf16 (round to nearest even)\n"
    "#if PARR\n"
    "    global void *Cb = ((global void *global const *)C0v)[bz];\n"
    "#else\n"
    "    global void *Cb = C0v;\n"
    "#endif\n"
    "#if OUT == 0\n"
    "    global float *C = (global float *)Cb + (PARR ? 0 : bz * sc);\n"
    "#else\n"
    "    global ushort *C = (global ushort *)Cb + (PARR ? 0 : bz * sc);\n"
    "#endif\n"
    "    float8 acc[4][4];\n"
    "    for (int i = 0; i < 4; ++i)\n"
    "        for (int j = 0; j < 4; ++j) acc[i][j] = (float8)0.0f;\n"
    "\n"
    "    // Global -> registers (8 values per load, 16 bytes) -> local memory. Each work-item moves\n"
    "    // 4 groups of 8 A values and 2 x (8 B values of two k rows) per k-step of 32.\n"
    "    ushort8 ra[4];\n"
    "    ushort8 rb[4];\n"
    "#define LOAD_A(k0)                                                                              \\\n"
    "    for (int r = 0; r < 4; ++r) {                                                               \\\n"
    "        const int idx = tid + r * NT; /* 512 groups of 8 */                                     \\\n"
    "        int mm, kk; /* first element; the 8 run along k (TA=0) or along m (TA=1) */             \\\n"
    "        if (TA) { mm = (idx % (BM / 8)) * 8; kk = idx / (BM / 8); } else { kk = (idx % (BK / 8)) * 8; mm = idx / (BK / 8); } \\\n"
    "        const int gm = m0 + mm, gk = (k0) + kk;                                                 \\\n"
    "        global const ushort *p = TA ? A + (long)gk * lda + gm : A + (long)gm * lda + gk;        \\\n"
    "        const bool full = TA ? (gk < K && gm + 7 < M) : (gm < M && gk + 7 < K);                 \\\n"
    "        if (full && ((((ulong)p) & 15) == 0)) ra[r] = vload8(0, p);                             \\\n"
    "        else {                                                                                  \\\n"
    "            ushort t[8];                                                                        \\\n"
    "            for (int e = 0; e < 8; ++e) {                                                       \\\n"
    "                const int em = TA ? gm + e : gm, ek = TA ? gk : gk + e;                         \\\n"
    "                t[e] = (em < M && ek < K) ? A_AT(em, ek) : (ushort)0;                           \\\n"
    "            }                                                                                   \\\n"
    "            ra[r] = (ushort8)(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);                  \\\n"
    "        }                                                                                       \\\n"
    "    }\n"
    "#define LOAD_B(k0)                                                                              \\\n"
    "    for (int r = 0; r < 4; ++r) {                                                               \\\n"
    "        const int idx = tid + r * NT; /* 512 groups of 8 */                                     \\\n"
    "        int nn, kk;                                                                             \\\n"
    "        if (TB) { kk = (idx % (BK / 8)) * 8; nn = idx / (BK / 8); } else { nn = (idx % (BN / 8)) * 8; kk = idx / (BN / 8); } \\\n"
    "        const int gn = n0 + nn, gk = (k0) + kk;                                                 \\\n"
    "        global const ushort *p = TB ? B + (long)gn * ldb + gk : B + (long)gk * ldb + gn;        \\\n"
    "        const bool full = TB ? (gn < N && gk + 7 < K) : (gk < K && gn + 7 < N);                 \\\n"
    "        if (full && ((((ulong)p) & 15) == 0)) rb[r] = vload8(0, p);                             \\\n"
    "        else {                                                                                  \\\n"
    "            ushort t[8];                                                                        \\\n"
    "            for (int e = 0; e < 8; ++e) {                                                       \\\n"
    "                const int en = TB ? gn : gn + e, ek = TB ? gk + e : gk;                         \\\n"
    "                t[e] = (en < N && ek < K) ? B_AT(ek, en) : (ushort)0;                           \\\n"
    "            }                                                                                   \\\n"
    "            rb[r] = (ushort8)(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);                  \\\n"
    "        }                                                                                       \\\n"
    "    }\n"
    "    LOAD_A(0)\n"
    "    LOAD_B(0)\n"
    "    for (int k0 = 0; k0 < K; k0 += BK) {\n"
    "        for (int r = 0; r < 4; ++r) {\n"
    "            const int idx = tid + r * NT;\n"
    "            const ushort v[8] = {ra[r].s0, ra[r].s1, ra[r].s2, ra[r].s3, ra[r].s4, ra[r].s5, ra[r].s6, ra[r].s7};\n"
    "            if (TA) {\n"
    "                const int mm = (idx % (BM / 8)) * 8, kk = idx / (BM / 8);\n"
    "                for (int e = 0; e < 8; ++e) As[kk / 16][mm + e][kk % 16] = v[e];\n"
    "            } else {\n"
    "                const int kk = (idx % (BK / 8)) * 8, mm = idx / (BK / 8);\n"
    "                vstore8(ra[r], 0, &As[kk / 16][mm][kk % 16]);\n"
    "            }\n"
    "        }\n"
    "        if (TB) {\n"
    "            // 8 consecutive k of one column n: pairs (k, k+1) -> 4 VNNI ints of column n\n"
    "            for (int r = 0; r < 4; ++r) {\n"
    "                const int idx = tid + r * NT;\n"
    "                const int kk = (idx % (BK / 8)) * 8, nn = idx / (BK / 8);\n"
    "                const ushort v[8] = {rb[r].s0, rb[r].s1, rb[r].s2, rb[r].s3, rb[r].s4, rb[r].s5, rb[r].s6, rb[r].s7};\n"
    "                for (int e = 0; e < 4; ++e) Bv[nn / 16][kk / 2 + e][nn % 16] = (uint)v[2 * e] | ((uint)v[2 * e + 1] << 16);\n"
    "            }\n"
    "        } else {\n"
    "            // row k (even) and row k+1 come from two different groups: use local memory staging\n"
    "            // through the half-precision tile, then pack (simple, conflict-free enough)\n"
    "            for (int r = 0; r < 4; ++r) {\n"
    "                const int idx = tid + r * NT;\n"
    "                const int nn = (idx % (BN / 8)) * 8, kk = idx / (BN / 8);\n"
    "                const ushort v[8] = {rb[r].s0, rb[r].s1, rb[r].s2, rb[r].s3, rb[r].s4, rb[r].s5, rb[r].s6, rb[r].s7};\n"
    "                local ushort *bh = (local ushort *)&Bv[0][0][0];\n"
    "                // ushort view: Bv[nb][kp][n16] = halves [nb][kp][n16][2]; this row's half is (kk & 1)\n"
    "                for (int e = 0; e < 8; ++e) bh[((((nn + e) / 16) * (BK / 2) + kk / 2) * 16 + (nn + e) % 16) * 2 + (kk & 1)] = v[e];\n"
    "            }\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        if (k0 + BK < K) { LOAD_A(k0 + BK) LOAD_B(k0 + BK) }\n"
    "        #pragma unroll\n"
    "        for (int kb = 0; kb < BK / 16; ++kb) {\n"
    "            short8 a[4];\n"
    "            int8 b[4];\n"
    "            #pragma unroll\n"
    "            for (int i = 0; i < 4; ++i)\n"
    "                a[i] = as_short8(intel_sub_group_block_read_us8(&As[kb][sgm * 32 + i * 8][0]));\n"
    "            #pragma unroll\n"
    "            for (int j = 0; j < 4; ++j)\n"
    "                b[j] = as_int8(intel_sub_group_block_read8(&Bv[sgn * 4 + j][kb * 8][0]));\n"
    "            #pragma unroll\n"
    "            for (int i = 0; i < 4; ++i)\n"
    "                #pragma unroll\n"
    "                for (int j = 0; j < 4; ++j) acc[i][j] = MAD(a[i], b[j], acc[i][j]);\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    #pragma unroll\n"
    "    for (int i = 0; i < 4; ++i) {\n"
    "        #pragma unroll\n"
    "        for (int j = 0; j < 4; ++j) {\n"
    "            const int gn = n0 + sgn * 64 + j * 16 + lane;\n"
    "            const float v8[8] = {acc[i][j].s0, acc[i][j].s1, acc[i][j].s2, acc[i][j].s3,\n"
    "                                 acc[i][j].s4, acc[i][j].s5, acc[i][j].s6, acc[i][j].s7};\n"
    "            for (int r = 0; r < 8; ++r) {\n"
    "                const int gm = m0 + sgm * 32 + i * 8 + r;\n"
    "                if (gm < M && gn < N) {\n"
    "                    const long at = (long)gm * ldc + gn;\n"
    "                    float v = alpha * v8[r];\n"
    "#if OUT == 0\n"
    "                    if (beta != 0.0f) v = fma(beta, C[at], v);\n"
    "                    C[at] = v;\n"
    "#elif OUT == 1\n"
    "                    if (beta != 0.0f) v = fma(beta, vload_half(at, (global const half *)C), v);\n"
    "                    vstore_half_rte(v, at, (global half *)C);\n"
    "#else\n"
    "                    if (beta != 0.0f) v = fma(beta, as_float((uint)C[at] << 16), v);\n"
    "                    uint u = as_uint(v);\n"
    "                    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u); /* RNE */\n"
    "                    else if (u & 0x007fffffu) u |= 0x00400000u;                              /* quiet NaN */\n"
    "                    C[at] = (ushort)(u >> 16);\n"
    "#endif\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n";

static const char k_hgemm4_src[] =
    "// XMX GEMM with 2D block loads (Xe2, cl_intel_subgroup_2d_block_io): no shared memory; the\n"
    "// hardware reads A tiles in DPAS layout and B ([N x K] rows) transposed into VNNI pairs.\n"
    "// Row-major C[MxN] = alpha * A[MxK] * op(B) + beta * C; TB=1: B is [N x K] (op = transpose), TB=0: B is\n"
    "// [K x N], read with the hardware's VNNI transform. Conditions (checked by the host): K, lda, ldb\n"
    "// multiples of 8; A, B (and batch strides) 64-byte aligned; K >= 32 (TB=0: N >= 32, even).\n"
    "// Split-K: z = batch * S + slice; a slice's k range is clipped by the block reads' surface size\n"
    "// (reads outside it return 0), and its partial sums go to C at slice offset (the host reduces).\n"
    "#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable\n"
    "#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable\n"
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "#define TM 32\n"
    "#define TN 64\n"
    "#define WM (SGM * TM)\n"
    "#define WN (SGN * TN)\n"
    "#if BF16\n"
    "#define MAD intel_sub_group_bf16_bf16_matrix_mad_k16\n"
    "#else\n"
    "#define MAD intel_sub_group_f16_f16_matrix_mad_k16\n"
    "#endif\n"
    "__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * SGM * SGN, 1, 1)))\n"
    "kernel void ocu_hgemm4(int M, int N, int K, float alpha, float beta,\n"
    "                       global const ushort *A0, long lda, long sa,\n"
    "                       global const ushort *B0, long ldb, long sb,\n"
    "                       global void *C0v, long ldc, long sc, int S, int kc)\n"
    "{\n"
    "    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();\n"
    "    const int sgm = sg % SGM, sgn = sg / SGM;\n"
    "    const int m0 = get_group_id(1) * WM + sgm * TM, n0 = get_group_id(0) * WN + sgn * TN;\n"
    "    const long bz = get_group_id(2) / S;\n"
    "    const int ks = (int)(get_group_id(2) % S);\n"
    "    const int kbeg = ks * kc, kend = min(K, kbeg + kc);\n"
    "    global const ushort *A = A0 + bz * sa;\n"
    "    global const ushort *B = B0 + bz * sb;\n"
    "#if OUT == 0\n"
    "    global float *C = (global float *)C0v + (S > 1 ? (long)get_group_id(2) : bz) * sc;\n"
    "#else\n"
    "    global ushort *C = (global ushort *)C0v + bz * sc;\n"
    "#endif\n"
    "    const int wA = kend * 2, pA = (int)lda * 2, pB = (int)ldb * 2;\n"
    "#if TB\n"
    "    const int wB = kend * 2, hB = N;\n"
    "#else\n"
    "    const int wB = N * 2, hB = kend;\n"
    "#endif\n"
    "    float8 acc[4][4];\n"
    "    for (int i = 0; i < 4; ++i)\n"
    "        for (int j = 0; j < 4; ++j) acc[i][j] = (float8)0.0f;\n"
    "    for (int k0 = kbeg; k0 < kend; k0 += 16) {\n"
    "        ushort8 a[4];\n"
    "        uint8 b[4];\n"
    "        intel_sub_group_2d_block_read_16b_32r16x1c((global void *)A, wA, M, pA, (int2)(k0, m0), (private ushort *)a);\n"
    "        for (int j = 0; j < 4; ++j)\n"
    "#if TB\n"
    "            intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)B, wB, hB, pB, (int2)(k0 / 2, n0 + 16 * j),\n"
    "                                                                (private uint *)&b[j]);\n"
    "#else\n"
    "            intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)B, wB, hB, pB, (int2)(n0 + 16 * j, k0),\n"
    "                                                                 (private uint *)&b[j]);\n"
    "#endif\n"
    "        intel_sub_group_2d_block_prefetch_16b_32r16x1c((global void *)A, wA, M, pA, (int2)(k0 + 32, m0));\n"
    "#if TB\n"
    "        intel_sub_group_2d_block_prefetch_16b_32r16x1c((global void *)B, wB, hB, pB, (int2)(k0 + 32, n0));\n"
    "        intel_sub_group_2d_block_prefetch_16b_32r16x1c((global void *)B, wB, hB, pB, (int2)(k0 + 32, n0 + 32));\n"
    "#else\n"
    "        intel_sub_group_2d_block_prefetch_16b_16r16x2c((global void *)B, wB, hB, pB, (int2)(n0, k0 + 32));\n"
    "        intel_sub_group_2d_block_prefetch_16b_16r16x2c((global void *)B, wB, hB, pB, (int2)(n0 + 32, k0 + 32));\n"
    "#endif\n"
    "        for (int i = 0; i < 4; ++i)\n"
    "            for (int j = 0; j < 4; ++j) acc[i][j] = MAD(as_short8(a[i]), as_int8(b[j]), acc[i][j]);\n"
    "    }\n"
    "    for (int i = 0; i < 4; ++i)\n"
    "        for (int j = 0; j < 4; ++j) {\n"
    "            const int gn = n0 + 16 * j + lane;\n"
    "            const float v8[8] = {acc[i][j].s0, acc[i][j].s1, acc[i][j].s2, acc[i][j].s3,\n"
    "                                 acc[i][j].s4, acc[i][j].s5, acc[i][j].s6, acc[i][j].s7};\n"
    "            for (int r = 0; r < 8; ++r) {\n"
    "                const int gm = m0 + 8 * i + r;\n"
    "                if (gm < M && gn < N) {\n"
    "                    const long at = (long)gm * ldc + gn;\n"
    "                    float v = alpha * v8[r];\n"
    "#if OUT == 0\n"
    "                    if (beta != 0.0f) v = fma(beta, C[at], v);\n"
    "                    C[at] = v;\n"
    "#elif OUT == 1\n"
    "                    if (beta != 0.0f) v = fma(beta, vload_half(at, (global const half *)C), v);\n"
    "                    vstore_half_rte(v, at, (global half *)C);\n"
    "#else\n"
    "                    if (beta != 0.0f) v = fma(beta, as_float((uint)C[at] << 16), v);\n"
    "                    uint u = as_uint(v);\n"
    "                    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);\n"
    "                    else if (u & 0x007fffffu) u |= 0x00400000u;\n"
    "                    C[at] = (ushort)(u >> 16);\n"
    "#endif\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "}\n"
;

static const char k_nf4_src[] =
    "// Fused NF4 GEMM on XMX (cl_intel_subgroup_2d_block_io): C[M,N] (f32) = A[M,K] (bf16) * W[N,K]^T + beta * C,\n"
    "// W in bitsandbytes' NF4 layout: row-major, two values per byte (first in the high nibble), one f32\n"
    "// absmax per 64 consecutive values of a row (K % 64 == 0). A sub-group owns 64 rows x 16 columns;\n"
    "// per 64 k, one transposed 2D block read gives each lane its row of W (32 bytes); a byte is two\n"
    "// consecutive k, i.e. one VNNI pair, looked up in a 256-entry table of bf16 code pairs. The scale is\n"
    "// per output column and 64-k block, so the DPAS runs on the unscaled codes and the accumulator is\n"
    "// kept in units of the current block's scale (times s_prev / s_new between blocks, s at the end).\n"
    "#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable\n"
    "#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable\n"
    "#define TMI 8\n"
    "#define SGM 2\n"
    "#define SGN 4\n"
    "#define SM (8 * TMI)\n"
    "#define WM (SM * SGM)\n"
    "#define WN (16 * SGN)\n"
    "#define NTH (16 * SGM * SGN)\n"
    "constant float NF4[16] = {-1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,\n"
    "    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f, 0.07958029955625534f,\n"
    "    0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f, 0.44070982933044434f,\n"
    "    0.5626170039176941f, 0.7229568362236023f, 1.0f};\n"
    "inline uint bf16_rne(float f) { uint u = as_uint(f); return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16; }\n"
    "__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(NTH, 1, 1)))\n"
    "kernel void ocu_nf4_gemm(int M, int N, int K, global const ushort *A, int lda, global const uchar *W,\n"
    "                         global const float *absmax, global float *C, int ldc, float beta)\n"
    "{\n"
    "    local uint lut[256];\n"
    "    for (int i = get_local_id(0); i < 256; i += NTH) lut[i] = bf16_rne(NF4[i >> 4]) | (bf16_rne(NF4[i & 15]) << 16);\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();\n"
    "    const int sgm = sg % SGM, sgn = sg / SGM;\n"
    "    const int m0 = get_group_id(1) * WM + sgm * SM, n0 = get_group_id(0) * WN + sgn * 16;\n"
    "    const int wB = K / 2, kb = K / 64, n = n0 + lane;\n"
    "    float8 acc[TMI];\n"
    "    for (int i = 0; i < TMI; ++i) acc[i] = (float8)0.0f;\n"
    "    float sp = 1.0f;\n"
    "    for (int k0 = 0; k0 < K; k0 += 64) {\n"
    "        uint8 q;\n"
    "        intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)W, wB, N, wB, (int2)(k0 / 8, n0), (private uint *)&q);\n"
    "        intel_sub_group_2d_block_prefetch_32b_16r8x1c((global void *)W, wB, N, wB, (int2)(k0 / 8 + 8, n0));\n"
    "        const float sn = n < N ? absmax[n * kb + k0 / 64] : 0.0f;\n"
    "        const int zero = sn == 0.0f;\n"
    "        const float snew = zero ? sp : sn;\n"
    "        const float r = sp / snew;\n"
    "        if (sub_group_any(r != 1.0f))\n"
    "            for (int i = 0; i < TMI; ++i) acc[i] *= r;\n"
    "        sp = snew;\n"
    "        const uint w[8] = {q.s0, q.s1, q.s2, q.s3, q.s4, q.s5, q.s6, q.s7};\n"
    "        for (int s = 0; s < 4; ++s) {\n"
    "            ushort8 a[TMI];\n"
    "            for (int i = 0; i < TMI; i += 4)\n"
    "                intel_sub_group_2d_block_read_16b_32r16x1c((global void *)A, K * 2, M, lda * 2, (int2)(k0 + 16 * s, m0 + 8 * i), (private ushort *)&a[i]);\n"
    "            const uint w0 = w[2 * s], w1 = w[2 * s + 1];\n"
    "            int8 b = (int8)(lut[w0 & 255u], lut[(w0 >> 8) & 255u], lut[(w0 >> 16) & 255u], lut[w0 >> 24],\n"
    "                            lut[w1 & 255u], lut[(w1 >> 8) & 255u], lut[(w1 >> 16) & 255u], lut[w1 >> 24]);\n"
    "            if (zero) b = (int8)0;\n"
    "            for (int i = 0; i < TMI; ++i) acc[i] = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(a[i]), b, acc[i]);\n"
    "        }\n"
    "    }\n"
    "    for (int i = 0; i < TMI; ++i) {\n"
    "        const float8 v = acc[i] * sp;\n"
    "        const float v8[8] = {v.s0, v.s1, v.s2, v.s3, v.s4, v.s5, v.s6, v.s7};\n"
    "        for (int r = 0; r < 8; ++r) {\n"
    "            const int gm = m0 + 8 * i + r;\n"
    "            if (gm < M && n < N) {\n"
    "                global float *o = C + (size_t)gm * ldc + n;\n"
    "                *o = beta == 0.0f ? v8[r] : fma(beta, *o, v8[r]);\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n"
;

typedef struct ocu_op_plan {
    ocu_device *dev;
    cl_kernel   k;
    int         M, N, K, batch;
    long        lda, ldb, ldc, sa, sb, sc;
    float       alpha, beta;
    int         S, kc;      /* split-K: S slices of kc (S == 1: none) */
    ocu_dtype   in, out;    /* input and output types */
    int         parr;       /* OCU_OP_FLAG_PTR_ARRAY: A, B, C are pointer arrays */
    cl_kernel   k4;         /* 2D-block-load XMX kernel when the shape allows it (else NULL) */
    int         sgm4, sgn4; /* its sub-group grid */
    int         bm, bn;     /* FP32 kernel tile (128 x 128 unless the shape is small) */
    int         qm;         /* OCU_OP_QMATMUL (NF4) plan */
    void       *part;       /* S x M x N partial sums when S > 1 */
    cl_kernel   reduce;
    char        name[96];   /* for OCU_OPENCL_TRACE */
} ocl_gemm_plan;

static pthread_mutex_t ops_lock = PTHREAD_MUTEX_INITIALIZER;
typedef struct { ocu_device *dev; cl_kernel k[4]; cl_kernel kp[4]; cl_kernel kt[4][4][4][2]; cl_kernel reduce; int state; int xmx; cl_kernel hk[2][3][4][2]; cl_kernel hk4[2][3][3][2]; cl_kernel nf4; int blk; } ocl_ops_dev; /* state: 0 new, 1 ok, -1 no */
static ocl_ops_dev ops_devs[16];

static int ops_enabled(ocu_device *d)
{
    const char *e = getenv("OCU_OPENCL_OPS");
    if (e != NULL && *e == '0') return 0;
    char ext[16384] = {0};
    if (ocl_load()->GetDeviceInfo(d->entry->device, CL_DEVICE_EXTENSIONS, sizeof(ext) - 1, ext, NULL) != CL_SUCCESS)
        return 0;
    return strstr(ext, "cl_intel_subgroups") && strstr(ext, "cl_intel_subgroup_local_block_io") &&
           strstr(ext, "cl_intel_required_subgroup_size");
}

/* The four transpose variants, built once per device. Called with ops_lock held. */
static ocl_ops_dev *ops_dev(ocu_device *d)
{
    ocl_ops_dev *o = NULL;
    for (int i = 0; i < 16; ++i) {
        if (ops_devs[i].dev == d) return ops_devs[i].state == 1 ? &ops_devs[i] : NULL;
        if (o == NULL && ops_devs[i].dev == NULL) o = &ops_devs[i];
    }
    if (o == NULL) return NULL;
    o->dev = d;
    o->state = -1;
    if (!ops_enabled(d)) return NULL;
    const ocl_api *a = ocl_load();
    for (int v = 0; v < 4; ++v) {
        cl_int err = CL_SUCCESS;
        const char *src = k_sgemm_src;
        cl_program prog = a->CreateProgramWithSource(d->ctx, 1, &src, NULL, &err);
        if (prog == NULL) return NULL;
        char opts[256];
        const char *grf[2] = {" -cl-intel-256-GRF-per-thread", ""};
        err = CL_BUILD_PROGRAM_FAILURE;
        for (int g = 0; g < 2 && err != CL_SUCCESS; ++g) {
            snprintf(opts, sizeof opts, "-cl-std=CL3.0 -DTA=%d -DTB=%d -DKNAME=ocu_sgemm%s", v >> 1, v & 1, grf[g]);
            err = a->BuildProgram(prog, 1, &d->entry->device, opts, NULL, NULL);
        }
        if (err != CL_SUCCESS) {
            if (verbose()) fprintf(stderr, "ocu opencl: the GEMM kernels do not build here (%d); CUDA kernels are used\n", (int)err);
            a->ReleaseProgram(prog);
            return NULL;
        }
        o->k[v] = a->CreateKernel(prog, "ocu_sgemm", &err);
        if (v == 0) o->reduce = a->CreateKernel(prog, "ocu_sgemm_reduce", &err);
        a->ReleaseProgram(prog); /* the kernel keeps it */
        if (o->k[v] == NULL || o->reduce == NULL) return NULL;
    }
    o->state = 1;
    return o;
}

static int xmx_enabled(ocu_device *d)
{
    char ext[16384] = {0};
    if (ocl_load()->GetDeviceInfo(d->entry->device, CL_DEVICE_EXTENSIONS, sizeof(ext) - 1, ext, NULL) != CL_SUCCESS)
        return 0;
    return strstr(ext, "cl_intel_subgroup_matrix_multiply_accumulate") != NULL && strstr(ext, "cl_khr_fp16") != NULL;
}

/* The XMX kernel for (input type, output type, transposes), built on first use. ops_lock held. */
/* The 2D-block-load XMX GEMM; cfg 0: 256 x 256 tiles, 1: 128 x 256, 2: 256 x 64 (short M: every
   work-group covers all of M, so B -- the weights, usually -- is read once). */
static cl_kernel hgemm4_kernel(ocu_device *d, ocl_ops_dev *o, int bf, int out, int cfg, int tb)
{
    if (o->hk4[bf][out][cfg][tb] != NULL) return o->hk4[bf][out][cfg][tb];
    if (o->blk == 0) {
        char ext[16384] = {0};
        ocl_load()->GetDeviceInfo(d->entry->device, CL_DEVICE_EXTENSIONS, sizeof ext - 1, ext, NULL);
        const char *off = getenv("OCU_OPENCL_BLOCK2D");
        o->blk = strstr(ext, "cl_intel_subgroup_2d_block_io") != NULL && !(off != NULL && *off == '0') ? 1 : -1;
    }
    if (o->blk < 0) return NULL;
    const ocl_api *a = ocl_load();
    cl_int err = CL_SUCCESS;
    const char *src = k_hgemm4_src;
    cl_program prog = a->CreateProgramWithSource(d->ctx, 1, &src, NULL, &err);
    if (prog == NULL) return NULL;
    char opts[256];
    snprintf(opts, sizeof opts, "-cl-std=CL3.0 -DBF16=%d -DOUT=%d -DSGM=%d -DSGN=%d -DTB=%d -cl-intel-256-GRF-per-thread",
             bf, out, cfg == 1 ? 4 : 8, cfg == 2 ? 1 : 4, tb);
    err = a->BuildProgram(prog, 1, &d->entry->device, opts, NULL, NULL);
    cl_kernel k = err == CL_SUCCESS ? a->CreateKernel(prog, "ocu_hgemm4", &err) : NULL;
    a->ReleaseProgram(prog);
    if (k == NULL) {
        if (verbose()) fprintf(stderr, "ocu opencl: the 2D-block XMX GEMM does not build here (%d)\n", (int)err);
        o->blk = -1;
        return NULL;
    }
    o->hk4[bf][out][cfg][tb] = k;
    return k;
}

/* The FP32 GEMM with a BM x BN tile (16 .. 128 each; 128 x 128 is o->k / o->kp), built on first use. */
static cl_kernel sgemm_tile_kernel(ocu_device *d, ocl_ops_dev *o, int v, int bm, int bn, int parr)
{
    const int im = bm == 16 ? 0 : bm == 32 ? 1 : bm == 64 ? 2 : 3, in = bn == 16 ? 0 : bn == 32 ? 1 : bn == 64 ? 2 : 3;
    if (o->kt[v][im][in][parr] != NULL) return o->kt[v][im][in][parr];
    const ocl_api *a = ocl_load();
    cl_int err = CL_SUCCESS;
    const char *src = k_sgemm_src;
    cl_program prog = a->CreateProgramWithSource(d->ctx, 1, &src, NULL, &err);
    if (prog == NULL) return NULL;
    char opts[256];
    snprintf(opts, sizeof opts, "-cl-std=CL3.0 -DTA=%d -DTB=%d -DPARR=%d -DBM=%d -DBN=%d -DKNAME=ocu_sgemm", v >> 1, v & 1,
             parr, bm, bn);
    err = a->BuildProgram(prog, 1, &d->entry->device, opts, NULL, NULL);
    cl_kernel k = err == CL_SUCCESS ? a->CreateKernel(prog, "ocu_sgemm", &err) : NULL;
    a->ReleaseProgram(prog);
    o->kt[v][im][in][parr] = k;
    return k;
}

/* The FP32 GEMM over pointer arrays (cublas*Batched), built on first use. */
static cl_kernel sgemm_parr_kernel(ocu_device *d, ocl_ops_dev *o, int v)
{
    if (o->kp[v] != NULL) return o->kp[v];
    const ocl_api *a = ocl_load();
    cl_int err = CL_SUCCESS;
    const char *src = k_sgemm_src;
    cl_program prog = a->CreateProgramWithSource(d->ctx, 1, &src, NULL, &err);
    if (prog == NULL) return NULL;
    char opts[256];
    const char *grf[2] = {" -cl-intel-256-GRF-per-thread", ""};
    err = CL_BUILD_PROGRAM_FAILURE;
    for (int g = 0; g < 2 && err != CL_SUCCESS; ++g) {
        snprintf(opts, sizeof opts, "-cl-std=CL3.0 -DTA=%d -DTB=%d -DPARR=1 -DKNAME=ocu_sgemm%s", v >> 1, v & 1, grf[g]);
        err = a->BuildProgram(prog, 1, &d->entry->device, opts, NULL, NULL);
    }
    cl_kernel k = err == CL_SUCCESS ? a->CreateKernel(prog, "ocu_sgemm", &err) : NULL;
    a->ReleaseProgram(prog);
    o->kp[v] = k;
    return k;
}

static cl_kernel xmx_kernel(ocu_device *d, ocl_ops_dev *o, int bf, int out, int v, int parr)
{
    if (o->hk[bf][out][v][parr] != NULL) return o->hk[bf][out][v][parr];
    if (o->xmx == 0) o->xmx = xmx_enabled(d) ? 1 : -1;
    if (o->xmx < 0) return NULL;
    const ocl_api *a = ocl_load();
    cl_int err = CL_SUCCESS;
    const char *src = k_hgemm_src;
    cl_program prog = a->CreateProgramWithSource(d->ctx, 1, &src, NULL, &err);
    if (prog == NULL) return NULL;
    char opts[256];
    snprintf(opts, sizeof opts, "-cl-std=CL3.0 -DTA=%d -DTB=%d -DBF16=%d -DOUT=%d -DPARR=%d -DKNAME=ocu_hgemm -cl-intel-256-GRF-per-thread",
             v >> 1, v & 1, bf, out, parr);
    err = a->BuildProgram(prog, 1, &d->entry->device, opts, NULL, NULL);
    cl_kernel k = err == CL_SUCCESS ? a->CreateKernel(prog, "ocu_hgemm", &err) : NULL;
    a->ReleaseProgram(prog);
    if (k == NULL) {
        if (verbose()) fprintf(stderr, "ocu opencl: the XMX GEMM does not build here (%d)\n", (int)err);
        o->xmx = -1;
        return NULL;
    }
    o->hk[bf][out][v][parr] = k;
    return k;
}

static int gemm_matrix(const ocu_tensor_desc *t, int *rows, int *cols, long *ld, long *bs, int *batch)
{
    if ((t->dtype != OCU_DT_F32 && t->dtype != OCU_DT_F16 && t->dtype != OCU_DT_BF16) || t->rank < 2 || t->rank > 3 ||
        t->stride[t->rank - 1] != 1) return 0;
    *rows = (int)t->shape[t->rank - 2];
    *cols = (int)t->shape[t->rank - 1];
    *ld = (long)t->stride[t->rank - 2];
    *batch = t->rank == 3 ? (int)t->shape[0] : 1;
    *bs = t->rank == 3 ? (long)t->stride[0] : 0;
    return *rows >= 0 && *cols >= 0 && *ld >= (*cols > 1 ? *cols : 1);
}

static ocu_status ops_support(ocu_device *dev, ocu_op op, ocu_dtype dt, ocu_op_support *out)
{
    if (dev == NULL || out == NULL) return OCU_ERR_INVALID_VALUE;
    if (op == OCU_OP_QMATMUL && dt == OCU_DT_BF16) {
        pthread_mutex_lock(&ops_lock);
        ocl_ops_dev *o = ops_dev(dev);
        pthread_mutex_unlock(&ops_lock);
        if (o == NULL || !xmx_enabled(dev)) return OCU_ERR_NOT_SUPPORTED;
        memset(out, 0, sizeof *out);
        out->layouts = OCU_LAYOUT_DENSE | OCU_LAYOUT_ROW_STRIDED;
        out->max_rank = 2;
        out->align_bytes = 64;
        out->accum_dtype = OCU_DT_F32;
        out->out_dtypes = 1u << OCU_DT_F32;
        return OCU_OK;
    }
    if (op != OCU_OP_GEMM || (dt != OCU_DT_F32 && dt != OCU_DT_F16 && dt != OCU_DT_BF16)) return OCU_ERR_NOT_SUPPORTED;
    if (dt != OCU_DT_F32 && !xmx_enabled(dev)) return OCU_ERR_NOT_SUPPORTED;
    pthread_mutex_lock(&ops_lock);
    ocl_ops_dev *o = ops_dev(dev);
    pthread_mutex_unlock(&ops_lock);
    if (o == NULL) return OCU_ERR_NOT_SUPPORTED;
    memset(out, 0, sizeof *out);
    out->layouts = OCU_LAYOUT_DENSE | OCU_LAYOUT_ROW_STRIDED;
    out->max_rank = 3;
    out->align_bytes = 4;
    out->accum_dtype = OCU_DT_F32;
    out->out_dtypes = dt == OCU_DT_F32 ? 1u << OCU_DT_F32
                                       : (1u << OCU_DT_F32) | (1u << OCU_DT_F16) | (1u << OCU_DT_BF16);
    return OCU_OK;
}

static ocu_status gemm_shape(const ocu_op_desc *d, ocl_gemm_plan *p, int *ta, int *tb)
{
    memset(p, 0, sizeof *p);   /* every field starts defined (qm, k4, part ...) */
    if (d == NULL || d->op != OCU_OP_GEMM || d->num_tensors != 3 || (d->flags & ~OCU_OP_FLAG_PTR_ARRAY))
        return OCU_ERR_NOT_SUPPORTED;
    int ar, ac, br, bc, cr, cc, ab, bb, cb;
    if (!gemm_matrix(&d->tensors[0], &ar, &ac, &p->lda, &p->sa, &ab) ||
        !gemm_matrix(&d->tensors[1], &br, &bc, &p->ldb, &p->sb, &bb) ||
        !gemm_matrix(&d->tensors[2], &cr, &cc, &p->ldc, &p->sc, &cb))
        return OCU_ERR_NOT_SUPPORTED;
    *ta = d->p.gemm.trans_a != 0;
    *tb = d->p.gemm.trans_b != 0;
    const int m = *ta ? ac : ar, k = *ta ? ar : ac, kb = *tb ? bc : br, n = *tb ? br : bc;
    if (k != kb || m != cr || n != cc || ab != cb || bb != cb) return OCU_ERR_NOT_SUPPORTED;
    if (cb > 65535 || m <= 0 || n <= 0 || k <= 0) return OCU_ERR_NOT_SUPPORTED;
    const ocu_dtype in = d->tensors[0].dtype, cout = d->tensors[2].dtype;
    if (d->tensors[1].dtype != in) return OCU_ERR_NOT_SUPPORTED;
    if (in == OCU_DT_F32 && cout != OCU_DT_F32) return OCU_ERR_NOT_SUPPORTED;
    p->parr = (d->flags & OCU_OP_FLAG_PTR_ARRAY) != 0;
    if (p->parr) p->sa = p->sb = p->sc = 0;
    p->in = in;
    p->out = cout;
    p->M = m; p->N = n; p->K = k; p->batch = cb;
    p->alpha = d->p.gemm.alpha;
    p->beta = d->p.gemm.beta;
    return OCU_OK;
}

/* OCU_OP_QMATMUL: A bf16 [M, K] (row stride lda), W I8 [N, K/2] (NF4 bytes, dense), absmax F32
   [N, K/64] (dense), C F32 [M, N] (row stride ldc); C = A * dequant(W)^T + beta * C. */
static ocu_status qm_shape(const ocu_op_desc *d, ocl_gemm_plan *p)
{
    if (d == NULL || d->op != OCU_OP_QMATMUL || d->num_tensors != 4 || d->flags != 0) return OCU_ERR_NOT_SUPPORTED;
    const ocu_tensor_desc *a = &d->tensors[0], *w = &d->tensors[1], *s = &d->tensors[2], *c = &d->tensors[3];
    if (a->dtype != OCU_DT_BF16 || w->dtype != OCU_DT_I8 || s->dtype != OCU_DT_F32 || c->dtype != OCU_DT_F32) return OCU_ERR_NOT_SUPPORTED;
    if (a->rank != 2 || w->rank != 2 || s->rank != 2 || c->rank != 2) return OCU_ERR_NOT_SUPPORTED;
    const long M = a->shape[0], K = a->shape[1], N = w->shape[0];
    if (M <= 0 || N <= 0 || K < 128 || K % 64 != 0 || K > 0x7fffffffL / 2 || w->shape[1] != K / 2 || s->shape[0] != N ||
        s->shape[1] != K / 64 || c->shape[0] != M || c->shape[1] != N)
        return OCU_ERR_NOT_SUPPORTED;
    if (a->stride[1] != 1 || a->stride[0] % 8 != 0 || w->stride[0] != K / 2 || s->stride[0] != K / 64 || c->stride[1] != 1 ||
        c->stride[0] < N || N * (K / 64) > 0x7fffffffL)
        return OCU_ERR_NOT_SUPPORTED;
    memset(p, 0, sizeof *p);
    p->qm = 1;
    p->M = (int)M; p->N = (int)N; p->K = (int)K; p->batch = 1;
    p->lda = a->stride[0]; p->ldc = c->stride[0];
    p->beta = d->p.gemm.beta;
    p->in = OCU_DT_BF16; p->out = OCU_DT_F32;
    return OCU_OK;
}

static cl_kernel nf4_kernel(ocu_device *d, ocl_ops_dev *o)
{
    if (o->nf4 != NULL) return o->nf4;
    const ocl_api *a = ocl_load();
    cl_int err = CL_SUCCESS;
    const char *src = k_nf4_src;
    cl_program prog = a->CreateProgramWithSource(d->ctx, 1, &src, NULL, &err);
    if (prog == NULL) return NULL;
    err = a->BuildProgram(prog, 1, &d->entry->device, "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread", NULL, NULL);
    cl_kernel k = err == CL_SUCCESS ? a->CreateKernel(prog, "ocu_nf4_gemm", &err) : NULL;
    if (k == NULL && verbose()) fprintf(stderr, "ocu opencl: the NF4 GEMM does not build here (%d)\n", (int)err);
    a->ReleaseProgram(prog);
    o->nf4 = k;
    return k;
}

static ocu_status ops_query(ocu_device *dev, const ocu_op_desc *d, ocu_op_caps *caps)
{
    if (d != NULL && d->op == OCU_OP_QMATMUL) {
        ocl_gemm_plan p;
        ocu_op_support sup;
        if (dev == NULL || caps == NULL) return OCU_ERR_INVALID_VALUE;
        ocu_status st = qm_shape(d, &p);
        if (st != OCU_OK) return st;
        if (ops_support(dev, OCU_OP_QMATMUL, OCU_DT_BF16, &sup) != OCU_OK) return OCU_ERR_NOT_SUPPORTED;
        memset(caps, 0, sizeof *caps);
        caps->tol = OCU_TOL_DOT;
        caps->accum_dtype = OCU_DT_F32;
        return OCU_OK;
    }
    ocl_gemm_plan p;
    int ta, tb;
    ocu_op_support sup;
    if (dev == NULL || caps == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = gemm_shape(d, &p, &ta, &tb);
    if (st != OCU_OK) return st;
    if (ops_support(dev, OCU_OP_GEMM, p.in, &sup) != OCU_OK) return OCU_ERR_NOT_SUPPORTED;
    memset(caps, 0, sizeof *caps);
    caps->tol = OCU_TOL_DOT;
    caps->accum_dtype = OCU_DT_F32;
    caps->scratch_bytes = 2 * 32 * 128 * 4;
    caps->tiles = (uint64_t)((p.M + 127) / 128) * ((p.N + 127) / 128) * p.batch;
    return OCU_OK;
}

static ocu_status ops_plan(ocu_device *dev, const ocu_op_desc *d, ocu_op_plan **out)
{
    if (dev == NULL || out == NULL) return OCU_ERR_INVALID_VALUE;
    if (d != NULL && d->op == OCU_OP_QMATMUL) {
        ocl_gemm_plan p;
        ocu_status st = qm_shape(d, &p);
        if (st != OCU_OK) return st;
        pthread_mutex_lock(&ops_lock);
        ocl_ops_dev *o = ops_dev(dev);
        cl_kernel k = o != NULL && xmx_enabled(dev) ? nf4_kernel(dev, o) : NULL;
        pthread_mutex_unlock(&ops_lock);
        if (k == NULL) return OCU_ERR_NOT_SUPPORTED;
        ocl_gemm_plan *pl = malloc(sizeof *pl);
        if (pl == NULL) return OCU_ERR_OUT_OF_MEMORY;
        *pl = p;
        pl->dev = dev;
        pl->k = k;
        snprintf(pl->name, sizeof pl->name, "ocu_nf4_gemm(xmx) M%d N%d K%d", pl->M, pl->N, pl->K);
        *out = pl;
        return OCU_OK;
    }
    ocl_gemm_plan p;
    int ta, tb;
    ocu_status st = gemm_shape(d, &p, &ta, &tb);
    if (st != OCU_OK) return st;
    pthread_mutex_lock(&ops_lock);
    ocl_ops_dev *o = ops_dev(dev);
    pthread_mutex_unlock(&ops_lock);
    if (o == NULL) return OCU_ERR_NOT_SUPPORTED;
    ocl_gemm_plan *pl = malloc(sizeof *pl);
    if (pl == NULL) return OCU_ERR_OUT_OF_MEMORY;
    *pl = p;
    pl->dev = dev;
    pl->k = o->k[ta * 2 + tb];
    pl->k4 = NULL;
    if (p.in == OCU_DT_F32 && p.parr) {
        pthread_mutex_lock(&ops_lock);
        pl->k = sgemm_parr_kernel(dev, o, ta * 2 + tb);
        pthread_mutex_unlock(&ops_lock);
        if (pl->k == NULL) { free(pl); return OCU_ERR_NOT_SUPPORTED; }
    }
    pl->reduce = o->reduce;
    if (p.in != OCU_DT_F32) {
        pthread_mutex_lock(&ops_lock);
        pl->k = xmx_kernel(dev, o, p.in == OCU_DT_BF16, p.out == OCU_DT_F32 ? 0 : p.out == OCU_DT_F16 ? 1 : 2, ta * 2 + tb, p.parr);
        pl->k4 = NULL;
        if (pl->k != NULL && !p.parr && ta == 0 && (tb == 1 || (pl->N >= 32 && pl->N % 2 == 0)) && pl->K >= 32 &&
            pl->K % 8 == 0 && pl->lda % 8 == 0 &&
            pl->ldb % 8 == 0 && (pl->batch == 1 || (pl->sa % 32 == 0 && pl->sb % 32 == 0)) &&
            (long)pl->K * 2 <= 0x7fffffffL && pl->lda * 2 <= 0x7fffffffL && pl->ldb * 2 <= 0x7fffffffL) {
            const long big = (long)((pl->M + 255) / 256) * ((pl->N + 255) / 256) * pl->batch;
            int cfg = big >= 64 ? 0 : 1;
            /* Short M: 256 x 64 tiles read B once, but only pay when there are enough of them. */
            const long tall = (long)((pl->M + 255) / 256) * ((pl->N + 63) / 64) * pl->batch;
            if (pl->M <= 512 && tall >= 32 && getenv("OCU_OPENCL_NO_TALL") == NULL) cfg = 2;
            pl->k4 = hgemm4_kernel(dev, o, p.in == OCU_DT_BF16, p.out == OCU_DT_F32 ? 0 : p.out == OCU_DT_F16 ? 1 : 2, cfg, tb);
            pl->sgm4 = cfg == 1 ? 4 : 8;
            pl->sgn4 = cfg == 2 ? 1 : 4;
        }
        pthread_mutex_unlock(&ops_lock);
        if (pl->k == NULL) { free(pl); return OCU_ERR_NOT_SUPPORTED; }
    }
    pl->S = 1;
    pl->kc = pl->K;
    pl->part = NULL;
    pl->bm = pl->bn = 128;
    if (p.in == OCU_DT_F32 && getenv("OCU_OPENCL_SGEMM_TILES") == NULL) {
        /* Small or skinny shapes (LoRA's rank-16 sides, attention heads): smaller tiles, so
           that the GPU gets enough work-groups and few computed values are thrown away. */
        int bm = 128, bn = 128;
        while (bm > 16 && bm / 2 >= pl->M) bm /= 2;
        while (bn > 16 && bn / 2 >= pl->N) bn /= 2;
        /* The 128 x 128 tile (8 x 8 values per work-item) is several times faster per
           value; smaller ones only when the shape is very small. */
        for (;;) {
            const long wg = (long)((pl->M + bm - 1) / bm) * ((pl->N + bn - 1) / bn) * pl->batch;
            if (wg >= 16 || (bm <= 64 && bn <= 64)) break;
            if (bm >= bn && bm > 64) bm /= 2; else if (bn > 64) bn /= 2; else break;
        }
        if (bm != 128 || bn != 128) {
            pthread_mutex_lock(&ops_lock);
            cl_kernel k = sgemm_tile_kernel(dev, o, ta * 2 + tb, bm, bn, p.parr);
            pthread_mutex_unlock(&ops_lock);
            if (k != NULL) { pl->k = k; pl->bm = bm; pl->bn = bn; }
        }
    }
    /* Split-K when the C tiles cannot fill the GPU and K is long (for example
       a 256 x 128 result over K = 50304: 2 tiles). Slices of at least 512. */
    const long tiles = (long)((pl->M + pl->bm - 1) / pl->bm) * ((pl->N + pl->bn - 1) / pl->bn) * pl->batch;
    /* The 2D-block XMX kernel: split K when its tiles cannot fill the GPU (f32 output). */
    if (pl->k4 != NULL && p.out == OCU_DT_F32 && pl->batch == 1 && getenv("OCU_OPENCL_SPLITK") == NULL) {
        const long t4 = (long)((pl->M + 32 * pl->sgm4 - 1) / (32 * pl->sgm4)) * ((pl->N + 64 * pl->sgn4 - 1) / (64 * pl->sgn4));
        /* Only when the tiles are very few: the partial sums and their reduction cost a pass over C. */
        int S = t4 < 16 && pl->K >= 2048 ? (int)((32 + t4 - 1) / t4) : 1;
        /* A very long K (a vocabulary: the LM head in the backward pass) with a few hundred tiles: each
           work-group would walk the whole K. Slices of 16K or more, at most 8. */
        if (S == 1 && pl->K >= 32768 && t4 < 1024) { S = (int)((pl->K + 16383) / 16384); if (S > 8) S = 8; }
        if (S > pl->K / 1024) S = pl->K / 1024;
        { const char *fs = getenv("OCU_OPENCL_SPLIT4"); if (fs != NULL) S = atoi(fs); }
        if (S > 1) {
            int kc = (pl->K + S - 1) / S;
            kc = (kc + 31) / 32 * 32;
            S = (pl->K + kc - 1) / kc;
            void *part = NULL;
            if (S > 1 && ocl_mem_alloc(dev, (size_t)S * pl->M * pl->N * sizeof(float), &part) == OCU_OK) {
                pl->S = S;
                pl->kc = kc;
                pl->part = part;
            }
        }
    }
    /* Also for accuracy: a very long K (a vocabulary) is summed in slices of
       at most 8192, so no output adds 150000 products in one chain. */
    const int long_k = pl->K > 16384;
    /* Skinny results (LoRA's rank-16 side: a few work-groups) split even a short K, in slices of 128 or more. */
    const int skinny = tiles < 32 && pl->K >= 512;
    if (p.in == OCU_DT_F32 && pl->batch == 1 && !pl->parr && (tiles < 128 || long_k) && (pl->K >= 2048 || skinny) &&
        getenv("OCU_OPENCL_SPLITK") == NULL) {
        int S = tiles < 128 ? (int)((256 + tiles - 1) / tiles) : 1;
        if (S > pl->K / (skinny && pl->K < 2048 ? 128 : 512)) S = pl->K / (skinny && pl->K < 2048 ? 128 : 512);
        if (long_k && S < (pl->K + 8191) / 8192) S = (pl->K + 8191) / 8192;
        if (S > 1) {
            int kc = (pl->K + S - 1) / S;
            kc = (kc + 31) / 32 * 32;
            S = (pl->K + kc - 1) / kc;
            void *part = NULL;
            if (S > 1 && ocl_mem_alloc(dev, (size_t)S * pl->M * pl->N * sizeof(float), &part) == OCU_OK) {
                pl->S = S;
                pl->kc = kc;
                pl->part = part;
            }
        }
    }
    snprintf(pl->name, sizeof pl->name, pl->bm != 128 || pl->bn != 128 ? "%s %c%c M%d N%d K%d x%d split %d t" : "%s %c%c M%d N%d K%d x%d split %d",
             p.in == OCU_DT_F32 ? "ocu_sgemm" : p.in == OCU_DT_F16 ? "ocu_hgemm(xmx f16)" : "ocu_hgemm(xmx bf16)",
             ta ? 'T' : 'N', tb ? 'T' : 'N', pl->M, pl->N, pl->K, pl->batch, pl->S);
    *out = pl;
    return OCU_OK;
}

static ocu_status ops_execute(ocu_op_plan *pl, ocu_queue *q, const ocu_tensor *t, uint32_t n)
{
    if (pl != NULL && pl->qm) {
        if (t == NULL || n != 4) return OCU_ERR_INVALID_VALUE;
        ocu_device *d = pl->dev;
        if (q != NULL && q->dev != d) return OCU_ERR_INVALID_HANDLE;
        for (int i = 0; i < 4; ++i)
            if (!table_has(d, t[i].data, 1)) return OCU_ERR_NOT_SUPPORTED;
        if ((((uintptr_t)t[0].data | (uintptr_t)t[1].data) & 63) != 0) return OCU_ERR_NOT_SUPPORTED;
        ocu_queue *queue = q != NULL ? q : d->sync_q;
        const ocl_api *a = ocl_load();
        const int lda = (int)pl->lda, ldc = (int)pl->ldc;
        size_t lws[3] = {128, 1, 1}, gws[3] = {(size_t)((pl->N + 63) / 64) * 128, (size_t)((pl->M + 127) / 128), 1};
        cl_event ev = NULL;
        pthread_mutex_lock(&ops_lock);
        cl_uint i = 0;
        cl_int err = a->SetKernelArg(pl->k, i++, sizeof(int), &pl->M);
        err |= a->SetKernelArg(pl->k, i++, sizeof(int), &pl->N);
        err |= a->SetKernelArg(pl->k, i++, sizeof(int), &pl->K);
        err |= a->SetKernelArgSVMPointer(pl->k, i++, t[0].data);
        err |= a->SetKernelArg(pl->k, i++, sizeof(int), &lda);
        err |= a->SetKernelArgSVMPointer(pl->k, i++, t[1].data);
        err |= a->SetKernelArgSVMPointer(pl->k, i++, t[2].data);
        err |= a->SetKernelArgSVMPointer(pl->k, i++, t[3].data);
        err |= a->SetKernelArg(pl->k, i++, sizeof(int), &ldc);
        err |= a->SetKernelArg(pl->k, i++, sizeof(float), &pl->beta);
        if (err == CL_SUCCESS) err = a->EnqueueNDRangeKernel(queue->q, pl->k, 3, NULL, gws, lws, 0, NULL, &ev);
        pthread_mutex_unlock(&ops_lock);
        if (err == CL_SUCCESS && trace_on()) trace_add_name(pl->name, ev);
        return finish_enqueue(queue, q == NULL, err, ev);
    }
    if (pl == NULL || t == NULL || n != 3) return OCU_ERR_INVALID_VALUE;
    ocu_device *d = pl->dev;
    if (q != NULL && q->dev != d) return OCU_ERR_INVALID_HANDLE;
    /* only memory this backend allocated (SVM pointers) */
    if (!table_has(d, t[0].data, 2) || !table_has(d, t[1].data, 2) || !table_has(d, t[2].data, 2))
        return OCU_ERR_NOT_SUPPORTED;
    ocu_queue *queue = q != NULL ? q : d->sync_q;
    const ocl_api *a = ocl_load();
    const int xmx = pl->in != OCU_DT_F32;
    size_t lws[3] = {xmx ? 128 : 16, xmx ? 1 : 16, 1};
    size_t gws[3] = {(size_t)((pl->N + 127) / 128) * (xmx ? 128 : 16), (size_t)((pl->M + 127) / 128) * (xmx ? 1 : 16),
                     (size_t)pl->batch * pl->S};
    if (!xmx) {
        gws[0] = (size_t)((pl->N + pl->bn - 1) / pl->bn) * 16;
        gws[1] = (size_t)((pl->M + pl->bm - 1) / pl->bm) * 16;
    }
    /* The block loads need 64-byte aligned matrices. */
    cl_kernel kern = pl->k;
    if (pl->k4 != NULL && ((uintptr_t)t[0].data & 63) == 0 && ((uintptr_t)t[1].data & 63) == 0) {
        kern = pl->k4;
        const int wm = 32 * pl->sgm4, wn = 64 * pl->sgn4;
        lws[0] = (size_t)16 * pl->sgm4 * pl->sgn4;
        gws[0] = (size_t)((pl->N + wn - 1) / wn) * lws[0];
        gws[1] = (size_t)((pl->M + wm - 1) / wm);
    }
    /* A split plan of the 2D kernel runs unsplit on the fallback kernel (misaligned matrices). */
    const int S_run = (xmx && kern != pl->k4) ? 1 : pl->S;
    const int split = S_run > 1;
    gws[2] = (size_t)pl->batch * S_run;
    const float one = 1.0f, zero = 0.0f;
    const long ldp = pl->N, sp = (long)pl->M * pl->N;
    cl_event ev = NULL;
    pthread_mutex_lock(&ops_lock);
    cl_uint i = 0;
    cl_int err = a->SetKernelArg(kern, i++, sizeof(int), &pl->M);
    err |= a->SetKernelArg(kern, i++, sizeof(int), &pl->N);
    err |= a->SetKernelArg(kern, i++, sizeof(int), &pl->K);
    err |= a->SetKernelArg(kern, i++, sizeof(float), split ? &one : &pl->alpha);
    err |= a->SetKernelArg(kern, i++, sizeof(float), split ? &zero : &pl->beta);
    err |= a->SetKernelArgSVMPointer(kern, i++, t[0].data);
    err |= a->SetKernelArg(kern, i++, sizeof(long), &pl->lda);
    err |= a->SetKernelArg(kern, i++, sizeof(long), &pl->sa);
    err |= a->SetKernelArgSVMPointer(kern, i++, t[1].data);
    err |= a->SetKernelArg(kern, i++, sizeof(long), &pl->ldb);
    err |= a->SetKernelArg(kern, i++, sizeof(long), &pl->sb);
    err |= a->SetKernelArgSVMPointer(kern, i++, split ? pl->part : t[2].data);
    err |= a->SetKernelArg(kern, i++, sizeof(long), split ? &ldp : &pl->ldc);
    err |= a->SetKernelArg(kern, i++, sizeof(long), split ? &sp : &pl->sc);
    if (!xmx || kern == pl->k4) {
        const int kc_run = split ? pl->kc : pl->K;
        err |= a->SetKernelArg(kern, i++, sizeof(int), &S_run);
        err |= a->SetKernelArg(kern, i++, sizeof(int), &kc_run);
    }
    if (err == CL_SUCCESS && pl->parr) {
        /* The matrices are reached through the arrays: name every allocation. */
        pthread_mutex_lock(&d->lock);
        size_t na = d->count;
        void **ptrs = malloc((na ? na : 1) * sizeof(void *));
        if (ptrs == NULL) err = CL_OUT_OF_HOST_MEMORY;
        else {
            for (size_t j = 0; j < na; ++j) ptrs[j] = d->allocs[j].ptr;
            if (na > 0) err = a->SetKernelExecInfo(kern, CL_KERNEL_EXEC_INFO_SVM_PTRS, na * sizeof(void *), ptrs);
            free(ptrs);
        }
        pthread_mutex_unlock(&d->lock);
    }
    if (err == CL_SUCCESS) err = a->EnqueueNDRangeKernel(queue->q, kern, 3, NULL, gws, lws, 0, NULL, split ? NULL : &ev);
    if (err == CL_SUCCESS && split) {
        cl_uint r = 0;
        err = a->SetKernelArg(pl->reduce, r++, sizeof(int), &pl->M);
        err |= a->SetKernelArg(pl->reduce, r++, sizeof(int), &pl->N);
        err |= a->SetKernelArg(pl->reduce, r++, sizeof(int), &pl->S);
        err |= a->SetKernelArg(pl->reduce, r++, sizeof(float), &pl->alpha);
        err |= a->SetKernelArg(pl->reduce, r++, sizeof(float), &pl->beta);
        err |= a->SetKernelArgSVMPointer(pl->reduce, r++, pl->part);
        err |= a->SetKernelArgSVMPointer(pl->reduce, r++, t[2].data);
        err |= a->SetKernelArg(pl->reduce, r++, sizeof(long), &pl->ldc);
        size_t rl[3] = {16, 16, 1}, rg[3] = {(size_t)(pl->N + 15) / 16 * 16, (size_t)(pl->M + 15) / 16 * 16, 1};
        if (err == CL_SUCCESS) err = a->EnqueueNDRangeKernel(queue->q, pl->reduce, 3, NULL, rg, rl, 0, NULL, &ev);
    }
    pthread_mutex_unlock(&ops_lock);
    if (err == CL_SUCCESS && trace_on()) {
        if (kern == pl->k4 && pl->name[0] != '2') { memmove(pl->name + 3, pl->name, sizeof pl->name - 3); memcpy(pl->name, "2d ", 3); pl->name[sizeof pl->name - 1] = 0; }
        trace_add_name(pl->name, ev);
    }
    return finish_enqueue(queue, q == NULL, err, ev);
}

static ocu_status ops_destroy(ocu_op_plan *pl)
{
    if (pl != NULL && pl->part != NULL) ocl_mem_free(pl->dev, pl->part); /* waits for queued work */
    free(pl);
    return OCU_OK;
}

static const ocu_op_table ocl_ops = {
    .version = OCU_OPS_VERSION,
    .struct_size = sizeof(ocu_op_table),
    .support = ops_support,
    .query = ops_query,
    .plan = ops_plan,
    .execute = ops_execute,
    .destroy = ops_destroy,
    .stats = NULL,
};

static const void *ocl_get_provider(const char *library)
{
    if (library != NULL && strcmp(library, OCU_OPS_PROVIDER) == 0) return &ocl_ops;
    return NULL;
}

static ocu_status ocl_kernel_launch(ocu_queue *q, ocu_kernel *k, const ocu_launch_params *p)
{
    HP_B(HP_LAUNCH);
    ocu_status r_ = ocl_kernel_launch_impl(q, k, p);
    HP_E(HP_LAUNCH);
    return r_;
}

static ocu_status ocl_mem_copy(ocu_device *d, ocu_queue *q, void *dst, const void *src, size_t bytes, ocu_copy_dir dir)
{
    HP_B(HP_COPY);
    ocu_status r_ = ocl_mem_copy_impl(d, q, dst, src, bytes, dir);
    HP_E(HP_COPY);
    return r_;
}

static ocu_status ocl_mem_set(ocu_device *d, ocu_queue *q, void *dst, int value, size_t bytes)
{
    HP_B(HP_SET);
    ocu_status r_ = ocl_mem_set_impl(d, q, dst, value, bytes);
    HP_E(HP_SET);
    return r_;
}

static ocu_status ocl_queue_sync(ocu_queue *q)
{
    HP_B(HP_SYNC);
    ocu_status r_ = ocl_queue_sync_impl(q);
    HP_E(HP_SYNC);
    return r_;
}

static ocu_status ocl_marker_record(ocu_marker *m, ocu_queue *q)
{
    HP_B(HP_MARK);
    ocu_status r_ = ocl_marker_record_impl(m, q);
    HP_E(HP_MARK);
    return r_;
}

static ocu_status ocl_mem_alloc(ocu_device *d, size_t bytes, void **ptr)
{
    HP_B(HP_ALLOC);
    ocu_status r_ = ocl_mem_alloc_impl(d, bytes, ptr);
    HP_E(HP_ALLOC);
    return r_;
}

static ocu_status ocl_mem_free(ocu_device *d, void *ptr)
{
    HP_B(HP_FREE);
    ocu_status r_ = ocl_mem_free_impl(d, ptr);
    HP_E(HP_FREE);
    return r_;
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

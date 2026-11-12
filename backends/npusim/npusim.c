/*
 * npusim: a backend that behaves like a non-SIMT accelerator (NPU), on the
 * host, so the layers above can be tested against NPU limits without the
 * hardware. Rules: docs/spec/op-table.md section 7.
 *
 *   - no kernels: module_load and kernel_launch are NOT_SUPPORTED
 *   - work is done only by the operator table and by copies/fills
 *   - device memory is an address range the host cannot touch; its
 *     contents live in a separate shadow mapping only this file reaches
 *   - execute must match the planned shapes
 *   - a configurable scratchpad bounds the working set of one tile
 *   - counters for op calls, plans, refused launches and DMA bytes
 *
 * Every call runs synchronously under one lock (one DMA/compute engine),
 * so queue order holds trivially and queue_query is always OK.
 */
#if !defined(_WIN32) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE /* MAP_ANONYMOUS, MAP_NORESERVE, nanosleep */
#endif

#include "ocu_backend_npusim.h"
#include "ocu_ops.h"
#include "ocu_ops_ref.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#define NS_ALIGN        256u  /* allocation granularity */
#define NS_TENSOR_ALIGN 64u   /* required alignment of op tensors */
#define NS_MAGIC_QUEUE  0x6e737171u
#define NS_MAGIC_MARKER 0x6e736d6bu
#define NS_MAGIC_PLAN   0x6e73706cu

/* ---- objects ------------------------------------------------------------ */

struct ocu_device { int unused; };
struct ocu_queue  { uint32_t magic; };
struct ocu_marker { uint32_t magic; int recorded; double time_ms; };

struct ocu_op_plan {
    uint32_t     magic;
    ocu_op_plan *next;
    ocu_op_desc  desc;
    ocu_op_caps  caps;
};

typedef struct ns_alloc {
    uintptr_t        base;  /* device address */
    size_t           bytes;
    struct ns_alloc *next;  /* sorted by base */
} ns_alloc;

/* ---- state (under g_lock) ------------------------------------------------- */

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_once = PTHREAD_ONCE_INIT;
static ocu_device      g_device;

static struct {
    /* configuration */
    size_t   mem_bytes;     /* OCU_NPUSIM_MEM_MB */
    size_t   scratch_bytes; /* OCU_NPUSIM_SCRATCH_KB */
    int      tiling;        /* OCU_NPUSIM_TILING */
    uint32_t dtype_mask;    /* OCU_NPUSIM_DTYPES, bit per ocu_dtype */
    uint32_t plan_ms;       /* OCU_NPUSIM_PLAN_MS */
    int      report;        /* OCU_NPUSIM_REPORT */
    /* memory */
    uintptr_t      base;    /* reserved, inaccessible device range */
    unsigned char *shadow;  /* backing store, same size */
    int            ready;   /* both mappings exist */
    ns_alloc      *allocs;
    size_t         in_use;
    /* plans and counters */
    ocu_op_plan   *plans;
    ocu_op_stats   stats;
} G;

static double now_ms(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void sleep_ms(uint32_t ms)
{
#if defined(_WIN32)
    Sleep(ms);
#else
    struct timespec ts = {(time_t)(ms / 1000u), (long)(ms % 1000u) * 1000000L};
    while (nanosleep(&ts, &ts) != 0) {
    }
#endif
}

static long env_long(const char *name, long dflt, long lo, long hi)
{
    const char *s = getenv(name);
    if (s == NULL || *s == '\0') return dflt;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < lo || v > hi) return dflt;
    return v;
}

static uint32_t parse_dtypes(const char *s)
{
    uint32_t mask = 0;
    while (s != NULL && *s != '\0') {
        const char *e = strchr(s, ',');
        size_t n = e ? (size_t)(e - s) : strlen(s);
        if (n == 3 && strncmp(s, "f32", 3) == 0) mask |= 1u << OCU_DT_F32;
        if (n == 3 && strncmp(s, "f16", 3) == 0) mask |= 1u << OCU_DT_F16;
        if (n == 4 && strncmp(s, "bf16", 4) == 0) mask |= 1u << OCU_DT_BF16;
        s = e ? e + 1 : NULL;
    }
    return mask;
}

static void init_once(void)
{
    G.mem_bytes = (size_t)env_long("OCU_NPUSIM_MEM_MB", 256, 1, 65536) << 20;
    G.scratch_bytes = (size_t)env_long("OCU_NPUSIM_SCRATCH_KB", 256, 1, 1 << 20) << 10;
    G.tiling = env_long("OCU_NPUSIM_TILING", 1, 0, 1) != 0;
    const char *dt = getenv("OCU_NPUSIM_DTYPES");
    G.dtype_mask = (dt != NULL && *dt != '\0')
        ? parse_dtypes(dt)
        : (1u << OCU_DT_F32) | (1u << OCU_DT_F16) | (1u << OCU_DT_BF16);
    G.plan_ms = (uint32_t)env_long("OCU_NPUSIM_PLAN_MS", 0, 0, 60000);
    G.report = env_long("OCU_NPUSIM_REPORT", 0, 0, 1) != 0;

#if defined(_WIN32)
    void *r = VirtualAlloc(NULL, G.mem_bytes, MEM_RESERVE, PAGE_NOACCESS);
    void *s = VirtualAlloc(NULL, G.mem_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (r == NULL || s == NULL) {
        if (r) VirtualFree(r, 0, MEM_RELEASE);
        if (s) VirtualFree(s, 0, MEM_RELEASE);
        return;
    }
#else
    void *r = mmap(NULL, G.mem_bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    void *s = mmap(NULL, G.mem_bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (r == MAP_FAILED || s == MAP_FAILED) {
        if (r != MAP_FAILED) munmap(r, G.mem_bytes);
        if (s != MAP_FAILED) munmap(s, G.mem_bytes);
        return;
    }
#endif
    G.base = (uintptr_t)r;
    G.shadow = s;
    G.ready = 1;
}

static void ensure_init(void) { pthread_once(&g_once, init_once); }

/* ---- device memory -------------------------------------------------------- */

static int in_space(uintptr_t p) { return G.ready && p >= G.base && p - G.base < G.mem_bytes; }

/* Caller holds g_lock. The allocation holding [p, p + bytes), or NULL. */
static const ns_alloc *find_range(uintptr_t p, uint64_t bytes)
{
    for (const ns_alloc *a = G.allocs; a != NULL && a->base <= p; a = a->next) {
        if (p - a->base <= a->bytes && bytes <= a->bytes - (p - a->base)) return a;
    }
    return NULL;
}

static void *shadow_of(uintptr_t p) { return G.shadow + (p - G.base); }

static ocu_status ns_mem_alloc(ocu_device *dev, size_t bytes, void **ptr)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    *ptr = NULL;
    if (bytes > G.mem_bytes) return OCU_ERR_OUT_OF_MEMORY;
    size_t need = (bytes == 0 ? 1 : bytes);
    need = (need + NS_ALIGN - 1) / NS_ALIGN * NS_ALIGN;
    ns_alloc *n = malloc(sizeof(*n));
    if (n == NULL) return OCU_ERR_OUT_OF_MEMORY;
    pthread_mutex_lock(&g_lock);
    /* First fit in the gaps of the sorted list. */
    uintptr_t at = G.base;
    ns_alloc **link = &G.allocs;
    while (*link != NULL) {
        if ((*link)->base - at >= need) break;
        uintptr_t end = (*link)->base + (*link)->bytes;
        at = (end + NS_ALIGN - 1) / NS_ALIGN * NS_ALIGN;
        link = &(*link)->next;
    }
    if (G.base + G.mem_bytes - at < need) {
        pthread_mutex_unlock(&g_lock);
        free(n);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    n->base = at;
    n->bytes = bytes == 0 ? 1 : bytes;
    n->next = *link;
    *link = n;
    G.in_use += need;
    pthread_mutex_unlock(&g_lock);
    *ptr = (void *)at;
    return OCU_OK;
}

static ocu_status ns_mem_free(ocu_device *dev, void *ptr)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (ptr == NULL) return OCU_OK;
    pthread_mutex_lock(&g_lock);
    ns_alloc **link = &G.allocs;
    while (*link != NULL && (*link)->base != (uintptr_t)ptr) link = &(*link)->next;
    if (*link == NULL) {
        pthread_mutex_unlock(&g_lock);
        return OCU_ERR_INVALID_VALUE;
    }
    ns_alloc *a = *link;
    *link = a->next;
    G.in_use -= (a->bytes + NS_ALIGN - 1) / NS_ALIGN * NS_ALIGN;
    pthread_mutex_unlock(&g_lock);
    free(a);
    return OCU_OK;
}

static int queue_ok(const ocu_queue *q) { return q == NULL || q->magic == NS_MAGIC_QUEUE; }

static ocu_status ns_mem_copy(ocu_device *dev, ocu_queue *queue, void *dst,
                              const void *src, size_t bytes, ocu_copy_dir dir)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (!queue_ok(queue)) return OCU_ERR_INVALID_HANDLE;
    if (bytes == 0) return OCU_OK;
    if (dst == NULL || src == NULL) return OCU_ERR_INVALID_VALUE;
    uintptr_t d = (uintptr_t)dst, s = (uintptr_t)src;
    ocu_status st = OCU_OK;
    pthread_mutex_lock(&g_lock);
    switch (dir) {
    case OCU_COPY_HOST_TO_DEVICE:
        if (in_space(s) || find_range(d, bytes) == NULL) { st = OCU_ERR_INVALID_VALUE; break; }
        memcpy(shadow_of(d), src, bytes);
        G.stats.bytes_to_device += bytes;
        break;
    case OCU_COPY_DEVICE_TO_HOST:
        if (in_space(d) || find_range(s, bytes) == NULL) { st = OCU_ERR_INVALID_VALUE; break; }
        memcpy(dst, shadow_of(s), bytes);
        G.stats.bytes_to_host += bytes;
        break;
    case OCU_COPY_DEVICE_TO_DEVICE:
        if (find_range(s, bytes) == NULL || find_range(d, bytes) == NULL) {
            st = OCU_ERR_INVALID_VALUE;
            break;
        }
        memmove(shadow_of(d), shadow_of(s), bytes);
        G.stats.bytes_on_device += bytes;
        break;
    default:
        st = OCU_ERR_INVALID_VALUE;
        break;
    }
    pthread_mutex_unlock(&g_lock);
    return st;
}

static ocu_status ns_mem_set(ocu_device *dev, ocu_queue *queue, void *dst,
                             int value, size_t bytes)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (!queue_ok(queue)) return OCU_ERR_INVALID_HANDLE;
    if (bytes == 0) return OCU_OK;
    ocu_status st = OCU_OK;
    pthread_mutex_lock(&g_lock);
    if (find_range((uintptr_t)dst, bytes) == NULL) {
        st = OCU_ERR_INVALID_VALUE;
    } else {
        memset(shadow_of((uintptr_t)dst), value, bytes);
    }
    pthread_mutex_unlock(&g_lock);
    return st;
}

static ocu_status ns_mem_info(ocu_device *dev, uint64_t *free_bytes, uint64_t *total_bytes)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (free_bytes == NULL || total_bytes == NULL) return OCU_ERR_INVALID_VALUE;
    pthread_mutex_lock(&g_lock);
    *total_bytes = G.mem_bytes;
    *free_bytes = G.mem_bytes - G.in_use;
    pthread_mutex_unlock(&g_lock);
    return OCU_OK;
}

/* Pinned host memory is ordinary host memory here. */
static ocu_status ns_host_alloc_pinned(ocu_device *dev, size_t bytes, void **ptr)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (ptr == NULL) return OCU_ERR_INVALID_VALUE;
    *ptr = malloc(bytes == 0 ? 1 : bytes);
    return *ptr ? OCU_OK : OCU_ERR_OUT_OF_MEMORY;
}

static ocu_status ns_host_free_pinned(ocu_device *dev, void *ptr)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    free(ptr);
    return OCU_OK;
}

/* ---- device --------------------------------------------------------------- */

static ocu_status ns_dev_count(uint32_t *count)
{
    if (count == NULL) return OCU_ERR_INVALID_VALUE;
    ensure_init();
    *count = G.ready ? 1u : 0u;
    return OCU_OK;
}

static ocu_status ns_dev_info(uint32_t index, ocu_dev_info *info)
{
    if (info == NULL) return OCU_ERR_INVALID_VALUE;
    ensure_init();
    if (index != 0u || !G.ready) return OCU_ERR_INVALID_DEVICE;
    memset(info, 0, sizeof(*info));
    snprintf(info->name, sizeof(info->name), "npusim (NPU constraint simulator)");
    info->total_memory_bytes = G.mem_bytes;
    info->warp_size = 1;
    info->max_threads_per_block = 1;
    for (int i = 0; i < 3; ++i) {
        info->max_block_dim[i] = 1;
        info->max_grid_dim[i] = 1;
    }
    info->max_shared_memory_per_block = G.scratch_bytes;
    info->supports_unified_memory = 0;
    return OCU_OK;
}

static ocu_status ns_dev_open(uint32_t index, ocu_device **dev)
{
    if (dev == NULL) return OCU_ERR_INVALID_VALUE;
    ensure_init();
    if (index != 0u || !G.ready) return OCU_ERR_INVALID_DEVICE;
    *dev = &g_device;
    return OCU_OK;
}

static ocu_status ns_dev_close(ocu_device *dev)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (G.report) {
        pthread_mutex_lock(&g_lock);
        ocu_op_stats s = G.stats;
        pthread_mutex_unlock(&g_lock);
        fprintf(stderr, "npusim: dma to_device=%llu to_host=%llu on_device=%llu bytes\n",
                (unsigned long long)s.bytes_to_device, (unsigned long long)s.bytes_to_host,
                (unsigned long long)s.bytes_on_device);
        fprintf(stderr, "npusim: plans created=%llu refused=%llu; kernel launches refused=%llu\n",
                (unsigned long long)s.plans_created, (unsigned long long)s.plans_refused,
                (unsigned long long)s.kernel_launches_refused);
        for (uint32_t op = 1; op < OCU_OP_COUNT; ++op) {
            if (s.op_calls[op]) {
                fprintf(stderr, "npusim: op %s calls=%llu\n", ocu_op_name((ocu_op)op),
                        (unsigned long long)s.op_calls[op]);
            }
        }
    }
    return OCU_OK;
}

/* ---- queues and markers (synchronous engine) ------------------------------ */

static ocu_status ns_queue_create(ocu_device *dev, ocu_queue **queue)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (queue == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_queue *q = calloc(1, sizeof(*q));
    if (q == NULL) return OCU_ERR_OUT_OF_MEMORY;
    q->magic = NS_MAGIC_QUEUE;
    *queue = q;
    return OCU_OK;
}

static ocu_status ns_queue_destroy(ocu_queue *queue)
{
    if (queue == NULL || queue->magic != NS_MAGIC_QUEUE) return OCU_ERR_INVALID_HANDLE;
    queue->magic = 0;
    free(queue);
    return OCU_OK;
}

static ocu_status ns_queue_sync(ocu_queue *queue)
{
    return queue_ok(queue) ? OCU_OK : OCU_ERR_INVALID_HANDLE;
}

static ocu_status ns_queue_query(ocu_queue *queue)
{
    return queue_ok(queue) ? OCU_OK : OCU_ERR_INVALID_HANDLE;
}

static ocu_status ns_marker_create(ocu_device *dev, ocu_marker **marker)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (marker == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_marker *m = calloc(1, sizeof(*m));
    if (m == NULL) return OCU_ERR_OUT_OF_MEMORY;
    m->magic = NS_MAGIC_MARKER;
    *marker = m;
    return OCU_OK;
}

static int marker_ok(const ocu_marker *m) { return m != NULL && m->magic == NS_MAGIC_MARKER; }

static ocu_status ns_marker_destroy(ocu_marker *m)
{
    if (!marker_ok(m)) return OCU_ERR_INVALID_HANDLE;
    m->magic = 0;
    free(m);
    return OCU_OK;
}

static ocu_status ns_marker_record(ocu_marker *m, ocu_queue *queue)
{
    if (!marker_ok(m) || !queue_ok(queue)) return OCU_ERR_INVALID_HANDLE;
    pthread_mutex_lock(&g_lock); /* after everything submitted before it */
    m->time_ms = now_ms();
    m->recorded = 1;
    pthread_mutex_unlock(&g_lock);
    return OCU_OK;
}

static ocu_status ns_marker_wait(ocu_marker *m)
{
    return marker_ok(m) ? OCU_OK : OCU_ERR_INVALID_HANDLE;
}

static ocu_status ns_marker_query(ocu_marker *m)
{
    return marker_ok(m) ? OCU_OK : OCU_ERR_INVALID_HANDLE;
}

static ocu_status ns_marker_elapsed_ms(ocu_marker *start, ocu_marker *end, float *ms)
{
    if (!marker_ok(start) || !marker_ok(end)) return OCU_ERR_INVALID_HANDLE;
    if (ms == NULL) return OCU_ERR_INVALID_VALUE;
    if (!start->recorded || !end->recorded) return OCU_ERR_INVALID_HANDLE;
    *ms = (float)(end->time_ms - start->time_ms);
    return OCU_OK;
}

/* ---- code: none ------------------------------------------------------------ */

static ocu_status ns_module_load(ocu_device *dev, const void *image, size_t bytes,
                                 ocu_module **module)
{
    (void)image; (void)bytes;
    if (module != NULL) *module = NULL;
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    return OCU_ERR_NOT_SUPPORTED;
}

static ocu_status ns_module_unload(ocu_module *module)
{
    (void)module;
    return OCU_ERR_NOT_SUPPORTED;
}

static ocu_status ns_module_get_kernel(ocu_module *module, const char *name, ocu_kernel **kernel)
{
    (void)module; (void)name;
    if (kernel != NULL) *kernel = NULL;
    return OCU_ERR_NOT_SUPPORTED;
}

static ocu_status ns_kernel_launch(ocu_queue *queue, ocu_kernel *kernel,
                                   const ocu_launch_params *params)
{
    (void)queue; (void)kernel; (void)params;
    pthread_mutex_lock(&g_lock);
    G.stats.kernel_launches_refused++;
    pthread_mutex_unlock(&g_lock);
    return OCU_ERR_NOT_SUPPORTED;
}

/* ---- operator table --------------------------------------------------------- */

static int is_float(ocu_dtype dt)
{
    return dt == OCU_DT_F32 || dt == OCU_DT_F16 || dt == OCU_DT_BF16;
}

static int dtype_allowed(ocu_dtype dt)
{
    return is_float(dt) && (G.dtype_mask & (1u << dt)) != 0;
}

static ocu_tol_class tol_of(ocu_op op)
{
    switch (op) {
    case OCU_OP_GEMM:                          return OCU_TOL_DOT;
    case OCU_OP_RMS_NORM: case OCU_OP_SOFTMAX: return OCU_TOL_ROW;
    case OCU_OP_ROPE:                          return OCU_TOL_ULP;
    default:                                   return OCU_TOL_EXACT;
    }
}

/* Working set of a t x t x t GEMM tile (spec 7, rule 5). */
static uint64_t gemm_tile_bytes(uint64_t tm, uint64_t tn, uint64_t tk, uint64_t in)
{
    return (tm * tk + tk * tn) * in + tm * tn * 4u;
}

static uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

/* Scratchpad planning for a valid descriptor; fills caps or NOT_SUPPORTED. */
static ocu_status fit_scratch(const ocu_op_desc *d, ocu_op_caps *caps)
{
    const uint64_t S = G.scratch_bytes;
    const ocu_tensor_desc *y = &d->tensors[d->num_tensors - 1];
    uint64_t in = ocu_dtype_size(d->tensors[0].dtype);
    uint64_t out = ocu_dtype_size(y->dtype);
    uint64_t rows = (uint64_t)ocu_ref_rows(d);
    uint64_t n = (uint64_t)y->shape[y->rank - 1];
    switch (d->op) {
    case OCU_OP_GEMM: {
        const ocu_tensor_desc *a = &d->tensors[0];
        uint32_t o = a->rank - 2;
        uint64_t bt = o ? (uint64_t)y->shape[0] : 1u;
        uint64_t m = (uint64_t)y->shape[o];
        uint64_t k = (uint64_t)(d->p.gemm.trans_a ? a->shape[o] : a->shape[o + 1]);
        if (!G.tiling) {
            uint64_t ws = gemm_tile_bytes(m, n, k, in);
            if (ws > S) return OCU_ERR_NOT_SUPPORTED;
            caps->scratch_bytes = ws;
            caps->tiles = bt;
            return OCU_OK;
        }
        for (uint64_t t = 128; t >= 16; t /= 2) {
            uint64_t ws = gemm_tile_bytes(t, t, t, in);
            if (ws <= S) {
                caps->scratch_bytes = ws;
                caps->tiles = bt * ceil_div(m, t) * ceil_div(n, t) * ceil_div(k, t);
                return OCU_OK;
            }
        }
        return OCU_ERR_NOT_SUPPORTED;
    }
    case OCU_OP_RMS_NORM:
    case OCU_OP_SOFTMAX: {
        uint64_t ws = n * (in + out) + 64u;
        if (d->op == OCU_OP_RMS_NORM && d->num_tensors == 3) ws += n * in;
        if (ws > S) return OCU_ERR_NOT_SUPPORTED;
        caps->scratch_bytes = ws;
        caps->tiles = rows;
        return OCU_OK;
    }
    default: /* element-wise: streamed */
        caps->scratch_bytes = 0;
        caps->tiles = rows;
        return OCU_OK;
    }
}

static ocu_status ns_op_support(ocu_device *dev, ocu_op op, ocu_dtype dtype, ocu_op_support *out)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    memset(out, 0, sizeof(*out));
    switch (op) {
    case OCU_OP_GEMM: case OCU_OP_ADD: case OCU_OP_MUL:
    case OCU_OP_RMS_NORM: case OCU_OP_SOFTMAX:
        break;
    case OCU_OP_ROPE:    return OCU_ERR_NOT_IMPLEMENTED;
    case OCU_OP_QMATMUL: return OCU_ERR_NOT_SUPPORTED;
    default:             return OCU_ERR_INVALID_VALUE;
    }
    if (!dtype_allowed(dtype)) return OCU_ERR_NOT_SUPPORTED;
    uint64_t in = ocu_dtype_size(dtype);
    if (op == OCU_OP_GEMM && gemm_tile_bytes(16, 16, 16, in) > G.scratch_bytes) {
        return OCU_ERR_NOT_SUPPORTED;
    }
    out->layouts = OCU_LAYOUT_DENSE | OCU_LAYOUT_ROW_STRIDED;
    out->max_rank = op == OCU_OP_GEMM ? 3u : OCU_OPS_MAX_RANK;
    if (op == OCU_OP_RMS_NORM || op == OCU_OP_SOFTMAX) {
        /* Longest row that fits, weight included, input dtype on both sides. */
        uint64_t per = op == OCU_OP_RMS_NORM ? 3u * in : 2u * in;
        if (G.scratch_bytes < 64u + per) return OCU_ERR_NOT_SUPPORTED;
        out->max_row = (int64_t)((G.scratch_bytes - 64u) / per);
    }
    out->align_bytes = NS_TENSOR_ALIGN;
    out->accum_dtype = OCU_DT_F32;
    out->out_dtypes = 1u << dtype;
    if (op == OCU_OP_GEMM && (G.dtype_mask & (1u << OCU_DT_F32))) out->out_dtypes |= 1u << OCU_DT_F32;
    out->needs_plan_ms = G.plan_ms;
    out->var_rows = 1;
    return OCU_OK;
}

static ocu_status query_locked(const ocu_op_desc *desc, ocu_op_caps *caps)
{
    memset(caps, 0, sizeof(*caps));
    ocu_status st = ocu_ref_check_desc(desc);
    if (st != OCU_OK) return st;
    for (uint32_t i = 0; i < desc->num_tensors; ++i) {
        /* inputs, and C of GEMM (read when beta != 0) */
        int is_input = i + 1 < desc->num_tensors;
        if (is_input && !dtype_allowed(desc->tensors[i].dtype)) return OCU_ERR_NOT_SUPPORTED;
    }
    st = fit_scratch(desc, caps);
    if (st != OCU_OK) {
        memset(caps, 0, sizeof(*caps));
        return st;
    }
    caps->tol = tol_of(desc->op);
    caps->accum_dtype = OCU_DT_F32;
    return OCU_OK;
}

static ocu_status ns_op_query(ocu_device *dev, const ocu_op_desc *desc, ocu_op_caps *caps)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (desc == NULL || caps == NULL) return OCU_ERR_INVALID_VALUE;
    return query_locked(desc, caps); /* reads only configuration */
}

static ocu_status ns_op_plan(ocu_device *dev, const ocu_op_desc *desc, ocu_op_plan **plan)
{
    if (plan != NULL) *plan = NULL;
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (desc == NULL || plan == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_op_caps caps;
    ocu_status st = query_locked(desc, &caps);
    ocu_op_plan *p = NULL;
    if (st == OCU_OK) {
        p = calloc(1, sizeof(*p));
        if (p == NULL) st = OCU_ERR_OUT_OF_MEMORY;
    }
    if (st != OCU_OK) {
        pthread_mutex_lock(&g_lock);
        G.stats.plans_refused++;
        pthread_mutex_unlock(&g_lock);
        return st;
    }
    if (G.plan_ms) sleep_ms(G.plan_ms); /* stands in for compilation */
    p->magic = NS_MAGIC_PLAN;
    p->desc = *desc;
    p->desc.struct_size = sizeof(ocu_op_desc);
    p->caps = caps;
    pthread_mutex_lock(&g_lock);
    p->next = G.plans;
    G.plans = p;
    G.stats.plans_created++;
    pthread_mutex_unlock(&g_lock);
    *plan = p;
    return OCU_OK;
}

static int plan_live_locked(const ocu_op_plan *p)
{
    for (const ocu_op_plan *q = G.plans; q != NULL; q = q->next) {
        if (q == p) return 1;
    }
    return 0;
}

static ocu_status ns_op_destroy(ocu_op_plan *plan)
{
    if (plan == NULL) return OCU_ERR_INVALID_HANDLE;
    pthread_mutex_lock(&g_lock);
    ocu_op_plan **link = &G.plans;
    while (*link != NULL && *link != plan) link = &(*link)->next;
    if (*link == NULL) {
        pthread_mutex_unlock(&g_lock);
        return OCU_ERR_INVALID_HANDLE;
    }
    *link = plan->next;
    pthread_mutex_unlock(&g_lock);
    plan->magic = 0;
    free(plan);
    return OCU_OK;
}

static ocu_status ns_op_execute(ocu_op_plan *plan, ocu_queue *queue,
                                const ocu_tensor *tensors, uint32_t num_tensors)
{
    if (!queue_ok(queue)) return OCU_ERR_INVALID_HANDLE;
    void *data[OCU_OPS_MAX_TENSORS] = {0};
    ocu_op_desc actual;
    const ocu_op_desc *d = NULL;
    ocu_status st = OCU_OK;
    pthread_mutex_lock(&g_lock);
    if (plan == NULL || !plan_live_locked(plan)) {
        st = OCU_ERR_INVALID_HANDLE;
        goto out;
    }
    st = ocu_ref_check_tensors(&plan->desc, tensors, num_tensors, &actual);
    if (st != OCU_OK) goto out;
    d = &actual;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        uintptr_t p = (uintptr_t)tensors[i].data;
        /* Host pointers, stray device pointers and misaligned tensors. */
        if (!in_space(p) || p % NS_TENSOR_ALIGN != 0 ||
            find_range(p, ocu_ref_span_bytes(&tensors[i].desc)) == NULL) {
            st = OCU_ERR_INVALID_VALUE;
            goto out;
        }
        data[i] = shadow_of(p);
    }
    ocu_ref_run(d, data, 0, ocu_ref_rows(d));
    G.stats.op_calls[d->op]++;
out:
    pthread_mutex_unlock(&g_lock);
    return st;
}

static ocu_status ns_op_stats(ocu_device *dev, ocu_op_stats *out)
{
    if (dev != &g_device) return OCU_ERR_INVALID_DEVICE;
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    pthread_mutex_lock(&g_lock);
    *out = G.stats;
    pthread_mutex_unlock(&g_lock);
    return OCU_OK;
}

static const ocu_op_table ns_ops = {
    .version = OCU_OPS_VERSION,
    .struct_size = sizeof(ocu_op_table),
    .support = ns_op_support,
    .query = ns_op_query,
    .plan = ns_op_plan,
    .execute = ns_op_execute,
    .destroy = ns_op_destroy,
    .stats = ns_op_stats,
};

static const void *ns_get_provider(const char *library)
{
    if (library != NULL && strcmp(library, OCU_OPS_PROVIDER) == 0) return &ns_ops;
    return NULL;
}

static const ocu_backend ns_backend = {
    .abi = OCU_BACKEND_ABI,
    .name = "npusim",
    .dev_count = ns_dev_count,
    .dev_info = ns_dev_info,
    .dev_open = ns_dev_open,
    .dev_close = ns_dev_close,
    .mem_alloc = ns_mem_alloc,
    .mem_free = ns_mem_free,
    .mem_copy = ns_mem_copy,
    .mem_set = ns_mem_set,
    .mem_info = ns_mem_info,
    .host_alloc_pinned = ns_host_alloc_pinned,
    .host_free_pinned = ns_host_free_pinned,
    .queue_create = ns_queue_create,
    .queue_destroy = ns_queue_destroy,
    .queue_sync = ns_queue_sync,
    .queue_query = ns_queue_query,
    .module_load = ns_module_load,
    .module_unload = ns_module_unload,
    .module_get_kernel = ns_module_get_kernel,
    .kernel_launch = ns_kernel_launch,
    .marker_create = ns_marker_create,
    .marker_destroy = ns_marker_destroy,
    .marker_record = ns_marker_record,
    .marker_wait = ns_marker_wait,
    .marker_query = ns_marker_query,
    .marker_elapsed_ms = ns_marker_elapsed_ms,
    .get_provider = ns_get_provider,
    .module_get_global = NULL,
    .kernel_arg_sizes = NULL,
    .mem_alloc_shared = NULL, /* no memory the host can reach */
    .host_register = NULL,
    .host_unregister = NULL,
};

const ocu_backend *ocu_backend_npusim(void) { return &ns_backend; }

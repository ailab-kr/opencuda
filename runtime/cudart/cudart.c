/*
 * The runtime API (cuda_runtime_api.h) on top of the backend interface.
 * Chip-neutral: nothing here knows which backend is active.
 *
 * Ordering model (the documented behaviour of the legacy default stream):
 *   - stream 0 (and cudaStreamLegacy) is one backend queue per device;
 *   - work submitted to the default stream first waits for every blocking
 *     stream of the device; work submitted to a blocking stream first waits
 *     for the default stream; non-blocking streams take part in neither;
 *   - cudaMemcpy (not ...Async) returns when the copy is done.
 * Waiting is done on the host by synchronising the queues involved, which
 * keeps the order exact at some cost in overlap.
 *
 * Errors: every call records a failure as the calling thread's last error
 * (cudaGetLastError returns and clears it, cudaPeekAtLastError only
 * returns it). Failures of queued work surface at the next synchronising
 * call on that queue as cudaErrorLaunchFailure or the backend's error.
 */
#define _POSIX_C_SOURCE 200809L

#include "ocu_rt.h"
#include "ocu_core.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DEVICES 16

/* ---- objects ------------------------------------------------------------ */

struct ocu_rt_stream {
    ocu_queue           *queue;
    unsigned             flags;
    int                  priority; /* recorded; the CPU backend has one level */
    int                  device;
    struct ocu_rt_stream *next;   /* the device's list of streams */
};

struct ocu_rt_event {
    ocu_marker *marker;
    unsigned    flags;
    int         device;
};

/* What the runtime allocated, so pointer kinds can be told apart
   (cudaPointerGetAttributes, cudaMemcpyDefault, pageable-copy rule). */
typedef struct mem_range {
    char                *base;
    size_t               size;
    enum cudaMemoryType  type; /* Host (page-locked), Device or Managed */
    void                *dev;  /* address device code uses (registered host memory) */
} mem_range;

typedef struct rt_device {
    pthread_mutex_t       lock;
    bool                  open;
    ocu_device           *dev;
    ocu_queue            *legacy;   /* the default stream */
    struct ocu_rt_stream *streams;
    /* Held from the ordering check to the enqueue of work on the default
       stream or a blocking stream, so the legacy rule holds across host
       threads. Order: submit before lock. */
    pthread_mutex_t       submit;
    /* A kernel failure stays visible to every later synchronising call
       until cudaDeviceReset, as the API describes. 0 = none. */
    _Atomic int           sticky;
    /* Allocations made through the runtime, guarded by `lock`. Only
       page-locked host memory may be read by an asynchronous copy after
       the call returns; any other host memory is copied before. */
    mem_range            *ranges;
    size_t                range_count, range_cap;
} rt_device;

static const ocu_backend *g_backend;
static int                g_device_count = -1;
static rt_device          g_devices[MAX_DEVICES];
static pthread_once_t     g_init_once = PTHREAD_ONCE_INIT;
static cudaError_t        g_init_error = cudaSuccess;

static _Thread_local int         tl_device;
/* Set while a cudaLaunchHostFunc callback runs on this thread. */
static _Thread_local bool        tl_in_host_fn;
static _Thread_local cudaError_t tl_last_error = cudaSuccess;

/* ---- errors -------------------------------------------------------------- */

cudaError_t ocu_rt_error(ocu_status st)
{
    switch (st) {
    case OCU_OK:                  return cudaSuccess;
    case OCU_ERR_INVALID_VALUE:   return cudaErrorInvalidValue;
    case OCU_ERR_INVALID_DEVICE:  return cudaErrorInvalidDevice;
    case OCU_ERR_INVALID_HANDLE:  return cudaErrorInvalidResourceHandle;
    case OCU_ERR_OUT_OF_MEMORY:   return cudaErrorMemoryAllocation;
    case OCU_ERR_NOT_READY:       return cudaErrorNotReady;
    case OCU_ERR_LAUNCH_FAILED:   return cudaErrorLaunchFailure;
    case OCU_ERR_NOT_SUPPORTED:   return cudaErrorNotSupported;
    case OCU_ERR_NOT_IMPLEMENTED: return cudaErrorNotSupported;
    case OCU_ERR_NOT_FOUND:       return cudaErrorInvalidDeviceFunction;
    case OCU_ERR_UNKNOWN:         return cudaErrorUnknown;
    }
    return cudaErrorUnknown;
}

/* Record a failure as the thread's last error; pass the value through. */
static cudaError_t set_error(cudaError_t e)
{
    if (e != cudaSuccess && e != cudaErrorNotReady) {
        tl_last_error = e;
    }
    return e;
}

static cudaError_t from(ocu_status st) { return set_error(ocu_rt_error(st)); }

/* A backend result from synchronising work on device d: a kernel failure
   becomes the device's sticky error; an earlier sticky error wins. */
static cudaError_t dev_result(struct rt_device *d, ocu_status st)
{
    cudaError_t e = ocu_rt_error(st);
    if (e == cudaErrorLaunchFailure) {
        int none = 0;
        atomic_compare_exchange_strong(&d->sticky, &none, (int)e);
    }
    int sticky = atomic_load(&d->sticky);
    if (sticky != 0 && e != cudaErrorNotReady) {
        e = (cudaError_t)sticky;
    }
    return set_error(e);
}

cudaError_t cudaGetLastError(void)
{
    cudaError_t e = tl_last_error;
    tl_last_error = cudaSuccess;
    return e;
}

cudaError_t cudaPeekAtLastError(void) { return tl_last_error; }

const char *cudaGetErrorName(cudaError_t e)
{
    switch (e) {
#define NAME_CASE(name, value, text) case name: return #name;
        OCU_RT_ERRORS(NAME_CASE)
#undef NAME_CASE
    default: break;
    }
    return "cudaErrorUnknown";
}

const char *cudaGetErrorString(cudaError_t e)
{
    switch (e) {
#define TEXT_CASE(name, value, text) case name: return text;
        OCU_RT_ERRORS(TEXT_CASE)
#undef TEXT_CASE
    default: break;
    }
    return "unrecognized error code";
}

/* ---- devices ------------------------------------------------------------- */

static void init_once(void)
{
    ocu_device *ignored = NULL;
    if (ocu_core_device(&g_backend, &ignored) != OCU_OK || g_backend == NULL) {
        g_init_error = cudaErrorNoDevice;
        g_device_count = 0;
        return;
    }
    uint32_t n = 0;
    if (g_backend->dev_count(&n) != OCU_OK) {
        g_init_error = cudaErrorInitializationError;
        n = 0;
    }
    g_device_count = n > MAX_DEVICES ? MAX_DEVICES : (int)n;
    for (int i = 0; i < MAX_DEVICES; ++i) {
        pthread_mutex_init(&g_devices[i].lock, NULL);
        pthread_mutex_init(&g_devices[i].submit, NULL);
    }
    if (g_device_count == 0 && g_init_error == cudaSuccess) {
        g_init_error = cudaErrorNoDevice;
    }
}

static cudaError_t init(void)
{
    pthread_once(&g_init_once, init_once);
    return g_init_error;
}

/* The device record for `index`, opened on first use. */
static cudaError_t device_at(int index, rt_device **out)
{
    cudaError_t e = init();
    if (e != cudaSuccess) {
        return e;
    }
    if (index < 0 || index >= g_device_count) {
        return cudaErrorInvalidDevice;
    }
    rt_device *d = &g_devices[index];
    pthread_mutex_lock(&d->lock);
    ocu_status st = OCU_OK;
    if (!d->open) {
        st = g_backend->dev_open((uint32_t)index, &d->dev);
        if (st == OCU_OK) {
            st = g_backend->queue_create(d->dev, &d->legacy);
            if (st != OCU_OK) {
                g_backend->dev_close(d->dev);
            }
        }
        d->open = st == OCU_OK;
    }
    pthread_mutex_unlock(&d->lock);
    if (st != OCU_OK) {
        return ocu_rt_error(st);
    }
    *out = d;
    return cudaSuccess;
}

static cudaError_t current(rt_device **out) { return device_at(tl_device, out); }

const ocu_backend *ocu_rt_backend(void)
{
    rt_device *d;
    if (set_error(current(&d)) != cudaSuccess) {
        return NULL;
    }
    return g_backend;
}

ocu_device *ocu_rt_device(void)
{
    rt_device *d;
    if (set_error(current(&d)) != cudaSuccess) {
        return NULL;
    }
    return d->dev;
}

cudaError_t cudaGetDeviceCount(int *count)
{
    if (count == NULL) {
        return set_error(cudaErrorInvalidValue);
    }
    cudaError_t e = init();
    *count = g_device_count > 0 ? g_device_count : 0;
    return set_error(e);
}

cudaError_t cudaGetDevice(int *device)
{
    if (device == NULL) {
        return set_error(cudaErrorInvalidValue);
    }
    *device = tl_device;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device)
{
    cudaError_t e = init();
    if (e != cudaSuccess) {
        return set_error(e);
    }
    if (device < 0 || device >= g_device_count) {
        return set_error(cudaErrorInvalidDevice);
    }
    tl_device = device;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    if (prop == NULL) {
        return set_error(cudaErrorInvalidValue);
    }
    cudaError_t e = init();
    if (e != cudaSuccess) {
        return set_error(e);
    }
    if (device < 0 || device >= g_device_count) {
        return set_error(cudaErrorInvalidDevice);
    }
    ocu_dev_info info;
    ocu_status st = g_backend->dev_info((uint32_t)device, &info);
    if (st != OCU_OK) {
        return from(st);
    }
    memset(prop, 0, sizeof(*prop));
    memcpy(prop->name, info.name, sizeof(prop->name));
    prop->name[sizeof(prop->name) - 1] = '\0';
    prop->totalGlobalMem = (size_t)info.total_memory_bytes;
    prop->sharedMemPerBlock = (size_t)info.max_shared_memory_per_block;
    prop->sharedMemPerBlockOptin = (size_t)info.max_shared_memory_per_block;
    prop->sharedMemPerMultiprocessor = (size_t)info.max_shared_memory_per_block;
    prop->regsPerBlock = 65536;
    prop->regsPerMultiprocessor = 65536;
    prop->warpSize = (int)info.warp_size;
    prop->memPitch = SIZE_MAX / 2;
    prop->maxThreadsPerBlock = (int)info.max_threads_per_block;
    prop->maxThreadsPerMultiProcessor = (int)info.max_threads_per_block;
    prop->maxBlocksPerMultiProcessor = 32;
    for (int i = 0; i < 3; ++i) {
        prop->maxThreadsDim[i] = (int)info.max_block_dim[i];
        prop->maxGridSize[i] = (int)(info.max_grid_dim[i] > 2147483647u
                                         ? 2147483647u : info.max_grid_dim[i]);
    }
    prop->clockRate = 1000000;
    prop->memoryClockRate = 1000000;
    prop->memoryBusWidth = 64;
    prop->totalConstMem = 65536;
    /*
     * Reported device policy (what programs such as llama.cpp's CUDA
     * backend pick kernels from):
     *   compute capability 5.0 -- the generic tier: no tensor-core, async
     *     copy or half-arithmetic paths, which the CPU path cannot run and
     *     which are not needed for correctness. OCU_COMPUTE_CAPABILITY=
     *     "M.m" (for example "8.0") overrides it;
     *   warpSize = the backend's value (32 on CPU and NVIDIA);
     *   sharedMemPerBlockOptin >= 48 KiB on CPU and NVIDIA (Metal reports
     *     its smaller real value);
     *   cudaFuncGetAttributes().ptxVersion = 10*major+minor, below 90, so
     *     programs do not take their Hopper-only launch paths;
     *   cooperative launch: not supported (attribute 0);
     *   PCI ids: synthetic, see cudaDeviceGetPCIBusId.
     */
    prop->major = 5;
    prop->minor = 0;
    {
        const char *cc = getenv("OCU_COMPUTE_CAPABILITY");
        int M, m;
        if (cc != NULL && sscanf(cc, "%d.%d", &M, &m) == 2 && M >= 1 && M < 20 && m >= 0 && m < 10) {
            prop->major = M;
            prop->minor = m;
        }
    }
    prop->pciDomainID = 0;
    prop->pciBusID = 1 + device;
    prop->pciDeviceID = 0;
    prop->textureAlignment = 512;
    prop->deviceOverlap = 1;
    prop->asyncEngineCount = 1;
    prop->multiProcessorCount = 1;
    prop->concurrentKernels = 1;
    prop->unifiedAddressing = info.supports_unified_memory ? 1 : 0;
    prop->integrated = info.supports_unified_memory ? 1 : 0;
    prop->managedMemory = (info.supports_unified_memory || g_backend->mem_alloc_shared) ? 1 : 0;
    prop->concurrentManagedAccess = info.supports_unified_memory ? 1 : 0;
    prop->canMapHostMemory = 1;
    prop->l2CacheSize = 4 * 1024 * 1024;
    return cudaSuccess;
}

cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
    const unsigned known = cudaDeviceScheduleMask | cudaDeviceMapHost | cudaDeviceLmemResizeToMax;
    if ((flags & ~known) != 0) return set_error(cudaErrorInvalidValue);
    unsigned sched = flags & cudaDeviceScheduleMask;
    if (sched != cudaDeviceScheduleAuto && sched != cudaDeviceScheduleSpin &&
        sched != cudaDeviceScheduleYield && sched != cudaDeviceScheduleBlockingSync) {
        return set_error(cudaErrorInvalidValue);
    }
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    return cudaSuccess; /* how the host waits: a hint here */
}

cudaError_t cudaDeviceCanAccessPeer(int *can, int device, int peer)
{
    if (can == NULL) return set_error(cudaErrorInvalidValue);
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) return e;
    if (device < 0 || device >= n || peer < 0 || peer >= n) return set_error(cudaErrorInvalidDevice);
    *can = 0; /* no peer access between devices yet; a device is not its own peer */
    return cudaSuccess;
}

/*
 * Peer access. Devices here never reach each other's memory
 * (cudaDeviceCanAccessPeer reports 0), so enabling it is the documented
 * failure for an incapable pair: cudaErrorPeerAccessUnsupported. Checks
 * first, in the documented order: flags must be 0; the peer must be a valid
 * device other than the current one (cudaErrorInvalidDevice).
 */
cudaError_t cudaDeviceEnablePeerAccess(int peer, unsigned int flags)
{
    if (flags != 0) return set_error(cudaErrorInvalidValue);
    int can = 0;
    cudaError_t e = cudaDeviceCanAccessPeer(&can, tl_device, peer);
    if (e != cudaSuccess) return e;
    if (peer == tl_device) return set_error(cudaErrorInvalidDevice);
    (void)can; /* can is always 0 until a backend reports peers */
    return set_error(cudaErrorPeerAccessUnsupported);
}

cudaError_t cudaDeviceDisablePeerAccess(int peer)
{
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) return e;
    if (peer < 0 || peer >= n || peer == tl_device) return set_error(cudaErrorInvalidDevice);
    return set_error(cudaErrorPeerAccessNotEnabled); /* never enabled */
}

/*
 * Synthetic PCI identifier "0000:BB:00.0" (domain, bus, device, function in
 * hex, as the documented format): bus = 1 + device index, so it is stable
 * across runs, unique per device and never the host bridge's bus 0. The
 * backends do not report real PCI locations. The string needs 13 bytes with
 * the terminator; a shorter len is cudaErrorInvalidValue.
 */
cudaError_t cudaDeviceGetPCIBusId(char *id, int len, int device)
{
    if (id == NULL || len < 13) return set_error(cudaErrorInvalidValue);
    struct cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, device);
    if (e != cudaSuccess) return e;
    snprintf(id, (size_t)len, "%04x:%02x:%02x.%x", (unsigned)p.pciDomainID,
             (unsigned)p.pciBusID & 0xffu, (unsigned)p.pciDeviceID & 0x1fu, 0u);
    return cudaSuccess;
}

/* No backend can promise that every block of a grid is resident at once,
   which cooperative launch requires: report "not supported", as for a
   device whose cooperative-launch attribute is 0. */
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 grid, dim3 block, void **args,
                                        size_t shared_mem, cudaStream_t stream)
{
    (void)func; (void)grid; (void)block; (void)args; (void)shared_mem; (void)stream;
    return set_error(cudaErrorNotSupported);
}

/* Shared memory a kernel may ask for at launch, raised per function up to
   the device's opt-in limit. Kernel attributes come from the device limits:
   the CPU path compiles kernels without static resource counts. */
cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value)
{
    if (func == NULL) return set_error(cudaErrorInvalidDeviceFunction);
    struct cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, tl_device);
    if (e != cudaSuccess) return e;
    switch (attr) {
    case cudaFuncAttributeMaxDynamicSharedMemorySize:
        if (value < 0 || (size_t)value > p.sharedMemPerBlockOptin) {
            return set_error(cudaErrorInvalidValue);
        }
        return cudaSuccess;
    case cudaFuncAttributePreferredSharedMemoryCarveout:
        if (value < -1 || value > 100) return set_error(cudaErrorInvalidValue);
        return cudaSuccess;
    default:
        return set_error(cudaErrorInvalidValue);
    }
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)
{
    if (attr == NULL) return set_error(cudaErrorInvalidValue);
    if (func == NULL) return set_error(cudaErrorInvalidDeviceFunction);
    struct cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, tl_device);
    if (e != cudaSuccess) return e;
    memset(attr, 0, sizeof(*attr));
    attr->maxThreadsPerBlock = p.maxThreadsPerBlock;
    attr->maxDynamicSharedSizeBytes = (int)p.sharedMemPerBlock;
    attr->ptxVersion = p.major * 10 + p.minor;
    attr->binaryVersion = p.major * 10 + p.minor;
    attr->preferredShmemCarveout = -1;
    return cudaSuccess;
}

/* Resident blocks per multiprocessor, from the block size and dynamic
   shared memory against the device limits (no register limit: none is
   known for CPU-path kernels). */
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, const void *func,
                                                          int block_size, size_t dynamic_smem)
{
    if (blocks == NULL || block_size <= 0) return set_error(cudaErrorInvalidValue);
    if (func == NULL) return set_error(cudaErrorInvalidDeviceFunction);
    struct cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, tl_device);
    if (e != cudaSuccess) return e;
    if (block_size > p.maxThreadsPerBlock || dynamic_smem > p.sharedMemPerBlock) {
        *blocks = 0;
        return cudaSuccess;
    }
    /* Resources are given out in whole warps. */
    int warps = (block_size + p.warpSize - 1) / p.warpSize;
    int by_threads = p.maxThreadsPerMultiProcessor / (warps * p.warpSize);
    int by_smem = dynamic_smem ? (int)(p.sharedMemPerMultiprocessor / dynamic_smem) : by_threads;
    int n = by_threads < by_smem ? by_threads : by_smem;
    if (n > p.maxBlocksPerMultiProcessor) n = p.maxBlocksPerMultiProcessor;
    *blocks = n;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
    if (value == NULL) return set_error(cudaErrorInvalidValue);
    struct cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, device);
    if (e != cudaSuccess) return e;
    switch (attr) {
    case cudaDevAttrMaxThreadsPerBlock:          *value = p.maxThreadsPerBlock; break;
    case cudaDevAttrMaxBlockDimX:                *value = p.maxThreadsDim[0]; break;
    case cudaDevAttrMaxBlockDimY:                *value = p.maxThreadsDim[1]; break;
    case cudaDevAttrMaxBlockDimZ:                *value = p.maxThreadsDim[2]; break;
    case cudaDevAttrMaxGridDimX:                 *value = p.maxGridSize[0]; break;
    case cudaDevAttrMaxGridDimY:                 *value = p.maxGridSize[1]; break;
    case cudaDevAttrMaxGridDimZ:                 *value = p.maxGridSize[2]; break;
    case cudaDevAttrMaxSharedMemoryPerBlock:     *value = (int)p.sharedMemPerBlock; break;
    case cudaDevAttrTotalConstantMemory:         *value = (int)p.totalConstMem; break;
    case cudaDevAttrWarpSize:                    *value = p.warpSize; break;
    case cudaDevAttrMaxRegistersPerBlock:        *value = p.regsPerBlock; break;
    case cudaDevAttrClockRate:                   *value = p.clockRate; break;
    case cudaDevAttrMultiProcessorCount:         *value = p.multiProcessorCount; break;
    case cudaDevAttrIntegrated:                  *value = p.integrated; break;
    case cudaDevAttrConcurrentKernels:           *value = p.concurrentKernels; break;
    case cudaDevAttrMemoryClockRate:             *value = p.memoryClockRate; break;
    case cudaDevAttrGlobalMemoryBusWidth:        *value = p.memoryBusWidth; break;
    case cudaDevAttrL2CacheSize:                 *value = p.l2CacheSize; break;
    case cudaDevAttrMaxThreadsPerMultiProcessor: *value = p.maxThreadsPerMultiProcessor; break;
    case cudaDevAttrAsyncEngineCount:            *value = p.asyncEngineCount; break;
    case cudaDevAttrUnifiedAddressing:           *value = p.unifiedAddressing; break;
    case cudaDevAttrComputeCapabilityMajor:      *value = p.major; break;
    case cudaDevAttrComputeCapabilityMinor:      *value = p.minor; break;
    case cudaDevAttrMaxSharedMemoryPerMultiprocessor: *value = (int)p.sharedMemPerMultiprocessor; break;
    case cudaDevAttrManagedMemory:               *value = p.managedMemory; break;
    case cudaDevAttrConcurrentManagedAccess:     *value = p.concurrentManagedAccess; break;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = (int)p.sharedMemPerBlockOptin; break;
    case cudaDevAttrPciBusId:                    *value = p.pciBusID; break;
    case cudaDevAttrPciDeviceId:                 *value = p.pciDeviceID; break;
    case cudaDevAttrPciDomainId:                 *value = p.pciDomainID; break;
    case cudaDevAttrCooperativeLaunch:           *value = 0; break;
    case cudaDevAttrCooperativeMultiDeviceLaunch: *value = 0; break;
    default:
        return set_error(cudaErrorInvalidValue);
    }
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int *version)
{
    if (version == NULL) {
        return set_error(cudaErrorInvalidValue);
    }
    *version = CUDART_VERSION;
    return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int *version)
{
    if (version == NULL) {
        return set_error(cudaErrorInvalidValue);
    }
    *version = CUDART_VERSION;
    return cudaSuccess;
}

/* Synchronise the default stream and every stream of the device. The first
   failure wins; all queues are still waited for. */
static cudaError_t sync_device(rt_device *d)
{
    ocu_status first = g_backend->queue_sync(d->legacy);
    pthread_mutex_lock(&d->lock);
    for (struct ocu_rt_stream *s = d->streams; s != NULL; s = s->next) {
        ocu_status st = g_backend->queue_sync(s->queue);
        if (first == OCU_OK) first = st;
    }
    pthread_mutex_unlock(&d->lock);
    return dev_result(d, first);
}

cudaError_t cudaDeviceSynchronize(void)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) {
        return set_error(e);
    }
    return set_error(sync_device(d));
}

cudaError_t cudaThreadSynchronize(void) { return cudaDeviceSynchronize(); }

cudaError_t cudaDeviceReset(void)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) {
        return set_error(e);
    }
    /* Finish all work; streams and events the program still holds stay
       valid (their queues keep working). */
    (void)sync_device(d);
    atomic_store(&d->sticky, 0);
    tl_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaThreadExit(void) { return cudaDeviceReset(); }

static bool valid_cache(enum cudaFuncCache c)
{
    return c == cudaFuncCachePreferNone || c == cudaFuncCachePreferShared ||
           c == cudaFuncCachePreferL1 || c == cudaFuncCachePreferEqual;
}

/* Per device, as the API describes; read back by cudaDeviceGetCacheConfig. */
static _Atomic int g_cache[MAX_DEVICES];

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache config)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    if (!valid_cache(config)) return set_error(cudaErrorInvalidValue);
    atomic_store(&g_cache[tl_device], (int)config);
    return cudaSuccess;
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *config)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    if (config == NULL) return set_error(cudaErrorInvalidValue);
    *config = (enum cudaFuncCache)atomic_load(&g_cache[tl_device]);
    return cudaSuccess;
}

cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache config)
{
    if (func == NULL) return set_error(cudaErrorInvalidDeviceFunction);
    if (!valid_cache(config)) return set_error(cudaErrorInvalidValue);
    return cudaSuccess; /* a hint: no cache split to choose on this device */
}

/* ---- stream resolution and the default-stream rules ---------------------- */

/*
 * The backend queue for `stream`, after applying the legacy ordering rule
 * for work about to be submitted to it.
 */
typedef struct submission {
    rt_device *d;
    ocu_queue *q;
    bool       held; /* d->submit is held until submit_end */
} submission;

/*
 * Start submitting work to `stream`: apply the legacy ordering rule and
 * pick the queue. For the default stream and blocking streams the
 * submission lock is held until submit_end(), so no other host thread can
 * enqueue in between the ordering check and this enqueue.
 */
static cudaError_t submit_begin(rt_device *d, cudaStream_t stream, submission *sub)
{
    sub->d = d;
    sub->q = NULL;
    sub->held = false;
    if (tl_in_host_fn) {
        /* The API forbids runtime calls from a host function; submitting
           here would wait on itself. */
        return set_error(cudaErrorNotPermitted);
    }
    int sticky = atomic_load(&d->sticky);
    if (sticky != 0) {
        return set_error((cudaError_t)sticky);
    }
    if (stream == NULL || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        pthread_mutex_lock(&d->submit);
        sub->held = true;
        /* Default stream: wait for every blocking stream first. */
        pthread_mutex_lock(&d->lock);
        ocu_status first = OCU_OK;
        for (struct ocu_rt_stream *s = d->streams; s != NULL; s = s->next) {
            if (!(s->flags & cudaStreamNonBlocking)) {
                ocu_status st = g_backend->queue_sync(s->queue);
                if (first == OCU_OK) first = st;
            }
        }
        pthread_mutex_unlock(&d->lock);
        sub->q = d->legacy;
        if (first != OCU_OK) {
            pthread_mutex_unlock(&d->submit);
            sub->held = false;
            return dev_result(d, first);
        }
        return cudaSuccess;
    }
    if (!(stream->flags & cudaStreamNonBlocking)) {
        pthread_mutex_lock(&d->submit);
        sub->held = true;
        /* Blocking stream: wait for the default stream first. */
        ocu_status st = g_backend->queue_sync(d->legacy);
        if (st != OCU_OK) {
            pthread_mutex_unlock(&d->submit);
            sub->held = false;
            return dev_result(d, st);
        }
    }
    sub->q = stream->queue;
    return cudaSuccess;
}

static void submit_end(submission *sub)
{
    if (sub->held) {
        pthread_mutex_unlock(&sub->d->submit);
        sub->held = false;
    }
}

/* The queue a synchronising call on `stream` waits for. */
static ocu_queue *queue_of(rt_device *d, cudaStream_t stream)
{
    if (stream == NULL || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        return d->legacy;
    }
    return stream->queue;
}

/* ---- memory -------------------------------------------------------------- */

static bool range_add(rt_device *d, void *p, size_t size, enum cudaMemoryType type);
static void range_remove(rt_device *d, void *p);
static bool range_find(rt_device *d, const void *p, size_t n, mem_range *out);
static bool range_overlaps(rt_device *d, const void *p, size_t n);

static cudaError_t device_alloc(void **ptr, size_t size, enum cudaMemoryType type)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    if (ptr == NULL) return set_error(cudaErrorInvalidValue);
    ocu_status st = (type == cudaMemoryTypeManaged && g_backend->mem_alloc_shared != NULL)
                        ? g_backend->mem_alloc_shared(d->dev, size, ptr)
                        : g_backend->mem_alloc(d->dev, size, ptr);
    if (st != OCU_OK) return from(st);
    if (*ptr != NULL && !range_add(d, *ptr, size, type)) {
        (void)g_backend->mem_free(d->dev, *ptr);
        *ptr = NULL;
        return set_error(cudaErrorMemoryAllocation);
    }
    return cudaSuccess;
}

cudaError_t cudaMalloc(void **ptr, size_t size)
{
    return device_alloc(ptr, size, cudaMemoryTypeDevice);
}

/* Memory both the host and the device use directly: only where the device
   shares the host's memory (the CPU backend); elsewhere not supported. */
cudaError_t cudaMallocManaged(void **ptr, size_t size, unsigned int flags)
{
    if (flags != cudaMemAttachGlobal && flags != cudaMemAttachHost) {
        return set_error(cudaErrorInvalidValue);
    }
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    ocu_dev_info info;
    ocu_status st = g_backend->dev_info((uint32_t)tl_device, &info);
    if (st != OCU_OK) return from(st);
    if (!info.supports_unified_memory && g_backend->mem_alloc_shared == NULL) {
        return set_error(cudaErrorNotSupported);
    }
    return device_alloc(ptr, size, cudaMemoryTypeManaged);
}

cudaError_t cudaFree(void *ptr)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    /* Forget the range first: once freed, another thread may get the same
       address. Put it back if the backend refuses the free. */
    mem_range r;
    bool known = ptr != NULL && range_find(d, ptr, 1, &r) && r.base == (char *)ptr &&
                 r.type != cudaMemoryTypeHost;
    if (known) range_remove(d, ptr);
    ocu_status st = g_backend->mem_free(d->dev, ptr); /* waits for queued work */
    if (st != OCU_OK && known) (void)range_add(d, r.base, r.size, r.type);
    return from(st);
}

/*
 * Page-lock existing host memory. On a device that shares host memory the
 * range only needs to be known as page-locked (asynchronous copies may then
 * read it later); elsewhere not supported yet.
 */
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
    const unsigned known = cudaHostRegisterPortable | cudaHostRegisterMapped |
                           cudaHostRegisterIoMemory | cudaHostRegisterReadOnly;
    if (ptr == NULL || size == 0 || (flags & ~known) != 0) return set_error(cudaErrorInvalidValue);
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    ocu_dev_info info;
    ocu_status st = g_backend->dev_info((uint32_t)tl_device, &info);
    if (st != OCU_OK) return from(st);
    bool via_backend = !info.supports_unified_memory;
    if ((via_backend && g_backend->host_register == NULL) || (flags & cudaHostRegisterIoMemory)) {
        return set_error(cudaErrorNotSupported);
    }
    if (size > UINTPTR_MAX - (uintptr_t)ptr) return set_error(cudaErrorInvalidValue);
    if (range_overlaps(d, ptr, size)) {
        return set_error(cudaErrorHostMemoryAlreadyRegistered);
    }
    void *dev = ptr;
    if (via_backend) {
        st = g_backend->host_register(d->dev, ptr, size, &dev);
        if (st != OCU_OK) return from(st);
    }
    if (!range_add(d, ptr, size, cudaMemoryTypeHost)) {
        if (via_backend) (void)g_backend->host_unregister(d->dev, ptr);
        return set_error(cudaErrorMemoryAllocation);
    }
    pthread_mutex_lock(&d->lock);
    d->ranges[d->range_count - 1].dev = dev;
    pthread_mutex_unlock(&d->lock);
    return cudaSuccess;
}

cudaError_t cudaHostUnregister(void *ptr)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    mem_range r;
    if (ptr == NULL || !range_find(d, ptr, 1, &r) || r.base != (char *)ptr ||
        r.type != cudaMemoryTypeHost) {
        return set_error(cudaErrorHostMemoryNotRegistered);
    }
    /* Work that may still read it finishes first. */
    cudaError_t s = sync_device(d);
    range_remove(d, ptr);
    if (r.dev != r.base && g_backend->host_unregister != NULL) {
        ocu_status st = g_backend->host_unregister(d->dev, ptr);
        if (st != OCU_OK && s == cudaSuccess) s = from(st);
    }
    return s;
}

/* The device address of page-locked host memory: the same address where
   the device shares host memory. */
cudaError_t cudaHostGetDevicePointer(void **dptr, void *hptr, unsigned int flags)
{
    if (dptr == NULL || flags != 0) return set_error(cudaErrorInvalidValue);
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    mem_range r;
    if (hptr == NULL || !range_find(d, hptr, 1, &r) || r.type == cudaMemoryTypeDevice) {
        return set_error(cudaErrorInvalidValue);
    }
    if (r.dev != r.base || r.type == cudaMemoryTypeManaged) { /* registered or managed */
        *dptr = (char *)r.dev + ((char *)hptr - r.base);
        return cudaSuccess;
    }
    ocu_dev_info info;
    ocu_status st = g_backend->dev_info((uint32_t)tl_device, &info);
    if (st != OCU_OK) return from(st);
    if (!info.supports_unified_memory) return set_error(cudaErrorNotSupported);
    *dptr = hptr;
    return cudaSuccess;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attr, const void *ptr)
{
    if (attr == NULL) return set_error(cudaErrorInvalidValue);
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    memset(attr, 0, sizeof(*attr));
    mem_range r;
    if (ptr == NULL || !range_find(d, ptr, 0, &r)) {
        /* Memory the runtime did not allocate: ordinary host memory. */
        attr->type = cudaMemoryTypeUnregistered;
        attr->device = -2;
        attr->hostPointer = (void *)(uintptr_t)ptr;
        return cudaSuccess;
    }
    attr->type = r.type;
    attr->device = tl_device;
    ocu_dev_info info;
    bool unified = g_backend->dev_info((uint32_t)tl_device, &info) == OCU_OK &&
                   info.supports_unified_memory;
    if (r.type == cudaMemoryTypeHost || r.type == cudaMemoryTypeManaged || unified) {
        attr->hostPointer = (void *)(uintptr_t)ptr;
    }
    attr->devicePointer = (void *)(uintptr_t)ptr;
    return cudaSuccess;
}

static bool range_add(rt_device *d, void *p, size_t size, enum cudaMemoryType type)
{
    pthread_mutex_lock(&d->lock);
    bool ok = true;
    if (d->range_count == d->range_cap) {
        size_t cap = d->range_cap ? d->range_cap * 2 : 16;
        mem_range *n = realloc(d->ranges, cap * sizeof(*n));
        if (n == NULL) {
            ok = false;
        } else {
            d->ranges = n;
            d->range_cap = cap;
        }
    }
    if (ok) {
        d->ranges[d->range_count].base = p;
        d->ranges[d->range_count].size = size;
        d->ranges[d->range_count].type = type;
        d->ranges[d->range_count].dev = p;
        d->range_count++;
    }
    pthread_mutex_unlock(&d->lock);
    return ok;
}

static void range_remove(rt_device *d, void *p)
{
    pthread_mutex_lock(&d->lock);
    for (size_t i = 0; i < d->range_count; ++i) {
        if (d->ranges[i].base == (char *)p) {
            d->ranges[i] = d->ranges[--d->range_count];
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
}

/* The allocation holding [p, p + n), or false. *out gets a copy. */
static bool range_find(rt_device *d, const void *p, size_t n, mem_range *out)
{
    const char *c = p;
    bool found = false;
    pthread_mutex_lock(&d->lock);
    for (size_t i = 0; i < d->range_count && !found; ++i) {
        const mem_range *r = &d->ranges[i];
        if (c >= r->base && (size_t)(c - r->base) <= r->size &&
            n <= r->size - (size_t)(c - r->base) && (n > 0 || c < r->base + r->size || r->size == 0)) {
            *out = *r;
            found = true;
        }
    }
    pthread_mutex_unlock(&d->lock);
    return found;
}

/* True when [p, p + n) shares a byte with any recorded range. */
static bool range_overlaps(rt_device *d, const void *p, size_t n)
{
    uintptr_t a = (uintptr_t)p, b = a + n;
    bool hit = false;
    pthread_mutex_lock(&d->lock);
    for (size_t i = 0; i < d->range_count && !hit; ++i) {
        uintptr_t ra = (uintptr_t)d->ranges[i].base, rb = ra + d->ranges[i].size;
        hit = a < rb && ra < b;
    }
    pthread_mutex_unlock(&d->lock);
    return hit;
}

static bool pinned_contains(rt_device *d, const void *p, size_t n)
{
    mem_range r;
    return range_find(d, p, n, &r) && r.type == cudaMemoryTypeHost;
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    if (ptr == NULL) return set_error(cudaErrorInvalidValue);
    ocu_status st = g_backend->host_alloc_pinned(d->dev, size, ptr);
    if (st != OCU_OK) return from(st);
    if (*ptr != NULL && !range_add(d, *ptr, size, cudaMemoryTypeHost)) {
        (void)g_backend->host_free_pinned(d->dev, *ptr);
        *ptr = NULL;
        return set_error(cudaErrorMemoryAllocation);
    }
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void *ptr)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    ocu_status st = g_backend->host_free_pinned(d->dev, ptr);
    if (st == OCU_OK) range_remove(d, ptr);
    return from(st);
}

cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags)
{
    const unsigned known = cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined;
    if ((flags & ~known) != 0) return set_error(cudaErrorInvalidValue);
    return cudaMallocHost(ptr, size);
}

/* Rows start on this boundary, the alignment every allocation has. */
#define PITCH_ALIGN 256u

cudaError_t cudaMallocPitch(void **ptr, size_t *pitch, size_t width, size_t height)
{
    if (ptr == NULL || pitch == NULL) return set_error(cudaErrorInvalidValue);
    if (width > SIZE_MAX - PITCH_ALIGN) return set_error(cudaErrorMemoryAllocation);
    size_t p = (width + PITCH_ALIGN - 1) / PITCH_ALIGN * PITCH_ALIGN;
    if (height != 0 && p > SIZE_MAX / height) return set_error(cudaErrorMemoryAllocation);
    cudaError_t e = cudaMalloc(ptr, p * height);
    if (e == cudaSuccess) *pitch = p;
    return e;
}

cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    if (free_bytes == NULL || total_bytes == NULL) return set_error(cudaErrorInvalidValue);
    uint64_t f = 0, t = 0;
    ocu_status st = g_backend->mem_info(d->dev, &f, &t);
    if (st == OCU_OK) {
        *free_bytes = (size_t)f;
        *total_bytes = (size_t)t;
    }
    return from(st);
}

static cudaError_t copy_dir(enum cudaMemcpyKind kind, ocu_copy_dir *dir)
{
    switch (kind) {
    case cudaMemcpyHostToDevice:   *dir = OCU_COPY_HOST_TO_DEVICE; return cudaSuccess;
    case cudaMemcpyDeviceToHost:   *dir = OCU_COPY_DEVICE_TO_HOST; return cudaSuccess;
    case cudaMemcpyDeviceToDevice: *dir = OCU_COPY_DEVICE_TO_DEVICE; return cudaSuccess;
    case cudaMemcpyHostToHost:     return cudaErrorNotReady; /* sentinel: plain copy */
    case cudaMemcpyDefault:        return cudaErrorNotSupported; /* resolved by copy() */
    }
    return cudaErrorInvalidMemcpyDirection;
}

/* The direction cudaMemcpyDefault stands for: pointers the runtime
   allocated on the device (or as managed memory) are device side. */
static enum cudaMemcpyKind infer_kind(rt_device *d, void *dst, const void *src, size_t count)
{
    /* By the start address only: a copy running past the allocation must
       reach the backend's range check, not become a host copy. */
    (void)count;
    mem_range r;
    bool sd = range_find(d, src, 1, &r) && r.type != cudaMemoryTypeHost;
    bool dd = range_find(d, dst, 1, &r) && r.type != cudaMemoryTypeHost;
    return sd ? (dd ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost)
              : (dd ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost);
}

static cudaError_t copy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
                        cudaStream_t stream, bool wait)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    if (count == 0) return cudaSuccess;
    if (kind == cudaMemcpyDefault) kind = infer_kind(d, dst, src, count);
    ocu_copy_dir dir;
    e = copy_dir(kind, &dir);
    submission sub;
    if (e == cudaErrorNotReady) { /* host to host: ordered after the stream */
        e = submit_begin(d, stream, &sub);
        if (e != cudaSuccess) return e;
        ocu_status st = g_backend->queue_sync(sub.q);
        if (st == OCU_OK && (dst == NULL || src == NULL)) st = OCU_ERR_INVALID_VALUE;
        if (st == OCU_OK) memmove(dst, src, count);
        submit_end(&sub);
        return dev_result(d, st);
    }
    if (e != cudaSuccess) return set_error(e);
    /* Pageable host memory: the documented behaviour is that the source is
       consumed (host to device) or the result is in place (device to host)
       when the call returns, so such copies complete before returning. */
    if (!wait) {
        const void *host = dir == OCU_COPY_HOST_TO_DEVICE ? src
                         : dir == OCU_COPY_DEVICE_TO_HOST ? (const void *)dst : NULL;
        if (host != NULL && !pinned_contains(d, host, count)) wait = true;
    }
    e = submit_begin(d, stream, &sub);
    if (e != cudaSuccess) return e;
    ocu_status st = g_backend->mem_copy(d->dev, sub.q, dst, src, count, dir);
    submit_end(&sub);
    if (st != OCU_OK) return from(st); /* rejected: bad pointers or range */
    if (wait) return dev_result(d, g_backend->queue_sync(sub.q));
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return copy(dst, src, count, kind, NULL, true);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return copy(dst, src, count, kind, stream, false);
}

static cudaError_t copy2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                          size_t width, size_t height, enum cudaMemcpyKind kind,
                          cudaStream_t stream, bool wait)
{
    if (width > dpitch || width > spitch) return set_error(cudaErrorInvalidValue);
    if (width == 0 || height == 0) return cudaSuccess;
    for (size_t row = 0; row < height; ++row) {
        cudaError_t e = copy((char *)dst + row * dpitch, (const char *)src + row * spitch,
                             width, kind, stream, false);
        if (e != cudaSuccess) return e;
    }
    if (wait) {
        rt_device *d;
        cudaError_t e = current(&d);
        if (e != cudaSuccess) return set_error(e);
        return dev_result(d, g_backend->queue_sync(queue_of(d, stream)));
    }
    return cudaSuccess;
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                         size_t width, size_t height, enum cudaMemcpyKind kind)
{
    return copy2d(dst, dpitch, src, spitch, width, height, kind, NULL, true);
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                              size_t width, size_t height, enum cudaMemcpyKind kind,
                              cudaStream_t stream)
{
    return copy2d(dst, dpitch, src, spitch, width, height, kind, stream, false);
}

static cudaError_t fill(void *ptr, int value, size_t count, cudaStream_t stream)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    submission sub;
    e = submit_begin(d, stream, &sub);
    if (e != cudaSuccess) return e;
    ocu_status st = g_backend->mem_set(d->dev, sub.q, ptr, value, count);
    submit_end(&sub);
    return from(st);
}

cudaError_t cudaMemset(void *ptr, int value, size_t count)
{
    return fill(ptr, value, count, NULL);
}

cudaError_t cudaMemsetAsync(void *ptr, int value, size_t count, cudaStream_t stream)
{
    return fill(ptr, value, count, stream);
}

cudaError_t cudaMemset2D(void *ptr, size_t pitch, int value, size_t width, size_t height)
{
    if (width > pitch) return set_error(cudaErrorInvalidValue);
    if (width == 0 || height == 0) return cudaSuccess;
    for (size_t row = 0; row < height; ++row) {
        cudaError_t e = fill((char *)ptr + row * pitch, value, width, NULL);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

/*
 * Symbols: the address of a __device__ / __constant__ variable. On
 * backends with unified memory (the CPU backend) that host address is the
 * device address and the copy is a plain copy after the default stream.
 * Other backends need module-level symbol lookup: not supported yet.
 */
/*
 * Copy to or from a __device__ / __constant__ variable, ordered after the
 * stream's work and complete on return. Variables registered by compiled
 * device code are copied through the backend (module_get_global); on a
 * backend whose device shares host memory the host address is the device
 * address and the copy is a plain copy.
 */
static cudaError_t symbol_copy(const void *symbol, size_t offset, void *host, size_t count,
                               bool to_device, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    void *dptr = NULL;
    size_t bytes = 0;
    ocu_status st = ocu_rt_find_symbol(symbol, &dptr, &bytes);
    if (st == OCU_OK) {
        if (offset > bytes || count > bytes - offset) return set_error(cudaErrorInvalidValue);
        if (count == 0) return cudaSuccess;
        ocu_copy_dir dir;
        if (kind == cudaMemcpyDeviceToDevice) {
            dir = OCU_COPY_DEVICE_TO_DEVICE;
        } else if (kind == cudaMemcpyDefault) {
            mem_range r;
            bool other_dev = range_find(d, host, 1, &r) && r.type != cudaMemoryTypeHost;
            dir = other_dev ? OCU_COPY_DEVICE_TO_DEVICE
                            : (to_device ? OCU_COPY_HOST_TO_DEVICE : OCU_COPY_DEVICE_TO_HOST);
        } else {
            dir = to_device ? OCU_COPY_HOST_TO_DEVICE : OCU_COPY_DEVICE_TO_HOST;
        }
        char *dev = (char *)dptr + offset;
        submission sub;
        e = submit_begin(d, stream, &sub);
        if (e != cudaSuccess) return e;
        st = to_device ? g_backend->mem_copy(d->dev, sub.q, dev, host, count, dir)
                       : g_backend->mem_copy(d->dev, sub.q, host, dev, count, dir);
        submit_end(&sub);
        if (st != OCU_OK) return from(st);
        return dev_result(d, g_backend->queue_sync(sub.q));
    }
    if (st != OCU_ERR_NOT_FOUND) return from(st);
    ocu_dev_info info;
    st = g_backend->dev_info((uint32_t)tl_device, &info);
    if (st != OCU_OK) return from(st);
    if (!info.supports_unified_memory) return set_error(cudaErrorInvalidSymbol);
    submission sub;
    e = submit_begin(d, stream, &sub); /* ordered after the stream's work */
    if (e != cudaSuccess) return e;
    st = g_backend->queue_sync(sub.q);
    char *addr = (char *)(uintptr_t)symbol + offset;
    if (st == OCU_OK && count > 0) {
        if (to_device) memmove(addr, host, count);
        else memmove(host, addr, count);
    }
    submit_end(&sub);
    return dev_result(d, st);
}

static cudaError_t to_symbol(const void *symbol, const void *src, size_t count,
                             size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    if (symbol == NULL || (src == NULL && count > 0)) return set_error(cudaErrorInvalidSymbol);
    if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToDevice &&
        kind != cudaMemcpyDefault) {
        return set_error(cudaErrorInvalidMemcpyDirection);
    }
    return symbol_copy(symbol, offset, (void *)(uintptr_t)src, count, true, kind, stream);
}

static cudaError_t from_symbol(void *dst, const void *symbol, size_t count, size_t offset,
                               enum cudaMemcpyKind kind, cudaStream_t stream)
{
    if (symbol == NULL || (dst == NULL && count > 0)) return set_error(cudaErrorInvalidSymbol);
    if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDeviceToDevice &&
        kind != cudaMemcpyDefault) {
        return set_error(cudaErrorInvalidMemcpyDirection);
    }
    return symbol_copy(symbol, offset, dst, count, false, kind, stream);
}

cudaError_t cudaGetSymbolAddress(void **dev_ptr, const void *symbol)
{
    if (dev_ptr == NULL || symbol == NULL) return set_error(cudaErrorInvalidSymbol);
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    size_t bytes = 0;
    ocu_status st = ocu_rt_find_symbol(symbol, dev_ptr, &bytes);
    if (st == OCU_OK) return cudaSuccess;
    if (st != OCU_ERR_NOT_FOUND) return from(st);
    ocu_dev_info info;
    if (g_backend->dev_info((uint32_t)tl_device, &info) == OCU_OK && info.supports_unified_memory) {
        *dev_ptr = (void *)(uintptr_t)symbol;
        return cudaSuccess;
    }
    return set_error(cudaErrorInvalidSymbol);
}

cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol)
{
    if (size == NULL || symbol == NULL) return set_error(cudaErrorInvalidSymbol);
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    void *p = NULL;
    ocu_status st = ocu_rt_find_symbol(symbol, &p, size);
    if (st == OCU_OK) return cudaSuccess;
    if (st != OCU_ERR_NOT_FOUND) return from(st);
    return set_error(cudaErrorInvalidSymbol); /* sizes are known for registered variables only */
}

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count,
                               size_t offset, enum cudaMemcpyKind kind)
{
    return to_symbol(symbol, src, count, offset, kind, NULL);
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset,
                                 enum cudaMemcpyKind kind)
{
    return from_symbol(dst, symbol, count, offset, kind, NULL);
}

/* The asynchronous forms complete before returning, which the API allows. */
cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count,
                                    size_t offset, enum cudaMemcpyKind kind,
                                    cudaStream_t stream)
{
    return to_symbol(symbol, src, count, offset, kind, stream);
}

cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count,
                                      size_t offset, enum cudaMemcpyKind kind,
                                      cudaStream_t stream)
{
    return from_symbol(dst, symbol, count, offset, kind, stream);
}

/* ---- streams ------------------------------------------------------------- */

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
{
    if (stream == NULL || (flags & ~cudaStreamNonBlocking) != 0) {
        return set_error(cudaErrorInvalidValue);
    }
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    struct ocu_rt_stream *s = calloc(1, sizeof(*s));
    if (s == NULL) return set_error(cudaErrorMemoryAllocation);
    ocu_status st = g_backend->queue_create(d->dev, &s->queue);
    if (st != OCU_OK) {
        free(s);
        return from(st);
    }
    s->flags = flags;
    s->device = tl_device;
    pthread_mutex_lock(&d->lock);
    s->next = d->streams;
    d->streams = s;
    pthread_mutex_unlock(&d->lock);
    *stream = s;
    return cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t *stream)
{
    return cudaStreamCreateWithFlags(stream, cudaStreamDefault);
}

/* One priority level: every requested priority is clamped to it (the
   documented behaviour for values outside the range). */
cudaError_t cudaDeviceGetStreamPriorityRange(int *least, int *greatest)
{
    if (least != NULL) *least = 0;
    if (greatest != NULL) *greatest = 0;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority)
{
    (void)priority;
    cudaError_t e = cudaStreamCreateWithFlags(stream, flags);
    if (e == cudaSuccess) (*stream)->priority = 0;
    return e;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority)
{
    if (priority == NULL) return set_error(cudaErrorInvalidValue);
    *priority = 0;
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags)
{
    if (flags == NULL) return set_error(cudaErrorInvalidValue);
    *flags = (stream == NULL || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
                 ? cudaStreamDefault : stream->flags;
    return cudaSuccess;
}

/* Stream capture (graphs) is not implemented: nothing is ever capturing,
   and starting a capture reports not-supported. */
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *status)
{
    (void)stream;
    if (status == NULL) return set_error(cudaErrorInvalidValue);
    *status = cudaStreamCaptureStatusNone;
    return cudaSuccess;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode)
{
    (void)stream;
    (void)mode;
    return set_error(cudaErrorNotSupported);
}

/*
 * A host function ordered in `stream`: work submitted before it finishes
 * first, and it finishes before this call returns, so it also runs before
 * anything submitted later. The caller waits, which the API allows.
 */
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *user_data)
{
    if (fn == NULL) return set_error(cudaErrorInvalidValue);
    rt_device *d;
    cudaError_t e = (stream == NULL || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
                        ? current(&d) : device_at(stream->device, &d);
    if (e != cudaSuccess) return set_error(e);
    submission sub;
    e = submit_begin(d, stream, &sub);
    if (e != cudaSuccess) return e;
    ocu_status st = g_backend->queue_sync(sub.q);
    if (st == OCU_OK) {
        tl_in_host_fn = true;
        fn(user_data);
        tl_in_host_fn = false;
    }
    submit_end(&sub);
    return dev_result(d, st);
}

static bool is_default(cudaStream_t s)
{
    return s == NULL || s == cudaStreamLegacy || s == cudaStreamPerThread;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    if (is_default(stream)) return set_error(cudaErrorInvalidResourceHandle);
    rt_device *d;
    cudaError_t e = device_at(stream->device, &d);
    if (e != cudaSuccess) return set_error(e);
    pthread_mutex_lock(&d->lock);
    bool found = false;
    for (struct ocu_rt_stream **pp = &d->streams; *pp != NULL; pp = &(*pp)->next) {
        if (*pp == stream) {
            *pp = stream->next;
            found = true;
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
    if (!found) return set_error(cudaErrorInvalidResourceHandle);
    ocu_status st = g_backend->queue_destroy(stream->queue); /* finishes queued work */
    free(stream);
    return from(st);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    rt_device *d;
    cudaError_t e = is_default(stream) ? current(&d) : device_at(stream->device, &d);
    if (e != cudaSuccess) return set_error(e);
    return dev_result(d, g_backend->queue_sync(queue_of(d, stream)));
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    rt_device *d;
    cudaError_t e = is_default(stream) ? current(&d) : device_at(stream->device, &d);
    if (e != cudaSuccess) return set_error(e);
    return dev_result(d, g_backend->queue_query(queue_of(d, stream)));
}

/*
 * Later work on `stream` waits for the event's current record. The wait is
 * done on the host here, which keeps the order and gives up overlap.
 */
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    (void)stream;
    if (event == NULL || flags != 0) return set_error(cudaErrorInvalidValue);
    rt_device *d;
    cudaError_t e = device_at(event->device, &d);
    if (e != cudaSuccess) return set_error(e);
    return dev_result(d, g_backend->marker_wait(event->marker));
}

/* ---- events -------------------------------------------------------------- */

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    const unsigned known = cudaEventBlockingSync | cudaEventDisableTiming | cudaEventInterprocess;
    if (event == NULL || (flags & ~known) != 0) return set_error(cudaErrorInvalidValue);
    if (flags & cudaEventInterprocess) return set_error(cudaErrorNotSupported);
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    struct ocu_rt_event *ev = calloc(1, sizeof(*ev));
    if (ev == NULL) return set_error(cudaErrorMemoryAllocation);
    ocu_status st = g_backend->marker_create(d->dev, &ev->marker);
    if (st != OCU_OK) {
        free(ev);
        return from(st);
    }
    ev->flags = flags;
    ev->device = tl_device;
    *event = ev;
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    if (event == NULL) return set_error(cudaErrorInvalidResourceHandle);
    ocu_status st = g_backend->marker_destroy(event->marker);
    free(event);
    return from(st);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    if (event == NULL) return set_error(cudaErrorInvalidResourceHandle);
    rt_device *d;
    cudaError_t e = device_at(event->device, &d);
    if (e != cudaSuccess) return set_error(e);
    submission sub;
    e = submit_begin(d, stream, &sub);
    if (e != cudaSuccess) return e;
    ocu_status st = g_backend->marker_record(event->marker, sub.q);
    submit_end(&sub);
    return from(st);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    if (event == NULL) return set_error(cudaErrorInvalidResourceHandle);
    rt_device *d;
    cudaError_t e = device_at(event->device, &d);
    if (e != cudaSuccess) return set_error(e);
    return dev_result(d, g_backend->marker_wait(event->marker));
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
    if (event == NULL) return set_error(cudaErrorInvalidResourceHandle);
    rt_device *d;
    cudaError_t e = device_at(event->device, &d);
    if (e != cudaSuccess) return set_error(e);
    return dev_result(d, g_backend->marker_query(event->marker));
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    if (ms == NULL) return set_error(cudaErrorInvalidValue);
    if (start == NULL || end == NULL) return set_error(cudaErrorInvalidResourceHandle);
    if ((start->flags | end->flags) & cudaEventDisableTiming) {
        return set_error(cudaErrorInvalidResourceHandle);
    }
    return from(g_backend->marker_elapsed_ms(start->marker, end->marker, ms));
}

/* ---- launches ------------------------------------------------------------ */

/*
 * A launch by the host address of a kernel (the compiler's launch stub, or
 * a user call of cudaLaunchKernel): the kernel registered under that
 * address, with argument sizes from the device code.
 */
cudaError_t cudaLaunchKernel(const void *func, dim3 grid, dim3 block, void **args,
                             size_t shared, cudaStream_t stream)
{
    if (func == NULL) return set_error(cudaErrorInvalidDeviceFunction);
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    ocu_kernel *k = NULL;
    uint32_t nargs = 0;
    const size_t *sizes = NULL;
    ocu_status st = ocu_rt_find_kernel(func, &k, &nargs, &sizes);
    if (st == OCU_ERR_NOT_FOUND) return set_error(cudaErrorInvalidDeviceFunction);
    if (st != OCU_OK) {
        return set_error(st == OCU_ERR_NOT_SUPPORTED ? cudaErrorNoKernelImageForDevice
                                                     : ocu_rt_error(st));
    }
    if (nargs > 0 && args == NULL) return set_error(cudaErrorInvalidValue);
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = grid.x; p.grid[1] = grid.y; p.grid[2] = grid.z;
    p.block[0] = block.x; p.block[1] = block.y; p.block[2] = block.z;
    p.shared_memory_bytes = shared;
    p.args = args;
    p.arg_sizes = sizes;
    p.num_args = nargs;
    return ocu_rt_launch(k, &p, stream);
}

/*
 * Extensible launch configuration (see cuda_runtime_api.h for what each
 * attribute does here). Programmatic dependent launch only relaxes stream
 * order; this layer keeps full stream order, which the attribute allows.
 */
cudaError_t ocu_rt_launch_config_check(const cudaLaunchConfig_t *config)
{
    if (config == NULL) return set_error(cudaErrorInvalidValue);
    if (config->numAttrs > 0 && config->attrs == NULL) return set_error(cudaErrorInvalidValue);
    for (unsigned int i = 0; i < config->numAttrs; ++i) {
        const cudaLaunchAttribute *a = &config->attrs[i];
        switch (a->id) {
        case cudaLaunchAttributeIgnore:
        case cudaLaunchAttributeAccessPolicyWindow:
        case cudaLaunchAttributeSynchronizationPolicy:
        case cudaLaunchAttributeClusterSchedulingPolicyPreference:
        case cudaLaunchAttributeProgrammaticStreamSerialization:
        case cudaLaunchAttributePriority:
        case cudaLaunchAttributeMemSyncDomainMap:
        case cudaLaunchAttributeMemSyncDomain:
        case cudaLaunchAttributePreferredClusterDimension:
        case cudaLaunchAttributePreferredSharedMemoryCarveout:
            break;
        case cudaLaunchAttributeProgrammaticEvent:
            if (a->val.programmaticEvent.event == NULL) {
                return set_error(cudaErrorInvalidResourceHandle);
            }
            break;
        case cudaLaunchAttributeLaunchCompletionEvent:
            if (a->val.launchCompletionEvent.event == NULL) {
                return set_error(cudaErrorInvalidResourceHandle);
            }
            break;
        case cudaLaunchAttributeCooperative:
            if (a->val.cooperative != 0) return set_error(cudaErrorNotSupported);
            break;
        case cudaLaunchAttributeClusterDimension: {
            unsigned long long n = (unsigned long long)a->val.clusterDim.x *
                                   a->val.clusterDim.y * a->val.clusterDim.z;
            if (n > 1) return set_error(cudaErrorNotSupported);
            break;
        }
        default:
            return set_error(cudaErrorInvalidValue);
        }
    }
    return cudaSuccess;
}

cudaError_t ocu_rt_launch_config_done(const cudaLaunchConfig_t *config)
{
    for (unsigned int i = 0; i < config->numAttrs; ++i) {
        const cudaLaunchAttribute *a = &config->attrs[i];
        cudaEvent_t ev = NULL;
        if (a->id == cudaLaunchAttributeProgrammaticEvent) {
            ev = a->val.programmaticEvent.event;
        } else if (a->id == cudaLaunchAttributeLaunchCompletionEvent) {
            ev = a->val.launchCompletionEvent.event;
        }
        if (ev != NULL) {
            cudaError_t e = cudaEventRecord(ev, config->stream);
            if (e != cudaSuccess) return e;
        }
    }
    return cudaSuccess;
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args)
{
    cudaError_t e = ocu_rt_launch_config_check(config);
    if (e != cudaSuccess) return e;
    e = cudaLaunchKernel(func, config->gridDim, config->blockDim, args,
                         config->dynamicSmemBytes, config->stream);
    if (e != cudaSuccess) return e;
    return ocu_rt_launch_config_done(config);
}

cudaError_t ocu_rt_launch(ocu_kernel *kernel, const ocu_launch_params *params,
                          cudaStream_t stream)
{
    rt_device *d;
    cudaError_t e = current(&d);
    if (e != cudaSuccess) return set_error(e);
    if (kernel == NULL || params == NULL) return set_error(cudaErrorInvalidDeviceFunction);
    submission sub;
    e = submit_begin(d, stream, &sub);
    if (e != cudaSuccess) return e;
    ocu_status st = g_backend->kernel_launch(sub.q, kernel, params);
    submit_end(&sub);
    if (st == OCU_ERR_INVALID_VALUE) {
        return set_error(cudaErrorInvalidConfiguration); /* dimensions, sizes */
    }
    return from(st);
}

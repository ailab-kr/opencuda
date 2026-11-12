/*
 * nv_driver.h -- the part of the NVIDIA driver API this backend calls.
 *
 * Written for this project from the public CUDA Driver API reference
 * (function names, parameters and the documented values of the constants
 * used here). Not derived from any vendor header. The functions are looked
 * up at run time in the installed driver library (libcuda.so.1); nothing
 * links against vendor files at build time.
 */
#ifndef OCU_NV_DRIVER_H
#define OCU_NV_DRIVER_H

#include <stddef.h>

typedef int nv_result;               /* 0 = success */
typedef int nv_device;
typedef unsigned long long nv_dptr;  /* device address */
typedef struct nv_ctx_st *nv_ctx;
typedef struct nv_mod_st *nv_module;
typedef struct nv_func_st *nv_function;
typedef struct nv_stream_st *nv_stream;
typedef struct nv_event_st *nv_event;

/* Result values used by the backend (documented values). */
enum {
    NV_SUCCESS = 0,
    NV_ERROR_INVALID_VALUE = 1,
    NV_ERROR_OUT_OF_MEMORY = 2,
    NV_ERROR_NOT_INITIALIZED = 3,
    NV_ERROR_NO_DEVICE = 100,
    NV_ERROR_INVALID_DEVICE = 101,
    NV_ERROR_INVALID_IMAGE = 200,
    NV_ERROR_INVALID_CONTEXT = 201,
    NV_ERROR_NO_BINARY_FOR_GPU = 209,
    NV_ERROR_INVALID_PTX = 218,
    NV_ERROR_INVALID_HANDLE = 400,
    NV_ERROR_NOT_FOUND = 500,
    NV_ERROR_NOT_READY = 600,
    NV_ERROR_ILLEGAL_ADDRESS = 700,
    NV_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    NV_ERROR_LAUNCH_TIMEOUT = 702,
    NV_ERROR_LAUNCH_FAILED = 719,
    NV_ERROR_NOT_SUPPORTED = 801
};

/* Device attributes (documented values). */
enum {
    NV_ATTR_MAX_THREADS_PER_BLOCK = 1,
    NV_ATTR_MAX_BLOCK_DIM_X = 2,
    NV_ATTR_MAX_BLOCK_DIM_Y = 3,
    NV_ATTR_MAX_BLOCK_DIM_Z = 4,
    NV_ATTR_MAX_GRID_DIM_X = 5,
    NV_ATTR_MAX_GRID_DIM_Y = 6,
    NV_ATTR_MAX_GRID_DIM_Z = 7,
    NV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    NV_ATTR_WARP_SIZE = 10,
    NV_ATTR_MULTIPROCESSOR_COUNT = 16,
    NV_ATTR_COMPUTE_CAPABILITY_MAJOR = 75,
    NV_ATTR_COMPUTE_CAPABILITY_MINOR = 76
};

/* Options of the PTX loader (documented values). */
enum {
    NV_JIT_INFO_LOG_BUFFER = 3,
    NV_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    NV_JIT_ERROR_LOG_BUFFER = 5,
    NV_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
};

#define NV_STREAM_NON_BLOCKING 0x1u
#define NV_EVENT_DEFAULT 0x0u
#define NV_MEMHOSTALLOC_PORTABLE 0x1u
#define NV_MEMHOSTREGISTER_PORTABLE 0x1u
#define NV_MEMHOSTREGISTER_DEVICEMAP 0x2u
#define NV_MEM_ATTACH_GLOBAL 0x1u

/* The driver entry points the backend uses, resolved by nv_load(). */
typedef struct nv_api {
    nv_result (*Init)(unsigned flags);
    nv_result (*DriverGetVersion)(int *version);
    nv_result (*DeviceGetCount)(int *count);
    nv_result (*DeviceGet)(nv_device *dev, int ordinal);
    nv_result (*DeviceGetName)(char *name, int len, nv_device dev);
    nv_result (*DeviceGetAttribute)(int *value, int attr, nv_device dev);
    nv_result (*DeviceTotalMem)(size_t *bytes, nv_device dev);
    nv_result (*DevicePrimaryCtxRetain)(nv_ctx *ctx, nv_device dev);
    nv_result (*DevicePrimaryCtxRelease)(nv_device dev);
    nv_result (*CtxSetCurrent)(nv_ctx ctx);
    nv_result (*CtxSynchronize)(void);
    nv_result (*MemAlloc)(nv_dptr *ptr, size_t bytes);
    nv_result (*MemFree)(nv_dptr ptr);
    nv_result (*MemGetInfo)(size_t *free_bytes, size_t *total_bytes);
    nv_result (*MemHostAlloc)(void **ptr, size_t bytes, unsigned flags);
    nv_result (*MemFreeHost)(void *ptr);
    nv_result (*MemcpyHtoDAsync)(nv_dptr dst, const void *src, size_t bytes, nv_stream s);
    nv_result (*MemcpyDtoHAsync)(void *dst, nv_dptr src, size_t bytes, nv_stream s);
    nv_result (*MemcpyDtoDAsync)(nv_dptr dst, nv_dptr src, size_t bytes, nv_stream s);
    nv_result (*MemsetD8Async)(nv_dptr dst, unsigned char value, size_t n, nv_stream s);
    nv_result (*StreamCreate)(nv_stream *s, unsigned flags);
    nv_result (*StreamDestroy)(nv_stream s);
    nv_result (*StreamSynchronize)(nv_stream s);
    nv_result (*StreamQuery)(nv_stream s);
    nv_result (*ModuleLoadDataEx)(nv_module *m, const void *image, unsigned n_options,
                                  int *options, void **values);
    nv_result (*ModuleUnload)(nv_module m);
    nv_result (*ModuleGetFunction)(nv_function *f, nv_module m, const char *name);
    nv_result (*ModuleGetGlobal)(nv_dptr *ptr, size_t *bytes, nv_module m, const char *name);
    nv_result (*LaunchKernel)(nv_function f, unsigned gx, unsigned gy, unsigned gz,
                              unsigned bx, unsigned by, unsigned bz, unsigned shared,
                              nv_stream s, void **params, void **extra);
    nv_result (*EventCreate)(nv_event *e, unsigned flags);
    nv_result (*EventDestroy)(nv_event e);
    nv_result (*EventRecord)(nv_event e, nv_stream s);
    nv_result (*EventSynchronize)(nv_event e);
    nv_result (*EventQuery)(nv_event e);
    nv_result (*EventElapsedTime)(float *ms, nv_event start, nv_event end);
    nv_result (*GetErrorName)(nv_result r, const char **name);
    nv_result (*MemAllocManaged)(nv_dptr *ptr, size_t bytes, unsigned flags);
    nv_result (*MemHostRegister)(void *ptr, size_t bytes, unsigned flags);
    nv_result (*MemHostUnregister)(void *ptr);
    nv_result (*MemHostGetDevicePointer)(nv_dptr *dptr, void *ptr, unsigned flags);
} nv_api;

/*
 * Load the driver library and resolve every entry point. Returns NULL when
 * no driver is installed or an entry point is missing (the backend then
 * reports no devices). Thread-safe; loads once.
 */
const nv_api *nv_load(void);

#endif /* OCU_NV_DRIVER_H */

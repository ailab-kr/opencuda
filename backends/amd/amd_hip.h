/*
 * amd_hip.h -- the part of AMD's HIP runtime API the AMD backend calls,
 * resolved at run time from the installed runtime library (amdhip64 on
 * Windows, shipped with the GPU driver; libamdhip64 on Linux, part of
 * ROCm), so building needs nothing from AMD.
 *
 * Written for this project from the public HIP runtime API documentation:
 * function names and parameters as documented. Numeric values that the
 * documentation gives only by name (attribute and error enumerations) were
 * read back from the runtime on a Radeon 780M (gfx1103, driver 3686.0,
 * HIP runtime 7.1, 2026-11-21); the evidence is in backends/amd/README.md
 * section 7. Not derived from any AMD header.
 */
#ifndef OCU_AMD_HIP_H
#define OCU_AMD_HIP_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define AMD_API __stdcall
#else
#define AMD_API
#endif

typedef int hip_result;
typedef int hip_device;
typedef void *hip_stream;
typedef void *hip_module;
typedef void *hip_function;
typedef void *hip_event;

/* results (the CUDA-compatible numbering; observed values) */
#define HIP_SUCCESS                     0
#define HIP_ERROR_INVALID_VALUE         1
#define HIP_ERROR_OUT_OF_MEMORY         2
#define HIP_ERROR_NOT_INITIALIZED       3
#define HIP_ERROR_NO_DEVICE             100
#define HIP_ERROR_INVALID_DEVICE        101
#define HIP_ERROR_INVALID_IMAGE         200
#define HIP_ERROR_INVALID_CONTEXT       201
#define HIP_ERROR_NO_BINARY_FOR_GPU     209
#define HIP_ERROR_INVALID_HANDLE        400
#define HIP_ERROR_NOT_FOUND             500
#define HIP_ERROR_NOT_READY             600
#define HIP_ERROR_ILLEGAL_ADDRESS       700
#define HIP_ERROR_LAUNCH_OUT_OF_RESOURCES 701
#define HIP_ERROR_NOT_SUPPORTED         801

/* device attributes (observed values, README section 7) */
#define HIP_ATTR_CAN_MAP_HOST_MEMORY      3
#define HIP_ATTR_INTEGRATED               16
#define HIP_ATTR_MANAGED_MEMORY           24
#define HIP_ATTR_MAX_BLOCK_DIM_X          26
#define HIP_ATTR_MAX_BLOCK_DIM_Y          27
#define HIP_ATTR_MAX_BLOCK_DIM_Z          28
#define HIP_ATTR_MAX_GRID_DIM_X           29
#define HIP_ATTR_MAX_GRID_DIM_Y           30
#define HIP_ATTR_MAX_GRID_DIM_Z           31
#define HIP_ATTR_MAX_THREADS_PER_BLOCK    56
#define HIP_ATTR_MULTIPROCESSOR_COUNT     63
#define HIP_ATTR_MAX_SHARED_MEMORY_PER_BLOCK 74
#define HIP_ATTR_WARP_SIZE                87

/* flags */
#define HIP_STREAM_NON_BLOCKING           0x1u
#define HIP_EVENT_DEFAULT                 0x0u
#define HIP_HOST_MALLOC_PORTABLE          0x1u
#define HIP_HOST_REGISTER_PORTABLE        0x1u
#define HIP_HOST_REGISTER_MAPPED          0x2u
#define HIP_MEM_ATTACH_GLOBAL             0x1u

/* copy kinds */
#define HIP_MEMCPY_HOST_TO_DEVICE         1
#define HIP_MEMCPY_DEVICE_TO_HOST         2
#define HIP_MEMCPY_DEVICE_TO_DEVICE       3

typedef struct amd_api {
    hip_result (AMD_API *Init)(unsigned flags);
    hip_result (AMD_API *RuntimeGetVersion)(int *version);
    hip_result (AMD_API *GetDeviceCount)(int *count);
    hip_result (AMD_API *DeviceGet)(hip_device *dev, int ordinal);
    hip_result (AMD_API *SetDevice)(int ordinal);
    hip_result (AMD_API *DeviceGetName)(char *name, int len, hip_device dev);
    hip_result (AMD_API *DeviceGetAttribute)(int *value, int attr, int ordinal);
    hip_result (AMD_API *DeviceTotalMem)(size_t *bytes, hip_device dev);
    hip_result (AMD_API *DeviceSynchronize)(void);
    hip_result (AMD_API *Malloc)(void **ptr, size_t bytes);
    hip_result (AMD_API *Free)(void *ptr);
    hip_result (AMD_API *MemGetInfo)(size_t *free_bytes, size_t *total_bytes);
    hip_result (AMD_API *HostMalloc)(void **ptr, size_t bytes, unsigned flags);
    hip_result (AMD_API *HostFree)(void *ptr);
    hip_result (AMD_API *MemcpyAsync)(void *dst, const void *src, size_t bytes, int kind, hip_stream s);
    hip_result (AMD_API *MemsetAsync)(void *dst, int value, size_t bytes, hip_stream s);
    hip_result (AMD_API *StreamCreateWithFlags)(hip_stream *s, unsigned flags);
    hip_result (AMD_API *StreamDestroy)(hip_stream s);
    hip_result (AMD_API *StreamSynchronize)(hip_stream s);
    hip_result (AMD_API *StreamQuery)(hip_stream s);
    hip_result (AMD_API *ModuleLoadData)(hip_module *m, const void *image);
    hip_result (AMD_API *ModuleUnload)(hip_module m);
    hip_result (AMD_API *ModuleGetFunction)(hip_function *f, hip_module m, const char *name);
    hip_result (AMD_API *ModuleGetGlobal)(void **ptr, size_t *bytes, hip_module m, const char *name);
    hip_result (AMD_API *ModuleLaunchKernel)(hip_function f, unsigned gx, unsigned gy, unsigned gz,
                                             unsigned bx, unsigned by, unsigned bz, unsigned shared,
                                             hip_stream s, void **params, void **extra);
    hip_result (AMD_API *EventCreateWithFlags)(hip_event *e, unsigned flags);
    hip_result (AMD_API *EventDestroy)(hip_event e);
    hip_result (AMD_API *EventRecord)(hip_event e, hip_stream s);
    hip_result (AMD_API *EventSynchronize)(hip_event e);
    hip_result (AMD_API *EventQuery)(hip_event e);
    hip_result (AMD_API *EventElapsedTime)(float *ms, hip_event start, hip_event end);
    hip_result (AMD_API *MallocManaged)(void **ptr, size_t bytes, unsigned flags);
    hip_result (AMD_API *HostRegister)(void *ptr, size_t bytes, unsigned flags);
    hip_result (AMD_API *HostUnregister)(void *ptr);
    hip_result (AMD_API *HostGetDevicePointer)(void **dev_ptr, void *host_ptr, unsigned flags);
} amd_api;

/* The runtime's entry points, or NULL when no HIP runtime is installed (the
   backend then reports no devices). OCU_AMD_HIP_LIBRARY names another file. */
const amd_api *amd_load(void);

/* Kernel parameter sizes from a code object's AMDGPU metadata note (the
   declared parameters, without the hidden ones the compiler adds).
   Returns the count, or -1 when the kernel or the note is not found; at
   most `max` sizes are written. amd_meta.c */
int amd_kernel_params(const void *elf, size_t bytes, const char *kernel, size_t *sizes, int max);

#endif /* OCU_AMD_HIP_H */

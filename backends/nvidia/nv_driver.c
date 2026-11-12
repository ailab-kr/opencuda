/*
 * nv_driver.c -- find the installed NVIDIA driver library at run time and
 * resolve the entry points listed in nv_driver.h.
 */
#define _GNU_SOURCE
#include "nv_driver.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define NV_LIB_DEFAULT "nvcuda.dll"
static void *lib_open(const char *p) { return (void *)LoadLibraryA(p); }
static void *lib_sym(void *lib, const char *n) { return (void *)(uintptr_t)GetProcAddress((HMODULE)lib, n); }
static void lib_close(void *lib) { FreeLibrary((HMODULE)lib); }
#else
#include <dlfcn.h>
#define NV_LIB_DEFAULT "libcuda.so.1"
static void *lib_open(const char *p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
static void *lib_sym(void *lib, const char *n) { return dlsym(lib, n); }
static void lib_close(void *lib) { dlclose(lib); }
#endif
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static nv_api g_api;
static const nv_api *g_loaded;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static int resolve(void *lib, void **slot, const char *name)
{
    *slot = lib_sym(lib, name);
    return *slot != NULL;
}

static void load_once(void)
{
    /* OCU_NV_DRIVER names another library file (tests, unusual installs). */
    const char *path = getenv("OCU_NV_DRIVER");
    void *lib = lib_open(path != NULL && path[0] != '\0' ? path : NV_LIB_DEFAULT);
    if (lib == NULL) {
        return;
    }
    nv_api a;
    memset(&a, 0, sizeof(a));
    int ok = 1;
#define NV_SYM(field, name) ok &= resolve(lib, (void **)&a.field, name)
    NV_SYM(Init, "cuInit");
    NV_SYM(DriverGetVersion, "cuDriverGetVersion");
    NV_SYM(DeviceGetCount, "cuDeviceGetCount");
    NV_SYM(DeviceGet, "cuDeviceGet");
    NV_SYM(DeviceGetName, "cuDeviceGetName");
    NV_SYM(DeviceGetAttribute, "cuDeviceGetAttribute");
    NV_SYM(DeviceTotalMem, "cuDeviceTotalMem_v2");
    NV_SYM(DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    NV_SYM(DevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
    NV_SYM(CtxSetCurrent, "cuCtxSetCurrent");
    NV_SYM(CtxSynchronize, "cuCtxSynchronize");
    NV_SYM(MemAlloc, "cuMemAlloc_v2");
    NV_SYM(MemFree, "cuMemFree_v2");
    NV_SYM(MemGetInfo, "cuMemGetInfo_v2");
    NV_SYM(MemHostAlloc, "cuMemHostAlloc");
    NV_SYM(MemFreeHost, "cuMemFreeHost");
    NV_SYM(MemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    NV_SYM(MemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    NV_SYM(MemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
    NV_SYM(MemsetD8Async, "cuMemsetD8Async");
    NV_SYM(StreamCreate, "cuStreamCreate");
    NV_SYM(StreamDestroy, "cuStreamDestroy_v2");
    NV_SYM(StreamSynchronize, "cuStreamSynchronize");
    NV_SYM(StreamQuery, "cuStreamQuery");
    NV_SYM(ModuleLoadDataEx, "cuModuleLoadDataEx");
    NV_SYM(ModuleUnload, "cuModuleUnload");
    NV_SYM(ModuleGetFunction, "cuModuleGetFunction");
    NV_SYM(ModuleGetGlobal, "cuModuleGetGlobal_v2");
    NV_SYM(LaunchKernel, "cuLaunchKernel");
    NV_SYM(EventCreate, "cuEventCreate");
    NV_SYM(EventDestroy, "cuEventDestroy_v2");
    NV_SYM(EventRecord, "cuEventRecord");
    NV_SYM(EventSynchronize, "cuEventSynchronize");
    NV_SYM(EventQuery, "cuEventQuery");
    NV_SYM(EventElapsedTime, "cuEventElapsedTime");
    NV_SYM(GetErrorName, "cuGetErrorName");
    NV_SYM(MemAllocManaged, "cuMemAllocManaged");
    NV_SYM(MemHostRegister, "cuMemHostRegister_v2");
    NV_SYM(MemHostUnregister, "cuMemHostUnregister");
    NV_SYM(MemHostGetDevicePointer, "cuMemHostGetDevicePointer_v2");
#undef NV_SYM
    if (!ok || a.Init(0) != NV_SUCCESS) {
        lib_close(lib);
        return;
    }
    g_api = a;
    g_loaded = &g_api; /* the library stays loaded for the process */
}

const nv_api *nv_load(void)
{
    pthread_once(&g_once, load_once);
    return g_loaded;
}

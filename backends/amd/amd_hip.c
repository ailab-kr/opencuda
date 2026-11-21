/*
 * amd_hip.c -- find the installed HIP runtime at run time and resolve the
 * entry points listed in amd_hip.h.
 *
 * Windows: the GPU driver installs amdhip64_7.dll (HIP 7) or amdhip64_6.dll
 * in System32. Linux: libamdhip64.so from ROCm.
 */
#define _GNU_SOURCE
#include "amd_hip.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static const char *const k_names[] = {"amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll", NULL};
static void *lib_open(const char *p) { return (void *)LoadLibraryA(p); }
static void *lib_sym(void *lib, const char *n) { return (void *)(uintptr_t)GetProcAddress((HMODULE)lib, n); }
static void lib_close(void *lib) { FreeLibrary((HMODULE)lib); }
#else
#include <dlfcn.h>
static const char *const k_names[] = {"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", NULL};
static void *lib_open(const char *p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
static void *lib_sym(void *lib, const char *n) { return dlsym(lib, n); }
static void lib_close(void *lib) { dlclose(lib); }
#endif
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static amd_api g_api;
static const amd_api *g_loaded;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static int resolve(void *lib, void **slot, const char *name)
{
    *slot = lib_sym(lib, name);
    return *slot != NULL;
}

static void load_once(void)
{
    const char *path = getenv("OCU_AMD_HIP_LIBRARY");
    void *lib = NULL;
    if (path != NULL && path[0] != '\0') {
        lib = lib_open(path);
    } else {
        for (int i = 0; lib == NULL && k_names[i] != NULL; ++i) lib = lib_open(k_names[i]);
    }
    if (lib == NULL) return;
    amd_api a;
    memset(&a, 0, sizeof(a));
    int ok = 1;
#define AMD_SYM(field, name) ok &= resolve(lib, (void **)&a.field, name)
    AMD_SYM(Init, "hipInit");
    AMD_SYM(RuntimeGetVersion, "hipRuntimeGetVersion");
    AMD_SYM(GetDeviceCount, "hipGetDeviceCount");
    AMD_SYM(DeviceGet, "hipDeviceGet");
    AMD_SYM(SetDevice, "hipSetDevice");
    AMD_SYM(DeviceGetName, "hipDeviceGetName");
    AMD_SYM(DeviceGetAttribute, "hipDeviceGetAttribute");
    AMD_SYM(DeviceTotalMem, "hipDeviceTotalMem");
    AMD_SYM(DeviceSynchronize, "hipDeviceSynchronize");
    AMD_SYM(Malloc, "hipMalloc");
    AMD_SYM(Free, "hipFree");
    AMD_SYM(MemGetInfo, "hipMemGetInfo");
    AMD_SYM(HostMalloc, "hipHostMalloc");
    AMD_SYM(HostFree, "hipHostFree");
    AMD_SYM(MemcpyAsync, "hipMemcpyAsync");
    AMD_SYM(MemsetAsync, "hipMemsetAsync");
    AMD_SYM(StreamCreateWithFlags, "hipStreamCreateWithFlags");
    AMD_SYM(StreamDestroy, "hipStreamDestroy");
    AMD_SYM(StreamSynchronize, "hipStreamSynchronize");
    AMD_SYM(StreamQuery, "hipStreamQuery");
    AMD_SYM(ModuleLoadData, "hipModuleLoadData");
    AMD_SYM(ModuleUnload, "hipModuleUnload");
    AMD_SYM(ModuleGetFunction, "hipModuleGetFunction");
    AMD_SYM(ModuleGetGlobal, "hipModuleGetGlobal");
    AMD_SYM(ModuleLaunchKernel, "hipModuleLaunchKernel");
    AMD_SYM(EventCreateWithFlags, "hipEventCreateWithFlags");
    AMD_SYM(EventDestroy, "hipEventDestroy");
    AMD_SYM(EventRecord, "hipEventRecord");
    AMD_SYM(EventSynchronize, "hipEventSynchronize");
    AMD_SYM(EventQuery, "hipEventQuery");
    AMD_SYM(EventElapsedTime, "hipEventElapsedTime");
    AMD_SYM(MallocManaged, "hipMallocManaged");
    AMD_SYM(HostRegister, "hipHostRegister");
    AMD_SYM(HostUnregister, "hipHostUnregister");
    AMD_SYM(HostGetDevicePointer, "hipHostGetDevicePointer");
#undef AMD_SYM
    if (!ok || a.Init(0) != HIP_SUCCESS) {
        lib_close(lib);
        return;
    }
    g_api = a;
    g_loaded = &g_api; /* the library stays loaded for the process */
}

const amd_api *amd_load(void)
{
    pthread_once(&g_once, load_once);
    return g_loaded;
}

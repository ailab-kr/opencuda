/*
 * ocl_load.c -- find the OpenCL library at run time and resolve the entry
 * points listed in ocl_api.h. The usual library is the ICD loader
 * (libOpenCL.so.1), which finds the installed drivers itself;
 * OCU_OPENCL_LIBRARY names another file (a driver library used directly).
 */
#define _GNU_SOURCE
#include "ocl_api.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static ocl_api g_api;
static const ocl_api *g_loaded;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void load_once(void)
{
    const char *path = getenv("OCU_OPENCL_LIBRARY");
    void *lib = dlopen(path != NULL && path[0] != '\0' ? path : "libOpenCL.so.1",
                       RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL && (path == NULL || path[0] == '\0')) {
        lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (lib == NULL) {
        return;
    }
    ocl_api a;
    memset(&a, 0, sizeof(a));
    int ok = 1;
#define CL_SYM(field) ok &= (*(void **)&a.field = dlsym(lib, "cl" #field)) != NULL
    CL_SYM(GetPlatformIDs);
    CL_SYM(GetPlatformInfo);
    CL_SYM(GetDeviceIDs);
    CL_SYM(GetDeviceInfo);
    CL_SYM(CreateContext);
    CL_SYM(ReleaseContext);
    CL_SYM(CreateCommandQueueWithProperties);
    CL_SYM(ReleaseCommandQueue);
    CL_SYM(Flush);
    CL_SYM(Finish);
    CL_SYM(SVMAlloc);
    CL_SYM(SVMFree);
    CL_SYM(EnqueueSVMMemcpy);
    CL_SYM(EnqueueSVMMemFill);
    CL_SYM(CreateBuffer);
    CL_SYM(ReleaseMemObject);
    CL_SYM(CreateProgramWithSource);
    CL_SYM(BuildProgram);
    CL_SYM(GetProgramBuildInfo);
    CL_SYM(ReleaseProgram);
    CL_SYM(CreateKernel);
    CL_SYM(ReleaseKernel);
    CL_SYM(SetKernelArg);
    CL_SYM(GetKernelInfo);
    CL_SYM(SetKernelArgSVMPointer);
    CL_SYM(SetKernelExecInfo);
    CL_SYM(GetKernelWorkGroupInfo);
    CL_SYM(EnqueueNDRangeKernel);
    CL_SYM(EnqueueMarkerWithWaitList);
    CL_SYM(WaitForEvents);
    CL_SYM(GetEventInfo);
    CL_SYM(GetEventProfilingInfo);
    CL_SYM(RetainEvent);
    CL_SYM(ReleaseEvent);
#undef CL_SYM
    *(void **)&a.CreateProgramWithIL = dlsym(lib, "clCreateProgramWithIL");
    *(void **)&a.GetExtensionFunctionAddressForPlatform = dlsym(lib, "clGetExtensionFunctionAddressForPlatform");
    *(void **)&a.SetEventCallback = dlsym(lib, "clSetEventCallback");
    *(void **)&a.EnqueueSVMFree = dlsym(lib, "clEnqueueSVMFree");
    if (!ok) {
        dlclose(lib);
        return;
    }
    g_api = a;
    g_loaded = &g_api; /* the library stays loaded for the process */
}

const ocl_api *ocl_load(void)
{
    pthread_once(&g_once, load_once);
    return g_loaded;
}

/* Experiment (OCU_OPENCL_DEVICE_USM=1): device memory from Intel USM device
   allocations. Pointer arguments, copies, fills and the indirect-access list
   then go through the cl_intel_unified_shared_memory entry points, which take
   the same arguments in the same order (and SVM pointers too on Intel's
   driver). Called once, while the devices are listed. */
void ocl_api_use_usm(void *set_arg_mem_pointer, void *enqueue_memcpy, void *enqueue_memfill)
{
    if (g_loaded == NULL || set_arg_mem_pointer == NULL || enqueue_memcpy == NULL || enqueue_memfill == NULL) return;
    *(void **)&g_api.SetKernelArgSVMPointer = set_arg_mem_pointer;
    *(void **)&g_api.EnqueueSVMMemcpy = enqueue_memcpy;
    *(void **)&g_api.EnqueueSVMMemFill = enqueue_memfill;
    g_api.usm_device_memory = 1;
}

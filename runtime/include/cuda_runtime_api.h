/*
 * cuda_runtime_api.h -- the runtime API as user programs see it.
 *
 * Written for this project from the public description of the CUDA
 * Runtime API (function names, parameters, documented behaviour). It is
 * not derived from any vendor header. Only the subset implemented by this
 * project is declared; anything else fails to compile instead of failing
 * at run time.
 *
 * C and C++. In C++ some functions gain the default arguments and
 * template overloads the public documentation describes.
 */
#ifndef OCU_CUDA_RUNTIME_API_H
#define OCU_CUDA_RUNTIME_API_H

#include <limits.h> /* INT_MAX and friends: CUDA programs use them without including it */
#include <stddef.h>

#include "ocu_vector_types.h"
#include "library_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The runtime version this layer reports (12.3), as major * 1000 + minor * 10.
   A program built with -DOCU_CUDART_VERSION=N sees N instead: a lower API
   profile, for code that picks optional features from this number (for
   example 11060 leaves out programmatic launch, FP8 and CUB code paths that
   test for 11.7 or 11.8). The functions declared here do not change. */
#ifdef OCU_CUDART_VERSION
#define CUDART_VERSION OCU_CUDART_VERSION
#else
#define CUDART_VERSION 12030
#endif

/* ---- errors ------------------------------------------------------------ */

/*
 * Values as listed in the public API reference. The runtime returns only
 * some of them (see cudart.c); the rest exist so that code switching over
 * error values compiles.
 */
#define OCU_RT_ERRORS(X) \
    X(cudaSuccess, 0, "no error") \
    X(cudaErrorInvalidValue, 1, "invalid argument") \
    X(cudaErrorMemoryAllocation, 2, "out of memory") \
    X(cudaErrorInitializationError, 3, "initialization error") \
    X(cudaErrorCudartUnloading, 4, "runtime is shutting down") \
    X(cudaErrorProfilerDisabled, 5, "profiler disabled") \
    X(cudaErrorProfilerNotInitialized, 6, "profiler not initialized") \
    X(cudaErrorProfilerAlreadyStarted, 7, "profiler already started") \
    X(cudaErrorProfilerAlreadyStopped, 8, "profiler already stopped") \
    X(cudaErrorInvalidConfiguration, 9, "invalid launch configuration") \
    X(cudaErrorInvalidPitchValue, 12, "invalid pitch") \
    X(cudaErrorInvalidSymbol, 13, "invalid device symbol") \
    X(cudaErrorInvalidHostPointer, 16, "invalid host pointer") \
    X(cudaErrorInvalidDevicePointer, 17, "invalid device pointer") \
    X(cudaErrorInvalidTexture, 18, "invalid texture") \
    X(cudaErrorInvalidTextureBinding, 19, "invalid texture binding") \
    X(cudaErrorInvalidChannelDescriptor, 20, "invalid channel descriptor") \
    X(cudaErrorInvalidMemcpyDirection, 21, "invalid copy direction") \
    X(cudaErrorAddressOfConstant, 22, "address of constant") \
    X(cudaErrorTextureFetchFailed, 23, "texture fetch failed") \
    X(cudaErrorTextureNotBound, 24, "texture not bound") \
    X(cudaErrorSynchronizationError, 25, "synchronization error") \
    X(cudaErrorInvalidFilterSetting, 26, "invalid filter setting") \
    X(cudaErrorInvalidNormSetting, 27, "invalid normalization setting") \
    X(cudaErrorMixedDeviceExecution, 28, "mixed device execution") \
    X(cudaErrorNotYetImplemented, 31, "not yet implemented") \
    X(cudaErrorMemoryValueTooLarge, 32, "memory value too large") \
    X(cudaErrorInsufficientDriver, 35, "driver too old") \
    X(cudaErrorInvalidSurface, 37, "invalid surface") \
    X(cudaErrorDuplicateVariableName, 43, "duplicate variable name") \
    X(cudaErrorDuplicateTextureName, 44, "duplicate texture name") \
    X(cudaErrorDuplicateSurfaceName, 45, "duplicate surface name") \
    X(cudaErrorDevicesUnavailable, 46, "devices unavailable") \
    X(cudaErrorIncompatibleDriverContext, 49, "incompatible driver context") \
    X(cudaErrorMissingConfiguration, 52, "missing launch configuration") \
    X(cudaErrorPriorLaunchFailure, 53, "prior launch failure") \
    X(cudaErrorInvalidDeviceFunction, 98, "invalid device function") \
    X(cudaErrorNoDevice, 100, "no device available") \
    X(cudaErrorInvalidDevice, 101, "invalid device ordinal") \
    X(cudaErrorStartupFailure, 127, "startup failure") \
    X(cudaErrorInvalidKernelImage, 200, "invalid kernel image") \
    X(cudaErrorMapBufferObjectFailed, 205, "map buffer object failed") \
    X(cudaErrorUnmapBufferObjectFailed, 206, "unmap buffer object failed") \
    X(cudaErrorNoKernelImageForDevice, 209, "no kernel image for this device") \
    X(cudaErrorECCUncorrectable, 214, "uncorrectable memory error") \
    X(cudaErrorUnsupportedLimit, 215, "unsupported limit") \
    X(cudaErrorPeerAccessUnsupported, 217, "peer access unsupported") \
    X(cudaErrorInvalidSource, 300, "invalid source") \
    X(cudaErrorFileNotFound, 301, "file not found") \
    X(cudaErrorSharedObjectSymbolNotFound, 302, "shared object symbol not found") \
    X(cudaErrorSharedObjectInitFailed, 303, "shared object initialization failed") \
    X(cudaErrorOperatingSystem, 304, "operating system call failed") \
    X(cudaErrorInvalidResourceHandle, 400, "invalid resource handle") \
    X(cudaErrorIllegalState, 401, "illegal state") \
    X(cudaErrorSymbolNotFound, 500, "symbol not found") \
    X(cudaErrorNotReady, 600, "device not ready") \
    X(cudaErrorIllegalAddress, 700, "illegal memory access") \
    X(cudaErrorLaunchOutOfResources, 701, "too many resources requested for launch") \
    X(cudaErrorLaunchTimeout, 702, "launch timed out") \
    X(cudaErrorPeerAccessAlreadyEnabled, 704, "peer access already enabled") \
    X(cudaErrorPeerAccessNotEnabled, 705, "peer access not enabled") \
    X(cudaErrorSetOnActiveProcess, 708, "cannot set while device is active") \
    X(cudaErrorAssert, 710, "device-side assert triggered") \
    X(cudaErrorTooManyPeers, 711, "too many peers") \
    X(cudaErrorHostMemoryAlreadyRegistered, 712, "host memory already registered") \
    X(cudaErrorHostMemoryNotRegistered, 713, "host memory not registered") \
    X(cudaErrorLaunchFailure, 719, "unspecified launch failure") \
    X(cudaErrorNotPermitted, 800, "operation not permitted") \
    X(cudaErrorNotSupported, 801, "operation not supported") \
    X(cudaErrorUnknown, 999, "unknown error")

typedef enum cudaError {
#define OCU_RT_ERROR_ENUM(name, value, text) name = value,
    OCU_RT_ERRORS(OCU_RT_ERROR_ENUM)
#undef OCU_RT_ERROR_ENUM
    ocu_rt_error_end_ = 10000 /* not an error value */
} cudaError_t;

/* ---- basic types ------------------------------------------------------- */

typedef struct ocu_rt_stream *cudaStream_t;
typedef struct ocu_rt_event *cudaEvent_t;


typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

/* Stream and event creation flags. */
#define cudaStreamDefault       0x00u
#define cudaStreamNonBlocking   0x01u
#define cudaEventDefault        0x00u
#define cudaEventBlockingSync   0x01u
#define cudaEventDisableTiming  0x02u
#define cudaEventInterprocess   0x04u

/* Host allocation flags (cudaHostAlloc). */
#define cudaHostAllocDefault       0x00u
#define cudaHostAllocPortable      0x01u
#define cudaHostAllocMapped        0x02u
#define cudaHostAllocWriteCombined 0x04u

/* Cache preference: a hint. Devices without the choice ignore it. */
enum cudaFuncCache {
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3
};

enum cudaComputeMode {
    cudaComputeModeDefault = 0,
    cudaComputeModeExclusive = 1,
    cudaComputeModeProhibited = 2,
    cudaComputeModeExclusiveProcess = 3
};

/* Kinds of memory a pointer can refer to (cudaPointerGetAttributes). */
enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
};

struct cudaPointerAttributes {
    enum cudaMemoryType type;
    int                 device;
    void               *devicePointer;
    void               *hostPointer;
};

/* cudaMallocManaged flags. */
#define cudaMemAttachGlobal 0x01u
#define cudaMemAttachHost   0x02u

/* Device attributes (cudaDeviceGetAttribute): the commonly queried subset. */
enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    /* Values as documented; the pci ones are synthetic (see cudart.c). */
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrCooperativeLaunch = 95,            /* always 0: no co-residency guarantee */
    cudaDevAttrCooperativeMultiDeviceLaunch = 96  /* always 0 */
};

/* cudaHostRegister flags. */
#define cudaHostRegisterDefault  0x00u
#define cudaHostRegisterPortable 0x01u
#define cudaHostRegisterMapped   0x02u
#define cudaHostRegisterIoMemory 0x04u
#define cudaHostRegisterReadOnly 0x08u

/* cudaSetDeviceFlags flags. */
#define cudaDeviceScheduleAuto         0x00u
#define cudaDeviceScheduleSpin         0x01u
#define cudaDeviceScheduleYield        0x02u
#define cudaDeviceScheduleBlockingSync 0x04u
#define cudaDeviceScheduleMask         0x07u
#define cudaDeviceMapHost              0x08u
#define cudaDeviceLmemResizeToMax      0x10u

enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2
};

enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2
};

typedef void (*cudaHostFn_t)(void *user_data);

enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9
};

/* What cudaFuncGetAttributes reports. */
struct cudaFuncAttributes {
    size_t sharedSizeBytes;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int    maxThreadsPerBlock;
    int    numRegs;
    int    ptxVersion;
    int    binaryVersion;
    int    cacheModeCA;
    int    maxDynamicSharedSizeBytes;
    int    preferredShmemCarveout;
};

/*
 * Extensible launch configuration (cudaLaunchKernelEx). Names and values
 * as in the public API reference (cudaLaunchAttributeID,
 * cudaLaunchAttributeValue, cudaLaunchConfig_t). What this layer does with
 * each attribute is described at cudaLaunchKernelExC below.
 */
typedef enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributePreferredClusterDimension = 11,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
    cudaLaunchAttributePreferredSharedMemoryCarveout = 14
} cudaLaunchAttributeID;

enum cudaAccessProperty {
    cudaAccessPropertyNormal = 0,
    cudaAccessPropertyStreaming = 1,
    cudaAccessPropertyPersisting = 2
};

struct cudaAccessPolicyWindow {
    void                   *base_ptr;
    size_t                  num_bytes;
    float                   hitRatio;
    enum cudaAccessProperty hitProp;
    enum cudaAccessProperty missProp;
};

enum cudaSynchronizationPolicy {
    cudaSyncPolicyAuto = 1,
    cudaSyncPolicySpin = 2,
    cudaSyncPolicyYield = 3,
    cudaSyncPolicyBlockingSync = 4
};

enum cudaClusterSchedulingPolicy {
    cudaClusterSchedulingPolicyDefault = 0,
    cudaClusterSchedulingPolicySpread = 1,
    cudaClusterSchedulingPolicyLoadBalancing = 2
};

typedef enum cudaLaunchMemSyncDomain {
    cudaLaunchMemSyncDomainDefault = 0,
    cudaLaunchMemSyncDomainRemote = 1
} cudaLaunchMemSyncDomain;

typedef struct cudaLaunchMemSyncDomainMap_st {
    unsigned char default_;
    unsigned char remote;
} cudaLaunchMemSyncDomainMap;

typedef union cudaLaunchAttributeValue {
    char pad[64];
    struct cudaAccessPolicyWindow accessPolicyWindow;
    int cooperative;
    enum cudaSynchronizationPolicy syncPolicy;
    struct { unsigned int x, y, z; } clusterDim;
    enum cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference;
    int programmaticStreamSerializationAllowed;
    struct { cudaEvent_t event; int flags; int triggerAtBlockStart; } programmaticEvent;
    int priority;
    cudaLaunchMemSyncDomainMap memSyncDomainMap;
    cudaLaunchMemSyncDomain memSyncDomain;
    struct { unsigned int x, y, z; } preferredClusterDim;
    struct { cudaEvent_t event; int flags; } launchCompletionEvent;
    unsigned int sharedMemCarveout;
} cudaLaunchAttributeValue;

typedef struct cudaLaunchAttribute_st {
    cudaLaunchAttributeID id;
    cudaLaunchAttributeValue val;
} cudaLaunchAttribute;

typedef struct cudaLaunchConfig_st {
    dim3 gridDim;
    dim3 blockDim;
    size_t dynamicSmemBytes;
    cudaStream_t stream;
    cudaLaunchAttribute *attrs;
    unsigned int numAttrs;
} cudaLaunchConfig_t;

/* The legacy default stream. */
#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamPerThread ((cudaStream_t)0x2)

/*
 * Device properties. The commonly read fields only; the layout is this
 * project's own. Values describe the device the backend reports.
 */
struct cudaDeviceProp {
    char   name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    regsPerBlock;
    int    warpSize;
    size_t memPitch;
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];
    int    maxGridSize[3];
    int    clockRate;              /* kHz */
    size_t totalConstMem;
    int    major;
    int    minor;
    size_t textureAlignment;
    int    deviceOverlap;
    int    multiProcessorCount;
    int    kernelExecTimeoutEnabled;
    int    integrated;
    int    canMapHostMemory;
    int    computeMode;
    int    concurrentKernels;
    int    ECCEnabled;
    int    pciBusID;
    int    pciDeviceID;
    int    pciDomainID;
    int    asyncEngineCount;
    int    unifiedAddressing;
    int    memoryClockRate;        /* kHz */
    int    memoryBusWidth;         /* bits */
    int    l2CacheSize;            /* bytes */
    int    maxThreadsPerMultiProcessor;
    size_t sharedMemPerMultiprocessor;
    int    regsPerMultiprocessor;
    int    managedMemory;
    int    isMultiGpuBoard;
    int    concurrentManagedAccess;
    size_t sharedMemPerBlockOptin;
    int    maxBlocksPerMultiProcessor;
};

/* ---- errors ------------------------------------------------------------ */

cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char *cudaGetErrorString(cudaError_t error);
const char *cudaGetErrorName(cudaError_t error);

/* ---- devices ----------------------------------------------------------- */

cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaDeviceCanAccessPeer(int *can_access, int device, int peer_device);
cudaError_t cudaDeviceEnablePeerAccess(int peer_device, unsigned int flags);
cudaError_t cudaDeviceDisablePeerAccess(int peer_device);
/* Synthetic, stable "dddd:bb:dd.f" identifier; see cudart.c. Needs len >= 13. */
cudaError_t cudaDeviceGetPCIBusId(char *pci_bus_id, int len, int device);
/* Always cudaErrorNotSupported: no backend guarantees that all blocks are resident. */
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 grid, dim3 block, void **args,
                                        size_t shared_mem, cudaStream_t stream);
cudaError_t cudaDeviceGetStreamPriorityRange(int *least, int *greatest);
cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, const void *func,
                                                          int block_size, size_t dynamic_smem);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaRuntimeGetVersion(int *version);
cudaError_t cudaDriverGetVersion(int *version);
/* Deprecated spellings kept by existing code. */
cudaError_t cudaThreadSynchronize(void);
cudaError_t cudaThreadExit(void);
/* Cache preferences are hints: recorded, validated, otherwise no effect. */
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache config);
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *config);
cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache config);

/* ---- memory ------------------------------------------------------------ */

cudaError_t cudaMalloc(void **ptr, size_t size);
cudaError_t cudaFree(void *ptr);
cudaError_t cudaMallocManaged(void **ptr, size_t size, unsigned int flags);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attr, const void *ptr);
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void *ptr);
cudaError_t cudaHostGetDevicePointer(void **dptr, void *hptr, unsigned int flags);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags);
cudaError_t cudaMallocPitch(void **ptr, size_t *pitch, size_t width, size_t height);
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                         size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                              size_t width, size_t height, enum cudaMemcpyKind kind,
                              cudaStream_t stream);
cudaError_t cudaMemset(void *ptr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *ptr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaMemset2D(void *ptr, size_t pitch, int value, size_t width, size_t height);
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count,
                               size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count,
                                 size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaGetSymbolAddress(void **dev_ptr, const void *symbol);
cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol);
cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count,
                                    size_t offset, enum cudaMemcpyKind kind,
                                    cudaStream_t stream);
cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count,
                                      size_t offset, enum cudaMemcpyKind kind,
                                      cudaStream_t stream);

/* ---- streams ----------------------------------------------------------- */

cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority);
cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority);
cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *status);
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *user_data);

/* ---- launches ---------------------------------------------------------- */

/* Launch the kernel whose host address (launch stub) is `func`; args[i]
   points at the i-th argument value. */
cudaError_t cudaLaunchKernel(const void *func, dim3 grid, dim3 block, void **args,
                             size_t shared, cudaStream_t stream);

/*
 * cudaLaunchKernel with an extensible configuration: grid, block, dynamic
 * shared memory and stream from `config`, then its attributes:
 *   Ignore, AccessPolicyWindow, SynchronizationPolicy, Priority,
 *   MemSyncDomain(Map), ClusterSchedulingPolicyPreference,
 *   PreferredClusterDimension, PreferredSharedMemoryCarveout:
 *     performance hints, accepted and without effect;
 *   ProgrammaticStreamSerialization: accepted. The kernel still starts
 *     only after the previous work in the stream has completed. The
 *     attribute only permits overlap with that work, so full stream order
 *     is one of the allowed executions; cudaGridDependencySynchronize()
 *     then has nothing left to wait for;
 *   ProgrammaticEvent, LaunchCompletionEvent: the event is recorded in
 *     the stream after the kernel, a later point than the one the
 *     attribute names, which waiters cannot tell apart from a trigger that
 *     happened to come late;
 *   Cooperative (non-zero): cudaErrorNotSupported, as
 *     cudaLaunchCooperativeKernel;
 *   ClusterDimension (more than one block per cluster): cudaErrorNotSupported;
 *   DeviceUpdatableKernelNode (graphs only) and unknown ids:
 *     cudaErrorInvalidValue.
 * A NULL config, or numAttrs > 0 with attrs NULL: cudaErrorInvalidValue.
 */
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args);

/* The attribute checks of cudaLaunchKernelExC, and what it does after the
   launch (events), for launch paths that do not go through it (the CPU
   path's typed launch). Internal names, not part of the CUDA API. */
cudaError_t ocu_rt_launch_config_check(const cudaLaunchConfig_t *config);
cudaError_t ocu_rt_launch_config_done(const cudaLaunchConfig_t *config);

/* ---- events ------------------------------------------------------------ */

cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

#ifdef __cplusplus
} /* extern "C" */

/* C++ conveniences described by the public documentation. */
template <class T>
static inline cudaError_t cudaMemcpyToSymbol(const T &symbol, const void *src, size_t count,
                                             size_t offset = 0,
                                             cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
    return cudaMemcpyToSymbol(static_cast<const void *>(&symbol), src, count, offset, kind);
}

template <class T>
static inline cudaError_t cudaMemcpyFromSymbol(void *dst, const T &symbol, size_t count,
                                               size_t offset = 0,
                                               cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
{
    return cudaMemcpyFromSymbol(dst, static_cast<const void *>(&symbol), count, offset, kind);
}

template <class T>
static inline cudaError_t cudaMalloc(T **ptr, size_t size)
{
    return cudaMalloc(reinterpret_cast<void **>(ptr), size);
}

template <class T>
static inline cudaError_t cudaMallocHost(T **ptr, size_t size)
{
    return cudaMallocHost(reinterpret_cast<void **>(ptr), size);
}

template <class T>
static inline cudaError_t cudaFuncSetAttribute(T *func, cudaFuncAttribute attr, int value)
{
    return cudaFuncSetAttribute(reinterpret_cast<const void *>(func), attr, value);
}

template <class T>
static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr, T *func)
{
    return cudaFuncGetAttributes(attr, reinterpret_cast<const void *>(func));
}

template <class T>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, T func,
                                                                        int block_size,
                                                                        size_t dynamic_smem)
{
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks, reinterpret_cast<const void *>(func), block_size, dynamic_smem);
}

template <class T>
static inline cudaError_t cudaMallocManaged(T **ptr, size_t size,
                                            unsigned int flags = cudaMemAttachGlobal)
{
    return cudaMallocManaged(reinterpret_cast<void **>(ptr), size, flags);
}

template <class T>
static inline cudaError_t cudaHostAlloc(T **ptr, size_t size, unsigned int flags)
{
    return cudaHostAlloc(reinterpret_cast<void **>(ptr), size, flags);
}

template <class T>
static inline cudaError_t cudaMallocPitch(T **ptr, size_t *pitch, size_t width, size_t height)
{
    return cudaMallocPitch(reinterpret_cast<void **>(ptr), pitch, width, height);
}

template <class T>
static inline cudaError_t cudaFuncSetCacheConfig(T *func, cudaFuncCache config)
{
    return cudaFuncSetCacheConfig(reinterpret_cast<const void *>(func), config);
}

template <class T>
static inline cudaError_t cudaMemcpyToSymbolAsync(const T &symbol, const void *src,
                                                  size_t count, size_t offset = 0,
                                                  cudaMemcpyKind kind = cudaMemcpyHostToDevice,
                                                  cudaStream_t stream = 0)
{
    return cudaMemcpyToSymbolAsync(static_cast<const void *>(&symbol), src, count, offset,
                                   kind, stream);
}

template <class T>
static inline cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const T &symbol,
                                                    size_t count, size_t offset = 0,
                                                    cudaMemcpyKind kind = cudaMemcpyDeviceToHost,
                                                    cudaStream_t stream = 0)
{
    return cudaMemcpyFromSymbolAsync(dst, static_cast<const void *>(&symbol), count, offset,
                                     kind, stream);
}

static inline cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                          cudaMemcpyKind kind)
{
    return cudaMemcpyAsync(dst, src, count, kind, static_cast<cudaStream_t>(0));
}

static inline cudaError_t cudaMemsetAsync(void *ptr, int value, size_t count)
{
    return cudaMemsetAsync(ptr, value, count, static_cast<cudaStream_t>(0));
}

static inline cudaError_t cudaEventRecord(cudaEvent_t event)
{
    return cudaEventRecord(event, static_cast<cudaStream_t>(0));
}

static inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event)
{
    return cudaStreamWaitEvent(stream, event, 0);
}

/* Kernel function pointers, as in the C++ overloads of the documented API
   (variadic: C++11 and later; older host code uses the const void * form). */
#if __cplusplus >= 201103L
template <typename... P>
static inline cudaError_t cudaLaunchCooperativeKernel(void (*func)(P...), dim3 grid, dim3 block,
                                                      void **args, size_t shared_mem = 0,
                                                      cudaStream_t stream = 0)
{
    return cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(func), grid, block, args,
                                       shared_mem, stream);
}
#endif
#endif /* __cplusplus */

#endif /* OCU_CUDA_RUNTIME_API_H */

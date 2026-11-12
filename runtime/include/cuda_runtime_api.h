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

#include <stddef.h>

/* Grid and block sizes. */
#ifndef OCU_DIM3_DEFINED
#define OCU_DIM3_DEFINED
struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
#endif
};
#ifndef __cplusplus
typedef struct dim3 dim3;
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The runtime version this layer reports (12.3), as major * 1000 + minor * 10. */
#define CUDART_VERSION 12030

/* ---- errors ------------------------------------------------------------ */

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorNotReady = 600,
    cudaErrorLaunchFailure = 719,
    cudaErrorNotSupported = 801,
    cudaErrorUnknown = 999
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
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaRuntimeGetVersion(int *version);
cudaError_t cudaDriverGetVersion(int *version);
/* Deprecated spellings kept by existing code. */
cudaError_t cudaThreadSynchronize(void);
cudaError_t cudaThreadExit(void);

/* ---- memory ------------------------------------------------------------ */

cudaError_t cudaMalloc(void **ptr, size_t size);
cudaError_t cudaFree(void *ptr);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaFreeHost(void *ptr);
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
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count,
                               size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count,
                                 size_t offset, enum cudaMemcpyKind kind);

/* ---- streams ----------------------------------------------------------- */

cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);

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

static inline cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                          cudaMemcpyKind kind)
{
    return cudaMemcpyAsync(dst, src, count, kind, static_cast<cudaStream_t>(nullptr));
}

static inline cudaError_t cudaMemsetAsync(void *ptr, int value, size_t count)
{
    return cudaMemsetAsync(ptr, value, count, static_cast<cudaStream_t>(nullptr));
}

static inline cudaError_t cudaEventRecord(cudaEvent_t event)
{
    return cudaEventRecord(event, static_cast<cudaStream_t>(nullptr));
}

static inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event)
{
    return cudaStreamWaitEvent(stream, event, 0);
}
#endif /* __cplusplus */

#endif /* OCU_CUDA_RUNTIME_API_H */

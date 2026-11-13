/*
 * cuda.h -- many CUDA programs include this out of habit and only use the
 * runtime API. This project does not provide the driver API (cu* calls);
 * this header gives the version number and the runtime API.
 *
 * Written for this project; not derived from any vendor header.
 */
#ifndef OCU_CUDA_H
#define OCU_CUDA_H

#define CUDA_VERSION 12030

#include "cuda_runtime.h"

/* The driver API's result type, for runtime-API programs that compare
   results with CUDA_SUCCESS (values as in the public Driver API
   reference). The only driver API function is cuMemGetInfo, which asks
   the runtime (cudaMemGetInfo) about the current device. */
#if !defined(OCU_NO_TEXTURE_DECLS) /* the translators' compat header defines CUDA_SUCCESS */
typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999
} CUresult;

static inline CUresult cuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    cudaError_t e = cudaMemGetInfo(free_bytes, total_bytes);
    switch (e) {
    case cudaSuccess: return CUDA_SUCCESS;
    case cudaErrorInvalidValue: return CUDA_ERROR_INVALID_VALUE;
    case cudaErrorNoDevice: return CUDA_ERROR_NO_DEVICE;
    case cudaErrorNotSupported: return CUDA_ERROR_NOT_SUPPORTED;
    default: return CUDA_ERROR_UNKNOWN;
    }
}
#endif

#endif /* OCU_CUDA_H */

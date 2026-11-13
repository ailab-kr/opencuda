/*
 * cuda_profiler_api.h -- profiler start/stop markers.
 *
 * Written for this project from the public CUDA Runtime API reference
 * ("Profiler Control"). cudaProfilerStart enables and cudaProfilerStop
 * disables profile collection by the active profiling tool for the current
 * context; with no tool attached they have no effect and return
 * cudaSuccess, which is what this runtime does (it has no profiling tool).
 * Not derived from any vendor header.
 */
#ifndef OCU_CUDA_PROFILER_API_H
#define OCU_CUDA_PROFILER_API_H

#include "cuda_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
static inline cudaError_t cudaProfilerStop(void) { return cudaSuccess; }

#ifdef __cplusplus
}
#endif

#endif /* OCU_CUDA_PROFILER_API_H */

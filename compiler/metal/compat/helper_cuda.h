// helper_cuda.h -- error-check helpers that programs written for CUDA include
// next to the runtime header. Written for this project from how the macros
// are used (wrap a runtime call, check the last launch) and from the public
// runtime API description (cudaError_t, cudaGetErrorString, cudaGetLastError).
//
// Used by cu2msl.py only to let the front end parse host code in a .cu file
// whose kernels are being translated; the host code is not translated.
#ifndef OCU_COMPAT_HELPER_CUDA_H
#define OCU_COMPAT_HELPER_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

inline void ocu_check_call(cudaError_t result, const char *what, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\"\n", file, line,
                (int)result, cudaGetErrorString(result), what);
        exit(EXIT_FAILURE);
    }
}

inline void ocu_check_last(const char *message, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s(%i) : error : %s : (%d) %s.\n", file, line, message,
                (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) ocu_check_call((val), #val, __FILE__, __LINE__)
#define getLastCudaError(msg) ocu_check_last((msg), __FILE__, __LINE__)

#endif

/*
 * ocu_backend_opencl.h -- the OpenCL backend's entry point.
 * Device code images are made by compiler/opencl/cu2ocl.py (README.md).
 */
#ifndef OCU_BACKEND_OPENCL_H
#define OCU_BACKEND_OPENCL_H

#include "ocu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

const ocu_backend *ocu_backend_opencl(void);

#ifdef __cplusplus
}
#endif

#endif /* OCU_BACKEND_OPENCL_H */

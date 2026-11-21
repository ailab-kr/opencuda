/*
 * ocu_backend_amd.h -- the AMD GPU backend's entry point.
 * Device code images for this backend are amdgcn code objects (ELF), made
 * by Clang in HIP mode with OpenCuda's headers (tools/amdgpu/ocu_amd_cc.py).
 */
#ifndef OCU_BACKEND_AMD_H
#define OCU_BACKEND_AMD_H

#include "ocu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

const ocu_backend *ocu_backend_amd(void);

#ifdef __cplusplus
}
#endif

#endif /* OCU_BACKEND_AMD_H */

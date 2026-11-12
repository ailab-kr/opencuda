/*
 * ocu_backend_nvidia.h -- the NVIDIA GPU backend's entry point.
 * Device code images for this backend are PTX text.
 */
#ifndef OCU_BACKEND_NVIDIA_H
#define OCU_BACKEND_NVIDIA_H

#include "ocu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

const ocu_backend *ocu_backend_nvidia(void);

#ifdef __cplusplus
}
#endif

#endif /* OCU_BACKEND_NVIDIA_H */

/*
 * ocu_backend_metal.h -- the Apple GPU (Metal) backend's entry point.
 * Device code images for this backend: see backends/metal/README.md.
 */
#ifndef OCU_BACKEND_METAL_H
#define OCU_BACKEND_METAL_H

#include "ocu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

const ocu_backend *ocu_backend_metal(void);

#ifdef __cplusplus
}
#endif

#endif /* OCU_BACKEND_METAL_H */

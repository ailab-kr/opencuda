#ifndef OCU_BACKEND_CPU_H
#define OCU_BACKEND_CPU_H

#include "ocu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The CPU reference backend table. Never NULL. */
const ocu_backend *ocu_backend_cpu(void);

#ifdef __cplusplus
}
#endif

#endif /* OCU_BACKEND_CPU_H */

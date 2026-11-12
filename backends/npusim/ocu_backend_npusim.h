/*
 * npusim: NPU constraint simulator backend (docs/spec/op-table.md, section 7).
 */
#ifndef OCU_BACKEND_NPUSIM_H
#define OCU_BACKEND_NPUSIM_H

#include "ocu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The simulator's backend table. Never NULL. */
const ocu_backend *ocu_backend_npusim(void);

#ifdef __cplusplus
}
#endif

#endif /* OCU_BACKEND_NPUSIM_H */

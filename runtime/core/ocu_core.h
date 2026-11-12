/*
 * ocu_core.h -- chip-neutral core used by the runtime and the libraries.
 * Internal header: not part of the surface user programs see.
 */
#ifndef OCU_CORE_H
#define OCU_CORE_H

#include "ocu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Add a backend. Fails when the ABI differs, a mandatory entry is missing,
 * or the name is already taken. The table must outlive the process.
 */
ocu_status ocu_register_backend(const ocu_backend *backend);

/* Make the named backend active. */
ocu_status ocu_select_backend(const char *name);

/*
 * The backend to use when nothing chose one before first use and
 * OCU_BACKEND is unset (e.g. the one whose device code a program carries).
 * No effect after the first use. OCU_ERR_NOT_FOUND when not built in.
 */
ocu_status ocu_prefer_backend(const char *name);

/*
 * The active backend. On first use it is chosen by the OCU_BACKEND
 * environment variable, or the first registered backend when unset.
 * Returns NULL when no backend is available or OCU_BACKEND names an
 * unknown one.
 */
const ocu_backend *ocu_active_backend(void);

/*
 * Device 0 of the active backend, opened on first use. The libraries use
 * this until the runtime layer manages devices per thread.
 */
ocu_status ocu_core_device(const ocu_backend **backend, ocu_device **device);

size_t      ocu_backend_count(void);
const char *ocu_backend_name_at(size_t index); /* NULL when out of range */

#ifdef __cplusplus
}
#endif

#endif /* OCU_CORE_H */

/*
 * ocu_rt.h -- what language support (the CPU path's kernel launch, later
 * the compiler-emitted launch entry points) needs from the runtime layer.
 * Not for user programs.
 */
#ifndef OCU_RT_H
#define OCU_RT_H

#include "cuda_runtime_api.h"
#include "ocu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Backend and device of the calling thread's current device (opened on
   first use). NULL when there is no usable device; the error is recorded
   as the thread's last error. */
const ocu_backend *ocu_rt_backend(void);
ocu_device *ocu_rt_device(void);

/*
 * Launch on `stream` (0 = the legacy default stream) with the CUDA
 * ordering rules, record the result as the thread's last error, and
 * return it. Invalid launch dimensions give cudaErrorInvalidConfiguration.
 */
cudaError_t ocu_rt_launch(ocu_kernel *kernel, const ocu_launch_params *params,
                          cudaStream_t stream);

/* Map a backend status to the runtime error it stands for. */
cudaError_t ocu_rt_error(ocu_status status);

#ifdef __cplusplus
}
#endif

#endif /* OCU_RT_H */

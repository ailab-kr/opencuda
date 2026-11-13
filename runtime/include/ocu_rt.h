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

/*
 * Submit work that is not a kernel (for libraries: an operator-table
 * execute, see docs/spec/op-table.md section 8) to `stream` with the same
 * ordering rules as a launch. `fn` is called once with the backend queue
 * of the stream while those rules hold; what it returns is stored in
 * *status (the runtime does not turn it into an error, so a library can
 * fall back to another path when the backend refuses). The return value is
 * the runtime error that prevented the call (no device, a sticky kernel
 * failure, ...), recorded as the thread's last error, or cudaSuccess.
 */
typedef ocu_status (*ocu_rt_submit_fn)(ocu_queue *queue, void *ctx);
cudaError_t ocu_rt_submit(cudaStream_t stream, ocu_rt_submit_fn fn, void *ctx,
                          ocu_status *status);

/*
 * Device code registered by compiled host code (runtime/cudart/fatbin.c).
 * OCU_ERR_NOT_FOUND when the address was never registered.
 */
ocu_status ocu_rt_find_kernel(const void *stub, ocu_kernel **kernel, uint32_t *nargs,
                              const size_t **arg_sizes);
ocu_status ocu_rt_find_symbol(const void *host, void **device, size_t *bytes);
int ocu_rt_has_device_code(void);

/* Map a backend status to the runtime error it stands for. */
cudaError_t ocu_rt_error(ocu_status status);

#ifdef __cplusplus
}
#endif

#endif /* OCU_RT_H */

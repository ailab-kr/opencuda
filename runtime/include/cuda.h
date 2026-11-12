/*
 * cuda.h -- many CUDA programs include this out of habit and only use the
 * runtime API. This project does not provide the driver API (cu* calls);
 * this header gives the version number and the runtime API.
 *
 * Written for this project; not derived from any vendor header.
 */
#ifndef OCU_CUDA_H
#define OCU_CUDA_H

#define CUDA_VERSION 12030

#include "cuda_runtime.h"

#endif /* OCU_CUDA_H */

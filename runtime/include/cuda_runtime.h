/*
 * cuda_runtime.h -- what a .cu file sees: the runtime API plus the language
 * support of the path it is compiled for.
 *
 * Written for this project from the public description of the CUDA
 * runtime and language. Not derived from any vendor header.
 *
 *   OCU_CPU_PATH   the CPU path (rewritten .cu compiled as C++17)
 *   __CUDA__       Clang's CUDA mode (host pass and NVPTX device pass)
 *   neither        plain host code: the runtime API only
 */
#ifndef OCU_CUDA_RUNTIME_H
#define OCU_CUDA_RUNTIME_H

#include "cuda_runtime_api.h"

#if defined(OCU_CPU_PATH)
#include "ocu_cuda_lang_cpu.h"
#elif defined(__CUDA__) && !defined(__HIP__)
#include "ocu_cuda_lang_nvptx.h"
#endif

#endif /* OCU_CUDA_RUNTIME_H */

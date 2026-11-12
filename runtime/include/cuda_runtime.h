/*
 * cuda_runtime.h -- what a .cu file sees: the runtime API plus the language
 * support of the path it is compiled for.
 *
 * Written for this project from the public description of the CUDA
 * runtime and language. Not derived from any vendor header.
 *
 *   OCU_CPU_PATH   the CPU path (rewritten .cu compiled as C++17)
 *   neither        plain host code: the runtime API only
 * The GPU paths (clang CUDA and HIP modes) get their language support from
 * the compiler driver; see compiler/.
 */
#ifndef OCU_CUDA_RUNTIME_H
#define OCU_CUDA_RUNTIME_H

#include "cuda_runtime_api.h"

#if defined(OCU_CPU_PATH)
#include "ocu_cuda_lang_cpu.h"
#endif

#endif /* OCU_CUDA_RUNTIME_H */

/*
 * cuda_runtime.h -- what a .cu file sees: the runtime API plus the language
 * support of the path it is compiled for.
 *
 * Written for this project from the public description of the CUDA
 * runtime and language. Not derived from any vendor header.
 *
 *   OCU_CPU_PATH   the CPU path (rewritten .cu compiled as C++17)
 *   __CUDA__       Clang's CUDA mode (host pass and NVPTX device pass)
 *   __HIP__        Clang's HIP mode: the AMD GPU path (amdgcn device pass)
 *   neither        plain host code: the runtime API only
 */
#ifndef OCU_CUDA_RUNTIME_H
#define OCU_CUDA_RUNTIME_H

#include "cuda_runtime_api.h"

#if defined(OCU_CPU_PATH)
#include "ocu_cuda_lang_cpu.h"
#elif defined(OCU_SPIRV_PATH)
#include "ocu_cuda_lang_spirv.h"
#elif defined(__CUDA__) && !defined(__HIP__)
#include "ocu_cuda_lang_nvptx.h"
#elif defined(__HIP__)
#include "ocu_cuda_lang_amdgcn.h"
#endif

/*
 * cudaLaunchKernelEx(config, kernel, args...): the C++ form described in
 * the public API reference. Each argument is converted to the type of the
 * kernel's parameter, then the launch goes the way cudaLaunchKernel goes
 * on this path, with the attribute handling of cudaLaunchKernelExC.
 */
#if defined(__cplusplus) && __cplusplus >= 201402L && \
    (!defined(__has_include) || (__has_include(<tuple>) && __has_include(<utility>) && __has_include(<type_traits>)))
#include <tuple>
#include <type_traits>
#include <utility>

namespace ocu_launch_ex_ {
template <typename... P, typename Tuple, std::size_t... I>
inline cudaError_t launch(const cudaLaunchConfig_t *config, void (*kernel)(P...), Tuple &values,
                          std::index_sequence<I...>)
{
    void *args[sizeof...(P) + 1] = {static_cast<void *>(&std::get<I>(values))..., nullptr};
#if defined(OCU_CPU_PATH)
    return cudaLaunchKernel(kernel, config->gridDim, config->blockDim, args,
                            config->dynamicSmemBytes, config->stream);
#else
    return cudaLaunchKernel(reinterpret_cast<const void *>(kernel), config->gridDim,
                            config->blockDim, args, config->dynamicSmemBytes, config->stream);
#endif
}
} // namespace ocu_launch_ex_

#if defined(OCU_SPIRV_PATH) && defined(__HIP_DEVICE_COMPILE__)
/* The SPIR-V device pass (Clang in HIP mode for spirv64) gives kernels a
   calling convention in their type, so `void (*)(P...)` does not match a
   kernel there; host code is parsed in that pass but never generated, so
   any kernel is accepted (the host pass has the real function above). */
template <typename F, typename... ActTypes>
static inline cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t *, F, ActTypes &&...)
{
    return cudaSuccess;
}
#else
template <typename... ExpTypes, typename... ActTypes>
static inline cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t *config,
                                             void (*kernel)(ExpTypes...), ActTypes &&...args)
{
    static_assert(sizeof...(ExpTypes) == sizeof...(ActTypes),
                  "cudaLaunchKernelEx: argument count differs from the kernel's parameters");
    cudaError_t e = ocu_rt_launch_config_check(config);
    if (e != cudaSuccess) return e;
    std::tuple<typename std::remove_cv<ExpTypes>::type...> values(
        static_cast<ExpTypes>(std::forward<ActTypes>(args))...);
    e = ocu_launch_ex_::launch(config, kernel, values, std::index_sequence_for<ExpTypes...>{});
    if (e != cudaSuccess) return e;
    return ocu_rt_launch_config_done(config);
}
#endif
#endif

/* OpenCL path (compiler/opencl): device code may call assert(); the C
   library's expansion calls the host function __assert_fail, so both Clang
   passes get a device-side declaration (the generated OpenCL source checks
   assert() with its prelude's macro). */
#if defined(__cplusplus) && defined(__CUDA__) && (defined(OCU_OPENCL_PATH) || defined(OCU_OPENCL_HOST_PASS) || defined(OCU_SPIRV_PATH)) && \
    !defined(OCU_OPENCL_ASSERT_DECL)
#define OCU_OPENCL_ASSERT_DECL 1
__device__ static inline void __assert_fail(const char *, const char *, unsigned int, const char *) {}
#endif
#endif /* OCU_CUDA_RUNTIME_H */

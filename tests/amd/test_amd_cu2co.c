/*
 * compiler/probe/probe_kernel.cu (saxpy, block_sum) compiled unmodified for
 * an AMD GPU by tools/amdgpu/ocu_amd_cc.py and run through the AMD backend
 * (HIP runtime): the checks of tests/backend/test_metal_cu2msl.c on another
 * backend.
 *   test_amd_cu2co <code object made by ocu_amd_cc.py>
 * Exits 77 (skip) when no AMD GPU is usable or the image is for another GPU.
 */
#include "ocu_backend_amd.h"
#define TEST_BACKEND ocu_backend_amd
#define TEST_NAME "test_amd_cu2co"
#define TEST_SKIP_CODE 77
#define TEST_SKIP_WHY "no AMD GPU with a HIP runtime"
#define TEST_BUILD_HINT "the runtime did not load the code object; rerun with OCU_AMD_VERBOSE=1"
#include "../backend/test_metal_cu2msl.c"

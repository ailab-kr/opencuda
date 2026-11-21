/*
 * tests/metal/cu2msl_features.cu (address spaces, templates, device
 * functions, __device__/__constant__ variables, struct arguments, shared
 * memory) compiled unmodified for an AMD GPU by tools/amdgpu/ocu_amd_cc.py
 * and run through the AMD backend: the checks of
 * tests/metal/test_metal_features.c on another backend.
 *   test_amd_features <code object made by ocu_amd_cc.py>
 * Exits 77 (skip) when no AMD GPU is usable.
 */
#include "ocu_backend_amd.h"
#define TEST_BACKEND ocu_backend_amd
#define TEST_NAME "test_amd_features"
#define TEST_SKIP_CODE 77
#define TEST_SKIP_WHY "no AMD GPU with a HIP runtime"
#define TEST_BUILD_HINT "the runtime did not load the code object; rerun with OCU_AMD_VERBOSE=1"
#include "../metal/test_metal_features.c"

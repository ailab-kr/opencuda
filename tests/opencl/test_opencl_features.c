/*
 * tests/metal/cu2msl_features.cu (address spaces, templates, device
 * functions, __device__/__constant__ variables, struct arguments, shared
 * memory) translated by compiler/opencl/cu2ocl.py and run through the
 * OpenCL backend: the checks of tests/metal/test_metal_features.c on
 * another backend.
 *   test_opencl_features <image made by cu2ocl.py>
 * Exits 77 (skip) when no OpenCL device is usable or takes the image.
 */
#include "ocu_backend_opencl.h"
#define TEST_BACKEND ocu_backend_opencl
#define TEST_NAME "test_opencl_features"
#define TEST_SKIP_CODE 77
#define TEST_SKIP_WHY "no usable OpenCL device"
#define TEST_BUILD_HINT "the device did not build the module; rerun with OCU_OPENCL_VERBOSE=1"
#include "../metal/test_metal_features.c"

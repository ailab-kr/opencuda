/*
 * compiler/probe/probe_kernel.cu (saxpy, block_sum) translated by
 * compiler/opencl/cu2ocl.py and run through the OpenCL backend: the checks
 * of tests/backend/test_metal_cu2msl.c on another backend.
 *   test_opencl_cu2ocl <image made by cu2ocl.py>
 * Exits 77 (skip) when no OpenCL device is usable or takes the image.
 */
#include "ocu_backend_opencl.h"
#define TEST_BACKEND ocu_backend_opencl
#define TEST_NAME "test_opencl_cu2ocl"
#define TEST_SKIP_CODE 77
#define TEST_SKIP_WHY "no usable OpenCL device"
#define TEST_BUILD_HINT "the device did not build the module; rerun with OCU_OPENCL_VERBOSE=1"
#include "../backend/test_metal_cu2msl.c"

// Local header, included with quotes from ptx_shift_run.cu: its device code is
// rewritten in a copy, the original is left alone (card: files included with
// quotes from the source's folder and below).
#pragma once
#include <cuda_runtime.h>

__device__ inline unsigned local_shl(unsigned a, int n) { return a << n; }

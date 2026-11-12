// Below the source's folder, included with quotes.
#pragma once
#include <cuda_runtime.h>

__device__ inline int sub_shr(int a, int n) { return a >> n; }

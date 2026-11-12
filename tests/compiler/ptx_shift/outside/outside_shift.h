// Not in the source's folder (found with -I): device code here is not
// rewritten, the shift is reported.
#pragma once
__device__ inline unsigned outside_shl(unsigned a, int n)
{
    return a << n; // MARK3
}

// An inline kernel defined in a header: included by two translation units
// that may be compiled with and without the whole-block pass, then linked
// together (the linker keeps one definition; both must work).
#pragma once
inline __global__ void k_lp_inl(int *o)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    o[t] = (int)t * 11 + 3;
}
inline void launch_inl(int *o, int n)
{
    k_lp_inl<<<(n + 31) / 32, 32>>>(o);
}

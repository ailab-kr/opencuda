// Local header of ptx_shift_cases.cu (case 8: a shift split between a macro
// and this file, reported with this file's name and line).
#pragma once
#define SHL_ARGS_H(a, b) ((a) << (b))

__device__ inline unsigned header_partial(unsigned a, int b)
{
    return SHL_ARGS_H(a, b); // MARK8
}

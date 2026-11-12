// Compile-only cases for ocu_cc.py --ptx-shift (driven by ptx_shift_cmd.cmake
// with -D CASE=<n>).  A line ending in `MARK<n>` is where the shift of case n
// is; the warning must name that file and line.
//
//  0  nothing to report (control)
//  1  operands passed to a macro:            ((a) << (b))          warns
//  2  an operand that is itself a macro:     x << AMOUNT           warns
//  3  device code in a file outside the source's folder (-I)       warns
//  4  compound shift of a bit-field:         s.f <<= n             warns
//  5  no warnings: whole expression in one macro body, literal
//     amount below 32, ordinary shifts
//  6  constexpr with an out-of-range amount (ill-formed C++): exit 1
//  8  split between a macro and a local header                      warns
// The host function host_args uses the same macro; host code is never
// changed, so it is never reported (it must not add a second warning).
#include <cuda_runtime.h>

#include "ptx_shift_cases.h"

#define SHL_ARGS(a, b) ((a) << (b))
#define AMOUNT 40
#define WHOLE (v << n)

struct BitsS {
    unsigned f : 5;
    unsigned g;
};

#if CASE == 3
#include "outside_shift.h"
#endif

__host__ unsigned host_args(unsigned a, int b) { return SHL_ARGS(a, b); }

__global__ void k(unsigned *o, int n)
{
    unsigned v = o[1];
#if CASE == 1
    o[0] = SHL_ARGS(v, n); // MARK1
#elif CASE == 2
    o[0] = v << AMOUNT; // MARK2
#elif CASE == 3
    o[0] = outside_shl(v, n);
#elif CASE == 4
    BitsS s = {3u, 0u};
    s.f <<= n; // MARK4
    o[0] = s.f;
#elif CASE == 5
    o[0] = WHOLE + (v >> 31) + (unsigned char)(v << 7);
    o[2] = v >> n;
#elif CASE == 6
    constexpr unsigned a = 1u << 32; // MARK6
    o[0] = a;
#elif CASE == 8
    o[0] = header_partial(v, n);
#else
    o[0] = v + (unsigned)n;
#endif
}

// Compile-time cases for the removed non-sync warp functions, the system
// scope atomics and the alignment attribute after a struct body. Selected
// with -DCASE=<n> by compile_case.cmake (optionally with -DMATCH=<regex>
// that the compiler's diagnostic must contain).
//   0        control: must compile (uses the _sync functions, scoped
//            atomics other than _system, and every guarded-out call)
//   1..7     must NOT compile; the message names the _sync replacement
//   8..10    must NOT compile (shapes other than the plain call)
//   11       must compile: the removed names are only mentioned in text
//   20       NVIDIA path only: atomicAdd_system must NOT compile
//   30..37   must compile: alignment after the struct body
//   38       must NOT compile: the alignment is not a power of two
#include <cuda_runtime.h>

#ifndef CASE
#define CASE 0
#endif

#if CASE >= 30
#define ALIGN_CASE 1
#endif

#ifdef ALIGN_CASE
#if CASE == 30
struct S { int a; int b; } __align__(8);
static_assert(alignof(S) == 8 && sizeof(S) == 8, "plain");
#elif CASE == 31
struct S { char c; } __align__(16);
static_assert(alignof(S) == 16 && sizeof(S) == 16, "16");
#elif CASE == 32
typedef struct { float x, y; } __align__(8) S;
static_assert(alignof(S) == 8, "typedef, attribute before the name");
#elif CASE == 33
struct S { struct { int q; } in; int r; } __align__(8);
static_assert(alignof(S) == 8 && sizeof(S) == 8, "braces before the attribute");
#elif CASE == 34
struct S { int a; int b; } __align__(8) g_s;
static_assert(alignof(S) == 8, "declarator after the attribute");
#elif CASE == 35
struct S
{
    int a;
    int b;
}
__align__(8);
static_assert(alignof(S) == 8, "attribute on its own line");
#elif CASE == 36
class S { public: int a; int b; } __align__(8);
static_assert(alignof(S) == 8, "class");
#elif CASE == 37
union S { int a; char b[4]; } __align__(8);
static_assert(alignof(S) == 8 && sizeof(S) == 8, "union");
#elif CASE == 38
struct S { int a; } __align__(6);
#else
#error unknown case
#endif
__global__ void k(S *p) { (void)p; }
#else
__global__ void k(float *out, int *iout, unsigned *uout)
{
    float v = (float)threadIdx.x;
    int lane = threadIdx.x & 31;
    // the supported spellings
    out[0] = __shfl_sync(0xffffffffu, v, 1);
    out[1] = __shfl_up_sync(0xffffffffu, v, 1);
    out[2] = __shfl_down_sync(0xffffffffu, v, 1);
    out[3] = __shfl_xor_sync(0xffffffffu, v, 1);
    uout[0] = __ballot_sync(0xffffffffu, lane & 1);
    iout[0] = __any_sync(0xffffffffu, lane > 3);
    iout[1] = __all_sync(0xffffffffu, lane >= 0);
    atomicAdd_block(iout + 2, 1);
    atomicExch_block(iout + 3, 1);
#if CASE == 1
    out[4] = __shfl(v, 1);
#elif CASE == 2
    out[4] = __shfl_up(v, 1);
#elif CASE == 3
    out[4] = __shfl_down(v, 1);
#elif CASE == 4
    out[4] = __shfl_xor(v, 1);
#elif CASE == 5
    uout[1] = __ballot(lane & 1);
#elif CASE == 6
    iout[4] = __any(lane > 3);
#elif CASE == 7
    iout[4] = __all(lane >= 0);
#elif CASE == 8
    out[4] = __shfl(v, 1, 16);           // with a width argument
#elif CASE == 9
    iout[4] = __shfl(lane, 1);           // integer operand
#elif CASE == 10
    double d = __shfl(2.0, 3);           // double operand
    out[4] = (float)d;
#elif CASE == 11
    // names inside a string or a comment only: __shfl __shfl_up __ballot
    const char *s = "__shfl __any __all";
    iout[4] = (int)s[0];
#elif CASE == 20
    atomicAdd_system(iout + 4, 1);
#endif
}
#endif

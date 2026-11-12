// Case 10 of ptx_shift_cmd.cmake: a macro whose body would be rewritten
// (operator and operand ends are its own text) but which is also used in an
// #if line, where the preprocessor cannot evaluate the rewritten form: it is
// left as C++ and reported (card: "#if·#elif 줄에도 쓰이는 것").  The report
// names the macro's #define line.
#include <cuda_runtime.h>

#define SHL_IF(a, b) ((a) << (b)) // MARK10

#if SHL_IF(1, 3) == 8
#define EIGHT_OK 1
#endif

__global__ void k(unsigned *o, int n)
{
    unsigned v = o[1];
    o[0] = SHL_IF(v, n);
#ifdef EIGHT_OK
    o[2] = 8;
#endif
}

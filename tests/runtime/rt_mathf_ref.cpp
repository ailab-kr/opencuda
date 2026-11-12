// Host side of rt_mathf_gpu.cu: the same float functions (ocu_device_math.h)
// evaluated by the host compiler, for the bit-for-bit comparison.
#include "ocu_device_math.h"

#include "rt_mathf_fns.h"

float ocu_mathf_host(int fn, float x, float y)
{
    using namespace ocu_dm;
    float s, c;
    switch (fn) {
    case F_EXP: return expf(x);
    case F_EXP2: return exp2f(x);
    case F_EXP10: return exp10f(x);
    case F_EXPM1: return expm1f(x);
    case F_LOG: return logf(x);
    case F_LOG2: return log2f(x);
    case F_LOG10: return log10f(x);
    case F_LOG1P: return log1pf(x);
    case F_SIN: return sinf(x);
    case F_COS: return cosf(x);
    case F_TAN: return tanf(x);
    case F_SINCOS_S: sincosf(x, &s, &c); return s;
    case F_SINCOS_C: sincosf(x, &s, &c); return c;
    case F_ASIN: return asinf(x);
    case F_ACOS: return acosf(x);
    case F_ATAN: return atanf(x);
    case F_SINH: return sinhf(x);
    case F_COSH: return coshf(x);
    case F_TANH: return tanhf(x);
    case F_CBRT: return cbrtf(x);
    case F_ERF: return erff(x);
    case F_ERFC: return erfcf(x);
    case F_POW: return powf(x, y);
    case F_ATAN2: return atan2f(x, y);
    case F_HYPOT: return hypotf(x, y);
    }
    return 0.0f;
}

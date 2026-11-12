// Function numbers shared by rt_mathf_gpu.cu and rt_mathf_ref.cpp.
#ifndef RT_MATHF_FNS_H
#define RT_MATHF_FNS_H
enum {
    F_EXP, F_EXP2, F_EXP10, F_EXPM1, F_LOG, F_LOG2, F_LOG10, F_LOG1P, F_SIN, F_COS, F_TAN,
    F_SINCOS_S, F_SINCOS_C, F_ASIN, F_ACOS, F_ATAN, F_SINH, F_COSH, F_TANH, F_CBRT, F_ERF,
    F_ERFC, F_POW, F_ATAN2, F_HYPOT, F_COUNT
};
static const char *const F_NAMES[F_COUNT] = {
    "expf", "exp2f", "exp10f", "expm1f", "logf", "log2f", "log10f", "log1pf", "sinf", "cosf",
    "tanf", "sincosf.s", "sincosf.c", "asinf", "acosf", "atanf", "sinhf", "coshf", "tanhf",
    "cbrtf", "erff", "erfcf", "powf", "atan2f", "hypotf"};
float ocu_mathf_host(int fn, float x, float y);
#endif

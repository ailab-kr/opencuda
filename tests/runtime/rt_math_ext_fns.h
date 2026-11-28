// Function numbers shared by rt_math_ext_gpu.cu and rt_math_ext_ref.cpp.
#ifndef RT_MATH_EXT_FNS_H
#define RT_MATH_EXT_FNS_H
// Exact results: GPU and host must give the same bits.
enum {
    E_FADD_RZ, E_FADD_RU, E_FADD_RD, E_FMUL_RZ, E_FMUL_RU, E_FMUL_RD, E_FDIV_RZ, E_FDIV_RU, E_FDIV_RD,
    E_FSQRT_RZ, E_FSQRT_RU, E_FSQRT_RD, E_FMAF_RZ, E_FMAF_RU, E_FMAF_RD, E_FRSQRT_RN,
    E_LL2F_RD, E_LL2F_RU, E_ULL2F_RZ, E_F2LL_RD, E_F2ULL_RU, E_REMQUOF, E_REMQUOF_Q, E_NEXTAFTERF,
    E_ILOGBF, E_SCALBNF, E_MODFF, E_MODFF_I,
    EF_COUNT
};
enum {
    E_DADD_RZ, E_DADD_RU, E_DADD_RD, E_DMUL_RZ, E_DMUL_RU, E_DMUL_RD, E_DDIV_RZ, E_DDIV_RU, E_DDIV_RD,
    E_DSQRT_RZ, E_DSQRT_RU, E_DSQRT_RD, E_LL2D_RU, E_ULL2D_RD, E_D2LL_RD, E_D2ULL_RZ, E_REMQUO,
    E_REMQUO_Q, E_REMAINDER, E_NEXTAFTER, E_ILOGB, E_SCALBN, E_MODF, E_MODF_I, E_LLRINT,
    ED_COUNT
};
// Approximations: measured against long double on the host.
enum {
    A_SINPI, A_COSPI, A_ACOSH, A_ASINH, A_ATANH, A_RCBRT, A_RHYPOT, A_NORM3D, A_RNORM4D,
    AD_COUNT
};
enum {
    A_SINPIF, A_COSPIF, A_ACOSHF, A_ASINHF, A_ATANHF, A_RCBRTF, A_RHYPOTF, A_NORM3DF, A_RNORM4DF,
    AF_COUNT
};
// host side (rt_math_ext_ref.cpp)
void ocu_mx_exact_f(const float *a, const float *b, const float *c, const long long *v, float *out /* EF_COUNT */);
void ocu_mx_exact_d(const double *a, const double *b, const long long *v, double *out /* ED_COUNT */);
// errors in ulps of the long double reference
double ocu_mx_err_d(int fn, double got, double a, double b, double c, double d);
double ocu_mx_err_f(int fn, float got, float a, float b, float c, float d);
#endif

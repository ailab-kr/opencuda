/*
 * cuComplex.h -- single and double precision complex numbers for host and
 * device code, as the public CUDA documentation describes them: the types
 * cuFloatComplex (= float2, x real, y imaginary), cuDoubleComplex (= double2)
 * and cuComplex (= cuFloatComplex), and the functions make_cu*Complex,
 * cuCreal/cuCimag, cuConj, cuCadd, cuCsub, cuCmul, cuCdiv, cuCabs, cuCfma
 * (each with an f-suffixed float form), cuComplexFloatToDouble and
 * cuComplexDoubleToFloat.
 *
 * Written for this project from that public description; not derived from
 * any vendor header. Division and absolute value avoid intermediate overflow
 * and underflow: Smith's algorithm (1962) for the quotient, scaling by the
 * larger component for the absolute value.
 */
#ifndef OCU_CUCOMPLEX_H
#define OCU_CUCOMPLEX_H

#include "cuda_runtime_api.h" /* float2, double2, make_float2, make_double2 */

#if defined(__cplusplus)
#include <cmath>
#define OCU_CX_FN static inline __host__ __device__
#define OCU_CX_SQRT(x) sqrt(x)
#define OCU_CX_SQRTF(x) sqrtf(x)
#define OCU_CX_FABS(x) fabs(x)
#define OCU_CX_FABSF(x) fabsf(x)
#else
#include <math.h>
#define OCU_CX_FN static inline
#define OCU_CX_SQRT(x) sqrt(x)
#define OCU_CX_SQRTF(x) sqrtf(x)
#define OCU_CX_FABS(x) fabs(x)
#define OCU_CX_FABSF(x) fabsf(x)
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

typedef float2 cuFloatComplex;
typedef double2 cuDoubleComplex;
typedef cuFloatComplex cuComplex;

/* ---- float -------------------------------------------------------------------------- */

OCU_CX_FN float cuCrealf(cuFloatComplex x) { return x.x; }
OCU_CX_FN float cuCimagf(cuFloatComplex x) { return x.y; }
OCU_CX_FN cuFloatComplex make_cuFloatComplex(float r, float i)
{
    cuFloatComplex z;
    z.x = r;
    z.y = i;
    return z;
}
OCU_CX_FN cuFloatComplex cuConjf(cuFloatComplex x) { return make_cuFloatComplex(x.x, -x.y); }
OCU_CX_FN cuFloatComplex cuCaddf(cuFloatComplex x, cuFloatComplex y) { return make_cuFloatComplex(x.x + y.x, x.y + y.y); }
OCU_CX_FN cuFloatComplex cuCsubf(cuFloatComplex x, cuFloatComplex y) { return make_cuFloatComplex(x.x - y.x, x.y - y.y); }
OCU_CX_FN cuFloatComplex cuCmulf(cuFloatComplex x, cuFloatComplex y)
{
    return make_cuFloatComplex(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}
/* Smith's algorithm: divide by the larger component of y first */
OCU_CX_FN cuFloatComplex cuCdivf(cuFloatComplex x, cuFloatComplex y)
{
    if (OCU_CX_FABSF(y.x) >= OCU_CX_FABSF(y.y)) {
        float r = y.y / y.x, d = y.x + y.y * r;
        if (y.x == 0.0f && y.y == 0.0f) return make_cuFloatComplex(x.x / OCU_CX_FABSF(y.x), x.y / OCU_CX_FABSF(y.x));
        return make_cuFloatComplex((x.x + x.y * r) / d, (x.y - x.x * r) / d);
    }
    float r = y.x / y.y, d = y.x * r + y.y;
    return make_cuFloatComplex((x.x * r + x.y) / d, (x.y * r - x.x) / d);
}
/* |x| without overflow or underflow of the squares */
OCU_CX_FN float cuCabsf(cuFloatComplex x)
{
    float a = OCU_CX_FABSF(x.x), b = OCU_CX_FABSF(x.y);
    if (a != a || b != b) return (a == INFINITY || b == INFINITY) ? INFINITY : a + b;
    float v = a > b ? a : b, w = a > b ? b : a;
    if (v == 0.0f || v == INFINITY) return v;
    float t = w / v;
    return v * OCU_CX_SQRTF(1.0f + t * t);
}
OCU_CX_FN cuComplex make_cuComplex(float x, float y) { return make_cuFloatComplex(x, y); }
/* x * y + d */
OCU_CX_FN cuFloatComplex cuCfmaf(cuFloatComplex x, cuFloatComplex y, cuFloatComplex d)
{
    float re = (x.x * y.x) + d.x;
    float im = (x.x * y.y) + d.y;
    re = -(x.y * y.y) + re;
    im = (x.y * y.x) + im;
    return make_cuFloatComplex(re, im);
}

/* ---- double ------------------------------------------------------------------------- */

OCU_CX_FN double cuCreal(cuDoubleComplex x) { return x.x; }
OCU_CX_FN double cuCimag(cuDoubleComplex x) { return x.y; }
OCU_CX_FN cuDoubleComplex make_cuDoubleComplex(double r, double i)
{
    cuDoubleComplex z;
    z.x = r;
    z.y = i;
    return z;
}
OCU_CX_FN cuDoubleComplex cuConj(cuDoubleComplex x) { return make_cuDoubleComplex(x.x, -x.y); }
OCU_CX_FN cuDoubleComplex cuCadd(cuDoubleComplex x, cuDoubleComplex y) { return make_cuDoubleComplex(x.x + y.x, x.y + y.y); }
OCU_CX_FN cuDoubleComplex cuCsub(cuDoubleComplex x, cuDoubleComplex y) { return make_cuDoubleComplex(x.x - y.x, x.y - y.y); }
OCU_CX_FN cuDoubleComplex cuCmul(cuDoubleComplex x, cuDoubleComplex y)
{
    return make_cuDoubleComplex(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}
OCU_CX_FN cuDoubleComplex cuCdiv(cuDoubleComplex x, cuDoubleComplex y)
{
    if (OCU_CX_FABS(y.x) >= OCU_CX_FABS(y.y)) {
        double r = y.y / y.x, d = y.x + y.y * r;
        if (y.x == 0.0 && y.y == 0.0) return make_cuDoubleComplex(x.x / OCU_CX_FABS(y.x), x.y / OCU_CX_FABS(y.x));
        return make_cuDoubleComplex((x.x + x.y * r) / d, (x.y - x.x * r) / d);
    }
    double r = y.x / y.y, d = y.x * r + y.y;
    return make_cuDoubleComplex((x.x * r + x.y) / d, (x.y * r - x.x) / d);
}
OCU_CX_FN double cuCabs(cuDoubleComplex x)
{
    double a = OCU_CX_FABS(x.x), b = OCU_CX_FABS(x.y);
    if (a != a || b != b) return (a == INFINITY || b == INFINITY) ? INFINITY : a + b;
    double v = a > b ? a : b, w = a > b ? b : a;
    if (v == 0.0 || v == INFINITY) return v;
    double t = w / v;
    return v * OCU_CX_SQRT(1.0 + t * t);
}
OCU_CX_FN cuDoubleComplex cuCfma(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex d)
{
    double re = (x.x * y.x) + d.x;
    double im = (x.x * y.y) + d.y;
    re = -(x.y * y.y) + re;
    im = (x.y * y.x) + im;
    return make_cuDoubleComplex(re, im);
}

OCU_CX_FN cuDoubleComplex cuComplexFloatToDouble(cuFloatComplex c) { return make_cuDoubleComplex((double)c.x, (double)c.y); }
OCU_CX_FN cuFloatComplex cuComplexDoubleToFloat(cuDoubleComplex c) { return make_cuFloatComplex((float)c.x, (float)c.y); }

#undef OCU_CX_FN
#undef OCU_CX_SQRT
#undef OCU_CX_SQRTF
#undef OCU_CX_FABS
#undef OCU_CX_FABSF

#endif /* OCU_CUCOMPLEX_H */

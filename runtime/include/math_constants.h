/*
 * math_constants.h -- named mathematical constants for device and host code.
 *
 * Written for this project from the public CUDA Math API reference
 * (single and double precision constants named CUDART_*). Every value is a
 * mathematical constant, computed here to 60 digits and rounded once to
 * the target type; the _HI/_LO pairs split a double (or float) value into
 * its nearest representable part and the rounded remainder, so that
 * HI + LO is the constant to about twice the precision.
 * Not derived from any vendor header.
 */
#ifndef OCU_MATH_CONSTANTS_H
#define OCU_MATH_CONSTANTS_H

/* ---- single precision --------------------------------------------------- */
#define CUDART_INF_F            __builtin_inff()
#define CUDART_NAN_F            __builtin_nanf("")
#define CUDART_MIN_DENORM_F     1.401298464e-45f
#define CUDART_MAX_NORMAL_F     3.402823466e+38f
#define CUDART_NEG_ZERO_F       (-0.0f)
#define CUDART_ZERO_F           0.0f
#define CUDART_ONE_F            1.0f
#define CUDART_SQRT_TWO_F       1.41421354f
#define CUDART_SQRT_HALF_F      0.707106769f
#define CUDART_SQRT_HALF_HI_F   0.707106769f
#define CUDART_SQRT_HALF_LO_F   1.21016175e-08f
#define CUDART_THIRD_F          0.333333343f
#define CUDART_PIO4_F           0.785398185f
#define CUDART_PIO2_F           1.57079637f
#define CUDART_3PIO4_F          2.3561945f
#define CUDART_2_OVER_PI_F      0.636619747f
#define CUDART_SQRT_2OPI_F      0.797884583f
#define CUDART_PI_F             3.14159274f
#define CUDART_L2E_F            1.44269502f
#define CUDART_L2T_F            3.32192802f
#define CUDART_LG2_F            0.30103001f
#define CUDART_LN2_F            0.693147182f
#define CUDART_LNPI_F           1.14472985f
#define CUDART_TWO_TO_23_F      8388608.0f
#define CUDART_TWO_TO_24_F      16777216.0f
#define CUDART_TWO_TO_31_F      2147483648.0f
#define CUDART_TWO_TO_32_F      4294967296.0f

/* ---- double precision --------------------------------------------------- */
#define CUDART_INF              __builtin_inf()
#define CUDART_NAN              __builtin_nan("")
#define CUDART_MIN_DENORM       4.9406564584124654e-324
#define CUDART_MAX_NORMAL       1.7976931348623157e+308
#define CUDART_NEG_ZERO         (-0.0)
#define CUDART_ZERO             0.0
#define CUDART_ONE              1.0
#define CUDART_SQRT_TWO         1.4142135623730951
#define CUDART_SQRT_HALF        0.70710678118654757
#define CUDART_SQRT_HALF_HI     0.70710678118654757
#define CUDART_SQRT_HALF_LO     -4.8336466567264567e-17
#define CUDART_THIRD            0.33333333333333331
#define CUDART_TWOTHIRD         0.66666666666666663
#define CUDART_PIO4             0.78539816339744828
#define CUDART_PIO4_HI          0.78539816339744828
#define CUDART_PIO4_LO          3.061616997868383e-17
#define CUDART_PIO2             1.5707963267948966
#define CUDART_PIO2_HI          1.5707963267948966
#define CUDART_PIO2_LO          6.123233995736766e-17
#define CUDART_3PIO4            2.3561944901923448
#define CUDART_2_OVER_PI        0.63661977236758138
#define CUDART_PI               3.1415926535897931
#define CUDART_PI_HI            3.1415926535897931
#define CUDART_PI_LO            1.2246467991473532e-16
#define CUDART_SQRT_2PI         2.5066282746310007
#define CUDART_SQRT_2PI_HI      2.5066282746310007
#define CUDART_SQRT_2PI_LO      -1.8328579980459167e-16
#define CUDART_SQRT_PIO2        1.2533141373155003
#define CUDART_SQRT_PIO2_HI     1.2533141373155003
#define CUDART_SQRT_PIO2_LO     -9.1642899902295834e-17
#define CUDART_SQRT_2OPI        0.79788456080286541
#define CUDART_L2E              1.4426950408889634
#define CUDART_L2T              3.3219280948873622
#define CUDART_LG2              0.3010299956639812
#define CUDART_LG2_HI           0.3010299956639812
#define CUDART_LG2_LO           -2.8037281277851704e-18
#define CUDART_LGE              0.43429448190325182
#define CUDART_LN2              0.69314718055994529
#define CUDART_LNT              2.3025850929940459
#define CUDART_LNPI             1.1447298858494002
#define CUDART_TWO_TO_23        8388608.0
#define CUDART_TWO_TO_24        16777216.0
#define CUDART_TWO_TO_31        2147483648.0
#define CUDART_TWO_TO_32        4294967296.0

#endif /* OCU_MATH_CONSTANTS_H */

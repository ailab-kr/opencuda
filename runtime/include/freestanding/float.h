/*
 * float.h for a device pass with no C library headers (see limits.h in
 * this folder). Values from the compiler's predefined __FLT_*__ and
 * __DBL_*__ macros.
 */
#ifndef OCU_FREESTANDING_FLOAT_H
#define OCU_FREESTANDING_FLOAT_H
#define FLT_RADIX __FLT_RADIX__
#define FLT_MANT_DIG __FLT_MANT_DIG__
#define DBL_MANT_DIG __DBL_MANT_DIG__
#define FLT_DIG __FLT_DIG__
#define DBL_DIG __DBL_DIG__
#define FLT_MIN_EXP __FLT_MIN_EXP__
#define DBL_MIN_EXP __DBL_MIN_EXP__
#define FLT_MAX_EXP __FLT_MAX_EXP__
#define DBL_MAX_EXP __DBL_MAX_EXP__
#define FLT_MAX __FLT_MAX__
#define DBL_MAX __DBL_MAX__
#define FLT_MIN __FLT_MIN__
#define DBL_MIN __DBL_MIN__
#define FLT_EPSILON __FLT_EPSILON__
#define DBL_EPSILON __DBL_EPSILON__
#define FLT_TRUE_MIN __FLT_DENORM_MIN__
#define DBL_TRUE_MIN __DBL_DENORM_MIN__
#endif

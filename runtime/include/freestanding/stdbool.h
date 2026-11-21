/*
 * stdbool.h for a device pass with no C library headers (see limits.h in
 * this folder). In C++ bool, true and false are keywords.
 */
#ifndef OCU_FREESTANDING_STDBOOL_H
#define OCU_FREESTANDING_STDBOOL_H
#ifndef __cplusplus
#define bool _Bool
#define true 1
#define false 0
#endif
#define __bool_true_false_are_defined 1
#endif

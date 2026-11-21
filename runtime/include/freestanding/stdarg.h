/*
 * stdarg.h for a device pass with no C library headers (see limits.h in
 * this folder): the compiler's variable-argument builtins.
 */
#ifndef OCU_FREESTANDING_STDARG_H
#define OCU_FREESTANDING_STDARG_H
typedef __builtin_va_list va_list;
#define va_start(ap, last) __builtin_va_start(ap, last)
#define va_end(ap) __builtin_va_end(ap)
#define va_arg(ap, type) __builtin_va_arg(ap, type)
#define va_copy(dst, src) __builtin_va_copy(dst, src)
#endif

/*
 * Minimal fiber contexts for the CPU backend.
 *
 * Fast path (x86-64, AArch64): a hand-written register switch
 * (cpu_fiber_switch.S) with no system calls. Fallback: ucontext.
 * OCU_CPU_FAST_SWITCH is defined by the build when the fast path is used.
 */
#ifndef OCU_CPU_FIBER_H
#define OCU_CPU_FIBER_H

#include <stddef.h>

#if !defined(OCU_CPU_FAST_SWITCH)
#if defined(__APPLE__)
#define _XOPEN_SOURCE 700
#endif
#include <ucontext.h>
#endif

typedef struct cpu_ctx {
#if defined(OCU_CPU_FAST_SWITCH)
    void *sp;
#else
    ucontext_t uc;
#endif
} cpu_ctx;

/*
 * Prepare `ctx` so that switching to it runs `entry` on the stack
 * [stack, stack + bytes). `entry` must never return.
 * Returns 0 on success.
 */
int cpu_ctx_make(cpu_ctx *ctx, void *stack, size_t bytes, void (*entry)(void));

/* Save the running context into `from` and resume `to`. */
void cpu_ctx_switch(cpu_ctx *from, cpu_ctx *to);

/* "fast" or "ucontext", for diagnostics and tests. */
const char *cpu_ctx_kind(void);

#endif /* OCU_CPU_FIBER_H */

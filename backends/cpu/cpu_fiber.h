/*
 * Minimal fiber contexts for the CPU backend.
 *
 * Fast path (x86-64, AArch64 on POSIX systems): a hand-written register
 * switch (cpu_fiber_switch.S) with no system calls. Windows: the system's
 * fibers, which keep the thread's stack bounds right. Fallback: ucontext.
 * OCU_CPU_FAST_SWITCH is defined by the build when the fast path is used.
 */
#ifndef OCU_CPU_FIBER_H
#define OCU_CPU_FIBER_H

#include <stddef.h>

#if !defined(OCU_CPU_FAST_SWITCH) && !defined(_WIN32)
#if defined(__APPLE__)
#define _XOPEN_SOURCE 700
#endif
#include <ucontext.h>
#endif

typedef struct cpu_ctx {
#if defined(_WIN32)
    void *fiber;          /* system fiber; created once, reused while possible */
    void (*entry)(void);
    int   reusable;       /* parked at its end by cpu_ctx_finish */
#elif defined(OCU_CPU_FAST_SWITCH)
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

/*
 * The last switch away from a context whose entry has finished its work.
 * Where contexts are reused (Windows fibers), the next cpu_ctx_make of
 * `from` restarts it cheaply; elsewhere this is cpu_ctx_switch and
 * `from` is never resumed.
 */
void cpu_ctx_finish(cpu_ctx *from, cpu_ctx *to);

/* Release what cpu_ctx_make allocated (the system fiber on Windows). */
void cpu_ctx_destroy(cpu_ctx *ctx);

/* True when a context that called cpu_ctx_finish returns from it to run
   its entry again (Windows), instead of never being resumed. */
#if defined(_WIN32)
#define CPU_CTX_REUSED 1
#else
#define CPU_CTX_REUSED 0
#endif

/* "fast" or "ucontext", for diagnostics and tests. */
const char *cpu_ctx_kind(void);

#endif /* OCU_CPU_FIBER_H */

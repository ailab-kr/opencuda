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

#include <stdint.h>

/*
 * Floating-point control state: rounding mode, flush-to-zero and
 * denormals-are-zero, x87 precision (x86-64: MXCSR control bits and the x87
 * control word; AArch64: FPCR; elsewhere the C11 <fenv.h> rounding mode).
 * Status flags are not part of it. Every GPU thread starts with the state of
 * the OS thread that runs its block (captured when the block starts): the
 * entry resets it before each thread, including a thread started on a used
 * stack (chaining in cpu_exec.c). A change a thread makes stays with that
 * thread where the switch saves and restores the state per context (the
 * fast switch, ucontext); Windows fibers are made without
 * FIBER_FLAG_FLOAT_SWITCH, so there it is seen by fibers resumed after it.
 */
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#define CPU_FPCTL_X86 1
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
#define CPU_FPCTL_ARM64 1
#else
#include <fenv.h>
#endif

typedef struct cpu_fpctl {
#if defined(CPU_FPCTL_X86)
    uint32_t mxcsr;
    uint16_t x87cw;
#elif defined(CPU_FPCTL_ARM64)
    uint64_t fpcr;
#else
    int round;
#endif
} cpu_fpctl;

#if defined(CPU_FPCTL_X86)
/* MXCSR bits 0-5 are sticky exception flags; 6-15 control (DAZ, masks,
   rounding, FTZ). */
#define CPU_MXCSR_CONTROL 0xffc0u
#endif

static inline void cpu_fpctl_get(cpu_fpctl *s)
{
#if defined(CPU_FPCTL_X86)
    __asm__ volatile("stmxcsr %0" : "=m"(s->mxcsr));
    __asm__ volatile("fnstcw %0" : "=m"(s->x87cw));
#elif defined(CPU_FPCTL_ARM64)
    uint64_t v;
    __asm__ volatile("mrs %0, fpcr" : "=r"(v));
    s->fpcr = v;
#else
    s->round = fegetround();
#endif
}

/*
 * Make `s` the current control state. Reads first and writes only what
 * differs: the writes serialise the pipeline (ldmxcsr, fldcw, msr fpcr), and
 * the state almost never differs.
 */
static inline void cpu_fpctl_reset(const cpu_fpctl *s)
{
#if defined(CPU_FPCTL_X86)
    uint32_t csr;
    uint16_t cw;
    __asm__ volatile("stmxcsr %0" : "=m"(csr));
    __asm__ volatile("fnstcw %0" : "=m"(cw));
    if ((csr ^ s->mxcsr) & CPU_MXCSR_CONTROL) {
        __asm__ volatile("ldmxcsr %0" : : "m"(s->mxcsr));
    }
    if (cw != s->x87cw) {
        __asm__ volatile("fldcw %0" : : "m"(s->x87cw));
    }
#elif defined(CPU_FPCTL_ARM64)
    uint64_t v;
    __asm__ volatile("mrs %0, fpcr" : "=r"(v));
    if (v != s->fpcr) {
        __asm__ volatile("msr fpcr, %0" : : "r"(s->fpcr));
    }
#else
    if (fegetround() != s->round) {
        fesetround(s->round);
    }
#endif
}


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
 * [stack, stack + bytes). `entry` must never return. With the fast switch
 * the context starts with the floating-point control state `fp`, whoever
 * calls this (a running fiber may prepare the next one); elsewhere it starts
 * with the caller's, and `entry` resets it (cpu_fpctl_reset).
 * Returns 0 on success.
 */
int cpu_ctx_make(cpu_ctx *ctx, void *stack, size_t bytes, void (*entry)(void),
                 const cpu_fpctl *fp);

/* Save the running context into `from` and resume `to`. */
void cpu_ctx_switch(cpu_ctx *from, cpu_ctx *to);

/*
 * The last switch away from a context whose entry has finished its work.
 * Where contexts are reused (Windows fibers), the next cpu_ctx_make of
 * `from` restarts it cheaply; elsewhere this is cpu_ctx_switch and
 * `from` is never resumed.
 *
 * Both switch functions must reach the actual switch in one call level and
 * the same one: the thread sanitizer moves to the target's shadow call
 * stack before the call (cpu_exec.c), so the switch function's entry is
 * recorded on the target and popped there when the target's own switch call
 * returns. An extra nested call here leaves one entry behind per finished
 * fiber, and the thread's shadow stack grows until the sanitizer slows to a
 * crawl and overflows it.
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

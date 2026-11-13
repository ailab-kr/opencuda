/*
 * Fiber contexts for the CPU backend. See cpu_fiber.h.
 */
#if !defined(OCU_CPU_FAST_SWITCH) && !defined(_WIN32)
#define _XOPEN_SOURCE 700
#define _DEFAULT_SOURCE
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#endif

#include "cpu_fiber.h"

#include <stdint.h>
#include <string.h>

#if defined(_WIN32)

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

const char *cpu_ctx_kind(void) { return "windows-fiber"; }

/* Runs the entry each time the fiber is (re)started; a finished entry
   parks in cpu_ctx_finish and comes back here for the next run. */
static VOID CALLBACK fiber_main(LPVOID param)
{
    cpu_ctx *ctx = (cpu_ctx *)param;
    for (;;) {
        ctx->entry();
    }
}

int cpu_ctx_make(cpu_ctx *ctx, void *stack, size_t bytes, void (*entry)(void),
                 const cpu_fpctl *fp)
{
    (void)stack; /* the system allocates fiber stacks */
    (void)fp;    /* the entry resets the control state (cpu_fiber.h) */
    if (ctx->fiber != NULL && ctx->reusable && ctx->entry == entry) {
        ctx->reusable = 0;
        return 0;
    }
    if (ctx->fiber != NULL) { /* abandoned mid-run (a failed block): start over */
        DeleteFiber(ctx->fiber);
        ctx->fiber = NULL;
    }
    ctx->entry = entry;
    ctx->reusable = 0;
    SIZE_T reserve = bytes < 65536 ? 65536 : bytes;
    ctx->fiber = CreateFiberEx(0, reserve, 0, fiber_main, ctx);
    return ctx->fiber != NULL ? 0 : -1;
}

static void make_current(cpu_ctx *from)
{
    if (from->fiber == NULL) { /* the scheduler: the thread itself */
        from->fiber = IsThreadAFiber() ? GetCurrentFiber() : ConvertThreadToFiber(NULL);
    }
}

void cpu_ctx_switch(cpu_ctx *from, cpu_ctx *to)
{
    make_current(from);
    SwitchToFiber(to->fiber);
}

void cpu_ctx_finish(cpu_ctx *from, cpu_ctx *to)
{
    from->reusable = 1;
    SwitchToFiber(to->fiber);
    /* Restarted by cpu_ctx_make: return so fiber_main runs the entry again. */
}

void cpu_ctx_destroy(cpu_ctx *ctx)
{
    /* Only contexts made by cpu_ctx_make own their fiber; a converted thread
       (the scheduler) is left alone. */
    if (ctx->fiber != NULL && ctx->entry != NULL) DeleteFiber(ctx->fiber);
    ctx->fiber = NULL;
}

#elif defined(OCU_CPU_FAST_SWITCH)

void ocu_cpu_ctx_switch(void **save_sp, void *load_sp);

const char *cpu_ctx_kind(void) { return "fast"; }

void cpu_ctx_switch(cpu_ctx *from, cpu_ctx *to)
{
    ocu_cpu_ctx_switch(&from->sp, to->sp);
}

/* The same switch, not a call to cpu_ctx_switch: see cpu_fiber.h. */
void cpu_ctx_finish(cpu_ctx *from, cpu_ctx *to)
{
    ocu_cpu_ctx_switch(&from->sp, to->sp);
}
void cpu_ctx_destroy(cpu_ctx *ctx) { (void)ctx; }

#if defined(__x86_64__)

int cpu_ctx_make(cpu_ctx *ctx, void *stack, size_t bytes, void (*entry)(void),
                 const cpu_fpctl *fp)
{
    if (bytes < 256) {
        return -1;
    }
    uintptr_t top = ((uintptr_t)stack + bytes) & ~(uintptr_t)15;
    uint64_t *t = (uint64_t *)top;
    /*
     * Layout read by ocu_cpu_ctx_switch, from the new stack pointer up:
     *   [mxcsr, x87 cw] r15 r14 r13 r12 rbx rbp [entry] [0]
     * After `ret` into entry, rsp = top - 8, which is the alignment the
     * System V ABI expects at a function's first instruction.
     */
    t[-1] = 0;                    /* fake return address of entry */
    t[-2] = (uint64_t)(uintptr_t)entry;
    for (int i = 3; i <= 8; ++i) {
        t[-i] = 0;                /* rbp rbx r12 r13 r14 r15 */
    }
    /* The given state, not the caller's: the caller may be a fiber that
       changed its own. */
    unsigned char *ctl = (unsigned char *)&t[-9];
    memset(ctl, 0, 8);
    memcpy(ctl, &fp->mxcsr, 4);
    memcpy(ctl + 4, &fp->x87cw, 2);
    ctx->sp = &t[-9];
    return 0;
}

#elif defined(__aarch64__)

int cpu_ctx_make(cpu_ctx *ctx, void *stack, size_t bytes, void (*entry)(void),
                 const cpu_fpctl *fp)
{
    if (bytes < 512) {
        return -1;
    }
    uintptr_t top = ((uintptr_t)stack + bytes) & ~(uintptr_t)15;
    /*
     * 176-byte frame read by ocu_cpu_ctx_switch:
     *   x19..x28 (0..79), x29 (80), x30 (88), d8..d15 (96..159), fpcr (160)
     * x30 = entry, so the final `ret` jumps there with sp = top.
     */
    uint64_t *f = (uint64_t *)(top - 176);
    memset(f, 0, 176);
    f[11] = (uint64_t)(uintptr_t)entry; /* x30 at offset 88 */
    f[20] = fp->fpcr;                    /* offset 160; not the caller's */
    ctx->sp = f;
    return 0;
}

#else
#error "OCU_CPU_FAST_SWITCH set for an unsupported architecture"
#endif

#else /* ucontext fallback */

const char *cpu_ctx_kind(void) { return "ucontext"; }

int cpu_ctx_make(cpu_ctx *ctx, void *stack, size_t bytes, void (*entry)(void),
                 const cpu_fpctl *fp)
{
    (void)fp; /* the entry resets the control state (cpu_fiber.h) */
    if (getcontext(&ctx->uc) != 0) {
        return -1;
    }
    ctx->uc.uc_stack.ss_sp = stack;
    ctx->uc.uc_stack.ss_size = bytes;
    ctx->uc.uc_link = NULL;
    makecontext(&ctx->uc, entry, 0);
    return 0;
}

void cpu_ctx_switch(cpu_ctx *from, cpu_ctx *to)
{
    swapcontext(&from->uc, &to->uc);
}

/* The same switch, not a call to cpu_ctx_switch: see cpu_fiber.h. */
void cpu_ctx_finish(cpu_ctx *from, cpu_ctx *to)
{
    swapcontext(&from->uc, &to->uc);
}
void cpu_ctx_destroy(cpu_ctx *ctx) { (void)ctx; }

#endif

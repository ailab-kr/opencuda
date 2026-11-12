/*
 * Fiber contexts for the CPU backend. See cpu_fiber.h.
 */
#if !defined(OCU_CPU_FAST_SWITCH)
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

#if defined(OCU_CPU_FAST_SWITCH)

void ocu_cpu_ctx_switch(void **save_sp, void *load_sp);

const char *cpu_ctx_kind(void) { return "fast"; }

void cpu_ctx_switch(cpu_ctx *from, cpu_ctx *to)
{
    ocu_cpu_ctx_switch(&from->sp, to->sp);
}

#if defined(__x86_64__)

int cpu_ctx_make(cpu_ctx *ctx, void *stack, size_t bytes, void (*entry)(void))
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
    uint32_t mxcsr;
    uint16_t fcw;
    __asm__ volatile("stmxcsr %0" : "=m"(mxcsr));
    __asm__ volatile("fnstcw %0" : "=m"(fcw));
    unsigned char *ctl = (unsigned char *)&t[-9];
    memset(ctl, 0, 8);
    memcpy(ctl, &mxcsr, 4);
    memcpy(ctl + 4, &fcw, 2);
    ctx->sp = &t[-9];
    return 0;
}

#elif defined(__aarch64__)

int cpu_ctx_make(cpu_ctx *ctx, void *stack, size_t bytes, void (*entry)(void))
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
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    f[20] = fpcr;                        /* offset 160 */
    ctx->sp = f;
    return 0;
}

#else
#error "OCU_CPU_FAST_SWITCH set for an unsupported architecture"
#endif

#else /* ucontext fallback */

const char *cpu_ctx_kind(void) { return "ucontext"; }

int cpu_ctx_make(cpu_ctx *ctx, void *stack, size_t bytes, void (*entry)(void))
{
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

#endif

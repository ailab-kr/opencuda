/*
 * Fiber context switch for the CPU backend, without the signal-mask system
 * calls that swapcontext makes.
 *
 *   void ocu_cpu_ctx_switch(void **save_sp, void *load_sp);
 *
 * Saves the callee-saved registers and floating-point control state of the
 * running context on its own stack, stores the stack pointer in *save_sp,
 * then switches to the stack at load_sp and restores the same set from it.
 * A fresh stack is prepared by ocu_cpu_ctx_make() in cpu_fiber.c with the
 * same layout, so the first switch "returns" into the fiber's entry.
 *
 * Supported: x86-64 (System V: Linux, macOS) and AArch64 (Linux, macOS).
 * Other targets use the ucontext fallback in cpu_fiber.c.
 */

#if defined(__APPLE__)
#define SYM(name) _##name
#else
#define SYM(name) name
#endif

#if defined(__x86_64__)

    .text
    .globl SYM(ocu_cpu_ctx_switch)
#if !defined(__APPLE__)
    .type ocu_cpu_ctx_switch, @function
#endif
    .p2align 4
SYM(ocu_cpu_ctx_switch):
    /* rdi = save_sp, rsi = load_sp */
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    subq  $8, %rsp
    stmxcsr (%rsp)
    fnstcw  4(%rsp)
    /* Loading the control words is slow (ldmxcsr and fldcw serialise);
       they almost never differ between contexts, so load only on change. */
    movl  (%rsp), %eax
    movzwl 4(%rsp), %ecx
    movq  %rsp, (%rdi)
    movq  %rsi, %rsp
    cmpl  (%rsp), %eax
    je    1f
    ldmxcsr (%rsp)
1:
    cmpw  4(%rsp), %cx
    je    2f
    fldcw   4(%rsp)
2:
    addq  $8, %rsp
    popq  %r15
    popq  %r14
    popq  %r13
    popq  %r12
    popq  %rbx
    popq  %rbp
    /* Into the other context by an indirect jump rather than `ret`: the
       return predictor holds this context's return address, so `ret`
       always mispredicts here, while the indirect predictor learns the
       few resume points. Measured on x86-64: a ping-pong round trip
       15 -> 4 ns; llama.cpp prompt processing on this path 16 -> 17.4
       tokens/s against `ret` with everything else equal. */
    popq  %rdx
    jmpq  *%rdx
#if !defined(__APPLE__)
    .size ocu_cpu_ctx_switch, .-ocu_cpu_ctx_switch
#endif

#elif defined(__aarch64__)

    .text
    .globl SYM(ocu_cpu_ctx_switch)
#if !defined(__APPLE__)
    .type ocu_cpu_ctx_switch, %function
#endif
    .p2align 4
SYM(ocu_cpu_ctx_switch):
    /* x0 = save_sp, x1 = load_sp */
    sub  sp, sp, #176
    stp  x19, x20, [sp, #0]
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    stp  x25, x26, [sp, #48]
    stp  x27, x28, [sp, #64]
    stp  x29, x30, [sp, #80]
    stp  d8,  d9,  [sp, #96]
    stp  d10, d11, [sp, #112]
    stp  d12, d13, [sp, #128]
    stp  d14, d15, [sp, #144]
    mrs  x9, fpcr
    str  x9, [sp, #160]
    mov  x9, sp
    str  x9, [x0]
    mov  sp, x1
    ldr  x9, [sp, #160]
    msr  fpcr, x9
    ldp  x19, x20, [sp, #0]
    ldp  x21, x22, [sp, #16]
    ldp  x23, x24, [sp, #32]
    ldp  x25, x26, [sp, #48]
    ldp  x27, x28, [sp, #64]
    ldp  x29, x30, [sp, #80]
    ldp  d8,  d9,  [sp, #96]
    ldp  d10, d11, [sp, #112]
    ldp  d12, d13, [sp, #128]
    ldp  d14, d15, [sp, #144]
    add  sp, sp, #176
    ret
#if !defined(__APPLE__)
    .size ocu_cpu_ctx_switch, .-ocu_cpu_ctx_switch
#endif

#endif

#if defined(__linux__) && defined(__ELF__)
    .section .note.GNU-stack, "", %progbits
#endif

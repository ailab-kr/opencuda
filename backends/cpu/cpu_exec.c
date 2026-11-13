/*
 * CPU backend: running a grid.
 *
 * Each GPU thread of a block is a fiber (cpu_fiber.h: a register switch on
 * x86-64 and AArch64, ucontext elsewhere). A block's fibers run on one OS
 * thread, switching only at the barrier, in fixed or shuffled order.
 * Blocks are spread over several OS threads.
 *
 * Stacks: one mapping per worker with a guard page at its low end, and a
 * canary at the bottom of every fiber stack, checked after each block. A
 * guard page per fiber would cost two kernel memory regions per GPU thread
 * and runs into vm.max_map_count with many workers.
 *
 * Barrier rule: a round runs every ready fiber until it either reaches the
 * barrier or returns. When no fiber is ready, the waiting ones are released
 * together. Fibers that returned count as arrived (what real programs that
 * `return` early before a barrier rely on); OCU_CPU_STRICT=1 makes that an
 * error instead.
 *
 * Address and thread sanitizers are told about every fiber switch, so both
 * can be used on code that runs here.
 *
 * Whole-block entries (image version 2, ocu_cpu_kernel.h): for kernels the
 * compiler found free of synchronisation points, one call runs every thread
 * of the block as a loop. It runs on one fiber of the block (so the stack
 * size, the canary check and the sanitizer hooks are those of a fiber), in
 * the order the fibers would run: fixed, or the same shuffle of the block's
 * seed. OCU_CPU_FORM=fiber turns it off.
 */
#define _DEFAULT_SOURCE
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif

#include "cpu_internal.h"
#include "cpu_fiber.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "cpu_os.h"

/* ---- sanitizer hooks ---------------------------------------------------- */

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define OCU_ASAN 1
#endif
#if __has_feature(thread_sanitizer)
#define OCU_TSAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define OCU_ASAN 1
#endif
#if defined(__SANITIZE_THREAD__)
#define OCU_TSAN 1
#endif

#ifdef OCU_ASAN
void __sanitizer_start_switch_fiber(void **fake_stack_save, const void *bottom,
                                    size_t size);
void __sanitizer_finish_switch_fiber(void *fake_stack_save,
                                     const void **bottom_old, size_t *size_old);
#endif
#ifdef OCU_TSAN
void *__tsan_get_current_fiber(void);
void *__tsan_create_fiber(unsigned flags);
void  __tsan_destroy_fiber(void *fiber);
void  __tsan_switch_to_fiber(void *fiber, unsigned flags);
#endif

/* ---- configuration ------------------------------------------------------ */

static cpu_config     g_config;
static pthread_once_t g_config_once = PTHREAD_ONCE_INIT;

static uint64_t env_u64(const char *name, uint64_t fallback)
{
    const char *v = getenv(name);
    if (v == NULL || *v == '\0') {
        return fallback;
    }
    char *end = NULL;
    errno = 0;
    unsigned long long n = strtoull(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0') {
        return fallback;
    }
    return (uint64_t)n;
}

static void load_config(void)
{
    const char *warp = getenv("OCU_CPU_WARP");
    g_config.warp_size = (warp != NULL && strcmp(warp, "64") == 0) ? 64u : 32u;

    const char *order = getenv("OCU_CPU_ORDER");
    g_config.order = (order != NULL && strcmp(order, "random") == 0)
                         ? CPU_ORDER_RANDOM
                         : CPU_ORDER_FIXED;
    g_config.seed = env_u64("OCU_CPU_SEED", 1);

    long cpus = cpu_os_cpu_count();
    uint64_t workers = env_u64("OCU_CPU_THREADS", cpus > 0 ? (uint64_t)cpus : 1);
    if (workers < 1) {
        workers = 1;
    }
    if (workers > 256) {
        workers = 256;
    }
    g_config.workers = (uint32_t)workers;

    uint64_t kb = env_u64("OCU_CPU_STACK_KB", 128);
    if (kb < 16) {
        kb = 16;
    }
    if (kb > 64 * 1024) {
        kb = 64 * 1024;
    }
    g_config.stack_bytes = (size_t)kb * 1024u;

    const char *strict = getenv("OCU_CPU_STRICT");
    g_config.strict_barrier = strict != NULL && strcmp(strict, "1") == 0;

    g_config.spin_us = (uint32_t)env_u64("OCU_CPU_SPIN_US", 50);
    if (g_config.spin_us > 100000) {
        g_config.spin_us = 100000;
    }

    const char *stats = getenv("OCU_CPU_STATS");
    g_config.stats = stats != NULL && strcmp(stats, "1") == 0;

    /* fiber: every kernel on fibers. loop (default): whole-block entries
       where the compiler made them. dispatch: the same as loop for now (a
       kernel without synchronisation points has one region, and the region
       dispatcher of docs/cpu-region-loops.md runs it the same way). */
    const char *trace = getenv("OCU_CPU_TRACE");
    g_config.trace = trace != NULL && *trace != '\0' ? trace : NULL;

    const char *form = getenv("OCU_CPU_FORM");
    g_config.loops = !(form != NULL && strcmp(form, "fiber") == 0);
}

const cpu_config *cpu_get_config(void)
{
    pthread_once(&g_config_once, load_config);
    return &g_config;
}

/* ---- launch validation -------------------------------------------------- */

ocu_status cpu_check_launch(const ocu_launch_params *p)
{
    if (p == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    static const uint32_t max_grid[3] = {CPU_MAX_GRID_X, CPU_MAX_GRID_Y,
                                         CPU_MAX_GRID_Z};
    static const uint32_t max_block[3] = {CPU_MAX_BLOCK_X, CPU_MAX_BLOCK_Y,
                                          CPU_MAX_BLOCK_Z};
    for (int d = 0; d < 3; ++d) {
        if (p->grid[d] == 0 || p->grid[d] > max_grid[d]) {
            return OCU_ERR_INVALID_VALUE;
        }
        if (p->block[d] == 0 || p->block[d] > max_block[d]) {
            return OCU_ERR_INVALID_VALUE;
        }
    }
    uint64_t threads = (uint64_t)p->block[0] * p->block[1] * p->block[2];
    if (threads > CPU_MAX_THREADS_PER_BLOCK) {
        return OCU_ERR_INVALID_VALUE;
    }
    if (p->shared_memory_bytes > CPU_MAX_SHARED_BYTES) {
        return OCU_ERR_INVALID_VALUE;
    }
    if (p->num_args > 0 && (p->args == NULL || p->arg_sizes == NULL)) {
        return OCU_ERR_INVALID_VALUE;
    }
    if (p->num_args > OCU_MAX_ARG_BYTES) {
        return OCU_ERR_INVALID_VALUE;
    }
    size_t total = 0;
    for (uint32_t i = 0; i < p->num_args; ++i) {
        if (p->args[i] == NULL && p->arg_sizes[i] != 0) {
            return OCU_ERR_INVALID_VALUE;
        }
        if (p->arg_sizes[i] > OCU_MAX_ARG_BYTES) {
            return OCU_ERR_INVALID_VALUE;
        }
        total += p->arg_sizes[i]; /* each term is bounded: no overflow */
        if (total > OCU_MAX_ARG_BYTES) {
            return OCU_ERR_INVALID_VALUE;
        }
    }
    return OCU_OK;
}

/* ---- fibers and blocks -------------------------------------------------- */

typedef enum fiber_state {
    FIBER_READY = 0,
    FIBER_WAITING,       /* at the block barrier */
    FIBER_WAITING_WARP,  /* at a warp-level operation */
    FIBER_DONE
} fiber_state;

typedef struct fiber {
    cpu_ctx      ctx;
    fiber_state  state;
    bool         started;    /* has a stack and a context in this block */
    bool         nonfull;    /* waits at a warp operation for part of its warp */
    uint32_t     slot;       /* its stack, while started */
    uint64_t     warp_seq;   /* warp operations done in this block */
    ocu_cpu_dim3 tid;        /* threadIdx */
    char        *stack;      /* usable stack of `slot`, above the guard page */
#ifdef OCU_ASAN
    void        *fake_stack;
#endif
#ifdef OCU_TSAN
    void        *tsan_fiber;
#endif
} fiber;

/*
 * What a lane declared at the warp operation it waits at. Read only while
 * the lane is parked there, by the lane that releases it.
 */
typedef struct warp_rec {
    uint64_t tag;
    uint64_t seq;    /* the lane's warp operations so far in this block */
    uint64_t named;  /* lanes it waits for (warp-relative) */
    bool     masked; /* a *_masked call (mask 0 included), not the unmasked
                        forms, which wait for the lanes still running */
} warp_rec;

/* Per warp: which lanes are parked (waiting at a warp operation or
   returned) and which of those wait, kept up to date at every arrival so a
   release costs no scan of the block. */
typedef struct warp_state {
    uint64_t lanes;   /* lanes that exist */
    uint64_t parked;  /* waiting at a warp operation, or returned */
    uint64_t waiting; /* waiting at a warp operation */
    uint32_t nonfull; /* waiting lanes whose mask names less than the warp */
} warp_state;

typedef struct block_exec {
    cpu_ctx           sched_ctx;
    fiber            *fibers;
    uint32_t         *order;
    uint32_t          nthreads;
    uint32_t          current;
    ocu_cpu_dim3      block_idx;
    ocu_cpu_dim3      block_dim;
    ocu_cpu_dim3      tid_dim;       /* block_dim the fibers' tid were made for */
    ocu_cpu_dim3      grid_dim;
    ocu_cpu_kernel_fn fn;
    ocu_cpu_block_fn  block_fn;      /* whole-block entry for this launch, or NULL */
    bool              loop_run;      /* fiber 0 runs block_fn for the block */
    bool              loop_declined; /* block_fn declined (ran no thread) */
    ocu_cpu_block_info info;         /* what block_fn is told */
    ocu_cpu_dim3     *tids;          /* threadIdx by linear index */
    void            **args;
    void             *dyn_shared;
    size_t            dyn_shared_bytes;
    warp_rec         *warp_recs;     /* one per thread */
    /* Values of a warp operation, two per thread picked by the parity of
       the lane's operation count: a lane cannot start operation s + 2
       before every lane it waits for has arrived at s + 1, so by then they
       have all read the values of s, and one wait per operation is enough. */
    uint64_t         *warp_vals;
    warp_state       *warps;
    uint32_t          nwarps;
    uint32_t          warp_size;
    uint32_t          warp_shift;    /* log2(warp_size): lane math without division */
    bool              warp_mismatch; /* lanes of a warp met at different calls */
    uint64_t         *ready_bits;    /* READY fibers not running now */
    uint64_t         *barrier_bits;  /* fibers waiting at the block barrier */
    uint32_t          nwords;        /* words in each bit set */
    uint32_t          n_barrier;     /* fibers waiting at the block barrier */
    uint32_t          n_warp;        /* fibers waiting at a warp operation */
    uint32_t          n_done;
    bool              chain;         /* CPU_CHAIN and fixed order */
    /* Stacks are handed out when a fiber first runs and taken back when it
       returns, last in first out: a block whose threads never wait runs
       them all on one warm stack. */
    uint32_t         *free_slots;
    uint32_t          nfree;
    uint32_t          lowest_slot;   /* lowest slot handed out in this block */
    char             *stack_region;  /* one mmap for all fiber stacks */
    size_t            stack_region_bytes;
    size_t            stack_bytes;   /* usable bytes per fiber */
    uint64_t          rng;
    /* Floating-point control state each GPU thread starts with: the OS
       thread's when the block starts (cpu_fiber.h). */
    cpu_fpctl         fp0;
    /* counters for OCU_CPU_STATS (cheap enough to keep always) */
    uint64_t          st_visits;     /* scheduler -> fiber switches */
    uint64_t          st_barriers;   /* block barrier waits */
    uint64_t          st_warp_waits; /* warp operation waits */
    uint64_t          st_rounds;     /* scheduler rounds */
#ifdef OCU_ASAN
    void             *sched_fake_stack;
    const void       *sched_bottom;
    size_t            sched_size;
#endif
#ifdef OCU_TSAN
    void             *sched_tsan_fiber;
#endif
} block_exec;

/*
 * Read by every device-side call. The library is position independent, so
 * the default model would look the variable up through __tls_get_addr on
 * each read; initial-exec is one load. It takes 8 bytes of the static TLS
 * reserve when the library ends up in a shared object loaded at run time.
 */
#if defined(__ELF__) && (defined(__GNUC__) || defined(__clang__))
#define CPU_TLS_FAST __attribute__((tls_model("initial-exec")))
#else
#define CPU_TLS_FAST
#endif
static _Thread_local block_exec *tl_exec CPU_TLS_FAST;

_Thread_local ocu_cpu_dim3 ocu_cpu_tls_thread_idx;
_Thread_local ocu_cpu_dim3 ocu_cpu_tls_block_idx;
_Thread_local ocu_cpu_dim3 ocu_cpu_tls_block_dim;
_Thread_local ocu_cpu_dim3 ocu_cpu_tls_grid_dim;
_Thread_local int          ocu_cpu_tls_warp_size;

static ocu_cpu_dim3 linear_to_dim3(uint32_t linear, ocu_cpu_dim3 dim);
static void fiber_entry(void);

static uint64_t rng_next(uint64_t *s)
{
    /* xorshift64*: small, fast, good enough for shuffling. */
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static void ready_set(block_exec *ex, uint32_t i)
{
    ex->ready_bits[i >> 6] |= 1ULL << (i & 63u);
}

static void ready_clear(block_exec *ex, uint32_t i)
{
    ex->ready_bits[i >> 6] &= ~(1ULL << (i & 63u));
}

/* First fiber at or after `i` that is ready; nthreads when none. */
static uint32_t next_ready(const block_exec *ex, uint32_t i)
{
    uint32_t wd = i >> 6;
    if (wd >= ex->nwords) {
        return ex->nthreads;
    }
    uint64_t m = ex->ready_bits[wd] & (~0ULL << (i & 63u));
    while (m == 0) {
        if (++wd >= ex->nwords) {
            return ex->nthreads;
        }
        m = ex->ready_bits[wd];
    }
    return (wd << 6) + (uint32_t)__builtin_ctzll(m);
}

/*
 * Chaining: a fiber whose thread returned starts the next thread itself, on
 * the same stack, when that is the thread the scheduler would run next
 * (fixed order, next index, not started yet). Saves two switches and a
 * context set-up per thread of blocks that never wait. Only where fibers
 * run on our own stacks, and not under the sanitizers, which are told
 * about each GPU thread as its own fiber.
 */
#if CPU_OS_OWN_STACKS && !CPU_CTX_REUSED && !defined(OCU_TSAN) && !defined(OCU_ASAN)
#define CPU_CHAIN 1
#else
#define CPU_CHAIN 0
#endif

/*
 * The register switch called directly, one call level less than through
 * cpu_ctx_switch: the switch returns by an indirect jump into the other
 * context, so every return on the way back out of it is mispredicted.
 * The sanitizer builds keep the wrappers (see cpu_fiber.h).
 */
#if defined(OCU_CPU_FAST_SWITCH) && !defined(OCU_TSAN) && !defined(OCU_ASAN)
void ocu_cpu_ctx_switch(void **save_sp, void *load_sp);
#define CTX_SWITCH(from, to) ocu_cpu_ctx_switch(&(from)->sp, (to)->sp)
#define CTX_FINISH(from, to) ocu_cpu_ctx_switch(&(from)->sp, (to)->sp)
#else
#define CTX_SWITCH(from, to) cpu_ctx_switch(from, to)
#define CTX_FINISH(from, to) cpu_ctx_finish(from, to)
#endif

static char *slot_stack(const block_exec *ex, uint32_t slot)
{
#if CPU_OS_OWN_STACKS
    size_t pg = cpu_os_page_size();
    return ex->stack_region + pg + (size_t)slot * (ex->stack_bytes + pg);
#else
    (void)ex;
    (void)slot;
    return NULL;
#endif
}

/*
 * Bytes of the slot's stack given to the fiber. Slots are a page longer
 * than the configured stack, and each fiber's stack top sits a different
 * number of cache lines into that page: tops a power of two apart would all
 * fall into the same cache sets.
 */
static size_t slot_bytes(const block_exec *ex, uint32_t slot)
{
    size_t lines = cpu_os_page_size() / 64u;
    return ex->stack_bytes + ((size_t)slot * 37u % lines) * 64u;
}

/* Give fiber `i` a stack and a fresh context: it starts running the kernel. */
static ocu_status start_fiber(block_exec *ex, uint32_t i)
{
    fiber *f = &ex->fibers[i];
    uint32_t slot = ex->free_slots[--ex->nfree];
    if (slot < ex->lowest_slot) {
        ex->lowest_slot = slot;
    }
    f->slot = slot;
    f->stack = slot_stack(ex, slot);
    if (cpu_ctx_make(&f->ctx, f->stack, slot_bytes(ex, slot), fiber_entry,
                     &ex->fp0) != 0) {
        ex->free_slots[ex->nfree++] = slot;
        return OCU_ERR_LAUNCH_FAILED;
    }
#ifdef OCU_TSAN
    f->tsan_fiber = __tsan_create_fiber(0);
#endif
    f->started = true;
    return OCU_OK;
}

/* Switch from the scheduler into fiber `i`; returns when it yields. */
static ocu_status switch_to_fiber(block_exec *ex, uint32_t i)
{
    fiber *f = &ex->fibers[i];
    if (!f->started) {
        ocu_status st = start_fiber(ex, i);
        if (st != OCU_OK) {
            return st;
        }
    }
    ex->current = i;
    ex->st_visits++;
    ocu_cpu_tls_thread_idx = f->tid;
#ifdef OCU_ASAN
    __sanitizer_start_switch_fiber(&ex->sched_fake_stack, f->stack,
                                   slot_bytes(ex, f->slot));
#endif
#ifdef OCU_TSAN
    __tsan_switch_to_fiber(f->tsan_fiber, 0);
#endif
    CTX_SWITCH(&ex->sched_ctx, &f->ctx);
#ifdef OCU_ASAN
    __sanitizer_finish_switch_fiber(ex->sched_fake_stack, NULL, NULL);
#endif
    /* The fiber that yielded: `i`, or a thread it chained to. */
    fiber *last = &ex->fibers[ex->current];
    if (last->state == FIBER_DONE) {
        ex->free_slots[ex->nfree++] = last->slot; /* its stack is free again */
    }
    return OCU_OK;
}

/*
 * Switch from the running fiber back to the scheduler. With chaining (fixed
 * order), hand over directly to the fiber the scheduler would run next, the
 * next ready one by index: one switch instead of two.
 */
static void yield_to_scheduler(block_exec *ex, bool exiting)
{
    fiber *f = &ex->fibers[ex->current];
#if CPU_CHAIN
    if (ex->chain) {
        uint32_t next = next_ready(ex, ex->current + 1);
        if (next < ex->nthreads &&
            (ex->fibers[next].started || start_fiber(ex, next) == OCU_OK)) {
            fiber *to = &ex->fibers[next];
            ready_clear(ex, next);
            ex->current = next;
            ex->st_visits++;
            ocu_cpu_tls_thread_idx = to->tid;
            if (exiting) {
                /* Free once the next one has its stack: nobody takes this
                   one before we have left it. */
                ex->free_slots[ex->nfree++] = f->slot;
                CTX_FINISH(&f->ctx, &to->ctx);
            } else {
                CTX_SWITCH(&f->ctx, &to->ctx);
            }
            return;
        }
        /* None left in this pass (or no stack for the next): the scheduler. */
    }
#endif
#ifdef OCU_ASAN
    __sanitizer_start_switch_fiber(exiting ? NULL : &f->fake_stack,
                                   ex->sched_bottom, ex->sched_size);
#else
    (void)exiting;
#endif
#ifdef OCU_TSAN
    __tsan_switch_to_fiber(ex->sched_tsan_fiber, 0);
#endif
    if (exiting) {
        CTX_FINISH(&f->ctx, &ex->sched_ctx);
    } else {
        CTX_SWITCH(&f->ctx, &ex->sched_ctx);
    }
#ifdef OCU_ASAN
    __sanitizer_finish_switch_fiber(f->fake_stack, NULL, NULL);
#endif
}

/*
 * The lanes `go` of warp `w` are about to be released: check that each met
 * the lanes it names at the same call, the same occurrence of it and with
 * the same mask (a lane that skipped an iteration has a different sequence
 * number). A named lane that returned before reaching the operation is
 * tolerated, or an error under OCU_CPU_STRICT=1 when the lanes were named by
 * a *_masked call: mask 0 there names the whole warp as 0xffffffff does.
 * The unmasked forms (ocu_cpu_warp_sync, ocu_cpu_warp_gather[_tagged]) mean
 * the lanes still running and tolerate returned lanes in every mode.
 */
static void check_release(block_exec *ex, uint32_t w, uint64_t go, bool full)
{
    const warp_state *ws = &ex->warps[w];
    uint32_t base = w * ex->warp_size;
    const warp_rec *recs = &ex->warp_recs[base];
    uint64_t returned = ws->parked & ~ws->waiting;
    bool strict = cpu_get_config()->strict_barrier;
    if (full) {
        /* Everyone names the whole warp: compare all against one. */
        const warp_rec *ref = &recs[__builtin_ctzll(go)];
        bool masked = false;
        for (uint64_t m = go; m != 0; m &= m - 1) {
            const warp_rec *r = &recs[__builtin_ctzll(m)];
            if (r->tag != ref->tag || r->seq != ref->seq) {
                ex->warp_mismatch = true;
            }
            masked |= r->masked;
        }
        if (strict && masked && returned != 0) {
            ex->warp_mismatch = true; /* explicitly named lane has returned */
        }
        return;
    }
    for (uint64_t m = go; m != 0; m &= m - 1) {
        const warp_rec *me = &recs[__builtin_ctzll(m)];
        for (uint64_t n = me->named; n != 0; n &= n - 1) {
            uint32_t b = (uint32_t)__builtin_ctzll(n);
            if (returned >> b & 1u) {
                if (strict && me->masked) {
                    ex->warp_mismatch = true;
                }
                continue;
            }
            const warp_rec *r = &recs[b];
            if (r->tag != me->tag || r->seq != me->seq || r->named != me->named) {
                ex->warp_mismatch = true; /* a different call, occurrence or mask */
            }
        }
    }
}

/*
 * Release the lanes of warp `w` whose named lanes are all parked. Decide
 * first, change states after: a decision must not see the result of
 * another one made in the same pass.
 */
static void try_release_warp(block_exec *ex, uint32_t w)
{
    warp_state *ws = &ex->warps[w];
    if (ws->waiting == 0) {
        return;
    }
    uint32_t base = w * ex->warp_size;
    uint64_t go;
    bool full = ws->nonfull == 0;
    if (full) {
        /* Every waiter waits for the whole warp. */
        go = ws->parked == ws->lanes ? ws->waiting : 0;
    } else {
        go = 0;
        for (uint64_t m = ws->waiting; m != 0; m &= m - 1) {
            uint32_t b = (uint32_t)__builtin_ctzll(m);
            if ((ex->warp_recs[base + b].named & ~ws->parked) == 0) {
                go |= 1ULL << b;
            }
        }
    }
    if (go == 0) {
        return;
    }
    check_release(ex, w, go, full);
    for (uint64_t m = go; m != 0; m &= m - 1) {
        uint32_t b = (uint32_t)__builtin_ctzll(m);
        fiber *f = &ex->fibers[base + b];
        f->state = FIBER_READY;
        if (base + b != ex->current) {
            ready_set(ex, base + b); /* the running lane just goes on */
        }
        if (f->nonfull) {
            f->nonfull = false;
            ws->nonfull--;
        }
        ex->n_warp--;
    }
    ws->parked &= ~go;
    ws->waiting &= ~go;
}

static void fiber_entry(void)
{
    block_exec *ex = tl_exec;
#ifdef OCU_ASAN
    __sanitizer_finish_switch_fiber(NULL, &ex->sched_bottom, &ex->sched_size);
#endif
    /* Each GPU thread starts with the block's initial floating-point control
       state (cpu_fiber.h). A context of the fast switch starts with it
       already (cpu_ctx_make); one from the system starts with the state of
       whoever made or last ran it. */
#if !defined(OCU_CPU_FAST_SWITCH)
    cpu_fpctl_reset(&ex->fp0);
#endif
    if (ex->loop_run) {
        /* The whole block as one loop. The kernel has no synchronisation
           point and does not change the control state (checked when it was
           compiled), so every thread starts with fp0 as on its own fiber. */
        ex->loop_declined = ex->block_fn(ex->args, &ex->info) != 0;
        ex->current = 0;
        ex->fibers[0].state = FIBER_DONE;
        ex->n_done = ex->nthreads;
        yield_to_scheduler(ex, true); /* nothing is ready: to the scheduler */
#if !CPU_CTX_REUSED
        abort();
#endif
        return;
    }
    for (;;) {
        ex->fn(ex->args);
        uint32_t me = ex->current;
        ex->fibers[me].state = FIBER_DONE;
        ex->n_done++;
        /* A returned lane counts as arrived for the warp operations of the
           others. */
        uint32_t w = me >> ex->warp_shift;
        ex->warps[w].parked |= 1ULL << (me & (ex->warp_size - 1));
        try_release_warp(ex, w);
#if CPU_CHAIN
        uint32_t next = me + 1;
        if (ex->chain && next < ex->nthreads && !ex->fibers[next].started) {
            fiber *n = &ex->fibers[next];
            n->started = true;
            n->slot = ex->fibers[me].slot; /* the stack passes on */
            n->stack = ex->fibers[me].stack;
            ready_clear(ex, next);
            ex->current = next;
            ex->st_visits++;
            ocu_cpu_tls_thread_idx = n->tid;
            /* The thread that ended here may have changed the control
               state; the next one must not see it. */
            cpu_fpctl_reset(&ex->fp0);
            continue;
        }
#endif
        break;
    }
    yield_to_scheduler(ex, true);
#if !CPU_CTX_REUSED
    abort(); /* a finished fiber is never resumed */
#endif
    /* Reused (Windows): return; the fiber runs this entry again next block. */
}

/* Fibers' threadIdx for the block shape `dim`. */
static void set_thread_indices(block_exec *ex, ocu_cpu_dim3 dim)
{
    for (uint32_t i = 0; i < ex->nthreads; ++i) {
        ex->fibers[i].tid = linear_to_dim3(i, dim);
        ex->tids[i] = ex->fibers[i].tid;
    }
    ex->tid_dim = dim;
}

/* Canary at the low end of each fiber stack: an overflow hits it first. */
#define CANARY_WORDS 32
#define CANARY_VALUE 0x6f63754361727921ULL

static void canary_set(char *stack)
{
    if (!CPU_OS_OWN_STACKS) return; /* system fiber stacks: the system guards them */
    uint64_t *w = (uint64_t *)(void *)stack;
    for (int i = 0; i < CANARY_WORDS; ++i) {
        w[i] = CANARY_VALUE;
    }
}

static bool canary_ok(const char *stack)
{
    if (!CPU_OS_OWN_STACKS) return true;
    const uint64_t *w = (const uint64_t *)(const void *)stack;
    for (int i = 0; i < CANARY_WORDS; ++i) {
        if (w[i] != CANARY_VALUE) {
            return false;
        }
    }
    return true;
}

static ocu_status block_exec_init(block_exec *ex, uint32_t nthreads,
                                  size_t shared_bytes)
{
    const cpu_config *cfg = cpu_get_config();
    memset(ex, 0, sizeof(*ex));
    ex->nthreads = nthreads;
    size_t pg = cpu_os_page_size();
    ex->stack_bytes = (cfg->stack_bytes + pg - 1) & ~(pg - 1);
#if CPU_OS_OWN_STACKS
    ex->stack_region_bytes = pg + (ex->stack_bytes + pg) * nthreads;
    void *region = cpu_os_map(ex->stack_region_bytes);
    if (region == NULL) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    ex->stack_region = region;
    if (cpu_os_guard(region, pg) != 0) { /* guard below fiber 0 */
        return OCU_ERR_OUT_OF_MEMORY;
    }
#endif
    ex->warp_size = cfg->warp_size;
    ex->warp_shift = (uint32_t)__builtin_ctz(ex->warp_size);
    ex->nwarps = (nthreads + ex->warp_size - 1) / ex->warp_size;
    ex->fibers = calloc(nthreads, sizeof(*ex->fibers));
    ex->order = calloc(nthreads, sizeof(*ex->order));
    ex->tids = calloc(nthreads, sizeof(*ex->tids));
    ex->warp_recs = calloc(nthreads, sizeof(*ex->warp_recs));
    ex->warp_vals = calloc(2 * (size_t)nthreads, sizeof(*ex->warp_vals));
    ex->warps = calloc(ex->nwarps, sizeof(*ex->warps));
    ex->free_slots = calloc(nthreads, sizeof(*ex->free_slots));
    ex->nwords = (nthreads + 63) / 64;
    ex->ready_bits = calloc(ex->nwords, sizeof(*ex->ready_bits));
    ex->barrier_bits = calloc(ex->nwords, sizeof(*ex->barrier_bits));
    ex->chain = CPU_CHAIN && cfg->order == CPU_ORDER_FIXED;
    if (ex->ready_bits == NULL || ex->barrier_bits == NULL ||
        ex->fibers == NULL || ex->order == NULL || ex->tids == NULL ||
        ex->warp_recs == NULL ||
        ex->warp_vals == NULL ||
        ex->warps == NULL || ex->free_slots == NULL) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < nthreads; ++i) {
        canary_set(slot_stack(ex, i));
    }
    for (uint32_t w = 0; w < ex->nwarps; ++w) {
        uint32_t count = nthreads - w * ex->warp_size;
        if (count > ex->warp_size) count = ex->warp_size;
        ex->warps[w].lanes = count == 64 ? ~0ULL : ((1ULL << count) - 1u);
    }
    if (shared_bytes > 0) {
        size_t a = CPU_SHARED_ALIGN;
        ex->dyn_shared = cpu_os_aligned_alloc(a, (shared_bytes + a - 1) & ~(a - 1));
        if (ex->dyn_shared == NULL) {
            return OCU_ERR_OUT_OF_MEMORY;
        }
        ex->dyn_shared_bytes = shared_bytes;
    }
#ifdef OCU_TSAN
    ex->sched_tsan_fiber = __tsan_get_current_fiber();
#endif
    return OCU_OK;
}

static void block_exec_release(block_exec *ex)
{
#if CPU_OS_OWN_STACKS
    if (ex->stack_region != NULL) {
        cpu_os_unmap(ex->stack_region, ex->stack_region_bytes);
    }
#endif
    if (ex->fibers != NULL) {
        for (uint32_t i = 0; i < ex->nthreads; ++i) cpu_ctx_destroy(&ex->fibers[i].ctx);
    }
    free(ex->fibers);
    free(ex->order);
    free(ex->tids);
    free(ex->warp_recs);
    free(ex->warp_vals);
    free(ex->warps);
    free(ex->free_slots);
    free(ex->ready_bits);
    free(ex->barrier_bits);
    cpu_os_aligned_free(ex->dyn_shared);
    memset(ex, 0, sizeof(*ex));
}

static void report_deadlock(void)
{
    static atomic_flag said = ATOMIC_FLAG_INIT;
    if (!atomic_flag_test_and_set(&said)) {
        fprintf(stderr, "ocu cpu backend: threads of a block wait on each "
                        "other at different barriers (warp operation vs "
                        "block barrier); the launch fails\n");
    }
}

static void report_warp_mismatch(void)
{
    static atomic_flag said = ATOMIC_FLAG_INIT;
    if (!atomic_flag_test_and_set(&said)) {
        fprintf(stderr, "ocu cpu backend: lanes of one warp met at different "
                        "warp operations (partial-mask code is not supported); "
                        "the launch fails\n");
    }
}

static void report_loop_wait(void)
{
    static atomic_flag said = ATOMIC_FLAG_INIT;
    if (!atomic_flag_test_and_set(&said)) {
        fprintf(stderr, "ocu cpu backend: a whole-block kernel entry reached a "
                        "synchronisation point; the launch fails\n");
    }
}

static void report_stack_overflow(void)
{
    static atomic_flag said = ATOMIC_FLAG_INIT;
    if (!atomic_flag_test_and_set(&said)) {
        fprintf(stderr, "ocu cpu backend: a kernel overflowed its fiber stack "
                        "(raise OCU_CPU_STACK_KB, now %zu)\n",
                cpu_get_config()->stack_bytes / 1024u);
    }
}

/*
 * The block through its whole-block entry, on fiber 0's context (so on the
 * stack the fibers would use first). Random order: the same shuffle as the
 * fibers' first round, which is the only round of a kernel that never waits.
 */
static ocu_status run_block_loop(block_exec *ex, const cpu_config *cfg)
{
    ex->info.order = NULL;
    if (cfg->order == CPU_ORDER_RANDOM) {
        for (uint32_t i = 0; i < ex->nthreads; ++i) {
            ex->order[i] = i;
        }
        for (uint32_t i = ex->nthreads; i > 1; --i) {
            uint32_t j = (uint32_t)(rng_next(&ex->rng) % i);
            uint32_t t = ex->order[i - 1];
            ex->order[i - 1] = ex->order[j];
            ex->order[j] = t;
        }
        ex->info.order = ex->order;
    }
    ex->info.tids = ex->tids;
    ex->info.current = &ex->current;
    ex->info.dim = ex->block_dim;
    ex->info.nthreads = ex->nthreads;
    for (uint32_t wd = 0; wd < ex->nwords; ++wd) {
        ex->ready_bits[wd] = 0; /* no fiber to chain to */
    }
    ex->loop_run = true;
    ex->loop_declined = false;
    ocu_status st = switch_to_fiber(ex, 0);
    ex->loop_run = false;
    ex->st_rounds++;
    if (st == OCU_OK && ex->fibers[0].state != FIBER_DONE) {
        /* The entry waited (a barrier or warp operation the compiler should
           have refused): the block did not finish. Never resumed. */
        report_loop_wait();
        st = OCU_ERR_LAUNCH_FAILED;
    }
    return st;
}

/*
 * Run one block to completion.
 *
 * A round runs every ready fiber until it waits or returns. Warp
 * operations release their lanes as soon as the last named lane arrives
 * (that lane goes on without a switch); in fixed order a round visits
 * threads 0, 1, 2, ... and also runs lanes released earlier in the same
 * round, in random order it runs the ready ones in a fresh shuffle. When no
 * fiber is ready, the block barrier's waiters are released together.
 */
/*
 * Per-block start. `fibers`: the fiber state of every thread; without it,
 * only what the whole-block entry's single fiber (fiber 0, on the slot the
 * fibers would take first) needs: the per-thread work here would cost
 * about 1 ns per GPU thread, as much as a small kernel body.
 */
static void block_prologue(block_exec *ex, uint64_t linear_block, bool fibers)
{
    const cpu_config *cfg = cpu_get_config();
    ex->rng = (cfg->seed ^ (linear_block * 0x9E3779B97F4A7C15ULL)) | 1u;
    cpu_fpctl_get(&ex->fp0);
    if (ex->dyn_shared != NULL) {
        /* Shared memory starts with garbage on a GPU; make it visible. */
        memset(ex->dyn_shared, 0xA5, ex->dyn_shared_bytes);
    }
    ex->warp_mismatch = false;
    ex->n_barrier = ex->n_warp = ex->n_done = 0;
    ex->lowest_slot = ex->nthreads;
    if (!fibers) {
        fiber *f = &ex->fibers[0];
        f->state = FIBER_READY;
        f->started = false;
        f->nonfull = false;
        f->warp_seq = 0;
        ex->free_slots[0] = ex->nthreads - 1;
        ex->nfree = 1;
        return;
    }
    for (uint32_t w = 0; w < ex->nwarps; ++w) {
        ex->warps[w].parked = ex->warps[w].waiting = 0;
        ex->warps[w].nonfull = 0;
    }
    ex->nfree = 0;
    /* The highest slot is handed out first: an overflow of the stack most
       used runs into a neighbour's memory and is caught by the canary,
       rather than into the guard page below slot 0. */
    for (uint32_t i = 0; i < ex->nthreads; ++i) {
        ex->free_slots[ex->nfree++] = i;
    }
    for (uint32_t i = 0; i < ex->nthreads; ++i) {
        fiber *f = &ex->fibers[i];
        f->state = FIBER_READY;
        f->started = false;
        f->nonfull = false;
        f->warp_seq = 0;
    }
    for (uint32_t wd = 0; wd < ex->nwords; ++wd) {
        uint32_t left = ex->nthreads - wd * 64u;
        ex->ready_bits[wd] = left >= 64 ? ~0ULL : ((1ULL << left) - 1u);
        ex->barrier_bits[wd] = 0;
    }
}

/* Fibers that ran in this block are forgotten (their stacks are free):
   the first `n` (only fiber 0 ran a whole-block entry). */
static void forget_fibers(block_exec *ex, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        fiber *f = &ex->fibers[i];
        if (!f->started) {
            continue;
        }
        f->started = false;
#ifdef OCU_TSAN
        __tsan_destroy_fiber(f->tsan_fiber);
#endif
    }
}

static ocu_status run_block(block_exec *ex, uint64_t linear_block)
{
    const cpu_config *cfg = cpu_get_config();
    bool fibers = ex->block_fn == NULL;
    block_prologue(ex, linear_block, fibers);
    ocu_status st = OCU_OK;
    if (!fibers) {
        st = run_block_loop(ex, cfg);
        if (st == OCU_OK && ex->loop_declined) {
            /* The entry ran no thread (its kernel pointer is not one it
               knows): this block and the rest of the launch on this thread
               run on fibers, from the same start. */
            ex->block_fn = NULL;
            forget_fibers(ex, 1);
            block_prologue(ex, linear_block, true);
            fibers = true;
        }
    }
    while (fibers) {
        bool ran = false;
        if (cfg->order == CPU_ORDER_RANDOM) {
            uint32_t ready = 0;
            for (uint32_t i = next_ready(ex, 0); i < ex->nthreads; i = next_ready(ex, i + 1)) {
                ex->order[ready++] = i;
            }
            for (uint32_t i = ready; i > 1; --i) {
                uint32_t j = (uint32_t)(rng_next(&ex->rng) % i);
                uint32_t t = ex->order[i - 1];
                ex->order[i - 1] = ex->order[j];
                ex->order[j] = t;
            }
            for (uint32_t k = 0; k < ready && st == OCU_OK; ++k) {
                ready_clear(ex, ex->order[k]);
                st = switch_to_fiber(ex, ex->order[k]);
            }
            ran = ready > 0;
        } else {
            /* Increasing index, including fibers released during the pass. */
            /* Fibers hand over to each other (yield_to_scheduler): go on
               after the one that came back. */
            for (uint32_t i = next_ready(ex, 0); i < ex->nthreads && st == OCU_OK;
                 i = next_ready(ex, ex->current + 1)) {
                ready_clear(ex, i);
                st = switch_to_fiber(ex, i);
                ran = true;
            }
        }
        ex->st_rounds++;
        if (st != OCU_OK) {
            break;
        }
        if (ran) {
            continue; /* go on until no fiber is ready */
        }

        /* No fiber is ready: each waits or has returned. Warp operations
           whose lanes could go have been released already. */
        if (ex->n_barrier == 0 && ex->n_warp == 0) {
            break; /* every fiber returned */
        }
        if (ex->n_warp > 0) {
            report_deadlock(); /* lanes split between warp op and barrier */
            st = OCU_ERR_LAUNCH_FAILED;
            break;
        }
        if (ex->n_done > 0 && cfg->strict_barrier) {
            st = OCU_ERR_LAUNCH_FAILED; /* barrier not reached by all */
            break;
        }
        for (uint32_t wd = 0; wd < ex->nwords; ++wd) {
            for (uint64_t m = ex->barrier_bits[wd]; m != 0; m &= m - 1) {
                ex->fibers[(wd << 6) + (uint32_t)__builtin_ctzll(m)].state = FIBER_READY;
            }
            ex->ready_bits[wd] |= ex->barrier_bits[wd];
            ex->barrier_bits[wd] = 0;
        }
        ex->n_barrier = 0;
    }
    bool overflow = false;
    forget_fibers(ex, fibers ? ex->nthreads : 1);
    for (uint32_t s = ex->lowest_slot; s < ex->nthreads; ++s) {
        char *stack = slot_stack(ex, s);
        if (!canary_ok(stack)) {
            overflow = true;
            canary_set(stack); /* for the next launch on these stacks */
        }
    }
    if (overflow) {
        report_stack_overflow();
        return OCU_ERR_LAUNCH_FAILED;
    }
    if (st == OCU_OK && ex->warp_mismatch) {
        report_warp_mismatch();
        return OCU_ERR_LAUNCH_FAILED;
    }
    return st;
}

/* ---- grid --------------------------------------------------------------- */

/*
 * A launch is a grid_job. The calling thread always works on its own job;
 * up to `workers - 1` threads of a persistent pool join it. Pool threads
 * are started once and serve whichever jobs are open, so a launch costs a
 * wake-up instead of thread creation. Each thread keeps its fiber stacks
 * between launches of the same block shape.
 */
typedef struct grid_job {
    ocu_cpu_kernel_fn fn;
    ocu_cpu_block_fn  block_fn;   /* NULL: fibers */
    _Atomic bool      declined;   /* block_fn declined: fibers after all */
    void            **args;
    uint32_t          grid[3];
    uint32_t          block[3];
    size_t            shared_bytes;
    uint64_t          nblocks;
    _Atomic uint64_t  next;
    _Atomic int       status;     /* first error, OCU_OK when none */
    /* guarded by g_pool_lock */
    uint32_t          helpers_wanted;
    uint32_t          attached;   /* pool threads working on it now */
    pthread_cond_t    detached_cv;
    struct grid_job  *next_open;
    /* OCU_CPU_STATS, summed over the threads that worked on it */
    _Atomic uint64_t  st_visits, st_barriers, st_warp_waits, st_rounds;
    _Atomic uint64_t  st_busy_ns;  /* time threads spent running blocks */
    _Atomic uint32_t  st_threads;  /* threads that ran at least one block */
} grid_job;

static pthread_mutex_t g_pool_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_pool_cv = PTHREAD_COND_INITIALIZER;
static grid_job       *g_open_jobs;
static uint32_t        g_pool_threads;  /* started so far */
static uint32_t        g_pool_idle;     /* waiting for work */
static _Atomic uint32_t g_pool_gen;     /* bumped whenever a job opens */

/*
 * Waking a sleeping thread costs tens of microseconds, about as long as a
 * small launch takes to run. Pool threads therefore watch for new work for
 * a short while (OCU_CPU_SPIN_US, default 50) before they sleep, and the
 * launching thread does the same while its helpers finish.
 */
static void cpu_relax(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause");
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}

/* Spin until `*gen` differs from `seen` or the spin time is up. */
static bool spin_for_new_job(uint32_t seen)
{
    double limit = cpu_get_config()->spin_us / 1e3;
    if (limit <= 0) {
        return false;
    }
    double start = cpu_now_ms();
    for (;;) {
        for (int i = 0; i < 64; ++i) {
            if (atomic_load_explicit(&g_pool_gen, memory_order_relaxed) != seen) {
                return true;
            }
            cpu_relax();
        }
        if (cpu_now_ms() - start > limit) {
            return false;
        }
    }
}

static void record_error(grid_job *job, ocu_status st)
{
    int expected = OCU_OK;
    atomic_compare_exchange_strong(&job->status, &expected, (int)st);
}

/*
 * Per-thread fiber stacks, reused while the block shape stays the same.
 * The cache lives on the heap and a thread-exit destructor releases it, so
 * threads that end (queue workers, user threads) do not leak it.
 */
typedef struct exec_cache {
    block_exec ex;
    uint32_t   threads;
    size_t     shared;
} exec_cache;

static pthread_key_t  g_cache_key;
static pthread_once_t g_cache_key_once = PTHREAD_ONCE_INIT;
static bool           g_cache_key_ok; /* false: no caching, a fresh exec per call */

static void cache_destroy(void *p)
{
    exec_cache *c = p;
    block_exec_release(&c->ex);
    free(c);
}

static void cache_key_init(void)
{
    g_cache_key_ok = pthread_key_create(&g_cache_key, cache_destroy) == 0;
}

/*
 * The exec to run blocks on. `*owned` is set when it is not cached and the
 * caller must cache_destroy it (only when no thread key could be made).
 */
static ocu_status cached_exec(uint32_t nthreads, size_t shared, block_exec **out,
                              exec_cache **owned)
{
    pthread_once(&g_cache_key_once, cache_key_init);
    *owned = NULL;
    if (!g_cache_key_ok) {
        exec_cache *c = calloc(1, sizeof(*c));
        if (c == NULL) {
            return OCU_ERR_OUT_OF_MEMORY;
        }
        ocu_status st = block_exec_init(&c->ex, nthreads, shared);
        if (st != OCU_OK) {
            cache_destroy(c);
            return st;
        }
        *owned = c;
        *out = &c->ex;
        return OCU_OK;
    }
    exec_cache *c = pthread_getspecific(g_cache_key);
    if (c != NULL && (c->threads != nthreads || c->shared != shared)) {
        pthread_setspecific(g_cache_key, NULL);
        cache_destroy(c);
        c = NULL;
    }
    if (c == NULL) {
        c = calloc(1, sizeof(*c));
        if (c == NULL) {
            return OCU_ERR_OUT_OF_MEMORY;
        }
        ocu_status st = block_exec_init(&c->ex, nthreads, shared);
        if (st != OCU_OK) {
            cache_destroy(c);
            return st;
        }
        c->threads = nthreads;
        c->shared = shared;
        if (pthread_setspecific(g_cache_key, c) != 0) {
            cache_destroy(c);
            return OCU_ERR_OUT_OF_MEMORY;
        }
    }
    *out = &c->ex;
    return OCU_OK;
}

/*
 * Run blocks of `job` until none is left or one fails. Returns false when
 * this thread could not even start (no memory for its stacks).
 */
static bool work_on(grid_job *job, bool primary)
{
    uint32_t nthreads = job->block[0] * job->block[1] * job->block[2];
    block_exec *ex = NULL;
    exec_cache *owned = NULL;
    ocu_status st = cached_exec(nthreads, job->shared_bytes, &ex, &owned);
    if (st != OCU_OK) {
        /* A helper that cannot start leaves its blocks to the others; only
           the calling thread's failure fails the launch. */
        if (primary) {
            record_error(job, st);
        }
        return false;
    }
    ex->fn = job->fn;
    ex->block_fn = job->block_fn;
    ex->args = job->args;
    ex->dyn_shared_bytes = job->shared_bytes;
    ex->block_dim = (ocu_cpu_dim3){job->block[0], job->block[1], job->block[2]};
    if (ex->tid_dim.x != ex->block_dim.x || ex->tid_dim.y != ex->block_dim.y ||
        ex->tid_dim.z != ex->block_dim.z) {
        set_thread_indices(ex, ex->block_dim);
    }
    ex->grid_dim = (ocu_cpu_dim3){job->grid[0], job->grid[1], job->grid[2]};
    tl_exec = ex;
    ocu_cpu_tls_block_dim = ex->block_dim;
    ocu_cpu_tls_grid_dim = ex->grid_dim;
    ocu_cpu_tls_warp_size = (int)ex->warp_size;
    ex->st_visits = ex->st_barriers = ex->st_warp_waits = ex->st_rounds = 0;
    bool stats = cpu_get_config()->stats;
    double t0 = stats ? cpu_now_ms() : 0.0;
    uint64_t ran = 0;
    for (;;) {
        if (atomic_load(&job->status) != OCU_OK) {
            break;
        }
        uint64_t b = atomic_fetch_add(&job->next, 1);
        if (b >= job->nblocks) {
            break;
        }
        uint64_t gx = job->grid[0], gy = job->grid[1];
        ex->block_idx.x = (uint32_t)(b % gx);
        ex->block_idx.y = (uint32_t)((b / gx) % gy);
        ex->block_idx.z = (uint32_t)(b / (gx * gy));
        ocu_cpu_tls_block_idx = ex->block_idx;
        ran++;
        st = run_block(ex, b);
        if (st != OCU_OK) {
            record_error(job, st);
            break;
        }
    }
    if (job->block_fn != NULL && ex->block_fn == NULL) {
        atomic_store(&job->declined, true);
    }
    if (stats && ran > 0) {
        atomic_fetch_add(&job->st_visits, ex->st_visits);
        atomic_fetch_add(&job->st_barriers, ex->st_barriers);
        atomic_fetch_add(&job->st_warp_waits, ex->st_warp_waits);
        atomic_fetch_add(&job->st_rounds, ex->st_rounds);
        atomic_fetch_add(&job->st_busy_ns, (uint64_t)((cpu_now_ms() - t0) * 1e6));
        atomic_fetch_add(&job->st_threads, 1u);
    }
    tl_exec = NULL;
    const ocu_cpu_dim3 zero = {0, 0, 0};
    ocu_cpu_tls_thread_idx = ocu_cpu_tls_block_idx = zero;
    ocu_cpu_tls_block_dim = ocu_cpu_tls_grid_dim = zero;
    ocu_cpu_tls_warp_size = 0;
    if (owned != NULL) {
        cache_destroy(owned);
    }
    return true;
}

/* Caller holds g_pool_lock. An open job that still wants a helper. */
static grid_job *job_wanting_help(void)
{
    for (grid_job *j = g_open_jobs; j != NULL; j = j->next_open) {
        if (j->attached < j->helpers_wanted && atomic_load(&j->next) < j->nblocks &&
            atomic_load(&j->status) == OCU_OK) {
            return j;
        }
    }
    return NULL;
}

static void *pool_thread(void *unused)
{
    (void)unused;
    pthread_mutex_lock(&g_pool_lock);
    for (;;) {
        grid_job *job;
        while ((job = job_wanting_help()) == NULL) {
            uint32_t seen = atomic_load(&g_pool_gen);
            pthread_mutex_unlock(&g_pool_lock);
            bool more = spin_for_new_job(seen);
            pthread_mutex_lock(&g_pool_lock);
            if (more) {
                continue; /* look again, under the lock */
            }
            g_pool_idle++;
            pthread_cond_wait(&g_pool_cv, &g_pool_lock);
            g_pool_idle--;
        }
        job->attached++;
        pthread_mutex_unlock(&g_pool_lock);
        bool started = work_on(job, false);
        pthread_mutex_lock(&g_pool_lock);
        if (!started) {
            /* No more helpers for this job: retrying would only spin. */
            job->helpers_wanted = 0;
        }
        if (--job->attached == 0) {
            pthread_cond_broadcast(&job->detached_cv);
        }
    }
    return NULL; /* not reached: pool threads live as long as the process */
}

/*
 * fork(): the child has none of the pool threads, and the lock may have
 * been held by a thread that does not exist there. Take the lock around
 * fork and give the child an empty pool, which it grows again on its next
 * launch. (Queues do not survive fork either: their worker threads are
 * gone, as with GPU runtimes.)
 */
static void pool_prepare(void) { pthread_mutex_lock(&g_pool_lock); }
static void pool_parent(void) { pthread_mutex_unlock(&g_pool_lock); }
static void pool_child(void)
{
    /* prepare() locked it on this very thread, the only one in the child. */
    pthread_mutex_unlock(&g_pool_lock);
    pthread_cond_init(&g_pool_cv, NULL); /* its waiters do not exist here */
    g_open_jobs = NULL;
    g_pool_threads = 0;
    g_pool_idle = 0;
}

static pthread_once_t g_atfork_once = PTHREAD_ONCE_INIT;
static void install_atfork(void)
{
#if CPU_OS_HAS_FORK
    (void)pthread_atfork(pool_prepare, pool_parent, pool_child);
#else
    (void)pool_prepare;
    (void)pool_parent;
    (void)pool_child;
#endif
}

/* Caller holds g_pool_lock. Start pool threads up to `want` in total. */
static void grow_pool(uint32_t want)
{
    while (g_pool_threads < want) {
        pthread_t t;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        int rc = pthread_create(&t, &attr, pool_thread, NULL);
        pthread_attr_destroy(&attr);
        if (rc != 0) {
            break; /* run with the threads we have */
        }
        g_pool_threads++;
    }
}

/* ---- OCU_CPU_STATS ------------------------------------------------------
 * Per kernel entry point: launches, blocks, fiber switches, waits and
 * time. Printed to stderr at exit, kernels as addresses relative to
 * cpu_run_grid (resolve with nm / addr2line on the executable).
 */
typedef struct kstat {
    ocu_cpu_kernel_fn fn;
    uint32_t threads_per_block;
    uint64_t launches, blocks, visits, barriers, warp_waits, rounds, threads;
    uint64_t loop_launches; /* launches run through the whole-block entry */
    double   wall_ms, busy_ms;
} kstat;

#define KSTAT_MAX 512
static pthread_mutex_t g_stats_lock = PTHREAD_MUTEX_INITIALIZER;
static kstat           g_kstats[KSTAT_MAX];
static uint32_t        g_nkstats;
static uint64_t        g_copies, g_copy_bytes;
static double          g_copy_ms;
static pthread_once_t  g_stats_once = PTHREAD_ONCE_INIT;

static int kstat_by_wall(const void *a, const void *b)
{
    double x = ((const kstat *)a)->wall_ms, y = ((const kstat *)b)->wall_ms;
    return x < y ? 1 : x > y ? -1 : 0;
}

static void stats_print(void)
{
    pthread_mutex_lock(&g_stats_lock);
    qsort(g_kstats, g_nkstats, sizeof(kstat), kstat_by_wall);
    kstat t;
    memset(&t, 0, sizeof(t));
    for (uint32_t i = 0; i < g_nkstats; ++i) {
        t.launches += g_kstats[i].launches;
        t.loop_launches += g_kstats[i].loop_launches;
        t.blocks += g_kstats[i].blocks;
        t.visits += g_kstats[i].visits;
        t.barriers += g_kstats[i].barriers;
        t.warp_waits += g_kstats[i].warp_waits;
        t.rounds += g_kstats[i].rounds;
        t.wall_ms += g_kstats[i].wall_ms;
        t.busy_ms += g_kstats[i].busy_ms;
    }
    fprintf(stderr, "ocu cpu stats: %llu launches (%llu as loops), %.1f ms in kernels (busy %.1f ms "
                    "over threads), %llu blocks, %llu fiber switches, %llu barrier "
                    "waits, %llu warp waits, %llu rounds; %llu copies %.1f MB %.1f ms\n",
            (unsigned long long)t.launches, (unsigned long long)t.loop_launches, t.wall_ms, t.busy_ms,
            (unsigned long long)t.blocks, (unsigned long long)t.visits,
            (unsigned long long)t.barriers, (unsigned long long)t.warp_waits,
            (unsigned long long)t.rounds, (unsigned long long)g_copies,
            (double)g_copy_bytes / 1e6, g_copy_ms);
    fprintf(stderr, "ocu cpu stats: kernel(+off) tpb launches blocks wall_ms busy_ms "
                    "thr/launch switches barriers warp_waits rounds form\n");
    for (uint32_t i = 0; i < g_nkstats && i < 40; ++i) {
        const kstat *k = &g_kstats[i];
        const char *form = k->loop_launches == 0              ? "fiber"
                           : k->loop_launches == k->launches ? "loop"
                                                             : "mixed";
        fprintf(stderr, "ocu cpu stats: %+lld %u %llu %llu %.1f %.1f %.1f %llu %llu %llu %llu %s\n",
                (long long)((intptr_t)k->fn - (intptr_t)cpu_run_grid),
                k->threads_per_block, (unsigned long long)k->launches,
                (unsigned long long)k->blocks, k->wall_ms, k->busy_ms,
                (double)k->threads / (double)k->launches,
                (unsigned long long)k->visits, (unsigned long long)k->barriers,
                (unsigned long long)k->warp_waits, (unsigned long long)k->rounds, form);
    }
    pthread_mutex_unlock(&g_stats_lock);
}

static void stats_install(void) { atexit(stats_print); }

static kstat *stats_slot(ocu_cpu_kernel_fn fn)
{
    for (uint32_t i = 0; i < g_nkstats; ++i) {
        if (g_kstats[i].fn == fn) return &g_kstats[i];
    }
    if (g_nkstats == KSTAT_MAX) return &g_kstats[KSTAT_MAX - 1]; /* lumped */
    g_kstats[g_nkstats].fn = fn;
    return &g_kstats[g_nkstats++];
}

static void stats_record(grid_job *job, double wall_ms)
{
    pthread_once(&g_stats_once, stats_install);
    pthread_mutex_lock(&g_stats_lock);
    kstat *k = stats_slot(job->fn);
    k->threads_per_block = job->block[0] * job->block[1] * job->block[2];
    k->launches++;
    k->loop_launches += job->block_fn != NULL && !atomic_load(&job->declined);
    k->blocks += job->nblocks;
    k->visits += atomic_load(&job->st_visits);
    k->barriers += atomic_load(&job->st_barriers);
    k->warp_waits += atomic_load(&job->st_warp_waits);
    k->rounds += atomic_load(&job->st_rounds);
    k->threads += atomic_load(&job->st_threads);
    k->busy_ms += (double)atomic_load(&job->st_busy_ns) / 1e6;
    k->wall_ms += wall_ms;
    pthread_mutex_unlock(&g_stats_lock);
}

void cpu_stats_copy(double ms, size_t bytes)
{
    pthread_once(&g_stats_once, stats_install);
    pthread_mutex_lock(&g_stats_lock);
    g_copies++;
    g_copy_bytes += bytes;
    g_copy_ms += ms;
    pthread_mutex_unlock(&g_stats_lock);
}

/*
 * OCU_CPU_TRACE=<file>: one line per launch, appended when it ends:
 * sequence number, grid, block, status and a hash of all device and pinned
 * memory. Two runs of a program whose launches happen in the same order
 * (one OS thread, one stream) can be compared line by line, for example
 * OCU_CPU_FORM=fiber against =loop (tests/cpu_forms). Slow: it reads all
 * device memory after every launch.
 */
static void trace_launch(grid_job *job)
{
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    static uint64_t seq;
    uint64_t h = cpu_mem_digest();
    pthread_mutex_lock(&lock);
    FILE *f = fopen(cpu_get_config()->trace, "a");
    if (f != NULL) {
        fprintf(f, "%llu grid %u,%u,%u block %u,%u,%u status %d form %s memory %016llx\n",
                (unsigned long long)++seq, job->grid[0], job->grid[1], job->grid[2],
                job->block[0], job->block[1], job->block[2], atomic_load(&job->status),
                job->block_fn != NULL && !atomic_load(&job->declined) ? "loop" : "fiber",
                (unsigned long long)h);
        fclose(f);
    }
    pthread_mutex_unlock(&lock);
}

ocu_status cpu_run_grid(ocu_cpu_kernel_fn fn, ocu_cpu_block_fn block_fn,
                        const uint32_t grid[3], const uint32_t block[3],
                        size_t shared_bytes, void **args)
{
    if (tl_exec != NULL) {
        return OCU_ERR_NOT_SUPPORTED; /* launch from inside a kernel */
    }
    grid_job job;
    memset(&job, 0, sizeof(job));
    job.fn = fn;
    job.block_fn = cpu_get_config()->loops ? block_fn : NULL;
    job.args = args;
    memcpy(job.grid, grid, sizeof(job.grid));
    memcpy(job.block, block, sizeof(job.block));
    job.shared_bytes = shared_bytes;
    job.nblocks = (uint64_t)grid[0] * grid[1] * grid[2];
    atomic_init(&job.next, 0);
    atomic_init(&job.status, OCU_OK);
    atomic_init(&job.declined, false);

    bool stats = cpu_get_config()->stats;
    double t_start = stats ? cpu_now_ms() : 0.0;
    uint64_t workers = cpu_get_config()->workers;
    uint64_t helpers = (workers < job.nblocks ? workers : job.nblocks);
    helpers = helpers > 0 ? helpers - 1 : 0;
    job.helpers_wanted = (uint32_t)helpers;

    if (helpers > 0) {
        pthread_once(&g_atfork_once, install_atfork);
        pthread_cond_init(&job.detached_cv, NULL);
        pthread_mutex_lock(&g_pool_lock);
        grow_pool(cpu_get_config()->workers - 1);
        job.next_open = g_open_jobs;
        g_open_jobs = &job;
        atomic_fetch_add(&g_pool_gen, 1u);
        pthread_cond_broadcast(&g_pool_cv);
        pthread_mutex_unlock(&g_pool_lock);
    }

    (void)work_on(&job, true);

    if (helpers > 0) {
        pthread_mutex_lock(&g_pool_lock);
        for (grid_job **pp = &g_open_jobs; *pp != NULL; pp = &(*pp)->next_open) {
            if (*pp == &job) {
                *pp = job.next_open; /* no new helpers can attach */
                break;
            }
        }
        /* Helpers usually finish within microseconds of us: poll a little
           before sleeping. */
        double limit = cpu_get_config()->spin_us / 1e3;
        double start = cpu_now_ms();
        while (job.attached > 0 && limit > 0 && cpu_now_ms() - start <= limit) {
            pthread_mutex_unlock(&g_pool_lock);
            for (int i = 0; i < 32; ++i) {
                cpu_relax();
            }
            pthread_mutex_lock(&g_pool_lock);
        }
        while (job.attached > 0) {
            pthread_cond_wait(&job.detached_cv, &g_pool_lock);
        }
        pthread_mutex_unlock(&g_pool_lock);
        pthread_cond_destroy(&job.detached_cv);
    }
    if (stats) {
        stats_record(&job, cpu_now_ms() - t_start);
    }
    if (cpu_get_config()->trace != NULL) {
        trace_launch(&job);
    }
    return (ocu_status)atomic_load(&job.status);
}

bool cpu_in_kernel(void)
{
    return tl_exec != NULL;
}

/* ---- device-side API ---------------------------------------------------- */

static ocu_cpu_dim3 linear_to_dim3(uint32_t linear, ocu_cpu_dim3 dim)
{
    ocu_cpu_dim3 r;
    r.x = linear % dim.x;
    r.y = (linear / dim.x) % dim.y;
    r.z = linear / (dim.x * dim.y);
    return r;
}

ocu_cpu_dim3 ocu_cpu_thread_idx(void)
{
    return tl_exec->fibers[tl_exec->current].tid;
}

ocu_cpu_dim3 ocu_cpu_block_idx(void) { return tl_exec->block_idx; }
ocu_cpu_dim3 ocu_cpu_block_dim(void) { return tl_exec->block_dim; }
ocu_cpu_dim3 ocu_cpu_grid_dim(void) { return tl_exec->grid_dim; }
uint32_t ocu_cpu_warp_size(void)
{
    const block_exec *ex = tl_exec;
    return ex != NULL ? ex->warp_size : cpu_get_config()->warp_size;
}

void ocu_cpu_barrier(void)
{
    block_exec *ex = tl_exec;
    uint32_t me = ex->current;
    ex->fibers[me].state = FIBER_WAITING;
    ex->barrier_bits[me >> 6] |= 1ULL << (me & 63u);
    ex->n_barrier++;
    ex->st_barriers++;
    yield_to_scheduler(ex, false);
}

void *ocu_cpu_dynamic_shared(void)
{
    return tl_exec->dyn_shared;
}

uint32_t ocu_cpu_lane_id(void)
{
    /* Outside a kernel: lane 0, as for thread 0 (warp_size gives the
       setting there). */
    const block_exec *ex = tl_exec;
    return ex != NULL ? ex->current & (ex->warp_size - 1) : 0;
}

/*
 * Every lane of a warp operation records (tag, sequence number, mask) and
 * its value, and waits until the lanes it names are parked; the last of
 * them checks that they all met at the same operation (check_release),
 * releases them and goes on without a switch.
 */
#define OCU_SYNC_TAG 0x53594e43ULL

static const uint64_t *warp_meet(block_exec *ex, uint64_t tag, uint32_t mask,
                                 bool masked, uint64_t value, uint32_t *count_out)
{
    uint32_t me = ex->current;
    uint32_t w = ex->warp_size;
    uint32_t wi = me >> ex->warp_shift;
    uint32_t base = wi * w;
    uint32_t lane = me - base;
    uint32_t count = ex->nthreads - base < w ? ex->nthreads - base : w;
    warp_state *ws = &ex->warps[wi];
    /* The lanes this call waits for, relative to the warp: the mask's 32
       lanes in the caller's own 32-lane group, or every lane (mask 0).
       The caller itself is always included. */
    uint64_t named;
    if (mask == 0) {
        named = ws->lanes;
    } else {
        uint32_t group = lane & ~31u;
        named = ((uint64_t)mask << group) & ws->lanes;
    }
    named |= 1ULL << lane;
    fiber *f = &ex->fibers[me];
    uint64_t seq = ++f->warp_seq;
    uint64_t *vals = &ex->warp_vals[(seq & 1) * ex->nthreads];
    vals[me] = value;
    warp_rec *mine = &ex->warp_recs[me];
    mine->tag = tag;
    mine->seq = seq;
    mine->named = named;
    mine->masked = masked;

    /* Arrive; wait unless this arrival releases us. */
    f->state = FIBER_WAITING_WARP;
    ex->n_warp++;
    ex->st_warp_waits++;
    ws->parked |= 1ULL << lane;
    ws->waiting |= 1ULL << lane;
    if (named != ws->lanes) {
        f->nonfull = true;
        ws->nonfull++;
    }
    try_release_warp(ex, wi);
    if (f->state == FIBER_WAITING_WARP) {
        yield_to_scheduler(ex, false);
    }
    *count_out = count;
    return &vals[base]; /* stable until this lane's next warp operation */
}

static uint32_t warp_gather(uint64_t tag, uint32_t mask, bool masked, uint64_t value,
                            uint64_t *all)
{
    uint32_t count;
    const uint64_t *vals = warp_meet(tl_exec, tag, mask, masked, value, &count);
    if (all != NULL) {
        memcpy(all, vals, count * sizeof(*all));
    }
    return count;
}

void ocu_cpu_warp_sync(void)
{
    (void)warp_gather(OCU_SYNC_TAG, 0, false, 0, NULL);
}

void ocu_cpu_warp_sync_masked(uint32_t mask)
{
    (void)warp_gather(OCU_SYNC_TAG ^ mask, mask, true, 0, NULL);
}

uint32_t ocu_cpu_warp_gather(uint64_t value, uint64_t *all)
{
    return warp_gather(0, 0, false, value, all);
}

uint32_t ocu_cpu_warp_gather_tagged(uint64_t value, uint64_t tag, uint64_t *all)
{
    return warp_gather(tag, 0, false, value, all);
}

uint32_t ocu_cpu_warp_gather_masked(uint64_t value, uint64_t tag, uint32_t mask,
                                    uint64_t *all)
{
    return warp_gather(tag, mask, true, value, all);
}

uint64_t ocu_cpu_warp_shuffle_masked(uint64_t value, uint64_t tag, uint32_t mask,
                                     uint32_t src)
{
    block_exec *ex = tl_exec;
    uint32_t count;
    const uint64_t *vals = warp_meet(ex, tag, mask, true, value, &count);
    if (src >= count) {
        src = ex->current & (ex->warp_size - 1); /* our own value */
    }
    return vals[src];
}

uint64_t ocu_cpu_warp_vote_masked(int pred, uint64_t tag, uint32_t mask)
{
    uint32_t count;
    const uint64_t *vals = warp_meet(tl_exec, tag, mask, true, pred != 0, &count);
    uint64_t bits = 0;
    for (uint32_t i = 0; i < count; ++i) {
        bits |= (uint64_t)(vals[i] != 0) << i;
    }
    return bits;
}

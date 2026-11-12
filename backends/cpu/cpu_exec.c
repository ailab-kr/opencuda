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
#include <sys/mman.h>
#include <unistd.h>

#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#if !defined(MAP_NORESERVE)
#define MAP_NORESERVE 0
#endif

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

    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
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
    FIBER_WAITING,  /* at the barrier */
    FIBER_DONE
} fiber_state;

typedef struct fiber {
    cpu_ctx     ctx;
    fiber_state state;
    char       *stack;      /* usable stack, above the guard page */
#ifdef OCU_ASAN
    void       *fake_stack;
#endif
#ifdef OCU_TSAN
    void       *tsan_fiber;
#endif
} fiber;

typedef struct block_exec {
    cpu_ctx           sched_ctx;
    fiber            *fibers;
    uint32_t         *order;
    uint32_t          nthreads;
    uint32_t          current;
    ocu_cpu_dim3      block_idx;
    ocu_cpu_dim3      block_dim;
    ocu_cpu_dim3      grid_dim;
    ocu_cpu_kernel_fn fn;
    void            **args;
    void             *dyn_shared;
    size_t            dyn_shared_bytes;
    char             *stack_region;  /* one mmap for all fiber stacks */
    size_t            stack_region_bytes;
    size_t            stack_bytes;   /* usable bytes per fiber */
    uint64_t          rng;
#ifdef OCU_ASAN
    void             *sched_fake_stack;
    const void       *sched_bottom;
    size_t            sched_size;
#endif
#ifdef OCU_TSAN
    void             *sched_tsan_fiber;
#endif
} block_exec;

static _Thread_local block_exec *tl_exec;

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

/* Switch from the scheduler into fiber `i`; returns when it yields. */
static void switch_to_fiber(block_exec *ex, uint32_t i)
{
    fiber *f = &ex->fibers[i];
    ex->current = i;
#ifdef OCU_ASAN
    __sanitizer_start_switch_fiber(&ex->sched_fake_stack, f->stack,
                                   ex->stack_bytes);
#endif
#ifdef OCU_TSAN
    __tsan_switch_to_fiber(f->tsan_fiber, 0);
#endif
    cpu_ctx_switch(&ex->sched_ctx, &f->ctx);
#ifdef OCU_ASAN
    __sanitizer_finish_switch_fiber(ex->sched_fake_stack, NULL, NULL);
#endif
}

/* Switch from the running fiber back to the scheduler. */
static void yield_to_scheduler(block_exec *ex, bool exiting)
{
    fiber *f = &ex->fibers[ex->current];
#ifdef OCU_ASAN
    __sanitizer_start_switch_fiber(exiting ? NULL : &f->fake_stack,
                                   ex->sched_bottom, ex->sched_size);
#else
    (void)exiting;
#endif
#ifdef OCU_TSAN
    __tsan_switch_to_fiber(ex->sched_tsan_fiber, 0);
#endif
    cpu_ctx_switch(&f->ctx, &ex->sched_ctx);
#ifdef OCU_ASAN
    __sanitizer_finish_switch_fiber(f->fake_stack, NULL, NULL);
#endif
}

static void fiber_entry(void)
{
    block_exec *ex = tl_exec;
#ifdef OCU_ASAN
    __sanitizer_finish_switch_fiber(NULL, &ex->sched_bottom, &ex->sched_size);
#endif
    ex->fn(ex->args);
    ex->fibers[ex->current].state = FIBER_DONE;
    yield_to_scheduler(ex, true);
    abort(); /* a finished fiber is never resumed */
}

static ocu_status block_exec_init(block_exec *ex, uint32_t nthreads,
                                  size_t shared_bytes)
{
    const cpu_config *cfg = cpu_get_config();
    memset(ex, 0, sizeof(*ex));
    ex->nthreads = nthreads;
    long page = sysconf(_SC_PAGESIZE);
    size_t pg = page > 0 ? (size_t)page : 4096u;
    ex->stack_bytes = (cfg->stack_bytes + pg - 1) & ~(pg - 1);
    ex->stack_region_bytes = pg + ex->stack_bytes * nthreads;
    void *region = mmap(NULL, ex->stack_region_bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (region == MAP_FAILED) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    ex->stack_region = region;
    if (mprotect(region, pg, PROT_NONE) != 0) { /* guard below fiber 0 */
        return OCU_ERR_OUT_OF_MEMORY;
    }
    ex->fibers = calloc(nthreads, sizeof(*ex->fibers));
    ex->order = calloc(nthreads, sizeof(*ex->order));
    if (ex->fibers == NULL || ex->order == NULL) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < nthreads; ++i) {
        ex->fibers[i].stack = ex->stack_region + pg + (size_t)i * ex->stack_bytes;
    }
    if (shared_bytes > 0) {
        size_t a = CPU_SHARED_ALIGN;
        ex->dyn_shared = aligned_alloc(a, (shared_bytes + a - 1) & ~(a - 1));
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
    if (ex->stack_region != NULL) {
        munmap(ex->stack_region, ex->stack_region_bytes);
    }
    free(ex->fibers);
    free(ex->order);
    free(ex->dyn_shared);
    memset(ex, 0, sizeof(*ex));
}

/* Canary at the low end of each fiber stack: an overflow hits it first. */
#define CANARY_WORDS 32
#define CANARY_VALUE 0x6f63754361727921ULL

static void canary_set(const fiber *f)
{
    uint64_t *w = (uint64_t *)(void *)f->stack;
    for (int i = 0; i < CANARY_WORDS; ++i) {
        w[i] = CANARY_VALUE;
    }
}

static bool canary_ok(const fiber *f)
{
    const uint64_t *w = (const uint64_t *)(const void *)f->stack;
    for (int i = 0; i < CANARY_WORDS; ++i) {
        if (w[i] != CANARY_VALUE) {
            return false;
        }
    }
    return true;
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

/* Run one block to completion. */
static ocu_status run_block(block_exec *ex, uint64_t linear_block)
{
    const cpu_config *cfg = cpu_get_config();
    ex->rng = (cfg->seed ^ (linear_block * 0x9E3779B97F4A7C15ULL)) | 1u;
    if (ex->dyn_shared != NULL) {
        /* Shared memory starts with garbage on a GPU; make it visible. */
        memset(ex->dyn_shared, 0xA5, ex->dyn_shared_bytes);
    }
    for (uint32_t i = 0; i < ex->nthreads; ++i) {
        fiber *f = &ex->fibers[i];
        f->state = FIBER_READY;
        canary_set(f);
        if (cpu_ctx_make(&f->ctx, f->stack, ex->stack_bytes, fiber_entry) != 0) {
            return OCU_ERR_LAUNCH_FAILED;
        }
#ifdef OCU_TSAN
        f->tsan_fiber = __tsan_create_fiber(0);
#endif
    }

    ocu_status st = OCU_OK;
    for (;;) {
        uint32_t ready = 0;
        for (uint32_t i = 0; i < ex->nthreads; ++i) {
            if (ex->fibers[i].state == FIBER_READY) {
                ex->order[ready++] = i;
            }
        }
        if (cfg->order == CPU_ORDER_RANDOM) {
            for (uint32_t i = ready; i > 1; --i) {
                uint32_t j = (uint32_t)(rng_next(&ex->rng) % i);
                uint32_t t = ex->order[i - 1];
                ex->order[i - 1] = ex->order[j];
                ex->order[j] = t;
            }
        }
        for (uint32_t k = 0; k < ready; ++k) {
            switch_to_fiber(ex, ex->order[k]);
        }

        uint32_t waiting = 0, done = 0;
        for (uint32_t i = 0; i < ex->nthreads; ++i) {
            waiting += ex->fibers[i].state == FIBER_WAITING;
            done += ex->fibers[i].state == FIBER_DONE;
        }
        if (waiting == 0) {
            break; /* every fiber returned */
        }
        if (done > 0 && cfg->strict_barrier) {
            st = OCU_ERR_LAUNCH_FAILED; /* barrier not reached by all */
            break;
        }
        for (uint32_t i = 0; i < ex->nthreads; ++i) {
            if (ex->fibers[i].state == FIBER_WAITING) {
                ex->fibers[i].state = FIBER_READY;
            }
        }
    }
#ifdef OCU_TSAN
    for (uint32_t i = 0; i < ex->nthreads; ++i) {
        __tsan_destroy_fiber(ex->fibers[i].tsan_fiber);
    }
#endif
    for (uint32_t i = 0; i < ex->nthreads; ++i) {
        if (!canary_ok(&ex->fibers[i])) {
            report_stack_overflow();
            return OCU_ERR_LAUNCH_FAILED;
        }
    }
    return st;
}

/* ---- grid --------------------------------------------------------------- */

typedef struct grid_job {
    ocu_cpu_kernel_fn fn;
    void            **args;
    uint32_t          grid[3];
    uint32_t          block[3];
    size_t            shared_bytes;
    uint64_t          nblocks;
    _Atomic uint64_t  next;
    _Atomic int       status;  /* first error, OCU_OK when none */
} grid_job;

static void record_error(grid_job *job, ocu_status st)
{
    int expected = OCU_OK;
    atomic_compare_exchange_strong(&job->status, &expected, (int)st);
}

typedef struct worker_arg {
    grid_job *job;
    bool      primary; /* the calling thread: its failure fails the launch */
} worker_arg;

static void *grid_worker(void *arg)
{
    worker_arg *wa = arg;
    grid_job *job = wa->job;
    uint32_t nthreads = job->block[0] * job->block[1] * job->block[2];
    block_exec ex;
    ocu_status st = block_exec_init(&ex, nthreads, job->shared_bytes);
    if (st != OCU_OK) {
        /* An extra worker that cannot start leaves its blocks to the
           others; only the calling thread's failure fails the launch. */
        if (wa->primary) {
            record_error(job, st);
        }
        block_exec_release(&ex);
        return NULL;
    }
    ex.fn = job->fn;
    ex.args = job->args;
    ex.dyn_shared_bytes = job->shared_bytes;
    ex.block_dim = (ocu_cpu_dim3){job->block[0], job->block[1], job->block[2]};
    ex.grid_dim = (ocu_cpu_dim3){job->grid[0], job->grid[1], job->grid[2]};
    block_exec *saved = tl_exec;
    tl_exec = &ex;
    for (;;) {
        if (atomic_load(&job->status) != OCU_OK) {
            break;
        }
        uint64_t b = atomic_fetch_add(&job->next, 1);
        if (b >= job->nblocks) {
            break;
        }
        uint64_t gx = job->grid[0], gy = job->grid[1];
        ex.block_idx.x = (uint32_t)(b % gx);
        ex.block_idx.y = (uint32_t)((b / gx) % gy);
        ex.block_idx.z = (uint32_t)(b / (gx * gy));
        st = run_block(&ex, b);
        if (st != OCU_OK) {
            record_error(job, st);
            break;
        }
    }
    tl_exec = saved;
    block_exec_release(&ex);
    return NULL;
}

ocu_status cpu_run_grid(ocu_cpu_kernel_fn fn, const uint32_t grid[3],
                        const uint32_t block[3], size_t shared_bytes,
                        void **args)
{
    if (tl_exec != NULL) {
        return OCU_ERR_NOT_SUPPORTED; /* launch from inside a kernel */
    }
    grid_job job;
    memset(&job, 0, sizeof(job));
    job.fn = fn;
    job.args = args;
    memcpy(job.grid, grid, sizeof(job.grid));
    memcpy(job.block, block, sizeof(job.block));
    job.shared_bytes = shared_bytes;
    job.nblocks = (uint64_t)grid[0] * grid[1] * grid[2];
    atomic_init(&job.next, 0);
    atomic_init(&job.status, OCU_OK);

    uint64_t want = cpu_get_config()->workers;
    if (want > job.nblocks) {
        want = job.nblocks;
    }
    uint32_t extra = (uint32_t)(want > 0 ? want - 1 : 0);
    pthread_t *threads = NULL;
    uint32_t started = 0;
    worker_arg helper = {&job, false};
    worker_arg primary = {&job, true};
    if (extra > 0) {
        threads = calloc(extra, sizeof(*threads));
        if (threads != NULL) {
            for (; started < extra; ++started) {
                if (pthread_create(&threads[started], NULL, grid_worker, &helper) != 0) {
                    break; /* run with the threads we got */
                }
            }
        }
    }
    grid_worker(&primary);
    for (uint32_t i = 0; i < started; ++i) {
        pthread_join(threads[i], NULL);
    }
    free(threads);
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
    return linear_to_dim3(tl_exec->current, tl_exec->block_dim);
}

ocu_cpu_dim3 ocu_cpu_block_idx(void) { return tl_exec->block_idx; }
ocu_cpu_dim3 ocu_cpu_block_dim(void) { return tl_exec->block_dim; }
ocu_cpu_dim3 ocu_cpu_grid_dim(void) { return tl_exec->grid_dim; }
uint32_t ocu_cpu_warp_size(void) { return cpu_get_config()->warp_size; }

void ocu_cpu_barrier(void)
{
    block_exec *ex = tl_exec;
    ex->fibers[ex->current].state = FIBER_WAITING;
    yield_to_scheduler(ex, false);
}

void *ocu_cpu_dynamic_shared(void)
{
    return tl_exec->dyn_shared;
}

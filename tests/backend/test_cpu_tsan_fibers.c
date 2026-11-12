/*
 * Regression test for the CPU backend defect "the thread sanitizer's shadow
 * stack grows by one frame per finished fiber" (commit 8cd73f6: cpu_ctx_finish
 * must switch at the same call depth as cpu_ctx_switch).
 *
 * Symptom: the long-lived worker (and the calling thread) kept one stale
 * shadow frame per finished fiber, so each block cost more than the one
 * before it, and a few thousand blocks later the sanitizer slowed to a
 * crawl or aborted ("nested bug"). In a build without the thread
 * sanitizer the same test is a plain "many small blocks" check and stays
 * fast; the cost-per-block comparison is meaningful in the TSan build of
 * tools/check.sh.
 *
 * Launches TOTAL_BLOCKS blocks of a few threads each in one process, in
 * CHUNKS launches, and compares the time per block of the first and the
 * last quarter of the launches. A leak makes the last quarter many times
 * slower; the allowed factor is generous.
 */
#define _POSIX_C_SOURCE 200809L

#include "ocu_backend.h"
#include "ocu_backend_cpu.h"
#include "ocu_cpu_kernel.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TOTAL_BLOCKS 20000
#define CHUNKS 80
#define THREADS_PER_BLOCK 8
#define WORK_ITERATIONS 2000u
#define ALLOWED_FACTOR 4.0
/* Timings below this are noise: a quarter must take at least this long
   (ms) before the ratio means anything; below it only the absolute bound
   applies. */
#define NOISE_FLOOR_MS 20.0
#define ABSOLUTE_LIMIT_MS 120000.0

static int g_failures, g_checks;
#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* Each thread counts itself; a block with a barrier in the middle makes the
   fibers switch at least twice per thread. */
static void k_count(void **a)
{
    _Atomic long *counter = *(_Atomic long **)a[0];
    atomic_fetch_add(counter, 1);
    /* Real work between the fiber switches: the cost of a deep shadow
       stack shows when many memory accesses are recorded (the sanitizer
       saves the stack at every trace-part switch). */
    volatile unsigned buf[64];
    for (unsigned i = 0; i < WORK_ITERATIONS; ++i) {
        buf[i & 63] = buf[(i * 7) & 63] + i;
    }
    ocu_cpu_barrier();
    atomic_fetch_add(counter, 1);
}

static const ocu_cpu_kernel_entry g_entries[] = {{"count", k_count}};
static const ocu_cpu_image g_image = {OCU_CPU_IMAGE_MAGIC, OCU_CPU_IMAGE_VERSION, 1, g_entries};

static double run_launches(const ocu_backend *B, ocu_queue *q, ocu_kernel *k, _Atomic long *counter, int from, int to,
                           uint32_t blocks_per_launch)
{
    void *args[] = {&counter};
    size_t sizes[] = {sizeof counter};
    ocu_launch_params p;
    memset(&p, 0, sizeof p);
    p.grid[0] = blocks_per_launch;
    p.grid[1] = p.grid[2] = 1;
    p.block[0] = THREADS_PER_BLOCK;
    p.block[1] = p.block[2] = 1;
    p.args = args;
    p.arg_sizes = sizes;
    p.num_args = 1;
    double t0 = now_ms();
    for (int i = from; i < to; ++i) {
        if (B->kernel_launch(q, k, &p) != OCU_OK || (q && B->queue_sync(q) != OCU_OK)) {
            fprintf(stderr, "launch %d failed\n", i);
            ++g_failures;
            return 0;
        }
    }
    return now_ms() - t0;
}

int main(void)
{
    const ocu_backend *B = ocu_backend_cpu();
    ocu_device *D = NULL;
    ocu_module *M = NULL;
    ocu_kernel *k = NULL;
    ocu_queue *q = NULL;
    if (B->dev_open(0, &D) != OCU_OK ||
        B->module_load(D, &g_image, sizeof g_image, &M) != OCU_OK ||
        B->module_get_kernel(M, "count", &k) != OCU_OK || B->queue_create(D, &q) != OCU_OK) {
        fprintf(stderr, "setup failed\n");
        return 1;
    }
    static _Atomic long counter;
    const uint32_t per_launch = TOTAL_BLOCKS / CHUNKS;
    const int quarter = CHUNKS / 4;

    /* A short warm-up outside the measurement (thread pool start, page faults). */
    run_launches(B, q, k, &counter, 0, 4, per_launch);
    atomic_store(&counter, 0);

    double first = run_launches(B, q, k, &counter, 0, quarter, per_launch);
    double middle = run_launches(B, q, k, &counter, quarter, 3 * quarter, per_launch);
    double last = run_launches(B, q, k, &counter, 3 * quarter, CHUNKS, per_launch);
    (void)middle;

    /* Every thread of every block ran both halves. */
    CHECK(atomic_load(&counter) == (long)TOTAL_BLOCKS * THREADS_PER_BLOCK * 2);

    printf("test_cpu_tsan_fibers: %d blocks in %d launches; first quarter %.1f ms, last quarter %.1f ms (ratio %.2f)\n",
           TOTAL_BLOCKS, CHUNKS, first, last, first > 0 ? last / first : 0.0);
    CHECK(first + middle + last < ABSOLUTE_LIMIT_MS);
    if (first >= NOISE_FLOOR_MS) {
        CHECK(last <= first * ALLOWED_FACTOR);
    } else {
        /* A fast first quarter: the last one may take a little longer in
           absolute terms, but not beyond the factor applied to the floor. */
        CHECK(last <= NOISE_FLOOR_MS * ALLOWED_FACTOR);
    }

    B->module_unload(M);
    if (g_failures == 0) {
        printf("test_cpu_tsan_fibers: ok (%d checks)\n", g_checks);
        return 0;
    }
    fprintf(stderr, "test_cpu_tsan_fibers: %d of %d checks failed\n", g_failures, g_checks);
    return 1;
}

/*
 * Regression tests for the CPU backend findings of the 2026-11-12 review.
 * Each test names its finding. Run under ASan/TSan by tools/check.sh.
 */
#define _POSIX_C_SOURCE 200809L

#include "ocu_backend.h"
#include "ocu_backend_cpu.h"
#include "ocu_cpu_kernel.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_failures, g_checks;
static const char *g_test = "";

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL [%s] %s:%d: %s\n", g_test, __FILE__,       \
                    __LINE__, #cond);                                        \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

#define CHECK_ST(expr, want)                                                 \
    do {                                                                     \
        ocu_status got_ = (expr);                                            \
        ++g_checks;                                                          \
        if (got_ != (want)) {                                                \
            fprintf(stderr, "FAIL [%s] %s:%d: %s -> %s, want %s\n", g_test,  \
                    __FILE__, __LINE__, #expr, ocu_status_name(got_),        \
                    ocu_status_name(want));                                  \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static const ocu_backend *B;
static ocu_device *D;
static ocu_module *M;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void sleep_ms(long ms)
{
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

/* ---- kernels -------------------------------------------------------------- */

static void k_gate(void **a)
{
    _Atomic int *gate = *(_Atomic int **)a[0];
    while (atomic_load(gate) == 0) {
        sleep_ms(1);
    }
}

static ocu_kernel *g_nested_target;
static ocu_queue *g_nested_queue;

/* Calls back into the backend from inside a kernel and records the results. */
static void k_nested(void **a)
{
    int *out = *(int **)a[0];
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = p.grid[1] = p.grid[2] = 1;
    p.block[0] = p.block[1] = p.block[2] = 1;
    out[0] = B->kernel_launch(NULL, g_nested_target, &p);
    out[1] = B->kernel_launch(g_nested_queue, g_nested_target, &p);
    out[2] = B->queue_sync(g_nested_queue);
    int x = 0;
    out[3] = B->mem_free(D, &x);
}

static void k_nop(void **a) { (void)a; }

typedef struct {
    _Alignas(64) double v[8];
} wide_arg;

static void k_alignment(void **a)
{
    int *out = *(int **)a[1];
    out[0] = ((uintptr_t)a[0] % 64) == 0;
    out[1] = (int)((const wide_arg *)a[0])->v[7];
}

/* Recurses with 1 KiB frames until `depth` reaches 0. */
static int __attribute__((noinline)) deep(int depth)
{
    volatile char frame[1024];
    for (int i = 0; i < 1024; i += 64) {
        frame[i] = (char)depth;
    }
    return depth == 0 ? frame[0] : deep(depth - 1) + frame[64];
}

static void k_deep(void **a)
{
    int depth = *(int *)a[0];
    int *sink = *(int **)a[1];
    if (ocu_cpu_thread_idx().x == 1) { /* fiber 0 has finished by now */
        sink[0] = deep(depth);
    }
}

static const ocu_cpu_kernel_entry g_entries[] = {
    {"gate", k_gate}, {"nested", k_nested}, {"nop", k_nop},
    {"alignment", k_alignment}, {"deep", k_deep},
};
static const ocu_cpu_image g_image = {OCU_CPU_IMAGE_MAGIC, OCU_CPU_IMAGE_VERSION,
                                      5, g_entries};

static ocu_kernel *kernel(const char *name)
{
    ocu_kernel *k = NULL;
    B->module_get_kernel(M, name, &k);
    return k;
}

static ocu_launch_params one(void **args, const size_t *sizes, uint32_t n)
{
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = p.grid[1] = p.grid[2] = 1;
    p.block[0] = p.block[1] = p.block[2] = 1;
    p.args = args;
    p.arg_sizes = sizes;
    p.num_args = n;
    return p;
}

/* ---- finding 2: free while queued work still uses the memory ------------- */

typedef struct {
    _Atomic int *gate;
    double opened_at;
} opener;

static void *open_later(void *arg)
{
    opener *o = arg;
    sleep_ms(50);
    o->opened_at = now_ms();
    atomic_store(o->gate, 1);
    return NULL;
}

static void test_free_waits_for_queued_work(void)
{
    g_test = "free_waits_for_queued_work";
    ocu_queue *q = NULL;
    CHECK_ST(B->queue_create(D, &q), OCU_OK);
    _Atomic int *gate = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(*gate), (void **)&gate), OCU_OK);
    atomic_store(gate, 0);
    void *ga[] = {&gate};
    size_t gs[] = {sizeof(gate)};
    ocu_launch_params gp = one(ga, gs, 1);
    CHECK_ST(B->kernel_launch(q, kernel("gate"), &gp), OCU_OK);

    static char host[1 << 16];
    memset(host, 1, sizeof(host));
    void *buf = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(host), &buf), OCU_OK);
    CHECK_ST(B->mem_copy(D, q, buf, host, sizeof(host), OCU_COPY_HOST_TO_DEVICE), OCU_OK);

    opener o = {gate, 0};
    pthread_t t;
    pthread_create(&t, NULL, open_later, &o);
    CHECK_ST(B->mem_free(D, buf), OCU_OK); /* must wait for the copy */
    double freed_at = now_ms();
    pthread_join(t, NULL);
    CHECK(freed_at >= o.opened_at);
    CHECK_ST(B->queue_sync(q), OCU_OK);
    CHECK_ST(B->queue_destroy(q), OCU_OK);
    CHECK_ST(B->mem_free(D, gate), OCU_OK);
}

/* ---- finding 3: argument sizes ------------------------------------------- */

static void test_argument_size_limits(void)
{
    g_test = "argument_size_limits";
    ocu_queue *q = NULL;
    CHECK_ST(B->queue_create(D, &q), OCU_OK);
    static unsigned char big[OCU_MAX_ARG_BYTES + 8];
    void *args[] = {big, big};
    size_t sizes[] = {SIZE_MAX - 7, 8};
    ocu_launch_params p = one(args, sizes, 1);
    CHECK_ST(B->kernel_launch(q, kernel("nop"), &p), OCU_ERR_INVALID_VALUE);
    sizes[0] = OCU_MAX_ARG_BYTES + 1;
    CHECK_ST(B->kernel_launch(q, kernel("nop"), &p), OCU_ERR_INVALID_VALUE);
    sizes[0] = OCU_MAX_ARG_BYTES; /* exactly at the limit is fine */
    CHECK_ST(B->kernel_launch(q, kernel("nop"), &p), OCU_OK);
    sizes[0] = OCU_MAX_ARG_BYTES - 4;
    p.num_args = 2; /* the sum crosses the limit */
    CHECK_ST(B->kernel_launch(q, kernel("nop"), &p), OCU_ERR_INVALID_VALUE);
    sizes[0] = SIZE_MAX / 2 + 1;
    sizes[1] = SIZE_MAX / 2 + 1; /* sum wraps around to a small number */
    CHECK_ST(B->kernel_launch(q, kernel("nop"), &p), OCU_ERR_INVALID_VALUE);
    CHECK_ST(B->queue_sync(q), OCU_OK);
    CHECK_ST(B->queue_destroy(q), OCU_OK);
}

/* ---- finding 7: calling back into the backend from a kernel ------------- */

static void test_calls_from_inside_a_kernel(void)
{
    g_test = "calls_from_inside_a_kernel";
    CHECK_ST(B->queue_create(D, &g_nested_queue), OCU_OK);
    g_nested_target = kernel("nop");
    int *out = NULL;
    CHECK_ST(B->mem_alloc(D, 4 * sizeof(int), (void **)&out), OCU_OK);
    void *args[] = {&out};
    size_t sizes[] = {sizeof(out)};
    ocu_launch_params p = one(args, sizes, 1);
    CHECK_ST(B->kernel_launch(NULL, kernel("nested"), &p), OCU_OK);
    CHECK(out[0] == OCU_ERR_NOT_SUPPORTED);
    CHECK(out[1] == OCU_ERR_NOT_SUPPORTED);
    CHECK(out[2] == OCU_ERR_NOT_SUPPORTED);
    CHECK(out[3] == OCU_ERR_NOT_SUPPORTED);
    /* The same, from a queued launch: no deadlock, same answers. */
    memset(out, 0, 4 * sizeof(int));
    CHECK_ST(B->kernel_launch(g_nested_queue, kernel("nested"), &p), OCU_OK);
    CHECK_ST(B->queue_sync(g_nested_queue), OCU_OK);
    CHECK(out[1] == OCU_ERR_NOT_SUPPORTED && out[2] == OCU_ERR_NOT_SUPPORTED);
    CHECK_ST(B->queue_destroy(g_nested_queue), OCU_OK);
    CHECK_ST(B->mem_free(D, out), OCU_OK);
}

/* ---- finding 8: argument alignment and the marker wait rule ------------- */

static void test_wide_argument_alignment(void)
{
    g_test = "wide_argument_alignment";
    ocu_queue *q = NULL;
    CHECK_ST(B->queue_create(D, &q), OCU_OK);
    wide_arg w;
    for (int i = 0; i < 8; ++i) {
        w.v[i] = i;
    }
    int *out = NULL;
    CHECK_ST(B->mem_alloc(D, 2 * sizeof(int), (void **)&out), OCU_OK);
    void *args[] = {&w, &out};
    size_t sizes[] = {sizeof(w), sizeof(out)};
    ocu_launch_params p = one(args, sizes, 2);
    CHECK_ST(B->kernel_launch(q, kernel("alignment"), &p), OCU_OK);
    CHECK_ST(B->queue_sync(q), OCU_OK);
    CHECK(out[0] == 1);
    CHECK(out[1] == 7);
    CHECK_ST(B->queue_destroy(q), OCU_OK);
    CHECK_ST(B->mem_free(D, out), OCU_OK);
}

typedef struct {
    ocu_marker *m;
    double returned_at;
} waiter;

static void *wait_marker(void *arg)
{
    waiter *w = arg;
    B->marker_wait(w->m);
    w->returned_at = now_ms();
    return NULL;
}

static void test_marker_wait_ignores_later_records(void)
{
    g_test = "marker_wait_ignores_later_records";
    ocu_queue *q1 = NULL, *q2 = NULL;
    CHECK_ST(B->queue_create(D, &q1), OCU_OK);
    CHECK_ST(B->queue_create(D, &q2), OCU_OK);
    _Atomic int *g1 = NULL, *g2 = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(*g1), (void **)&g1), OCU_OK);
    CHECK_ST(B->mem_alloc(D, sizeof(*g2), (void **)&g2), OCU_OK);
    atomic_store(g1, 0);
    atomic_store(g2, 0);
    void *a1[] = {&g1}, *a2[] = {&g2};
    size_t sz[] = {sizeof(g1)};
    ocu_launch_params p1 = one(a1, sz, 1), p2 = one(a2, sz, 1);
    ocu_marker *m = NULL;
    CHECK_ST(B->marker_create(D, &m), OCU_OK);

    CHECK_ST(B->kernel_launch(q1, kernel("gate"), &p1), OCU_OK);
    CHECK_ST(B->marker_record(m, q1), OCU_OK);
    waiter w = {m, 0};
    pthread_t t;
    pthread_create(&t, NULL, wait_marker, &w);
    sleep_ms(20); /* the waiter is now waiting for the first record */
    CHECK_ST(B->kernel_launch(q2, kernel("gate"), &p2), OCU_OK);
    CHECK_ST(B->marker_record(m, q2), OCU_OK); /* re-record, still blocked */
    double opened = now_ms();
    atomic_store(g1, 1);
    pthread_join(t, NULL); /* must return although q2 is still blocked */
    CHECK(w.returned_at >= opened);
    CHECK_ST(B->marker_query(m), OCU_ERR_NOT_READY); /* the newer record */
    atomic_store(g2, 1);
    CHECK_ST(B->marker_wait(m), OCU_OK);
    CHECK_ST(B->marker_query(m), OCU_OK);
    CHECK_ST(B->marker_destroy(m), OCU_OK);
    CHECK_ST(B->queue_destroy(q1), OCU_OK);
    CHECK_ST(B->queue_destroy(q2), OCU_OK);
    CHECK_ST(B->mem_free(D, g1), OCU_OK);
    CHECK_ST(B->mem_free(D, g2), OCU_OK);
}

/* ---- finding 4: stack overflow is reported, not silent ------------------ */

static void test_stack_overflow_reported(void)
{
    g_test = "stack_overflow_reported";
    const char *kb = getenv("OCU_CPU_STACK_KB");
    if (kb == NULL || strcmp(kb, "16") != 0) {
        return; /* runs in the ctest variant with 16 KiB stacks */
    }
#if defined(__SANITIZE_ADDRESS__)
    return; /* ASan moves locals to its heap "fake stack": no overflow */
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    return;
#endif
#endif
    int *sink = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(int), (void **)&sink), OCU_OK);
    int depth = 4; /* about 4 KiB: fits */
    void *args[] = {&depth, &sink};
    size_t sizes[] = {sizeof(depth), sizeof(sink)};
    ocu_launch_params p = one(args, sizes, 2);
    p.block[0] = 2;
    CHECK_ST(B->kernel_launch(NULL, kernel("deep"), &p), OCU_OK);
    depth = 15; /* about 15 KiB plus frames: reaches the canary */
    CHECK_ST(B->kernel_launch(NULL, kernel("deep"), &p), OCU_ERR_LAUNCH_FAILED);
    CHECK_ST(B->mem_free(D, sink), OCU_OK);
}

int main(void)
{
    B = ocu_backend_cpu();
    if (B->dev_open(0, &D) != OCU_OK ||
        B->module_load(D, &g_image, sizeof(g_image), &M) != OCU_OK) {
        fprintf(stderr, "setup failed\n");
        return 1;
    }
    test_free_waits_for_queued_work();
    test_argument_size_limits();
    test_calls_from_inside_a_kernel();
    test_wide_argument_alignment();
    test_marker_wait_ignores_later_records();
    test_stack_overflow_reported();
    B->module_unload(M);
    if (g_failures == 0) {
        printf("test_cpu_regressions: ok (%d checks)\n", g_checks);
        return 0;
    }
    fprintf(stderr, "test_cpu_regressions: %d of %d checks failed\n",
            g_failures, g_checks);
    return 1;
}

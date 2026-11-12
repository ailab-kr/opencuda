/*
 * Behaviour tests for the CPU backend, through the backend table only
 * (plus the CPU device-side API that kernels written for this backend use).
 *
 * Each test names the promise from ocu_backend.h or ARCHITECTURE.md it
 * checks. Run with OCU_CPU_ORDER=random as well as the default order.
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

static int g_failures;
static int g_checks;
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

static void sleep_ms(long ms)
{
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

/* ---- kernels ------------------------------------------------------------ */

static void k_saxpy(void **a)
{
    int n = *(int *)a[0];
    float alpha = *(float *)a[1];
    const float *x = *(const float **)a[2];
    float *y = *(float **)a[3];
    ocu_cpu_dim3 b = ocu_cpu_block_idx(), d = ocu_cpu_block_dim(),
                 t = ocu_cpu_thread_idx();
    int i = (int)(b.x * d.x + t.x);
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}

/* Every thread writes its global linear id at its own slot. */
static void k_ids(void **a)
{
    uint32_t *out = *(uint32_t **)a[0];
    ocu_cpu_dim3 t = ocu_cpu_thread_idx(), b = ocu_cpu_block_idx(),
                 bd = ocu_cpu_block_dim(), gd = ocu_cpu_grid_dim();
    uint32_t tl = t.x + bd.x * (t.y + bd.y * t.z);
    uint32_t bl = b.x + gd.x * (b.y + gd.y * b.z);
    uint32_t per = bd.x * bd.y * bd.z;
    uint32_t id = bl * per + tl;
    out[id] = id + 1u; /* +1 so an untouched slot (0) is visible */
}

/* Tree reduction in dynamic shared memory; needs a working barrier. */
static void k_block_sum(void **a)
{
    const float *in = *(const float **)a[0];
    float *out = *(float **)a[1];
    float *tile = ocu_cpu_dynamic_shared();
    uint32_t t = ocu_cpu_thread_idx().x, n = ocu_cpu_block_dim().x;
    uint32_t b = ocu_cpu_block_idx().x;
    tile[t] = in[b * n + t];
    ocu_cpu_barrier();
    for (uint32_t s = n / 2; s > 0; s >>= 1) {
        if (t < s) {
            tile[t] += tile[t + s];
        }
        ocu_cpu_barrier();
    }
    if (t == 0) {
        out[b] = tile[0];
    }
}

/* Records the order in which threads of block 0 reach this point. */
static void k_order(void **a)
{
    _Atomic uint32_t *counter = *(_Atomic uint32_t **)a[0];
    uint32_t *seen = *(uint32_t **)a[1];
    uint32_t slot = atomic_fetch_add(counter, 1);
    seen[slot] = ocu_cpu_thread_idx().x;
}

/* Threads past `n` return before the barrier (common in real code). */
static void k_early_exit(void **a)
{
    int n = *(int *)a[0];
    int *out = *(int **)a[1];
    int *tile = ocu_cpu_dynamic_shared();
    int t = (int)ocu_cpu_thread_idx().x;
    if (t >= n) {
        return;
    }
    tile[t] = t;
    ocu_cpu_barrier();
    out[t] = tile[(t + 1) % n];
}

/* Blocks until the host opens the gate: lets tests see a busy queue. */
static void k_gate(void **a)
{
    _Atomic int *gate = *(_Atomic int **)a[0];
    while (atomic_load(gate) == 0) {
        sleep_ms(1);
    }
}

/* Writes the value of its argument, to check arguments are copied. */
static void k_store_arg(void **a)
{
    int v = *(int *)a[0];
    int *out = *(int **)a[1];
    *out = v;
}

/* Atomic add across all blocks and threads. */
static void k_count(void **a)
{
    _Atomic uint64_t *c = *(_Atomic uint64_t **)a[0];
    atomic_fetch_add(c, 1);
}

static void k_warp(void **a)
{
    uint32_t *out = *(uint32_t **)a[0];
    *out = ocu_cpu_warp_size();
}

static const ocu_cpu_kernel_entry g_entries[] = {
    {"saxpy", k_saxpy},       {"ids", k_ids},
    {"block_sum", k_block_sum}, {"order", k_order},
    {"early_exit", k_early_exit}, {"gate", k_gate},
    {"store_arg", k_store_arg}, {"count", k_count},
    {"warp", k_warp},
};
static const ocu_cpu_image g_image = {
    OCU_CPU_IMAGE_MAGIC, OCU_CPU_IMAGE_VERSION,
    (uint32_t)(sizeof(g_entries) / sizeof(g_entries[0])), g_entries};

static ocu_module *M;

static ocu_kernel *kernel(const char *name)
{
    ocu_kernel *k = NULL;
    CHECK_ST(B->module_get_kernel(M, name, &k), OCU_OK);
    return k;
}

static ocu_launch_params params1(uint32_t grid, uint32_t block, void **args,
                                 const size_t *sizes, uint32_t n)
{
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = grid; p.grid[1] = 1; p.grid[2] = 1;
    p.block[0] = block; p.block[1] = 1; p.block[2] = 1;
    p.args = args;
    p.arg_sizes = sizes;
    p.num_args = n;
    return p;
}

/* ---- tests --------------------------------------------------------------- */

static void test_device(void)
{
    g_test = "device";
    uint32_t n = 0;
    CHECK_ST(B->dev_count(&n), OCU_OK);
    CHECK(n == 1);
    ocu_dev_info info;
    CHECK_ST(B->dev_info(0, &info), OCU_OK);
    CHECK(info.total_memory_bytes > 0);
    CHECK(info.max_threads_per_block == 1024);
    CHECK_ST(B->dev_info(1, &info), OCU_ERR_INVALID_DEVICE);
    ocu_device *d = NULL;
    CHECK_ST(B->dev_open(1, &d), OCU_ERR_INVALID_DEVICE);
    CHECK_ST(B->dev_open(0, NULL), OCU_ERR_INVALID_VALUE);
}

static void test_memory(void)
{
    g_test = "memory";
    void *p = NULL;
    CHECK_ST(B->mem_alloc(D, 1000, &p), OCU_OK);
    CHECK(p != NULL);
    CHECK(((uintptr_t)p % 256) == 0); /* GPU-like alignment */

    uint64_t free_before = 0, total = 0, free_after = 0;
    CHECK_ST(B->mem_info(D, &free_before, &total), OCU_OK);
    void *q = NULL;
    CHECK_ST(B->mem_alloc(D, 1 << 20, &q), OCU_OK);
    CHECK_ST(B->mem_info(D, &free_after, &total), OCU_OK);
    CHECK(free_before - free_after == (1u << 20));
    CHECK(total >= free_before);

    /* Only exact allocation starts can be freed; freeing twice fails. */
    CHECK_ST(B->mem_free(D, (char *)q + 8), OCU_ERR_INVALID_VALUE);
    CHECK_ST(B->mem_free(D, q), OCU_OK);
    CHECK_ST(B->mem_free(D, q), OCU_ERR_INVALID_VALUE);
    int stack_var = 0;
    CHECK_ST(B->mem_free(D, &stack_var), OCU_ERR_INVALID_VALUE);
    CHECK_ST(B->mem_free(D, NULL), OCU_OK);

    void *z = (void *)1;
    CHECK_ST(B->mem_alloc(D, 0, &z), OCU_OK);
    CHECK(z == NULL);
    CHECK_ST(B->mem_alloc(D, 16, NULL), OCU_ERR_INVALID_VALUE);

    /* Copies are checked against allocation bounds. */
    char host[1000];
    memset(host, 7, sizeof(host));
    CHECK_ST(B->mem_copy(D, NULL, p, host, 1000, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    CHECK_ST(B->mem_copy(D, NULL, p, host, 1001, OCU_COPY_HOST_TO_DEVICE),
             OCU_ERR_INVALID_VALUE);
    CHECK_ST(B->mem_copy(D, NULL, (char *)p + 999, host, 1, OCU_COPY_HOST_TO_DEVICE),
             OCU_OK);
    CHECK_ST(B->mem_copy(D, NULL, (char *)p + 1000, host, 1, OCU_COPY_HOST_TO_DEVICE),
             OCU_ERR_INVALID_VALUE);
    CHECK_ST(B->mem_copy(D, NULL, host, host + 10, 10, OCU_COPY_HOST_TO_DEVICE),
             OCU_ERR_INVALID_VALUE); /* destination is not device memory */
    CHECK_ST(B->mem_copy(D, NULL, host, p, 10, OCU_COPY_DEVICE_TO_DEVICE),
             OCU_ERR_INVALID_VALUE);
    CHECK_ST(B->mem_copy(D, NULL, p, host, 0, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    CHECK_ST(B->mem_copy(D, NULL, p, NULL, 4, OCU_COPY_HOST_TO_DEVICE),
             OCU_ERR_INVALID_VALUE);
    CHECK_ST(B->mem_copy(D, NULL, p, host, 4, (ocu_copy_dir)42), OCU_ERR_INVALID_VALUE);

    /* Device to device, overlapping, behaves like memmove. */
    unsigned char *u = p;
    for (int i = 0; i < 16; ++i) {
        host[i] = (char)i;
    }
    CHECK_ST(B->mem_copy(D, NULL, u, host, 16, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    CHECK_ST(B->mem_copy(D, NULL, u + 1, u, 8, OCU_COPY_DEVICE_TO_DEVICE), OCU_OK);
    char back[16];
    CHECK_ST(B->mem_copy(D, NULL, back, u, 16, OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK(back[0] == 0 && back[1] == 0 && back[8] == 7 && back[9] == 9);

    CHECK_ST(B->mem_set(D, NULL, p, 0x5A, 1000), OCU_OK);
    CHECK_ST(B->mem_copy(D, NULL, host, p, 1000, OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK(host[0] == 0x5A && host[999] == 0x5A);
    CHECK_ST(B->mem_set(D, NULL, p, 0, 1001), OCU_ERR_INVALID_VALUE);
    CHECK_ST(B->mem_free(D, p), OCU_OK);

    void *h = NULL;
    CHECK_ST(B->host_alloc_pinned(D, 64, &h), OCU_OK);
    CHECK(h != NULL);
    CHECK_ST(B->host_free_pinned(D, h), OCU_OK);
    CHECK_ST(B->host_free_pinned(D, h), OCU_ERR_INVALID_VALUE);
}

static void test_modules(void)
{
    g_test = "modules";
    ocu_module *m = NULL;
    ocu_cpu_image bad = g_image;
    bad.magic = 0;
    CHECK_ST(B->module_load(D, &bad, sizeof(bad), &m), OCU_ERR_INVALID_VALUE);
    CHECK_ST(B->module_load(D, &g_image, sizeof(g_image) - 1, &m),
             OCU_ERR_INVALID_VALUE);
    ocu_cpu_kernel_entry nofn[1] = {{"x", NULL}};
    ocu_cpu_image bad2 = {OCU_CPU_IMAGE_MAGIC, OCU_CPU_IMAGE_VERSION, 1, nofn};
    CHECK_ST(B->module_load(D, &bad2, sizeof(bad2), &m), OCU_ERR_INVALID_VALUE);
    ocu_kernel *k = NULL;
    CHECK_ST(B->module_get_kernel(M, "no_such_kernel", &k), OCU_ERR_NOT_FOUND);
    CHECK_ST(B->module_get_kernel(M, NULL, &k), OCU_ERR_INVALID_VALUE);
}

static void test_launch_saxpy(void)
{
    g_test = "launch_saxpy";
    const int n = 1000;
    float hx[1000], hy[1000];
    for (int i = 0; i < n; ++i) {
        hx[i] = (float)i;
        hy[i] = 1.0f;
    }
    float *dx = NULL, *dy = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(hx), (void **)&dx), OCU_OK);
    CHECK_ST(B->mem_alloc(D, sizeof(hy), (void **)&dy), OCU_OK);
    CHECK_ST(B->mem_copy(D, NULL, dx, hx, sizeof(hx), OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    CHECK_ST(B->mem_copy(D, NULL, dy, hy, sizeof(hy), OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    float alpha = 2.0f;
    void *args[] = {(void *)&n, &alpha, &dx, &dy};
    size_t sizes[] = {sizeof(n), sizeof(alpha), sizeof(dx), sizeof(dy)};
    ocu_launch_params p = params1((n + 127) / 128, 128, args, sizes, 4);
    CHECK_ST(B->kernel_launch(NULL, kernel("saxpy"), &p), OCU_OK);
    CHECK_ST(B->mem_copy(D, NULL, hy, dy, sizeof(hy), OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        bad += hy[i] != 2.0f * (float)i + 1.0f;
    }
    CHECK(bad == 0);
    B->mem_free(D, dx);
    B->mem_free(D, dy);
}

static void test_launch_indices_3d(void)
{
    g_test = "launch_indices_3d";
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = 3; p.grid[1] = 2; p.grid[2] = 2;
    p.block[0] = 4; p.block[1] = 3; p.block[2] = 2;
    uint32_t total = 3 * 2 * 2 * 4 * 3 * 2;
    uint32_t *out = NULL;
    CHECK_ST(B->mem_alloc(D, total * sizeof(uint32_t), (void **)&out), OCU_OK);
    CHECK_ST(B->mem_set(D, NULL, out, 0, total * sizeof(uint32_t)), OCU_OK);
    void *args[] = {&out};
    size_t sizes[] = {sizeof(out)};
    p.args = args;
    p.arg_sizes = sizes;
    p.num_args = 1;
    CHECK_ST(B->kernel_launch(NULL, kernel("ids"), &p), OCU_OK);
    int bad = 0;
    for (uint32_t i = 0; i < total; ++i) {
        bad += out[i] != i + 1;
    }
    CHECK(bad == 0); /* every (block, thread) pair ran exactly once */
    B->mem_free(D, out);
}

static void test_launch_limits(void)
{
    g_test = "launch_limits";
    void *args[] = {NULL};
    size_t sizes[] = {0};
    ocu_launch_params p = params1(1, 1, args, sizes, 0);
    ocu_kernel *k = kernel("count");
    p.block[0] = 1025;
    CHECK_ST(B->kernel_launch(NULL, k, &p), OCU_ERR_INVALID_VALUE);
    p.block[0] = 32; p.block[1] = 32; p.block[2] = 2; /* 2048 threads */
    CHECK_ST(B->kernel_launch(NULL, k, &p), OCU_ERR_INVALID_VALUE);
    p.block[0] = 1; p.block[1] = 1; p.block[2] = 65;
    CHECK_ST(B->kernel_launch(NULL, k, &p), OCU_ERR_INVALID_VALUE);
    p.block[2] = 1;
    p.grid[0] = 0;
    CHECK_ST(B->kernel_launch(NULL, k, &p), OCU_ERR_INVALID_VALUE);
    p.grid[0] = 1; p.grid[1] = 65536;
    CHECK_ST(B->kernel_launch(NULL, k, &p), OCU_ERR_INVALID_VALUE);
    p.grid[1] = 1;
    p.shared_memory_bytes = 48 * 1024 + 1;
    CHECK_ST(B->kernel_launch(NULL, k, &p), OCU_ERR_INVALID_VALUE);
    p.shared_memory_bytes = 0;
    p.num_args = 1;
    p.args = NULL;
    CHECK_ST(B->kernel_launch(NULL, k, &p), OCU_ERR_INVALID_VALUE);
    CHECK_ST(B->kernel_launch(NULL, NULL, &p), OCU_ERR_INVALID_HANDLE);
    CHECK_ST(B->kernel_launch(NULL, k, NULL), OCU_ERR_INVALID_VALUE);
}

static void test_barrier_reduction(void)
{
    g_test = "barrier_reduction";
    const uint32_t blocks = 37, threads = 256;
    float *in = NULL, *out = NULL;
    CHECK_ST(B->mem_alloc(D, blocks * threads * sizeof(float), (void **)&in), OCU_OK);
    CHECK_ST(B->mem_alloc(D, blocks * sizeof(float), (void **)&out), OCU_OK);
    for (uint32_t i = 0; i < blocks * threads; ++i) {
        in[i] = (float)(i % 7); /* small integers: the sum is exact */
    }
    void *args[] = {&in, &out};
    size_t sizes[] = {sizeof(in), sizeof(out)};
    ocu_launch_params p = params1(blocks, threads, args, sizes, 2);
    p.shared_memory_bytes = threads * sizeof(float);
    CHECK_ST(B->kernel_launch(NULL, kernel("block_sum"), &p), OCU_OK);
    int bad = 0;
    for (uint32_t b = 0; b < blocks; ++b) {
        float want = 0;
        for (uint32_t t = 0; t < threads; ++t) {
            want += (float)((b * threads + t) % 7);
        }
        bad += out[b] != want;
    }
    CHECK(bad == 0);
    B->mem_free(D, in);
    B->mem_free(D, out);
}

static void test_early_exit_before_barrier(void)
{
    g_test = "early_exit_before_barrier";
    int n = 20;
    int *out = NULL;
    CHECK_ST(B->mem_alloc(D, 32 * sizeof(int), (void **)&out), OCU_OK);
    void *args[] = {&n, &out};
    size_t sizes[] = {sizeof(n), sizeof(out)};
    ocu_launch_params p = params1(1, 32, args, sizes, 2);
    p.shared_memory_bytes = 32 * sizeof(int);
    const char *strict = getenv("OCU_CPU_STRICT");
    ocu_status want = (strict != NULL && strcmp(strict, "1") == 0)
                          ? OCU_ERR_LAUNCH_FAILED
                          : OCU_OK;
    CHECK_ST(B->kernel_launch(NULL, kernel("early_exit"), &p), want);
    if (want == OCU_OK) {
        int bad = 0;
        for (int t = 0; t < n; ++t) {
            bad += out[t] != (t + 1) % n;
        }
        CHECK(bad == 0);
    }
    B->mem_free(D, out);
}

static void test_thread_order(void)
{
    g_test = "thread_order";
    const uint32_t threads = 64;
    _Atomic uint32_t *counter = NULL;
    uint32_t *seen = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(*counter), (void **)&counter), OCU_OK);
    CHECK_ST(B->mem_alloc(D, threads * sizeof(uint32_t), (void **)&seen), OCU_OK);
    atomic_store(counter, 0);
    void *args[] = {&counter, &seen};
    size_t sizes[] = {sizeof(counter), sizeof(seen)};
    ocu_launch_params p = params1(1, threads, args, sizes, 2);
    CHECK_ST(B->kernel_launch(NULL, kernel("order"), &p), OCU_OK);
    uint32_t in_order = 0, mask_ok = 1;
    uint64_t mask = 0;
    for (uint32_t i = 0; i < threads; ++i) {
        in_order += seen[i] == i;
        mask |= 1ULL << seen[i];
    }
    mask_ok = mask == ~0ULL;
    CHECK(mask_ok); /* each thread ran once */
    const char *order = getenv("OCU_CPU_ORDER");
    if (order != NULL && strcmp(order, "random") == 0) {
        CHECK(in_order < threads); /* shuffled */
    } else {
        CHECK(in_order == threads); /* fixed: 0, 1, 2, ... */
    }
    B->mem_free(D, counter);
    B->mem_free(D, seen);
}

static void test_atomic_across_blocks(void)
{
    g_test = "atomic_across_blocks";
    _Atomic uint64_t *c = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(*c), (void **)&c), OCU_OK);
    atomic_store(c, 0);
    void *args[] = {&c};
    size_t sizes[] = {sizeof(c)};
    ocu_launch_params p = params1(500, 64, args, sizes, 1);
    CHECK_ST(B->kernel_launch(NULL, kernel("count"), &p), OCU_OK);
    CHECK(atomic_load(c) == 500u * 64u);
    B->mem_free(D, c);
}

static void test_queue_order_and_sync(void)
{
    g_test = "queue_order_and_sync";
    ocu_queue *q = NULL;
    CHECK_ST(B->queue_create(D, &q), OCU_OK);
    int *buf = NULL;
    CHECK_ST(B->mem_alloc(D, 4096 * sizeof(int), (void **)&buf), OCU_OK);
    /* Fill, overwrite, read back: only in-order execution gives 3s. */
    static int host[4096];
    for (int i = 0; i < 4096; ++i) {
        host[i] = 3;
    }
    CHECK_ST(B->mem_set(D, q, buf, 1, sizeof(int) * 4096), OCU_OK);
    CHECK_ST(B->mem_copy(D, q, buf, host, sizeof(host), OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    static int back[4096];
    CHECK_ST(B->mem_copy(D, q, back, buf, sizeof(back), OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK_ST(B->queue_sync(q), OCU_OK);
    int bad = 0;
    for (int i = 0; i < 4096; ++i) {
        bad += back[i] != 3;
    }
    CHECK(bad == 0);
    CHECK_ST(B->queue_query(q), OCU_OK);
    CHECK_ST(B->queue_destroy(q), OCU_OK);
    B->mem_free(D, buf);
    CHECK_ST(B->queue_sync(NULL), OCU_ERR_INVALID_HANDLE);
}

static void test_queue_busy_and_args_copied(void)
{
    g_test = "queue_busy_and_args_copied";
    ocu_queue *q = NULL;
    CHECK_ST(B->queue_create(D, &q), OCU_OK);
    _Atomic int *gate = NULL;
    int *out = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(*gate), (void **)&gate), OCU_OK);
    CHECK_ST(B->mem_alloc(D, sizeof(int), (void **)&out), OCU_OK);
    atomic_store(gate, 0);
    *out = 0;

    void *gargs[] = {&gate};
    size_t gsizes[] = {sizeof(gate)};
    ocu_launch_params gp = params1(1, 1, gargs, gsizes, 1);
    CHECK_ST(B->kernel_launch(q, kernel("gate"), &gp), OCU_OK);

    int value = 41;
    void *sargs[] = {&value, &out};
    size_t ssizes[] = {sizeof(value), sizeof(out)};
    ocu_launch_params sp = params1(1, 1, sargs, ssizes, 2);
    CHECK_ST(B->kernel_launch(q, kernel("store_arg"), &sp), OCU_OK);
    value = 99; /* changed after submission: the kernel must still see 41 */

    ocu_marker *m = NULL;
    CHECK_ST(B->marker_create(D, &m), OCU_OK);
    CHECK_ST(B->marker_query(m), OCU_OK); /* never recorded = complete */
    CHECK_ST(B->marker_wait(m), OCU_OK);
    CHECK_ST(B->marker_record(m, q), OCU_OK);

    CHECK_ST(B->queue_query(q), OCU_ERR_NOT_READY);
    CHECK_ST(B->marker_query(m), OCU_ERR_NOT_READY);
    atomic_store(gate, 1);
    CHECK_ST(B->marker_wait(m), OCU_OK);
    CHECK_ST(B->marker_query(m), OCU_OK);
    CHECK_ST(B->queue_sync(q), OCU_OK);
    CHECK(*out == 41);
    CHECK_ST(B->marker_destroy(m), OCU_OK);
    CHECK_ST(B->queue_destroy(q), OCU_OK);
    B->mem_free(D, gate);
    B->mem_free(D, out);
}

static void test_markers_timing(void)
{
    g_test = "markers_timing";
    ocu_queue *q = NULL;
    CHECK_ST(B->queue_create(D, &q), OCU_OK);
    ocu_marker *a = NULL, *b = NULL, *never = NULL;
    CHECK_ST(B->marker_create(D, &a), OCU_OK);
    CHECK_ST(B->marker_create(D, &b), OCU_OK);
    CHECK_ST(B->marker_create(D, &never), OCU_OK);
    _Atomic int *gate = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(*gate), (void **)&gate), OCU_OK);
    atomic_store(gate, 0);
    void *gargs[] = {&gate};
    size_t gsizes[] = {sizeof(gate)};
    ocu_launch_params gp = params1(1, 1, gargs, gsizes, 1);

    CHECK_ST(B->marker_record(a, q), OCU_OK);
    CHECK_ST(B->kernel_launch(q, kernel("gate"), &gp), OCU_OK);
    CHECK_ST(B->marker_record(b, q), OCU_OK);
    float ms = -1;
    CHECK_ST(B->marker_elapsed_ms(a, b, &ms), OCU_ERR_NOT_READY);
    sleep_ms(30);
    atomic_store(gate, 1);
    CHECK_ST(B->marker_wait(b), OCU_OK);
    CHECK_ST(B->marker_elapsed_ms(a, b, &ms), OCU_OK);
    CHECK(ms >= 25.0f && ms < 5000.0f);
    CHECK_ST(B->marker_elapsed_ms(a, never, &ms), OCU_ERR_INVALID_HANDLE);
    CHECK_ST(B->marker_elapsed_ms(a, b, NULL), OCU_ERR_INVALID_VALUE);

    /* Recording on no queue completes at once. */
    CHECK_ST(B->marker_record(never, NULL), OCU_OK);
    CHECK_ST(B->marker_query(never), OCU_OK);

    /* Destroying a marker whose record is still queued is safe. */
    atomic_store(gate, 0);
    CHECK_ST(B->kernel_launch(q, kernel("gate"), &gp), OCU_OK);
    CHECK_ST(B->marker_record(a, q), OCU_OK);
    CHECK_ST(B->marker_destroy(a), OCU_OK);
    atomic_store(gate, 1);
    CHECK_ST(B->queue_sync(q), OCU_OK);

    CHECK_ST(B->marker_destroy(b), OCU_OK);
    CHECK_ST(B->marker_destroy(never), OCU_OK);
    CHECK_ST(B->queue_destroy(q), OCU_OK);
    B->mem_free(D, gate);
}

static void test_two_queues_independent(void)
{
    g_test = "two_queues_independent";
    ocu_queue *q1 = NULL, *q2 = NULL;
    CHECK_ST(B->queue_create(D, &q1), OCU_OK);
    CHECK_ST(B->queue_create(D, &q2), OCU_OK);
    _Atomic int *gate = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(*gate), (void **)&gate), OCU_OK);
    atomic_store(gate, 0);
    void *gargs[] = {&gate};
    size_t gsizes[] = {sizeof(gate)};
    ocu_launch_params gp = params1(1, 1, gargs, gsizes, 1);
    CHECK_ST(B->kernel_launch(q1, kernel("gate"), &gp), OCU_OK);
    /* q2 must not wait for the blocked q1. */
    int *buf = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(int), (void **)&buf), OCU_OK);
    CHECK_ST(B->mem_set(D, q2, buf, 0, sizeof(int)), OCU_OK);
    CHECK_ST(B->queue_sync(q2), OCU_OK);
    CHECK_ST(B->queue_query(q1), OCU_ERR_NOT_READY);
    atomic_store(gate, 1);
    CHECK_ST(B->queue_sync(q1), OCU_OK);
    CHECK_ST(B->queue_destroy(q1), OCU_OK);
    CHECK_ST(B->queue_destroy(q2), OCU_OK);
    B->mem_free(D, gate);
    B->mem_free(D, buf);
}

static void test_queued_error_is_reported(void)
{
    g_test = "queued_error_is_reported";
    const char *strict = getenv("OCU_CPU_STRICT");
    if (strict == NULL || strcmp(strict, "1") != 0) {
        return; /* needs a kernel that fails, which only strict mode gives */
    }
    ocu_queue *q = NULL;
    CHECK_ST(B->queue_create(D, &q), OCU_OK);
    int n = 20;
    int *out = NULL;
    CHECK_ST(B->mem_alloc(D, 32 * sizeof(int), (void **)&out), OCU_OK);
    void *args[] = {&n, &out};
    size_t sizes[] = {sizeof(n), sizeof(out)};
    ocu_launch_params p = params1(1, 32, args, sizes, 2);
    p.shared_memory_bytes = 32 * sizeof(int);
    CHECK_ST(B->kernel_launch(q, kernel("early_exit"), &p), OCU_OK);
    CHECK_ST(B->queue_sync(q), OCU_ERR_LAUNCH_FAILED);
    CHECK_ST(B->queue_sync(q), OCU_OK); /* reported once, then cleared */
    CHECK_ST(B->queue_destroy(q), OCU_OK);
    B->mem_free(D, out);
}

static void test_warp_size(void)
{
    g_test = "warp_size";
    uint32_t *out = NULL;
    CHECK_ST(B->mem_alloc(D, sizeof(uint32_t), (void **)&out), OCU_OK);
    void *args[] = {&out};
    size_t sizes[] = {sizeof(out)};
    ocu_launch_params p = params1(1, 1, args, sizes, 1);
    CHECK_ST(B->kernel_launch(NULL, kernel("warp"), &p), OCU_OK);
    ocu_dev_info info;
    B->dev_info(0, &info);
    const char *w = getenv("OCU_CPU_WARP");
    uint32_t want = (w != NULL && strcmp(w, "64") == 0) ? 64u : 32u;
    CHECK(*out == want);
    CHECK(info.warp_size == want);
    B->mem_free(D, out);
}

int main(void)
{
    B = ocu_backend_cpu();
    if (B->dev_open(0, &D) != OCU_OK ||
        B->module_load(D, &g_image, sizeof(g_image), &M) != OCU_OK) {
        fprintf(stderr, "setup failed\n");
        return 1;
    }
    test_device();
    test_memory();
    test_modules();
    test_launch_saxpy();
    test_launch_indices_3d();
    test_launch_limits();
    test_barrier_reduction();
    test_early_exit_before_barrier();
    test_thread_order();
    test_atomic_across_blocks();
    test_queue_order_and_sync();
    test_queue_busy_and_args_copied();
    test_markers_timing();
    test_two_queues_independent();
    test_queued_error_is_reported();
    test_warp_size();
    B->module_unload(M);
    B->dev_close(D);
    if (g_failures == 0) {
        printf("test_cpu_backend: ok (%d checks)\n", g_checks);
        return 0;
    }
    fprintf(stderr, "test_cpu_backend: %d of %d checks failed\n", g_failures,
            g_checks);
    return 1;
}

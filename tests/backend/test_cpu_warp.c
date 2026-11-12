/*
 * Warp-level operations of the CPU backend: gather, warp barrier, their
 * interplay with the block barrier, partial warps, deadlock detection.
 * Run with OCU_CPU_WARP=32 and 64, fixed and random order.
 */
#define _POSIX_C_SOURCE 200809L

#include "ocu_backend.h"
#include "ocu_backend_cpu.h"
#include "ocu_cpu_kernel.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static const ocu_backend *B;
static ocu_device *D;
static ocu_module *M;
static uint32_t W;

/* Each lane checks what it gathered; errors counted in out[thread]. */
static void k_gather(void **a)
{
    int *out = *(int **)a[0];
    uint32_t t = ocu_cpu_thread_idx().x;
    uint32_t n = ocu_cpu_block_dim().x;
    uint32_t w = ocu_cpu_warp_size();
    uint64_t all[OCU_CPU_MAX_WARP];
    int bad = 0;
    for (int round = 0; round < 3; ++round) {
        uint64_t mine = (uint64_t)t * 1000u + (uint64_t)round;
        uint32_t count = ocu_cpu_warp_gather(mine, all);
        uint32_t base = t - t % w;
        uint32_t want = n - base < w ? n - base : w;
        bad += count != want;
        bad += ocu_cpu_lane_id() != t % w;
        for (uint32_t i = 0; i < count; ++i) {
            bad += all[i] != (uint64_t)(base + i) * 1000u + (uint64_t)round;
        }
    }
    out[ocu_cpu_block_idx().x * n + t] = bad; /* one slot per block and thread */
}

/* Odd lanes return first; even lanes keep using warp operations. */
static void k_early_exit(void **a)
{
    int *out = *(int **)a[0];
    uint32_t t = ocu_cpu_thread_idx().x;
    uint32_t slot = ocu_cpu_block_idx().x * ocu_cpu_block_dim().x + t;
    if (t % 2 == 1) {
        out[slot] = 0;
        return;
    }
    uint64_t all[OCU_CPU_MAX_WARP];
    ocu_cpu_warp_sync();
    uint32_t count = ocu_cpu_warp_gather(t, all);
    uint32_t base = t - t % ocu_cpu_warp_size();
    /* even lanes see each other's values; the exited lanes' slots are 0 */
    int bad = 0;
    for (uint32_t i = 0; i < count; i += 2) {
        bad += all[i] != base + i;
    }
    out[slot] = bad;
}

/* Lane 0 waits at the block barrier, lane 1 at a warp operation. */
static void k_split(void **a)
{
    (void)a;
    if (ocu_cpu_thread_idx().x == 0) {
        ocu_cpu_barrier();
    } else {
        ocu_cpu_warp_sync();
    }
}

/* Warp sums, then a block barrier, then lane 0s combine: two-level sum. */
static void k_two_level(void **a)
{
    const uint32_t *in = *(const uint32_t **)a[0];
    uint64_t *out = *(uint64_t **)a[1];
    uint64_t *partial = ocu_cpu_dynamic_shared();
    uint32_t t = ocu_cpu_thread_idx().x, n = ocu_cpu_block_dim().x;
    uint32_t w = ocu_cpu_warp_size();
    uint64_t all[OCU_CPU_MAX_WARP];
    for (int rep = 0; rep < 4; ++rep) {
        uint32_t count = ocu_cpu_warp_gather(in[ocu_cpu_block_idx().x * n + t], all);
        if (t % w == 0) {
            uint64_t s = 0;
            for (uint32_t i = 0; i < count; ++i) s += all[i];
            partial[t / w] = s;
        }
        ocu_cpu_barrier();
        if (t == 0) {
            uint64_t s = 0;
            for (uint32_t k = 0; k < (n + w - 1) / w; ++k) s += partial[k];
            out[ocu_cpu_block_idx().x * 4 + (uint32_t)rep] = s;
        }
        ocu_cpu_barrier();
    }
}

static const ocu_cpu_kernel_entry g_entries[] = {
    {"gather", k_gather}, {"early_exit", k_early_exit},
    {"split", k_split}, {"two_level", k_two_level},
};
static const ocu_cpu_image g_image = {OCU_CPU_IMAGE_MAGIC, OCU_CPU_IMAGE_VERSION,
                                      4, g_entries};

static ocu_kernel *kernel(const char *name)
{
    ocu_kernel *k = NULL;
    B->module_get_kernel(M, name, &k);
    return k;
}

static ocu_launch_params params(uint32_t grid, uint32_t block, void **args,
                                const size_t *sizes, uint32_t n)
{
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = grid; p.grid[1] = 1; p.grid[2] = 1;
    p.block[0] = block; p.block[1] = 1; p.block[2] = 1;
    p.args = args; p.arg_sizes = sizes; p.num_args = n;
    return p;
}

static void test_gather(void)
{
    g_test = "gather_full_and_partial_warps";
    const uint32_t sizes_[] = {1, 6, 32, 64, 70, 200, 1024};
    for (size_t s = 0; s < sizeof(sizes_) / sizeof(sizes_[0]); ++s) {
        uint32_t n = sizes_[s];
        int *out = NULL;
        B->mem_alloc(D, 3 * n * sizeof(int), (void **)&out);
        memset(out, 0xff, 3 * n * sizeof(int));
        void *args[] = {&out};
        size_t sz[] = {sizeof(out)};
        ocu_launch_params p = params(3, n, args, sz, 1);
        CHECK(B->kernel_launch(NULL, kernel("gather"), &p) == OCU_OK);
        int bad = 0;
        for (uint32_t i = 0; i < 3 * n; ++i) bad += out[i] != 0;
        CHECK(bad == 0);
        B->mem_free(D, out);
    }
}

static void test_early_exit(void)
{
    g_test = "warp_ops_after_lanes_return";
    uint32_t n = 96;
    int *out = NULL;
    B->mem_alloc(D, 2 * n * sizeof(int), (void **)&out);
    memset(out, 0xff, 2 * n * sizeof(int));
    void *args[] = {&out};
    size_t sz[] = {sizeof(out)};
    ocu_launch_params p = params(2, n, args, sz, 1);
    CHECK(B->kernel_launch(NULL, kernel("early_exit"), &p) == OCU_OK);
    int bad = 0;
    for (uint32_t i = 0; i < 2 * n; ++i) bad += out[i] != 0;
    CHECK(bad == 0);
    B->mem_free(D, out);
}

static void test_split_is_reported(void)
{
    g_test = "split_barriers_fail_not_hang";
    void *args[] = {NULL};
    size_t sz[] = {0};
    ocu_launch_params p = params(1, 2, args, sz, 0);
    CHECK(B->kernel_launch(NULL, kernel("split"), &p) == OCU_ERR_LAUNCH_FAILED);
    ocu_queue *q = NULL;
    B->queue_create(D, &q);
    CHECK(B->kernel_launch(q, kernel("split"), &p) == OCU_OK);
    CHECK(B->queue_sync(q) == OCU_ERR_LAUNCH_FAILED);
    B->queue_destroy(q);
}

static void test_two_level(void)
{
    g_test = "warp_then_block_barrier";
    const uint32_t blocks = 5, n = 160;
    uint32_t *in = NULL;
    uint64_t *out = NULL;
    B->mem_alloc(D, blocks * n * sizeof(uint32_t), (void **)&in);
    B->mem_alloc(D, blocks * 4 * sizeof(uint64_t), (void **)&out);
    for (uint32_t i = 0; i < blocks * n; ++i) in[i] = i * 3u + 1u;
    void *args[] = {&in, &out};
    size_t sz[] = {sizeof(in), sizeof(out)};
    ocu_launch_params p = params(blocks, n, args, sz, 2);
    p.shared_memory_bytes = ((n + W - 1) / W) * sizeof(uint64_t);
    CHECK(B->kernel_launch(NULL, kernel("two_level"), &p) == OCU_OK);
    int bad = 0;
    for (uint32_t b = 0; b < blocks; ++b) {
        uint64_t want = 0;
        for (uint32_t t = 0; t < n; ++t) want += in[b * n + t];
        for (int r = 0; r < 4; ++r) bad += out[b * 4 + (uint32_t)r] != want;
    }
    CHECK(bad == 0);
    B->mem_free(D, in);
    B->mem_free(D, out);
}

int main(void)
{
    B = ocu_backend_cpu();
    if (B->dev_open(0, &D) != OCU_OK ||
        B->module_load(D, &g_image, sizeof(g_image), &M) != OCU_OK) {
        return 1;
    }
    ocu_dev_info info;
    B->dev_info(0, &info);
    W = info.warp_size;
    test_gather();
    test_early_exit();
    test_split_is_reported();
    test_two_level();
    B->module_unload(M);
    if (g_failures == 0) {
        printf("test_cpu_warp: ok (%d checks, warp %u)\n", g_checks, W);
        return 0;
    }
    fprintf(stderr, "test_cpu_warp: %d of %d checks failed\n", g_failures, g_checks);
    return 1;
}

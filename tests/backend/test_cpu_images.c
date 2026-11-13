/*
 * Hand-made module images for the CPU backend (docs/cpu-region-loops.md,
 * "Tests for an independent session" item 11; ocu_cpu_kernel.h): version 2
 * with no whole-block entry, version 3 (refused), a whole-block entry that
 * declines, one that runs the threads itself (fixed and random order), one
 * that illegally calls the block barrier (a launch failure, no hang), and a
 * version 2 image with a missing thread entry.
 *
 * Built only from the public header of the image format and the backend
 * table. Run as is and with OCU_CPU_ORDER=random.
 */
#define _POSIX_C_SOURCE 200809L

#include "ocu_backend.h"
#include "ocu_backend_cpu.h"
#include "ocu_cpu_kernel.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_failures, g_checks;
static const char *g_test = "";

#define CHECK(cond)                                                                  \
    do {                                                                             \
        ++g_checks;                                                                  \
        if (!(cond)) {                                                               \
            fprintf(stderr, "FAIL [%s] %s:%d: %s\n", g_test, __FILE__, __LINE__, #cond); \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)
#define CHECK_ST(expr, want)                                                         \
    do {                                                                             \
        ocu_status got_ = (expr);                                                    \
        ++g_checks;                                                                  \
        if (got_ != (want)) {                                                        \
            fprintf(stderr, "FAIL [%s] %s:%d: %s -> %s, want %s\n", g_test, __FILE__, __LINE__, \
                    #expr, ocu_status_name(got_), ocu_status_name(want));           \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)

static const ocu_backend *B;
static ocu_device *D;

/* One thread: out[linear thread id of the whole launch] = id + 1 + (grid-wide salt). */
static void k_write(void **a)
{
    uint32_t *out = *(uint32_t **)a[0];
    ocu_cpu_dim3 t = ocu_cpu_thread_idx(), b = ocu_cpu_block_idx(), d = ocu_cpu_block_dim(), g = ocu_cpu_grid_dim();
    uint32_t bs = d.x * d.y * d.z;
    uint32_t blk = (b.z * g.y + b.y) * g.x + b.x;
    uint32_t tl = (t.z * d.y + t.y) * d.x + t.x;
    out[blk * bs + tl] = blk * bs + tl + 1;
}

/* A correct whole-block entry for k_write: runs every thread of the block. */
static int block_run(void **a, const ocu_cpu_block_info *info)
{
    for (uint32_t i = 0; i < info->nthreads; ++i) {
        uint32_t lin = info->order ? info->order[i] : i;
        ocu_cpu_dim3 t;
        if (info->tids) {
            t = info->tids[lin];
        } else {
            t.x = lin % info->dim.x;
            t.y = (lin / info->dim.x) % info->dim.y;
            t.z = lin / (info->dim.x * info->dim.y);
        }
        ocu_cpu_tls_thread_idx = t;
        if (info->current) *info->current = lin;
        k_write(a);
    }
    return 0;
}
/* Declines before running anything. */
static int block_decline(void **a, const ocu_cpu_block_info *info)
{
    (void)a; (void)info;
    return 1;
}
/* Calls the block barrier: not allowed in a whole-block entry. */
static int block_barrier(void **a, const ocu_cpu_block_info *info)
{
    (void)a; (void)info;
    ocu_cpu_barrier();
    return 0;
}

static ocu_launch_params lp(uint32_t grid, uint32_t blockx, uint32_t blocky, void **args, const size_t *sizes)
{
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = grid; p.grid[1] = 1; p.grid[2] = 1;
    p.block[0] = blockx; p.block[1] = blocky; p.block[2] = 1;
    p.args = args;
    p.arg_sizes = sizes;
    p.num_args = 1;
    return p;
}

/* Load a version 2 image with one kernel and the given block entry, launch it
   over 3 blocks of bx*by threads, return the launch status and whether every
   thread wrote its value. */
static ocu_status run_image(uint32_t version, ocu_cpu_kernel_fn fn, ocu_cpu_block_fn block, uint32_t bx, uint32_t by,
                            int *all_ok)
{
    ocu_cpu_kernel_entry2 entry = {"k", fn, block};
    ocu_cpu_image img = {OCU_CPU_IMAGE_MAGIC, version, 1, (const ocu_cpu_kernel_entry *)&entry};
    ocu_module *m = NULL;
    ocu_status st = B->module_load(D, &img, sizeof(img), &m);
    if (st != OCU_OK) return st;
    ocu_kernel *k = NULL;
    CHECK_ST(B->module_get_kernel(m, "k", &k), OCU_OK);
    uint32_t total = 3 * bx * by;
    uint32_t *out = NULL;
    CHECK_ST(B->mem_alloc(D, total * sizeof(uint32_t), (void **)&out), OCU_OK);
    CHECK_ST(B->mem_set(D, NULL, out, 0, total * sizeof(uint32_t)), OCU_OK);
    void *args[] = {&out};
    size_t sizes[] = {sizeof(out)};
    ocu_launch_params p = lp(3, bx, by, args, sizes);
    st = B->kernel_launch(NULL, k, &p);
    uint32_t *host = (uint32_t *)malloc(total * sizeof(uint32_t));
    B->mem_copy(D, NULL, host, out, total * sizeof(uint32_t), OCU_COPY_DEVICE_TO_HOST);
    int ok = 1;
    for (uint32_t i = 0; i < total; ++i) ok &= host[i] == i + 1;
    if (all_ok) *all_ok = ok;
    free(host);
    B->mem_free(D, out);
    B->module_unload(m);
    return st;
}

int main(void)
{
    alarm(120);     /* a hang is a failure */
    B = ocu_backend_cpu();
    if (B->dev_open(0, &D) != OCU_OK) {
        fprintf(stderr, "setup failed\n");
        return 1;
    }
    int ok;

    g_test = "v2 without a block entry";
    CHECK_ST(run_image(OCU_CPU_IMAGE_VERSION_2, k_write, NULL, 37, 1, &ok), OCU_OK);
    CHECK(ok);

    g_test = "v2 with a block entry";
    CHECK_ST(run_image(OCU_CPU_IMAGE_VERSION_2, k_write, block_run, 37, 1, &ok), OCU_OK);
    CHECK(ok);
    CHECK_ST(run_image(OCU_CPU_IMAGE_VERSION_2, k_write, block_run, 5, 7, &ok), OCU_OK);   /* 2D block */
    CHECK(ok);
    CHECK_ST(run_image(OCU_CPU_IMAGE_VERSION_2, k_write, block_run, 1, 1, &ok), OCU_OK);
    CHECK(ok);
    CHECK_ST(run_image(OCU_CPU_IMAGE_VERSION_2, k_write, block_run, 1024, 1, &ok), OCU_OK);
    CHECK(ok);

    g_test = "a block entry that declines";
    CHECK_ST(run_image(OCU_CPU_IMAGE_VERSION_2, k_write, block_decline, 37, 1, &ok), OCU_OK);
    CHECK(ok);   /* the launch ran on fibers, every thread wrote */
    CHECK_ST(run_image(OCU_CPU_IMAGE_VERSION_2, k_write, block_decline, 4, 9, &ok), OCU_OK);
    CHECK(ok);

    g_test = "version 3";
    CHECK_ST(run_image(3u, k_write, block_run, 8, 1, NULL), OCU_ERR_INVALID_VALUE);
    CHECK_ST(run_image(0u, k_write, NULL, 8, 1, NULL), OCU_ERR_INVALID_VALUE);

    g_test = "v2 without the thread entry";
    CHECK_ST(run_image(OCU_CPU_IMAGE_VERSION_2, NULL, block_run, 8, 1, NULL), OCU_ERR_INVALID_VALUE);

    g_test = "a block entry that calls the block barrier";
    {
        int ok2 = 0;
        ocu_status st = run_image(OCU_CPU_IMAGE_VERSION_2, k_write, block_barrier, 16, 1, &ok2);
        CHECK(st != OCU_OK);    /* a launch failure, not a hang (alarm) and not silent success */
        fprintf(stderr, "  barrier in a block entry: status %s\n", ocu_status_name(st));
        /* the backend still works afterwards */
        CHECK_ST(run_image(OCU_CPU_IMAGE_VERSION_2, k_write, block_run, 16, 1, &ok), OCU_OK);
        CHECK(ok);
    }
    B->dev_close(D);
    if (g_failures == 0) {
        printf("test_cpu_images: ok (%d checks)\n", g_checks);
        return 0;
    }
    fprintf(stderr, "test_cpu_images: %d of %d checks failed\n", g_failures, g_checks);
    return 1;
}

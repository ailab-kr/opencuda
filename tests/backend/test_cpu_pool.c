/*
 * Many launches at once: host threads launching directly and through
 * queues, with different block shapes, so several grids share the worker
 * pool and threads switch between cached stack shapes. Every result is
 * checked. Run under TSan and ASan by tools/check.sh.
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

#define HOSTS 8
#define ROUNDS 60

static const ocu_backend *B;
static ocu_device *D;
static ocu_module *M;
static _Atomic int g_bad;

/* out[block] = sum over threads of (block * 1000 + thread), via a barrier. */
static void k_sum(void **a)
{
    uint64_t *out = *(uint64_t **)a[0];
    uint64_t *tile = ocu_cpu_dynamic_shared();
    uint32_t t = ocu_cpu_thread_idx().x, n = ocu_cpu_block_dim().x;
    uint32_t b = ocu_cpu_block_idx().x;
    tile[t] = (uint64_t)b * 1000u + t;
    ocu_cpu_barrier();
    if (t == 0) {
        uint64_t s = 0;
        for (uint32_t i = 0; i < n; ++i) s += tile[i];
        out[b] = s;
    }
}

static const ocu_cpu_kernel_entry g_entries[] = {{"sum", k_sum}};
static const ocu_cpu_image g_image = {OCU_CPU_IMAGE_MAGIC, OCU_CPU_IMAGE_VERSION, 1, g_entries};

static void *host(void *arg)
{
    long id = (long)arg;
    ocu_kernel *k = NULL;
    B->module_get_kernel(M, "sum", &k);
    ocu_queue *q = NULL;
    if (id % 2 == 1) B->queue_create(D, &q);
    for (int r = 0; r < ROUNDS; ++r) {
        uint32_t threads = 16u << ((id + r) % 4);  /* 16, 32, 64, 128 */
        uint32_t blocks = 3u + (uint32_t)((id * 7 + r) % 40);
        uint64_t *out = NULL;
        if (B->mem_alloc(D, blocks * sizeof(uint64_t), (void **)&out) != OCU_OK) {
            atomic_fetch_add(&g_bad, 1);
            continue;
        }
        void *args[] = {&out};
        size_t sizes[] = {sizeof(out)};
        ocu_launch_params p;
        memset(&p, 0, sizeof(p));
        p.grid[0] = blocks; p.grid[1] = p.grid[2] = 1;
        p.block[0] = threads; p.block[1] = p.block[2] = 1;
        p.shared_memory_bytes = threads * sizeof(uint64_t);
        p.args = args; p.arg_sizes = sizes; p.num_args = 1;
        ocu_status st = B->kernel_launch(q, k, &p);
        if (st == OCU_OK && q != NULL) st = B->queue_sync(q);
        if (st != OCU_OK) atomic_fetch_add(&g_bad, 1);
        for (uint32_t b = 0; b < blocks && st == OCU_OK; ++b) {
            uint64_t want = (uint64_t)b * 1000u * threads + (uint64_t)threads * (threads - 1) / 2;
            if (out[b] != want) atomic_fetch_add(&g_bad, 1);
        }
        B->mem_free(D, out);
    }
    if (q != NULL) B->queue_destroy(q);
    return NULL;
}

int main(void)
{
    B = ocu_backend_cpu();
    if (B->dev_open(0, &D) != OCU_OK ||
        B->module_load(D, &g_image, sizeof(g_image), &M) != OCU_OK) {
        return 1;
    }
    pthread_t t[HOSTS];
    for (long i = 0; i < HOSTS; ++i) pthread_create(&t[i], NULL, host, (void *)i);
    for (int i = 0; i < HOSTS; ++i) pthread_join(t[i], NULL);
    B->module_unload(M);
    int bad = atomic_load(&g_bad);
    printf("test_cpu_pool: %s (%d host threads x %d launches, %d bad)\n",
           bad ? "FAIL" : "ok", HOSTS, ROUNDS, bad);
    return bad ? 1 : 0;
}

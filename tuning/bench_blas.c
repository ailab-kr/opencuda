/*
 * bench_blas: time sgemm and saxpy on the active backend at three sizes,
 * by the measurement rules of PLAN 6.4 (tuning/ocu_bench.h). Prints one
 * JSON object per line. The CPU backend's numbers describe the reference
 * backend only; they are not a speed claim.
 *
 *   OCU_BACKEND=cpu bench_blas
 */
#include "ocu_bench.h"
#include "ocu_blas.h"
#include "ocu_core.h"

#include <stdio.h>
#include <string.h>

static const ocu_backend *B;
static ocu_device *D;

typedef struct job {
    const char *op;
    int m, n, k;
    float *a, *b, *c;
} job;

static int run(void *p)
{
    job *j = p;
    if (strcmp(j->op, "sgemm") == 0) {
        return ocu_blas_sgemm(NULL, OCU_BLAS_OP_N, OCU_BLAS_OP_N, j->m, j->n, j->k, 1.0f,
                              j->a, j->m, j->b, j->k, 0.0f, j->c, j->m);
    }
    return ocu_blas_saxpy(NULL, j->n, 0.5f, j->a, 1, j->c, 1);
}

static float *alloc_filled(size_t count)
{
    float *p = NULL;
    if (B->mem_alloc(D, count * sizeof(float), (void **)&p) != OCU_OK) return NULL;
    float *h = p; /* the CPU backend's memory is host-visible; others: copy */
    ocu_dev_info info;
    if (B->dev_info(0, &info) == OCU_OK && strcmp(B->name, "cpu") == 0) {
        for (size_t i = 0; i < count; ++i) h[i] = (float)(i % 17) * 0.25f;
    } else {
        B->mem_set(D, NULL, p, 0, count * sizeof(float));
    }
    return p;
}

int main(void)
{
    if (ocu_core_device(&B, &D) != OCU_OK) {
        fprintf(stderr, "bench_blas: no device\n");
        return 1;
    }
    ocu_bench_config cfg;
    ocu_bench_default_config(&cfg);
    const struct { const char *op; int m, n, k; } shapes[] = {
        {"saxpy", 0, 1 << 16, 0}, {"saxpy", 0, 1 << 20, 0}, {"saxpy", 0, 1 << 22, 0},
        {"sgemm", 64, 64, 64},    {"sgemm", 128, 128, 128}, {"sgemm", 256, 256, 256},
    };
    int failed = 0;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        job j = {shapes[s].op, shapes[s].m, shapes[s].n, shapes[s].k, NULL, NULL, NULL};
        size_t na = strcmp(j.op, "sgemm") == 0 ? (size_t)j.m * j.k : (size_t)j.n;
        size_t nb = strcmp(j.op, "sgemm") == 0 ? (size_t)j.k * j.n : 1;
        size_t nc = strcmp(j.op, "sgemm") == 0 ? (size_t)j.m * j.n : (size_t)j.n;
        j.a = alloc_filled(na);
        j.b = alloc_filled(nb);
        j.c = alloc_filled(nc);
        ocu_bench_result r;
        if (j.a == NULL || j.b == NULL || j.c == NULL ||
            ocu_bench_run(&cfg, run, &j, NULL, NULL, &r) != 0 || r.status != 0) {
            fprintf(stderr, "bench_blas: %s failed\n", j.op);
            failed = 1;
        } else {
            /* Timing relies on the backend contract: queue == NULL runs the
               call to completion before returning. */
            double flop = strcmp(j.op, "sgemm") == 0 ? 2.0 * j.m * j.n * j.k : 2.0 * j.n;
            double gflops = r.median_ms > 0 ? flop / (r.median_ms * 1e6) : 0.0;
            printf("{\"backend\": \"%s\", \"op\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, "
                   "\"median_ms\": %.4f, \"p10_ms\": %.4f, \"p90_ms\": %.4f, "
                   "\"spread\": %.4f, \"valid\": %s, \"runs\": %d, \"gflops\": %.3f}\n",
                   B->name, j.op, j.m, j.n, j.k, r.median_ms, r.p10_ms, r.p90_ms, r.spread,
                   r.valid ? "true" : "false", r.runs, gflops);
        }
        B->mem_free(D, j.a);
        B->mem_free(D, j.b);
        B->mem_free(D, j.c);
    }
    return failed;
}

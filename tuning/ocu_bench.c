/*
 * Timing by the measurement rules of PLAN 6.4. See ocu_bench.h.
 */
#define _POSIX_C_SOURCE 200809L

#include "ocu_bench.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_ms(void *unused)
{
    (void)unused;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

void ocu_bench_default_config(ocu_bench_config *cfg)
{
    cfg->warmup = 3;
    cfg->reps = 10;
    cfg->repeats = 3;
    cfg->max_spread = 0.05;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

double ocu_bench_percentile(double *values, int n, double fraction)
{
    if (n <= 0) {
        return 0.0;
    }
    qsort(values, (size_t)n, sizeof(*values), cmp_double);
    if (fraction <= 0) return values[0];
    if (fraction >= 1) return values[n - 1];
    double pos = fraction * (n - 1);
    int lo = (int)pos;
    int hi = lo + 1 < n ? lo + 1 : lo;
    return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}

int ocu_bench_run(const ocu_bench_config *cfg_in, ocu_bench_fn fn, void *ctx,
                  ocu_bench_clock clock, void *clock_ctx, ocu_bench_result *r)
{
    ocu_bench_config cfg;
    if (cfg_in != NULL) {
        cfg = *cfg_in;
    } else {
        ocu_bench_default_config(&cfg);
    }
    if (fn == NULL || r == NULL || cfg.reps < 1 || cfg.reps > OCU_BENCH_MAX_REPS ||
        cfg.repeats < 1 || cfg.repeats > OCU_BENCH_MAX_REPEATS || cfg.warmup < 0) {
        return -1;
    }
    if (clock == NULL) {
        clock = monotonic_ms;
    }
    memset(r, 0, sizeof(*r));
    double *all = malloc(sizeof(double) * (size_t)cfg.reps * (size_t)cfg.repeats);
    double *rep = malloc(sizeof(double) * (size_t)cfg.reps);
    if (all == NULL || rep == NULL) {
        free(all);
        free(rep);
        return -1;
    }
    for (int w = 0; w < cfg.warmup; ++w) {
        int st = fn(ctx);
        if (st != 0) {
            r->status = st;
            free(all);
            free(rep);
            return 0;
        }
    }
    int count = 0;
    for (int k = 0; k < cfg.repeats; ++k) {
        for (int i = 0; i < cfg.reps; ++i) {
            double t0 = clock(clock_ctx);
            int st = fn(ctx);
            double t1 = clock(clock_ctx);
            if (st != 0) {
                r->status = st;
                free(all);
                free(rep);
                return 0;
            }
            rep[i] = t1 - t0;
            all[count++] = t1 - t0;
        }
        r->sample_ms[k] = ocu_bench_percentile(rep, cfg.reps, 0.5);
    }
    double samples[OCU_BENCH_MAX_REPEATS];
    memcpy(samples, r->sample_ms, sizeof(double) * (size_t)cfg.repeats);
    r->median_ms = ocu_bench_percentile(samples, cfg.repeats, 0.5);
    double lo = samples[0], hi = samples[cfg.repeats - 1]; /* sorted above */
    r->spread = r->median_ms > 0 ? (hi - lo) / r->median_ms : 0.0;
    r->valid = r->spread <= cfg.max_spread;
    r->p10_ms = ocu_bench_percentile(all, count, 0.1);
    r->p90_ms = ocu_bench_percentile(all, count, 0.9);
    r->runs = count;
    free(all);
    free(rep);
    return 0;
}

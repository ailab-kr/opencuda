/*
 * ocu_bench.h -- timing by the measurement rules of PLAN 6.4.
 *
 *   - `warmup` untimed runs first (default 3);
 *   - then `reps` timed runs (default 10); their median is one sample;
 *   - the whole thing `repeats` times (default 3);
 *   - the result is the median of the samples, and it is valid only when
 *     the spread (max - min) / median of the samples is at most
 *     `max_spread` (default 0.05).
 *
 * The caller's run function does one complete unit of work, including any
 * wait for queued work, so that wall time covers it.
 */
#ifndef OCU_BENCH_H
#define OCU_BENCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OCU_BENCH_MAX_REPS 1000
#define OCU_BENCH_MAX_REPEATS 20

typedef struct ocu_bench_config {
    int    warmup;     /* default 3 */
    int    reps;       /* default 10 */
    int    repeats;    /* default 3 */
    double max_spread; /* default 0.05 */
} ocu_bench_config;

typedef struct ocu_bench_result {
    double median_ms;                     /* median of the per-repeat medians */
    double p10_ms, p90_ms;                /* over every timed run */
    double spread;                        /* (max - min) / median of the samples */
    int    valid;                         /* spread <= max_spread */
    int    runs;                          /* timed runs in total */
    double sample_ms[OCU_BENCH_MAX_REPEATS];
    int    status;                        /* 0, or the first non-zero run result */
} ocu_bench_result;

/* Returns 0 on success; a non-zero value stops the measurement. */
typedef int (*ocu_bench_fn)(void *ctx);

/* Clock in milliseconds; NULL means the monotonic clock. For tests. */
typedef double (*ocu_bench_clock)(void *clock_ctx);

void ocu_bench_default_config(ocu_bench_config *cfg);

/* Returns 0 when the measurement ran; result->valid says if it counts. */
int ocu_bench_run(const ocu_bench_config *cfg, ocu_bench_fn fn, void *ctx,
                  ocu_bench_clock clock, void *clock_ctx, ocu_bench_result *result);

/* Percentile of `n` values (linear interpolation), `values` is reordered. */
double ocu_bench_percentile(double *values, int n, double fraction);

#ifdef __cplusplus
}
#endif

#endif /* OCU_BENCH_H */

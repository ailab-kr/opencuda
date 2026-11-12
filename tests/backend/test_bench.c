/*
 * tuning/ocu_bench: the statistics and the validity rule, with a fake
 * clock so results are exact.
 */
#include "ocu_bench.h"

#include <math.h>
#include <stdio.h>

static int g_fail;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

/* Fake clock: each run advances time by the next duration in the table. */
typedef struct fake {
    const double *durations;
    int n, next, calls;
    double now;
    int in_run;
} fake;

static double fake_clock(void *p)
{
    fake *f = p;
    if (f->in_run) { /* end of a run */
        f->now += f->durations[f->next % f->n];
        f->next++;
    }
    f->in_run = !f->in_run;
    f->calls++;
    return f->now;
}

static int counter_runs;
static int run_ok(void *p) { (void)p; ++counter_runs; return 0; }
static int run_fails_third(void *p) { (void)p; return ++counter_runs == 3 ? 7 : 0; }

int main(void)
{
    ocu_bench_config cfg;
    ocu_bench_default_config(&cfg);
    CHECK(cfg.warmup == 3 && cfg.reps == 10 && cfg.repeats == 3 && cfg.max_spread == 0.05);

    /* Steady 2 ms runs with one 50 ms outlier per repeat: median ignores it. */
    double steady[10] = {2, 2, 2, 2, 50, 2, 2, 2, 2, 2};
    fake f = {steady, 10, 0, 0, 0, 0};
    ocu_bench_result r;
    counter_runs = 0;
    CHECK(ocu_bench_run(&cfg, run_ok, NULL, fake_clock, &f, &r) == 0);
    CHECK(counter_runs == 3 + 30);    /* warm-up runs are not timed */
    CHECK(r.runs == 30);
    CHECK(fabs(r.median_ms - 2.0) < 1e-12);
    CHECK(r.valid && r.spread == 0.0);
    CHECK(fabs(r.p10_ms - 2.0) < 1e-12);
    CHECK(r.p90_ms > 2.0);             /* the outliers show in the tail */

    /* Repeats that drift by 20 %: measured, but not valid. */
    double drift[30];
    for (int i = 0; i < 30; ++i) drift[i] = i < 10 ? 10.0 : (i < 20 ? 11.0 : 12.0);
    fake g = {drift, 30, 0, 0, 0, 0};
    CHECK(ocu_bench_run(&cfg, run_ok, NULL, fake_clock, &g, &r) == 0);
    CHECK(fabs(r.median_ms - 11.0) < 1e-12);
    CHECK(fabs(r.spread - 2.0 / 11.0) < 1e-12);
    CHECK(!r.valid);

    /* A failing run stops the measurement and reports its status. */
    counter_runs = 0;
    fake h = {steady, 10, 0, 0, 0, 0};
    CHECK(ocu_bench_run(&cfg, run_fails_third, NULL, fake_clock, &h, &r) == 0);
    CHECK(r.status == 7 && r.runs == 0);

    /* Bad configurations are refused. */
    ocu_bench_config bad = cfg;
    bad.reps = 0;
    CHECK(ocu_bench_run(&bad, run_ok, NULL, fake_clock, &f, &r) != 0);
    bad = cfg;
    bad.repeats = OCU_BENCH_MAX_REPEATS + 1;
    CHECK(ocu_bench_run(&bad, run_ok, NULL, fake_clock, &f, &r) != 0);

    double v[5] = {5, 1, 4, 2, 3};
    CHECK(ocu_bench_percentile(v, 5, 0.5) == 3);
    CHECK(fabs(ocu_bench_percentile(v, 5, 0.1) - 1.4) < 1e-12);

    printf("test_bench: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail ? 1 : 0;
}

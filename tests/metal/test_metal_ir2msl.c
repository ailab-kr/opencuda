/* Runs the kernels of ir2msl_trial.cu from an OCUMSL1 image (made by
   compiler/metal/ir2msl.py from LLVM IR) on the Metal backend and compares
   every result with the same computation on the host. */
#include "ocu_backend.h"
#include "ocu_backend_metal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == OCU_OK)

static float ref_hard(int kind, int clampit, int n, const float *x, const int *idx, int i)
{
    int ns = kind + 2, scale = kind == 0 ? 1 : 3;
    float acc = x[idx[i]];
    for (int s = 0; s < ns; ++s) {
        float v = x[(i + s) % n] * scale + 0.5f;
        acc = kind == 0 ? acc + v : acc * v;
        if (acc > 1e6f) break;
    }
    if (clampit) acc = acc < -100.0f ? -100.0f : (acc > 100.0f ? 100.0f : acc);
    switch (i & 3) { case 0: acc += 1.0f; break; case 1: acc -= 1.0f; break; case 2: acc = -acc; break; default: break; }
    return acc;
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    const ocu_backend *b = ocu_backend_metal();
    uint32_t nd = 0;
    OK(b->dev_count(&nd));
    if (nd == 0) { printf("metal_ir2msl: SKIP (no Metal 3 device)\n"); return 0; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *img = malloc((size_t)len);
    if (fread(img, 1, (size_t)len, f) != (size_t)len) return 1;
    fclose(f);
    ocu_device *d = NULL;
    OK(b->dev_open(0, &d));
    ocu_module *m = NULL;
    OK(b->module_load(d, img, (size_t)len, &m));
    if (!m) { printf("the image did not compile (OCU_METAL_VERBOSE=1 shows the Metal errors)\n"); return 1; }

    enum { N = 100000, BLOCKS = 64, THREADS = 256 };
    float *hx = malloc(N * sizeof(float)), *hy = malloc(N * sizeof(float));
    int *hi = malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) { hx[i] = (float)((i * 37) % 1001) * 0.01f - 5.0f; hi[i] = (i * 7919) % N; }
    void *dx = NULL, *dy = NULL, *di = NULL, *dout = NULL;
    OK(b->mem_alloc(d, N * sizeof(float), &dx));
    OK(b->mem_alloc(d, N * sizeof(float), &dy));
    OK(b->mem_alloc(d, N * sizeof(int), &di));
    OK(b->mem_alloc(d, BLOCKS * sizeof(float), &dout));
    OK(b->mem_copy(d, NULL, dx, hx, N * sizeof(float), OCU_COPY_HOST_TO_DEVICE));
    OK(b->mem_copy(d, NULL, di, hi, N * sizeof(int), OCU_COPY_HOST_TO_DEVICE));
    int n = N, bad;
    ocu_kernel *k = NULL;

    /* saxpy */
    for (int i = 0; i < N; ++i) hy[i] = 2.0f;
    OK(b->mem_copy(d, NULL, dy, hy, N * sizeof(float), OCU_COPY_HOST_TO_DEVICE));
    OK(b->module_get_kernel(m, "_Z7k_saxpyifPKfPf", &k));
    float a = 1.5f;
    { void *args[] = {&n, &a, &dx, &dy}; size_t sz[] = {4, 4, 8, 8};
      ocu_launch_params p = {{(N + 127) / 128, 1, 1}, {128, 1, 1}, 0, args, sz, 4};
      OK(b->kernel_launch(NULL, k, &p)); }
    OK(b->mem_copy(d, NULL, hy, dy, N * sizeof(float), OCU_COPY_DEVICE_TO_HOST));
    bad = 0;
    for (int i = 0; i < N; ++i) bad += hy[i] != 1.5f * hx[i] + 2.0f;
    CHECK(bad == 0);
    printf("k_saxpy            %s (%d of %d differ)\n", bad ? "WRONG" : "ok", bad, N);

    /* k_hard, two instances */
    const char *names[2] = {"_Z6k_hardILi0ELb0EEviPKfPKiPf", "_Z6k_hardILi1ELb1EEviPKfPKiPf"};
    for (int v = 0; v < 2; ++v) {
        memset(hy, 0, N * sizeof(float));
        OK(b->mem_copy(d, NULL, dy, hy, N * sizeof(float), OCU_COPY_HOST_TO_DEVICE));
        k = NULL;
        OK(b->module_get_kernel(m, names[v], &k));
        void *args[] = {&n, &dx, &di, &dy}; size_t sz[] = {4, 8, 8, 8};
        ocu_launch_params p = {{(N + 127) / 128, 1, 1}, {128, 1, 1}, 0, args, sz, 4};
        OK(b->kernel_launch(NULL, k, &p));
        OK(b->mem_copy(d, NULL, hy, dy, N * sizeof(float), OCU_COPY_DEVICE_TO_HOST));
        bad = 0;
        float worst = 0;
        for (int i = 0; i < N; ++i) {
            float want = ref_hard(v, v, N, hx, hi, i);
            /* error against the size of the terms: sums that cancel make the error relative to the result meaningless */
            if (hy[i] != want) { ++bad; float e = fabsf(hy[i] - want) / (fabsf(want) + 10.0f); if (e > worst) worst = e; }
        }
        /* Metal compiles with fast math by default (sums may be reassociated): a few ulp of the terms */
        CHECK(worst <= 4e-7f);
        printf("k_hard<%d,%s>   %s (%d of %d not bit-equal, worst error relative to the terms %.2g)\n", v, v ? "true " : "false",
               worst <= 4e-7f ? "ok" : "WRONG", bad, N, worst);
    }

    /* block sum: shared memory, barrier, warp shuffles */
    k = NULL;
    OK(b->module_get_kernel(m, "_Z11k_block_sumPKfPf", &k));
    { void *args[] = {&dx, &dout}; size_t sz[] = {8, 8};
      ocu_launch_params p = {{BLOCKS, 1, 1}, {THREADS, 1, 1}, 0, args, sz, 2};
      OK(b->kernel_launch(NULL, k, &p)); }
    float hout[BLOCKS];
    OK(b->mem_copy(d, NULL, hout, dout, sizeof hout, OCU_COPY_DEVICE_TO_HOST));
    bad = 0;
    for (int blk = 0; blk < BLOCKS; ++blk) {
        double want = 0;
        for (int t = 0; t < THREADS; ++t) want += hx[blk * THREADS + t];
        bad += fabs((double)hout[blk] - want) > 1e-3 * (fabs(want) + 1.0);
    }
    CHECK(bad == 0);
    printf("k_block_sum        %s (%d of %d blocks wrong)\n", bad ? "WRONG" : "ok", bad, BLOCKS);

    printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

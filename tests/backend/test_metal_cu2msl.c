/*
 * CUDA kernels translated by compiler/metal/cu2msl.py, run on a real Apple
 * GPU through the Metal backend: compiler/probe/probe_kernel.cu (saxpy and
 * block_sum), checked against host computations.
 *   test_metal_cu2msl <image made by cu2msl.py>
 * Prints SKIP and exits 0 when no Metal device is usable.
 */
#include "ocu_backend.h"
#include "ocu_backend_metal.h"

#include <stdio.h>
#include <stdlib.h>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == OCU_OK)

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    const ocu_backend *b = ocu_backend_metal();
    uint32_t n = 0;
    OK(b->dev_count(&n));
    if (n == 0) {
        printf("test_metal_cu2msl: SKIP (no Metal 3 device)\n");
        return 0;
    }
    FILE *f = fopen(argv[1], "rb");
    if (f == NULL) return 1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *img = malloc((size_t)len);
    if (img == NULL || fread(img, 1, (size_t)len, f) != (size_t)len) return 1;
    fclose(f);

    ocu_device *d = NULL;
    OK(b->dev_open(0, &d));
    ocu_module *m = NULL;
    OK(b->module_load(d, img, (size_t)len, &m));
    if (m == NULL) {
        printf("test_metal_cu2msl: FAIL (the translated MSL did not compile; rerun with OCU_METAL_VERBOSE=1)\n");
        return 1;
    }
    ocu_kernel *saxpy = NULL, *bsum = NULL;
    OK(b->module_get_kernel(m, "_Z5saxpyifPKfPf", &saxpy));
    OK(b->module_get_kernel(m, "_Z9block_sumPKfPf", &bsum));
    uint32_t np = 0;
    size_t ps[8];
    OK(b->kernel_arg_sizes(saxpy, &np, ps, 8));
    CHECK(np == 4);

    const int count = 100000;
    float *hx = malloc(count * sizeof(float)), *hy = malloc(count * sizeof(float));
    for (int i = 0; i < count; ++i) {
        hx[i] = (float)(i % 777) * 0.25f;
        hy[i] = 2.0f;
    }
    void *dx = NULL, *dy = NULL, *dout = NULL;
    OK(b->mem_alloc(d, count * sizeof(float), &dx));
    OK(b->mem_alloc(d, count * sizeof(float), &dy));
    OK(b->mem_copy(d, NULL, dx, hx, count * sizeof(float), OCU_COPY_HOST_TO_DEVICE));
    OK(b->mem_copy(d, NULL, dy, hy, count * sizeof(float), OCU_COPY_HOST_TO_DEVICE));
    int nn = count;
    float a = 1.5f;
    void *args[] = {&nn, &a, &dx, &dy};
    size_t sizes[] = {sizeof(nn), sizeof(a), sizeof(dx), sizeof(dy)};
    ocu_launch_params p = {{(uint32_t)((count + 127) / 128), 1, 1}, {128, 1, 1}, 0, args, sizes, 4};
    OK(b->kernel_launch(NULL, saxpy, &p));
    OK(b->mem_copy(d, NULL, hy, dy, count * sizeof(float), OCU_COPY_DEVICE_TO_HOST));
    int bad = 0;
    for (int i = 0; i < count; ++i) bad += hy[i] != 1.5f * ((float)(i % 777) * 0.25f) + 2.0f;
    CHECK(bad == 0);
    printf("translated saxpy: %s\n", bad ? "WRONG" : "ok");

    const int blocks = 32, threads = 256; /* block_sum uses a 256-element shared tile */
    OK(b->mem_alloc(d, blocks * sizeof(float), &dout));
    void *args2[] = {&dx, &dout};
    size_t sizes2[] = {sizeof(dx), sizeof(dout)};
    ocu_launch_params p2 = {{blocks, 1, 1}, {threads, 1, 1}, 0, args2, sizes2, 2};
    OK(b->kernel_launch(NULL, bsum, &p2));
    float hout[32];
    OK(b->mem_copy(d, NULL, hout, dout, sizeof(hout), OCU_COPY_DEVICE_TO_HOST));
    bad = 0;
    for (int blk = 0; blk < blocks; ++blk) {
        double want = 0;
        for (int t = 0; t < threads; ++t) want += hx[blk * threads + t];
        bad += (double)hout[blk] != want;
    }
    CHECK(bad == 0);
    printf("translated block_sum: %s\n", bad ? "WRONG" : "ok");

    OK(b->mem_free(d, dx));
    OK(b->mem_free(d, dy));
    OK(b->mem_free(d, dout));
    OK(b->module_unload(m));
    OK(b->dev_close(d));
    printf("test_metal_cu2msl: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

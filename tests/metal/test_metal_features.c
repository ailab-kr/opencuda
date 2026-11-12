/*
 * cu2msl translator features on a real Apple GPU, through the Metal
 * backend: tests/metal/cu2msl_features.cu translated by
 * compiler/metal/cu2msl.py, every kernel checked against a host
 * computation. Values are small integers and quarters so that results are
 * exact whatever the GPU's floating-point contraction.
 *   test_metal_features <image made by cu2msl.py>
 * Prints SKIP and exits 0 when no Metal device is usable.
 */
#include "ocu_backend.h"
#ifndef TEST_BACKEND
/* tests/opencl/test_opencl_features.c runs these checks on another backend */
#include "ocu_backend_metal.h"
#define TEST_BACKEND ocu_backend_metal
#define TEST_NAME "test_metal_features"
#define TEST_SKIP_CODE 0
#define TEST_SKIP_WHY "no Metal 3 device"
#define TEST_BUILD_HINT "the translated MSL did not compile; rerun with OCU_METAL_VERBOSE=1"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == OCU_OK)

static const ocu_backend *b;
static ocu_device *d;

static void *dev_from(const void *host, size_t bytes)
{
    void *p = NULL;
    OK(b->mem_alloc(d, bytes, &p));
    OK(b->mem_copy(d, NULL, p, host, bytes, OCU_COPY_HOST_TO_DEVICE));
    return p;
}

static void to_host(void *host, void *dev, size_t bytes)
{
    OK(b->mem_copy(d, NULL, host, dev, bytes, OCU_COPY_DEVICE_TO_HOST));
}

static ocu_kernel *kernel(ocu_module *m, const char *name)
{
    ocu_kernel *k = NULL;
    OK(b->module_get_kernel(m, name, &k));
    if (k == NULL) printf("missing kernel %s\n", name);
    return k;
}

static void launch(ocu_kernel *k, uint32_t grid, uint32_t block, uint32_t shared, void **args,
                   size_t *sizes, uint32_t n)
{
    if (k == NULL) {
        CHECK(k != NULL);
        return;
    }
    ocu_launch_params p = {{grid, 1, 1}, {block, 1, 1}, shared, args, sizes, n};
    OK(b->kernel_launch(NULL, k, &p));
}

static void report(const char *what, int bad)
{
    CHECK(bad == 0);
    printf("%-28s %s\n", what, bad ? "WRONG" : "ok");
}

/* 1. device functions called with device, threadgroup and thread pointers,
      references, and a pointer-returning function */
static void test_spaces(ocu_module *m)
{
    enum { T = 64, B = 8, N = T * B };
    float in[N], out[N];
    for (int i = 0; i < N; ++i) in[i] = (float)(i % 37) * 0.25f;
    void *din = dev_from(in, sizeof(in)), *dout = dev_from(in, sizeof(in));
    void *args[] = {&din, &dout};
    size_t sizes[] = {8, 8};
    launch(kernel(m, "_Z6spacesPKfPf"), B, T, 0, args, sizes, 2);
    to_host(out, dout, sizeof(out));
    int bad = 0;
    for (int blk = 0; blk < B; ++blk) {
        for (int t = 0; t < T; ++t) {
            const float *p = in + blk * T + t;
            float r = 0.0f;
            if (t + 2 < T) r = (p[0] + p[1] + p[2]) + (p[0] + p[1] + p[2]) + 6.0f;
            r += 0.5f;
            float want = r + (p[0] + 1.0f);
            if (out[blk * T + t] != want) {
                if (bad < 3) printf("  spaces[%d] = %g, want %g\n", blk * T + t, out[blk * T + t], want);
                ++bad;
            }
        }
    }
    report("address-space overloads", bad);
    OK(b->mem_free(d, din));
    OK(b->mem_free(d, dout));
}

/* 2. kernel template instantiations and a device function template */
static void test_templates(ocu_module *m)
{
    enum { N = 1024 };
    float f[N], fo[N];
    int k[N], ko[N];
    for (int i = 0; i < N; ++i) {
        f[i] = (float)(i % 50) * 0.25f;
        k[i] = i % 300 - 20;
    }
    void *df = dev_from(f, sizeof(f)), *dk = dev_from(k, sizeof(k));
    float ff = 1.5f;
    int fk = 3;
    void *a1[] = {&df, &ff};
    size_t s1[] = {8, 4};
    launch(kernel(m, "_Z7scale_nIfLi1000ELb1EEvPT_S0_"), 4, 256, 0, a1, s1, 2);
    void *a2[] = {&dk, &fk};
    launch(kernel(m, "_Z7scale_nIiLi500ELb0EEvPT_S0_"), 4, 256, 0, a2, s1, 2);
    to_host(fo, df, sizeof(fo));
    to_host(ko, dk, sizeof(ko));
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        float wf = i < 1000 ? (f[i] * 1.5f) * 2.0f : f[i];
        int wk = i < 500 ? k[i] * 3 : k[i];
        bad += fo[i] != wf;
        bad += ko[i] != wk;
    }
    report("kernel templates", bad);

    /* clamp_to<int> and clamp_to<float> */
    int n = N;
    void *a3[] = {&dk, &df, &n};
    size_t s3[] = {8, 8, 4};
    launch(kernel(m, "_Z9clamp_allPiPfi"), 4, 256, 0, a3, s3, 3);
    int kc[N];
    float fc[N];
    to_host(kc, dk, sizeof(kc));
    to_host(fc, df, sizeof(fc));
    bad = 0;
    for (int i = 0; i < N; ++i) {
        bad += kc[i] != (ko[i] > 100 ? 100 : ko[i]);
        bad += fc[i] != (fo[i] > 0.5f ? 0.5f : fo[i]);
    }
    report("device function templates", bad);
    OK(b->mem_free(d, df));
    OK(b->mem_free(d, dk));
}

/* 3. __constant__ / __device__ variables: initial values, host copies
      through module_get_global, atomics on a __device__ variable */
static void test_globals(ocu_module *m)
{
    if (b->module_get_global == NULL) {
        CHECK(b->module_get_global != NULL);
        return;
    }
    void *gw = NULL, *go = NULL, *gc = NULL, *gt = NULL, *gx = NULL;
    size_t bw = 0, bo = 0, bc = 0, bt = 0, bx = 0;
    OK(b->module_get_global(m, "c_weights", &gw, &bw));
    OK(b->module_get_global(m, "c_offset", &go, &bo));
    OK(b->module_get_global(m, "d_counter", &gc, &bc));
    OK(b->module_get_global(m, "d_table", &gt, &bt));
    CHECK(b->module_get_global(m, "no_such_variable", &gx, &bx) == OCU_ERR_NOT_FOUND);
    CHECK(bw == 16 && bo == 4 && bc == 4 && bt == 32);
    float w[4] = {0};
    int off = 0;
    unsigned cnt = 1;
    if (gw != NULL) to_host(w, gw, sizeof(w));
    if (go != NULL) to_host(&off, go, sizeof(off));
    if (gc != NULL) to_host(&cnt, gc, sizeof(cnt));
    CHECK(w[0] == 0.5f && w[1] == -1.0f && w[2] == 2.0f && w[3] == 0.0f);
    CHECK(off == 7);
    CHECK(cnt == 0);
    float table[8];
    for (int i = 0; i < 8; ++i) table[i] = (float)i * 0.5f;
    if (gt != NULL) OK(b->mem_copy(d, NULL, gt, table, sizeof(table), OCU_COPY_HOST_TO_DEVICE));
    int newoff = -2;
    if (go != NULL) OK(b->mem_copy(d, NULL, go, &newoff, sizeof(newoff), OCU_COPY_HOST_TO_DEVICE));

    enum { N = 1000 };
    float in[N], out[N];
    unsigned positive = 0;
    for (int i = 0; i < N; ++i) {
        in[i] = (float)(i % 9 - 4) * 0.25f;
        positive += in[i] > 0.0f;
    }
    void *din = dev_from(in, sizeof(in)), *dout = dev_from(in, sizeof(in));
    int n = N;
    void *args[] = {&din, &dout, &n};
    size_t sizes[] = {8, 8, 4};
    launch(kernel(m, "_Z11use_globalsPKfPfi"), (N + 127) / 128, 128, 0, args, sizes, 3);
    to_host(out, dout, sizeof(out));
    if (gc != NULL) to_host(&cnt, gc, sizeof(cnt));
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        float want = in[i] * w[i % 4] + (float)newoff + table[i % 8];
        bad += out[i] != want;
    }
    bad += cnt != positive;
    report("__constant__/__device__ variables", bad);
    OK(b->mem_free(d, din));
    OK(b->mem_free(d, dout));
}

/* 4. struct arguments with mixed field sizes, packed as the host packs them */
typedef struct {
    uint64_t src, dst; /* device pointers */
    char mode;
    short tag;
    float scale;
    float shift[2];
} Params; /* 32 bytes, as in cu2msl_features.cuh */
typedef struct {
    float a;
    unsigned char b;
} Pair; /* 8 bytes */

static void test_structs(ocu_module *m)
{
    enum { N = 300 };
    float src[N], dst[N];
    for (int i = 0; i < N; ++i) src[i] = (float)(i % 11);
    void *dsrc = dev_from(src, sizeof(src)), *ddst = dev_from(src, sizeof(src));
    ocu_kernel *k = kernel(m, "_Z11with_struct6Paramsi4Pair");
    if (k != NULL) {
        uint32_t np = 0;
        size_t ps[4] = {0};
        OK(b->kernel_arg_sizes(k, &np, ps, 4));
        CHECK(np == 3 && ps[0] == sizeof(Params) && ps[1] == 4 && ps[2] == sizeof(Pair));
    }
    for (int mode = 1; mode <= 2; ++mode) {
        Params p;
        memset(&p, 0, sizeof(p));
        p.src = (uint64_t)(uintptr_t)dsrc;
        p.dst = (uint64_t)(uintptr_t)ddst;
        p.mode = (char)mode;
        p.tag = 3;
        p.scale = 2.0f;
        p.shift[0] = 100.0f;
        p.shift[1] = 0.25f;
        Pair q = {1.5f, 200};
        int n = N;
        void *args[] = {&p, &n, &q};
        size_t sizes[] = {sizeof(p), sizeof(n), sizeof(q)};
        launch(k, (N + 63) / 64, 64, 0, args, sizes, 3);
        to_host(dst, ddst, sizeof(dst));
        int bad = 0;
        for (int i = 0; i < N; ++i) {
            float v = src[i];
            if (mode == 1) v += 2.0f * 3 + 0.25f;
            else v *= 2.0f;
            float want = v + 1.5f + 200.0f + 3 + 1;
            bad += dst[i] != want;
        }
        report(mode == 1 ? "struct arguments (add)" : "struct arguments (mul)", bad);
    }
    OK(b->mem_free(d, dsrc));
    OK(b->mem_free(d, ddst));
}

/* 5. extern __shared__ reached through a device function */
static void test_dynamic_shared(ocu_module *m)
{
    enum { T = 128, B = 6, N = T * B };
    float in[N], out[N];
    for (int i = 0; i < N; ++i) in[i] = (float)i;
    void *din = dev_from(in, sizeof(in)), *dout = dev_from(in, sizeof(in));
    void *args[] = {&din, &dout};
    size_t sizes[] = {8, 8};
    launch(kernel(m, "_Z13reverse_blockPKfPf"), B, T, T * sizeof(float), args, sizes, 2);
    to_host(out, dout, sizeof(out));
    int bad = 0;
    for (int blk = 0; blk < B; ++blk)
        for (int t = 0; t < T; ++t) bad += out[blk * T + t] != in[blk * T + T - 1 - t];
    report("dynamic shared memory", bad);
    OK(b->mem_free(d, din));
    OK(b->mem_free(d, dout));
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    b = TEST_BACKEND();
    uint32_t n = 0;
    OK(b->dev_count(&n));
    if (n == 0) {
        printf(TEST_NAME ": SKIP (" TEST_SKIP_WHY ")\n");
        return TEST_SKIP_CODE;
    }
    FILE *f = fopen(argv[1], "rb");
    if (f == NULL) return 1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *img = malloc((size_t)len);
    if (img == NULL || fread(img, 1, (size_t)len, f) != (size_t)len) return 1;
    fclose(f);

    OK(b->dev_open(0, &d));
    ocu_module *m = NULL;
    ocu_status lst = b->module_load(d, img, (size_t)len, &m);
    if (lst == OCU_ERR_NOT_SUPPORTED && TEST_SKIP_CODE != 0) {
        printf(TEST_NAME ": SKIP (the device takes no form of code this image has)\n");
        return TEST_SKIP_CODE;
    }
    CHECK(lst == OCU_OK);
    if (m == NULL) {
        printf(TEST_NAME ": FAIL (" TEST_BUILD_HINT ")\n");
        return 1;
    }
    test_spaces(m);
    test_templates(m);
    test_globals(m);
    test_structs(m);
    test_dynamic_shared(m);

    /* a second module from the same image has its own variables */
    ocu_module *m2 = NULL;
    OK(b->module_load(d, img, (size_t)len, &m2));
    if (m2 != NULL && b->module_get_global != NULL) {
        void *g1 = NULL, *g2 = NULL;
        size_t s1 = 0, s2 = 0;
        int off = 0;
        OK(b->module_get_global(m, "c_offset", &g1, &s1));
        OK(b->module_get_global(m2, "c_offset", &g2, &s2));
        CHECK(g1 != g2);
        if (g2 != NULL) to_host(&off, g2, sizeof(off));
        CHECK(off == 7);
        OK(b->module_unload(m2));
    }
    OK(b->module_unload(m));
    OK(b->dev_close(d));
    free(img);
    printf(TEST_NAME ": %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

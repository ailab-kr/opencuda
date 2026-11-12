/*
 * The OpenCL backend against the backend interface, with hand-written
 * OpenCL C kernels in the backend's image format (backends/opencl/README.md):
 * arguments come from a byte block at the offsets of a second block, and
 * pointers are plain shared-virtual-memory addresses. Results are checked
 * against host computations. Exits 77 (skip) when no OpenCL device is usable.
 */
#define _POSIX_C_SOURCE 200809L
#include "ocu_backend.h"
#include "ocu_backend_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == OCU_OK)

#define CLC(...) #__VA_ARGS__ "\n"
static const char k_source[] =
    "#define ARG(T, i) (*(__constant T *)(args + offs[i]))\n"
    "typedef __global float *gfloat;\n"
    "typedef __global int *gint;\n"
    "typedef __global gfloat *gtable;\n"
    "typedef __global uint *guint;\n"
    CLC(__kernel void saxpy(__constant uchar *args, __constant uint *offs, __local uchar *dyn)
    {
        int n = ARG(int, 0);
        float a = ARG(float, 1);
        gfloat x = ARG(gfloat, 2);
        gfloat y = ARG(gfloat, 3);
        int i = (int)get_global_id(0);
        if (i < n) y[i] = a * x[i] + y[i];
    })
    CLC(__kernel void block_sum(__constant uchar *args, __constant uint *offs, __local uchar *dyn)
    {
        gfloat in = ARG(gfloat, 0);
        gfloat out = ARG(gfloat, 1);
        __local float *tile = (__local float *)dyn;
        uint t = get_local_id(0), b = get_group_id(0), n = get_local_size(0);
        tile[t] = in[b * n + t];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = n / 2; s > 0; s /= 2) {
            if (t < s) tile[t] += tile[t + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (t == 0) out[b] = tile[0];
    })
    /* atomics on global memory from every thread of every block */
    CLC(__kernel void histogram(__constant uchar *args, __constant uint *offs, __local uchar *dyn)
    {
        gint in = ARG(gint, 0);
        int n = ARG(int, 1);
        guint bins = ARG(guint, 2);
        int i = (int)get_global_id(0);
        if (i < n) atomic_inc(&bins[in[i] & 15]);
    })
    /* pointers stored in device memory: an array of pointers, each into a
       different allocation (needs shared virtual memory) */
    CLC(__kernel void gather_ptrs(__constant uchar *args, __constant uint *offs, __local uchar *dyn)
    {
        gtable table = ARG(gtable, 0);
        gfloat out = ARG(gfloat, 1);
        uint i = get_global_id(0);
        out[i] = table[i & 3][i];
    })
    /* a 2D grid of 2D blocks */
    CLC(__kernel void index2d(__constant uchar *args, __constant uint *offs, __local uchar *dyn)
    {
        gint out = ARG(gint, 0);
        int w = ARG(int, 1);
        uint x = get_global_id(0), y = get_global_id(1);
        out[y * w + x] = (int)(get_group_id(1) * 1000 + get_local_id(0) * 10 + get_local_id(1));
    })
    /* the module's variables block (kernel argument 2) */
    CLC(__kernel void bump_global(__constant uchar *args, __constant uint *offs, __global uchar *glob,
                                  __local uchar *dyn)
    {
        __global int *counter = (__global int *)(glob + 8);
        if (get_global_id(0) == 0) counter[0] += ARG(int, 0);
    });

static char *make_image(size_t *len)
{
    char head[256];
    int n = snprintf(head, sizeof(head),
                     "OCUOCL1\n"
                     "param saxpy 4 4 8 8\n"
                     "param block_sum 8 8\n"
                     "global counter 8 4\n"
                     "init 8 07000000\n"
                     "clc %zu\n"
                     "---\n", sizeof(k_source) - 1);
    char *img = malloc((size_t)n + sizeof(k_source));
    memcpy(img, head, (size_t)n);
    memcpy(img + n, k_source, sizeof(k_source));
    *len = (size_t)n + sizeof(k_source) - 1;
    return img;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(void)
{
    const ocu_backend *b = ocu_backend_opencl();
    CHECK(b->abi == OCU_BACKEND_ABI && strcmp(b->name, "opencl") == 0);
    uint32_t n = 0;
    OK(b->dev_count(&n));
    if (n == 0) {
        printf("test_opencl_backend: SKIP (no usable OpenCL device)\n");
        return 77;
    }
    ocu_dev_info info;
    OK(b->dev_info(0, &info));
    printf("device: %s, %.1f GiB, local memory %llu bytes, sub-group (warp) %u, block %u\n", info.name,
           (double)info.total_memory_bytes / (1u << 30), (unsigned long long)info.max_shared_memory_per_block,
           info.warp_size, info.max_threads_per_block);
    CHECK(info.warp_size >= 1 && info.max_threads_per_block >= 256);
    CHECK(b->dev_info(n, &info) == OCU_ERR_INVALID_DEVICE);

    ocu_device *d = NULL;
    OK(b->dev_open(0, &d));
    size_t img_len = 0;
    char *img = make_image(&img_len);
    ocu_module *m = NULL;
    OK(b->module_load(d, img, img_len, &m));
    CHECK(b->module_load(d, "OCUOCL1\nclc 9\n---\nnot code!", 27, &(ocu_module *){NULL}) != OCU_OK);
    CHECK(b->module_load(d, "OCUMSL1\n---\n", 12, &(ocu_module *){NULL}) == OCU_ERR_INVALID_VALUE);
    CHECK(b->module_load(d, "OCUOCL1\nclc 999\n---\nx", 21, &(ocu_module *){NULL}) == OCU_ERR_INVALID_VALUE);
    CHECK(b->module_load(d, "OCUOCL1\n---\n", 12, &(ocu_module *){NULL}) == OCU_ERR_NOT_SUPPORTED);
    ocu_kernel *saxpy = NULL, *bsum = NULL, *hist = NULL, *gather = NULL, *idx2 = NULL, *bump = NULL;
    OK(b->module_get_kernel(m, "saxpy", &saxpy));
    OK(b->module_get_kernel(m, "block_sum", &bsum));
    OK(b->module_get_kernel(m, "histogram", &hist));
    OK(b->module_get_kernel(m, "gather_ptrs", &gather));
    OK(b->module_get_kernel(m, "index2d", &idx2));
    OK(b->module_get_kernel(m, "bump_global", &bump));
    CHECK(b->module_get_kernel(m, "missing", &(ocu_kernel *){NULL}) == OCU_ERR_NOT_FOUND);
    uint32_t np = 0;
    size_t ps[8];
    OK(b->kernel_arg_sizes(saxpy, &np, ps, 8));
    CHECK(np == 4 && ps[0] == 4 && ps[2] == 8);

    /* saxpy, timed with markers */
    const int count = 1 << 20;
    float *hx = malloc(count * sizeof(float)), *hy = malloc(count * sizeof(float));
    for (int i = 0; i < count; ++i) {
        hx[i] = (float)(i % 1000) * 0.5f;
        hy[i] = 1.0f;
    }
    void *dx = NULL, *dy = NULL;
    OK(b->mem_alloc(d, count * sizeof(float), &dx));
    OK(b->mem_alloc(d, count * sizeof(float), &dy));
    CHECK(((uintptr_t)dx & 255) == 0);
    ocu_queue *q = NULL;
    OK(b->queue_create(d, &q));
    OK(b->mem_copy(d, q, dx, hx, count * sizeof(float), OCU_COPY_HOST_TO_DEVICE));
    OK(b->mem_copy(d, q, dy, hy, count * sizeof(float), OCU_COPY_HOST_TO_DEVICE));
    ocu_marker *t0 = NULL, *t1 = NULL;
    OK(b->marker_create(d, &t0));
    OK(b->marker_create(d, &t1));
    OK(b->marker_query(t0));
    OK(b->marker_wait(t0));
    float ms = 0;
    CHECK(b->marker_elapsed_ms(t0, t1, &ms) == OCU_ERR_INVALID_HANDLE);
    int nn = count;
    float a = 3.0f;
    void *args[] = {&nn, &a, &dx, &dy};
    size_t sizes[] = {sizeof(nn), sizeof(a), sizeof(dx), sizeof(dy)};
    ocu_launch_params p = {{(uint32_t)((count + 255) / 256), 1, 1}, {256, 1, 1}, 0, args, sizes, 4};
    OK(b->marker_record(t0, q));
    OK(b->kernel_launch(q, saxpy, &p));
    OK(b->marker_record(t1, q));
    OK(b->mem_copy(d, q, hy, dy, count * sizeof(float), OCU_COPY_DEVICE_TO_HOST));
    OK(b->queue_sync(q));
    OK(b->marker_wait(t1));
    OK(b->marker_query(t1));
    ocu_status est = b->marker_elapsed_ms(t0, t1, &ms);
    CHECK(est == OCU_OK);
    int bad = 0;
    for (int i = 0; i < count; ++i) bad += hy[i] != 3.0f * ((float)(i % 1000) * 0.5f) + 1.0f;
    CHECK(bad == 0);
    printf("saxpy on %d values: %s, %.3f ms (markers)\n", count, bad ? "WRONG" : "ok", ms);

    /* launch speed: 50 synchronous saxpy launches, median and slowest 10% */
    double t[50];
    for (int r = 0; r < 50; ++r) {
        double s = now_ms();
        OK(b->kernel_launch(NULL, saxpy, &p));
        t[r] = now_ms() - s;
    }
    qsort(t, 50, sizeof(double), cmp_double);
    printf("saxpy 1M, launch to done: median %.3f ms, slow 10%% from %.3f ms, max %.3f ms\n", t[25], t[45],
           t[49]);

    /* pointer arithmetic: a pointer into the middle of an allocation; dynamic local memory */
    const int blocks = 64, threads = 256;
    void *dout = NULL;
    OK(b->mem_alloc(d, blocks * sizeof(float), &dout));
    OK(b->mem_copy(d, NULL, dx, hx, count * sizeof(float), OCU_COPY_HOST_TO_DEVICE));
    void *in_off = (char *)dx + 1024 * sizeof(float);
    void *args2[] = {&in_off, &dout};
    size_t sizes2[] = {sizeof(in_off), sizeof(dout)};
    ocu_launch_params p2 = {{blocks, 1, 1}, {threads, 1, 1}, threads * sizeof(float), args2, sizes2, 2};
    OK(b->kernel_launch(q, bsum, &p2));
    float hout[64];
    OK(b->mem_copy(d, q, hout, dout, sizeof(hout), OCU_COPY_DEVICE_TO_HOST));
    OK(b->queue_sync(q));
    bad = 0;
    for (int blk = 0; blk < blocks; ++blk) {
        double want = 0;
        for (int k = 0; k < threads; ++k) want += hx[1024 + blk * threads + k];
        bad += (double)hout[blk] != want;
    }
    CHECK(bad == 0);
    printf("block sums with local memory and barriers: %s (block 0: %.1f)\n", bad ? "WRONG" : "ok", hout[0]);

    /* atomics */
    const int hn = 100000;
    int *hin = malloc(hn * sizeof(int));
    unsigned want_bins[16] = {0}, got_bins[16];
    for (int i = 0; i < hn; ++i) {
        hin[i] = (i * 7919) % 1009;
        want_bins[hin[i] & 15]++;
    }
    void *din = NULL, *dbins = NULL;
    OK(b->mem_alloc(d, hn * sizeof(int), &din));
    OK(b->mem_alloc(d, sizeof(got_bins), &dbins));
    OK(b->mem_copy(d, NULL, din, hin, hn * sizeof(int), OCU_COPY_HOST_TO_DEVICE));
    OK(b->mem_set(d, NULL, dbins, 0, sizeof(got_bins)));
    int hn_arg = hn;
    void *args3[] = {&din, &hn_arg, &dbins};
    size_t sizes3[] = {8, 4, 8};
    ocu_launch_params p3 = {{(hn + 127) / 128, 1, 1}, {128, 1, 1}, 0, args3, sizes3, 3};
    OK(b->kernel_launch(NULL, hist, &p3));
    OK(b->mem_copy(d, NULL, got_bins, dbins, sizeof(got_bins), OCU_COPY_DEVICE_TO_HOST));
    CHECK(memcmp(got_bins, want_bins, sizeof(got_bins)) == 0);
    printf("histogram with global atomics: %s (bin 0: %u)\n",
           memcmp(got_bins, want_bins, sizeof(got_bins)) ? "WRONG" : "ok", got_bins[0]);

    /* pointers stored in device memory */
    void *parts[4], *dtable = NULL, *dgather = NULL;
    float hpart[256], hg[256];
    for (int k = 0; k < 4; ++k) {
        OK(b->mem_alloc(d, sizeof(hpart), &parts[k]));
        for (int i = 0; i < 256; ++i) hpart[i] = (float)(k * 1000 + i);
        OK(b->mem_copy(d, NULL, parts[k], hpart, sizeof(hpart), OCU_COPY_HOST_TO_DEVICE));
    }
    OK(b->mem_alloc(d, sizeof(parts), &dtable));
    OK(b->mem_alloc(d, sizeof(hg), &dgather));
    OK(b->mem_copy(d, NULL, dtable, parts, sizeof(parts), OCU_COPY_HOST_TO_DEVICE));
    void *args4[] = {&dtable, &dgather};
    size_t sizes4[] = {8, 8};
    ocu_launch_params p4 = {{4, 1, 1}, {64, 1, 1}, 0, args4, sizes4, 2};
    OK(b->kernel_launch(NULL, gather, &p4));
    OK(b->mem_copy(d, NULL, hg, dgather, sizeof(hg), OCU_COPY_DEVICE_TO_HOST));
    bad = 0;
    for (int i = 0; i < 256; ++i) bad += hg[i] != (float)((i & 3) * 1000 + i);
    CHECK(bad == 0);
    printf("pointers read from device memory: %s\n", bad ? "WRONG" : "ok");

    /* 2D grid */
    int w = 32, h = 16, hidx[32 * 16];
    void *didx = NULL;
    OK(b->mem_alloc(d, sizeof(hidx), &didx));
    void *args5[] = {&didx, &w};
    size_t sizes5[] = {8, 4};
    ocu_launch_params p5 = {{2, 4, 1}, {16, 4, 1}, 0, args5, sizes5, 2};
    OK(b->kernel_launch(NULL, idx2, &p5));
    OK(b->mem_copy(d, NULL, hidx, didx, sizeof(hidx), OCU_COPY_DEVICE_TO_HOST));
    bad = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) bad += hidx[y * w + x] != (y / 4) * 1000 + (x % 16) * 10 + (y % 4);
    CHECK(bad == 0);
    printf("2D grid of 2D blocks: %s\n", bad ? "WRONG" : "ok");

    /* module variables: initial value, a kernel that changes it, host access */
    void *gptr = NULL;
    size_t gbytes = 0;
    OK(b->module_get_global(m, "counter", &gptr, &gbytes));
    CHECK(gbytes == 4);
    CHECK(b->module_get_global(m, "nothing", &gptr, &gbytes) == OCU_ERR_NOT_FOUND);
    OK(b->module_get_global(m, "counter", &gptr, &gbytes));
    int gval = 0, inc = 5;
    OK(b->mem_copy(d, NULL, &gval, gptr, 4, OCU_COPY_DEVICE_TO_HOST));
    CHECK(gval == 7);
    void *args6[] = {&inc};
    size_t sizes6[] = {4};
    ocu_launch_params p6 = {{1, 1, 1}, {1, 1, 1}, 0, args6, sizes6, 1};
    OK(b->kernel_launch(NULL, bump, &p6));
    OK(b->mem_copy(d, NULL, &gval, gptr, 4, OCU_COPY_DEVICE_TO_HOST));
    CHECK(gval == 12);
    printf("module variable: initial 7, after kernel %d\n", gval);

    /* memory both sides use at one address */
    float *shared = NULL;
    ocu_status sst = b->mem_alloc_shared(d, 64 * sizeof(float), (void **)&shared);
    CHECK(sst == OCU_OK || sst == OCU_ERR_NOT_SUPPORTED);
    if (sst == OCU_OK) {
        for (int i = 0; i < 64; ++i) shared[i] = (float)i;
        void *args7[] = {&shared, &dout};
        size_t sizes7[] = {8, 8};
        ocu_launch_params p7 = {{1, 1, 1}, {64, 1, 1}, 64 * sizeof(float), args7, sizes7, 2};
        OK(b->kernel_launch(NULL, bsum, &p7));
        float s = 0;
        OK(b->mem_copy(d, NULL, &s, dout, sizeof(s), OCU_COPY_DEVICE_TO_HOST));
        CHECK(s == 2016.0f);
        OK(b->mem_free(d, shared));
        printf("shared (fine-grained) memory: %s\n", s == 2016.0f ? "ok" : "WRONG");
    }

    /* ranges, sets, bad launches */
    CHECK(b->mem_copy(d, q, (char *)dy + 4, hy, count * sizeof(float), OCU_COPY_HOST_TO_DEVICE) ==
          OCU_ERR_INVALID_VALUE);
    CHECK(b->mem_copy(d, q, hy, hx, 16, OCU_COPY_DEVICE_TO_HOST) == OCU_ERR_INVALID_VALUE);
    CHECK(b->mem_free(d, (char *)dx + 4) == OCU_ERR_INVALID_VALUE);
    OK(b->mem_set(d, NULL, dy, 0, count * sizeof(float)));
    OK(b->mem_copy(d, NULL, dy, dx, 1024 * sizeof(float), OCU_COPY_DEVICE_TO_DEVICE));
    OK(b->mem_copy(d, NULL, hy, dy, 2048 * sizeof(float), OCU_COPY_DEVICE_TO_HOST));
    bad = 0;
    for (int i = 0; i < 2048; ++i) bad += hy[i] != (i < 1024 ? hx[i] : 0.0f);
    CHECK(bad == 0);
    ocu_launch_params big = p;
    big.block[0] = 1u << 16;
    CHECK(b->kernel_launch(q, saxpy, &big) == OCU_ERR_INVALID_VALUE);
    ocu_launch_params zero = p;
    zero.grid[0] = 0;
    CHECK(b->kernel_launch(q, saxpy, &zero) == OCU_ERR_INVALID_VALUE);
    ocu_launch_params toolocal = p2;
    toolocal.shared_memory_bytes = info.max_shared_memory_per_block + 16;
    CHECK(b->kernel_launch(q, bsum, &toolocal) == OCU_ERR_INVALID_VALUE);
    OK(b->queue_sync(q));
    OK(b->queue_query(q));
    uint64_t fb = 0, tb = 0;
    OK(b->mem_info(d, &fb, &tb));
    CHECK(tb > 0 && fb <= tb);
    void *pinned = NULL;
    OK(b->host_alloc_pinned(d, 1 << 16, &pinned));
    OK(b->mem_copy(d, q, pinned, dx, 1 << 16, OCU_COPY_DEVICE_TO_HOST));
    OK(b->queue_sync(q));
    CHECK(memcmp(pinned, hx, 1 << 16) == 0);
    OK(b->host_free_pinned(d, pinned));

    OK(b->marker_destroy(t0));
    OK(b->marker_destroy(t1));
    OK(b->queue_destroy(q));
    OK(b->mem_free(d, dx));
    OK(b->mem_free(d, dy));
    OK(b->mem_free(d, dout));
    OK(b->mem_free(d, din));
    OK(b->mem_free(d, dbins));
    for (int k = 0; k < 4; ++k) OK(b->mem_free(d, parts[k]));
    OK(b->mem_free(d, dtable));
    OK(b->mem_free(d, dgather));
    OK(b->mem_free(d, didx));
    OK(b->module_unload(m));
    OK(b->dev_close(d));
    free(img);
    free(hx);
    free(hy);
    free(hin);
    printf("test_opencl_backend: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

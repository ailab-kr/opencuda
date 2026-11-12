/*
 * The Metal backend against the backend interface, on a real Apple GPU,
 * with hand-written MSL kernels in the backend's image format
 * (backends/metal/README.md). Results are checked against host
 * computations. Prints SKIP and exits 0 when no Metal device is usable.
 */
#include "ocu_backend.h"
#include "ocu_backend_metal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == OCU_OK)

/* Kernels read argument i at args + offs[i]; pointers are GPU addresses. */
static const char k_image[] =
    "OCUMSL1\n"
    "param saxpy 4 4 8 8\n"
    "param block_sum 8 8\n"
    "---\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct ocu_ptr_f { device float *p; };\n"
    "template <typename T> T ocu_arg(constant uchar *a, constant uint *o, uint i)\n"
    "{ return *(constant T *)(a + o[i]); }\n"
    "kernel void saxpy(constant uchar *args [[buffer(0)]], constant uint *offs [[buffer(1)]],\n"
    "                  uint3 tid [[thread_position_in_threadgroup]],\n"
    "                  uint3 bid [[threadgroup_position_in_grid]],\n"
    "                  uint3 bdim [[threads_per_threadgroup]])\n"
    "{\n"
    "    int n = ocu_arg<int>(args, offs, 0);\n"
    "    float a = ocu_arg<float>(args, offs, 1);\n"
    "    device float *x = ocu_arg<ocu_ptr_f>(args, offs, 2).p;\n"
    "    device float *y = ocu_arg<ocu_ptr_f>(args, offs, 3).p;\n"
    "    int i = int(bid.x * bdim.x + tid.x);\n"
    "    if (i < n) y[i] = a * x[i] + y[i];\n"
    "}\n"
    "kernel void block_sum(constant uchar *args [[buffer(0)]], constant uint *offs [[buffer(1)]],\n"
    "                      threadgroup float *tile [[threadgroup(0)]],\n"
    "                      uint3 tid [[thread_position_in_threadgroup]],\n"
    "                      uint3 bid [[threadgroup_position_in_grid]],\n"
    "                      uint3 bdim [[threads_per_threadgroup]])\n"
    "{\n"
    "    device float *in = ocu_arg<ocu_ptr_f>(args, offs, 0).p;\n"
    "    device float *out = ocu_arg<ocu_ptr_f>(args, offs, 1).p;\n"
    "    uint t = tid.x;\n"
    "    tile[t] = in[bid.x * bdim.x + t];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint s = bdim.x / 2; s > 0; s /= 2) {\n"
    "        if (t < s) tile[t] += tile[t + s];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (t == 0) out[bid.x] = tile[0];\n"
    "}\n";

int main(void)
{
    const ocu_backend *b = ocu_backend_metal();
    uint32_t n = 0;
    OK(b->dev_count(&n));
    if (n == 0) {
        printf("test_metal_backend: SKIP (no Metal 3 device)\n");
        return 0;
    }
    ocu_dev_info info;
    OK(b->dev_info(0, &info));
    printf("device: %s, working set %.1f GiB, threadgroup memory %llu bytes\n", info.name,
           (double)info.total_memory_bytes / (1u << 30),
           (unsigned long long)info.max_shared_memory_per_block);
    CHECK(info.warp_size == 32 && info.max_threads_per_block >= 256);
    CHECK(b->dev_info(n, &info) == OCU_ERR_INVALID_DEVICE);

    ocu_device *d = NULL;
    OK(b->dev_open(0, &d));
    ocu_module *m = NULL;
    OK(b->module_load(d, k_image, sizeof(k_image) - 1, &m));
    CHECK(b->module_load(d, "OCUMSL1\n---\nnot msl", 19, &(ocu_module *){NULL}) != OCU_OK);
    ocu_kernel *saxpy = NULL, *bsum = NULL;
    OK(b->module_get_kernel(m, "saxpy", &saxpy));
    OK(b->module_get_kernel(m, "block_sum", &bsum));
    CHECK(b->module_get_kernel(m, "missing", &(ocu_kernel *){NULL}) == OCU_ERR_NOT_FOUND);
    uint32_t np = 0;
    size_t ps[8];
    OK(b->kernel_arg_sizes(saxpy, &np, ps, 8));
    CHECK(np == 4 && ps[0] == 4 && ps[2] == 8);

    const int count = 1 << 20;
    float *hx = malloc(count * sizeof(float)), *hy = malloc(count * sizeof(float));
    for (int i = 0; i < count; ++i) {
        hx[i] = (float)(i % 1000) * 0.5f;
        hy[i] = 1.0f;
    }
    void *dx = NULL, *dy = NULL;
    OK(b->mem_alloc(d, count * sizeof(float), &dx));
    OK(b->mem_alloc(d, count * sizeof(float), &dy));
    ocu_queue *q = NULL;
    OK(b->queue_create(d, &q));
    OK(b->mem_copy(d, q, dx, hx, count * sizeof(float), OCU_COPY_HOST_TO_DEVICE));
    OK(b->mem_copy(d, q, dy, hy, count * sizeof(float), OCU_COPY_HOST_TO_DEVICE));
    ocu_marker *t0 = NULL, *t1 = NULL;
    OK(b->marker_create(d, &t0));
    OK(b->marker_create(d, &t1));
    OK(b->marker_query(t0));
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
    OK(b->marker_elapsed_ms(t0, t1, &ms));
    int bad = 0;
    for (int i = 0; i < count; ++i) bad += hy[i] != 3.0f * ((float)(i % 1000) * 0.5f) + 1.0f;
    CHECK(bad == 0);
    printf("saxpy on %d values: %s, %.3f ms\n", count, bad ? "WRONG" : "ok", ms);

    /* Pointer arithmetic on device addresses, dynamic threadgroup memory. */
    const int blocks = 64, threads = 256;
    void *dout = NULL;
    OK(b->mem_alloc(d, blocks * sizeof(float), &dout));
    void *in_off = (char *)dx + 1024 * sizeof(float); /* start inside the allocation */
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
        for (int t = 0; t < threads; ++t) want += hx[1024 + blk * threads + t];
        bad += (double)hout[blk] != want;
    }
    CHECK(bad == 0);
    printf("block sums with threadgroup memory: %s\n", bad ? "WRONG" : "ok");

    /* Ranges, sets, bad launches. */
    CHECK(b->mem_copy(d, q, (char *)dy + 4, hy, count * sizeof(float), OCU_COPY_HOST_TO_DEVICE) ==
          OCU_ERR_INVALID_VALUE);
    CHECK(b->mem_free(d, (char *)dx + 4) == OCU_ERR_INVALID_VALUE);
    OK(b->mem_set(d, NULL, dy, 0, count * sizeof(float)));
    OK(b->mem_copy(d, NULL, dy, dx, 1024 * sizeof(float), OCU_COPY_DEVICE_TO_DEVICE));
    OK(b->mem_copy(d, NULL, hy, dy, 2048 * sizeof(float), OCU_COPY_DEVICE_TO_HOST));
    bad = 0;
    for (int i = 0; i < 2048; ++i) bad += hy[i] != (i < 1024 ? hx[i] : 0.0f);
    CHECK(bad == 0);
    ocu_launch_params big = p;
    big.block[0] = 4096;
    CHECK(b->kernel_launch(q, saxpy, &big) == OCU_ERR_INVALID_VALUE);
    ocu_launch_params zero = p;
    zero.grid[0] = 0;
    CHECK(b->kernel_launch(q, saxpy, &zero) == OCU_ERR_INVALID_VALUE);
    OK(b->queue_query(q));
    uint64_t fb = 0, tb = 0;
    OK(b->mem_info(d, &fb, &tb));
    CHECK(tb > 0 && fb <= tb);

    OK(b->marker_destroy(t0));
    OK(b->marker_destroy(t1));
    OK(b->queue_destroy(q));
    OK(b->mem_free(d, dx));
    OK(b->mem_free(d, dy));
    OK(b->mem_free(d, dout));
    OK(b->module_unload(m));
    OK(b->dev_close(d));
    free(hx);
    free(hy);
    printf("test_metal_backend: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

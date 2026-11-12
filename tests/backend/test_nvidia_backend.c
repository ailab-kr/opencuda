/*
 * The NVIDIA backend against the backend interface, on a real GPU. The
 * device code is PTX that Clang made from compiler/probe/probe_kernel.cu
 * (path given as argv[1]); results are checked against values computed
 * here on the host. Prints SKIP and exits 0 when no NVIDIA device exists.
 */
#include "ocu_backend.h"
#include "ocu_backend_nvidia.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == OCU_OK)

static char *read_file(const char *path, size_t *bytes)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (buf != NULL && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    if (buf != NULL) {
        buf[n] = '\0';
        *bytes = (size_t)n;
    }
    return buf;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: test_nvidia_backend <probe_kernel.ptx>\n");
        return 2;
    }
    const ocu_backend *b = ocu_backend_nvidia();
    uint32_t n = 0;
    OK(b->dev_count(&n));
    if (n == 0) {
        printf("test_nvidia_backend: SKIP (no NVIDIA device or driver)\n");
        return 0;
    }
    ocu_dev_info info;
    OK(b->dev_info(0, &info));
    CHECK(info.warp_size == 32 && info.max_threads_per_block >= 1024);
    CHECK(info.total_memory_bytes > 0 && info.max_shared_memory_per_block >= 48 * 1024);
    CHECK(b->dev_info(n, &info) == OCU_ERR_INVALID_DEVICE);
    printf("device: %s, %.1f GiB\n", info.name, (double)info.total_memory_bytes / (1u << 30));

    ocu_device *d = NULL;
    OK(b->dev_open(0, &d));
    size_t ptx_bytes = 0;
    char *ptx = read_file(argv[1], &ptx_bytes);
    CHECK(ptx != NULL);
    if (ptx == NULL) return 1;
    ocu_module *m = NULL;
    OK(b->module_load(d, ptx, ptx_bytes, &m));
    CHECK(b->module_load(d, "not ptx", 7, &(ocu_module *){NULL}) != OCU_OK);
    ocu_kernel *saxpy = NULL;
    OK(b->module_get_kernel(m, "_Z5saxpyifPKfPf", &saxpy));
    CHECK(b->module_get_kernel(m, "no_such_kernel", &(ocu_kernel *){NULL}) == OCU_ERR_NOT_FOUND);

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
    OK(b->marker_query(t0)); /* never recorded: complete */
    float ms = 0;
    CHECK(b->marker_elapsed_ms(t0, t1, &ms) == OCU_ERR_INVALID_HANDLE);
    int nn = count;
    float a = 3.0f;
    void *args[] = {&nn, &a, &dx, &dy};
    size_t sizes[] = {sizeof(nn), sizeof(a), sizeof(dx), sizeof(dy)};
    ocu_launch_params p = {{(uint32_t)((count + 255) / 256), 1, 1}, {256, 1, 1}, 0,
                           args, sizes, 4};
    OK(b->marker_record(t0, q));
    OK(b->kernel_launch(q, saxpy, &p));
    OK(b->marker_record(t1, q));
    OK(b->mem_copy(d, q, hy, dy, count * sizeof(float), OCU_COPY_DEVICE_TO_HOST));
    OK(b->queue_sync(q));
    OK(b->marker_elapsed_ms(t0, t1, &ms));
    CHECK(ms >= 0.0f);
    int bad = 0;
    for (int i = 0; i < count; ++i) bad += hy[i] != 3.0f * ((float)(i % 1000) * 0.5f) + 1.0f;
    CHECK(bad == 0);
    printf("saxpy on %d values: %s, kernel %.3f ms\n", count, bad ? "WRONG" : "ok", ms);

    /* Ranges, sets, device-to-device, bad launches. */
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
    big.block[0] = 2048;
    CHECK(b->kernel_launch(q, saxpy, &big) == OCU_ERR_INVALID_VALUE);
    ocu_launch_params zero = p;
    zero.grid[0] = 0;
    CHECK(b->kernel_launch(q, saxpy, &zero) == OCU_ERR_INVALID_VALUE);
    OK(b->queue_query(q));

    /* Pinned host memory and info. */
    void *pin = NULL;
    OK(b->host_alloc_pinned(d, 1 << 16, &pin));
    memset(pin, 7, 1 << 16);
    OK(b->mem_copy(d, q, dx, pin, 1 << 16, OCU_COPY_HOST_TO_DEVICE));
    OK(b->queue_sync(q));
    OK(b->host_free_pinned(d, pin));
    uint64_t fb = 0, tb = 0;
    OK(b->mem_info(d, &fb, &tb));
    CHECK(tb > 0 && fb <= tb);

    OK(b->marker_destroy(t0));
    OK(b->marker_destroy(t1));
    OK(b->queue_destroy(q));
    OK(b->mem_free(d, dx));
    OK(b->mem_free(d, dy));
    OK(b->module_unload(m));
    OK(b->dev_close(d));
    free(ptx);
    free(hx);
    free(hy);
    printf("test_nvidia_backend: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

/*
 * test_npusim.c -- the npusim profile rules (docs/spec/op-table.md, section
 * 7), driven through the backend table and the operator table only.
 *
 * Usage: test_npusim <mode>. The environment of each mode is set by ctest;
 * the program checks that it got the environment its mode needs.
 *   basic      defaults (scratchpad 256 KB, tiling on, all dtypes)
 *   scratch4   OCU_NPUSIM_SCRATCH_KB=4
 *   scratch1   OCU_NPUSIM_SCRATCH_KB=1 (not even a 16x16x16 GEMM tile fits)
 *   tiling0    OCU_NPUSIM_TILING=0
 *   dtypes_f16 OCU_NPUSIM_DTYPES=f16
 *   env        OCU_NPUSIM_MEM_MB=8, OCU_NPUSIM_PLAN_MS=40
 */
#include "ops_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t g_scratch = 256u * 1024u; /* S in bytes */
static int      g_tiling = 1;
static int      g_allowed[4] = { 0, 1, 1, 1 }; /* indexed by f32 / f16 / bf16 */
static uint64_t g_mem_mb = 256;
static long     g_plan_ms = 0;

static const ocu_dtype FLOATS[3] = { OCU_DT_F32, OCU_DT_F16, OCU_DT_BF16 };
static const ocu_op OPS5[5] = { OCU_OP_GEMM, OCU_OP_ADD, OCU_OP_MUL, OCU_OP_RMS_NORM,
                                OCU_OP_SOFTMAX };

static int env_is(const char *name, const char *want)
{
    const char *v = getenv(name);
    if (want == NULL) {
        return v == NULL || v[0] == '\0';
    }
    return v != NULL && strcmp(v, want) == 0;
}

static void *dalloc(size_t bytes)
{
    void *p = NULL;
    if (!CHECK_ST(G.b->mem_alloc(G.dev, bytes, &p), OCU_OK)) {
        return NULL;
    }
    return p;
}

static void dfree(void *p)
{
    if (p) {
        CHECK_ST(G.b->mem_free(G.dev, p), OCU_OK);
    }
}

static ocu_op_desc d_simple(ocu_op op, ocu_dtype dt, int64_t rows, int64_t n)
{
    ocu_op_desc d = op_desc(op);
    switch (op) {
    case OCU_OP_GEMM:
        d.num_tensors = 3;
        d.tensors[0] = tdesc(dt, 2, SHP(rows, n), 0);
        d.tensors[1] = tdesc(dt, 2, SHP(n, 8), 0);
        d.tensors[2] = tdesc(OCU_DT_F32, 2, SHP(rows, 8), 0);
        d.p.gemm.alpha = 1.0f;
        break;
    case OCU_OP_ADD:
    case OCU_OP_MUL:
        d.num_tensors = 3;
        for (int i = 0; i < 3; ++i) {
            d.tensors[i] = tdesc(dt, 2, SHP(rows, n), 0);
        }
        break;
    case OCU_OP_RMS_NORM:
        d.num_tensors = 2;
        d.tensors[0] = d.tensors[1] = tdesc(dt, 2, SHP(rows, n), 0);
        d.p.rms_norm.eps = 1e-6f;
        break;
    default:
        d.num_tensors = 2;
        d.tensors[0] = d.tensors[1] = tdesc(dt, 2, SHP(rows, n), 0);
        d.p.softmax.scale = 1.0f;
        break;
    }
    return d;
}

static ocu_op_desc d_gemm(ocu_dtype in, ocu_dtype out, int64_t m, int64_t n, int64_t k)
{
    ocu_op_desc d = op_desc(OCU_OP_GEMM);
    d.num_tensors = 3;
    d.tensors[0] = tdesc(in, 2, SHP(m, k), 0);
    d.tensors[1] = tdesc(in, 2, SHP(k, n), 0);
    d.tensors[2] = tdesc(out, 2, SHP(m, n), 0);
    d.p.gemm.alpha = 1.0f;
    return d;
}

/* ---- rule 1: no kernels; dev_info ---------------------------------------- */

static void test_no_kernels(void)
{
    set_case("rule 1: module_load / module_get_kernel / kernel_launch");
    ocu_op_stats s0, s1;
    CHECK(get_stats(&s0));
    uint8_t image[64];
    memset(image, 0, sizeof image);
    ocu_module *m = NULL;
    CHECK_ST(G.b->module_load(G.dev, image, sizeof image, &m), OCU_ERR_NOT_SUPPORTED);
    ocu_kernel *k = NULL;
    CHECK_ST(G.b->module_get_kernel(m, "k", &k), OCU_ERR_NOT_SUPPORTED);
    ocu_launch_params lp;
    memset(&lp, 0, sizeof lp);
    lp.grid[0] = lp.grid[1] = lp.grid[2] = 1;
    lp.block[0] = lp.block[1] = lp.block[2] = 1;
    CHECK_ST(G.b->kernel_launch(G.q, NULL, &lp), OCU_ERR_NOT_SUPPORTED);
    CHECK_ST(G.b->kernel_launch(NULL, NULL, &lp), OCU_ERR_NOT_SUPPORTED);
    CHECK(get_stats(&s1));
    CHECK(s1.kernel_launches_refused == s0.kernel_launches_refused + 2);
    CHECK_ST(G.b->queue_sync(G.q), OCU_OK);
}

static void test_dev_info(void)
{
    set_case("rule 1 / 10: dev_info");
    ocu_dev_info info;
    memset(&info, 0, sizeof info);
    CHECK_ST(G.b->dev_info(0, &info), OCU_OK);
    CHECK(info.warp_size == 1);
    CHECK(info.max_threads_per_block == 1);
    CHECK(info.max_grid_dim[0] == 1 && info.max_grid_dim[1] == 1 && info.max_grid_dim[2] == 1);
    CHECK(info.supports_unified_memory == 0);
    CHECK(info.max_shared_memory_per_block == g_scratch);
    CHECK(info.total_memory_bytes == g_mem_mb * 1024u * 1024u);
    CHECK(info.name[0] != '\0');
    uint64_t fr = 0, tot = 0;
    CHECK_ST(G.b->mem_info(G.dev, &fr, &tot), OCU_OK);
    CHECK(tot == g_mem_mb * 1024u * 1024u);
    CHECK(fr <= tot);
}

/* ---- rule 3: separate memory space --------------------------------------- */

static void test_memory_rules(void)
{
    set_case("rule 3: mem_copy / mem_set direction and range checks");
    const size_t n = 256;
    uint8_t *d1 = dalloc(n), *d2 = dalloc(n), *d3 = dalloc(n);
    if (!d1 || !d2 || !d3) {
        return;
    }
    uint8_t host[512], back[256];
    for (size_t i = 0; i < sizeof host; ++i) {
        host[i] = (uint8_t)(i * 7u + 1u);
    }
    for (int qi = 0; qi < 2; ++qi) {
        ocu_queue *q = qi ? G.q : NULL;
        set_case("rule 3: direction checks (queue=%d)", qi);
        /* host -> device */
        CHECK_ST(G.b->mem_copy(G.dev, q, host, host + 256, 64, OCU_COPY_HOST_TO_DEVICE),
                 OCU_ERR_INVALID_VALUE); /* dst is not device memory */
        CHECK_ST(G.b->mem_copy(G.dev, q, d1, d2, 64, OCU_COPY_HOST_TO_DEVICE),
                 OCU_ERR_INVALID_VALUE); /* host-side src is a device address */
        /* device -> host */
        CHECK_ST(G.b->mem_copy(G.dev, q, back, host, 64, OCU_COPY_DEVICE_TO_HOST),
                 OCU_ERR_INVALID_VALUE); /* src is not device memory */
        CHECK_ST(G.b->mem_copy(G.dev, q, d1, d2, 64, OCU_COPY_DEVICE_TO_HOST),
                 OCU_ERR_INVALID_VALUE); /* host-side dst is a device address */
        /* device -> device */
        CHECK_ST(G.b->mem_copy(G.dev, q, d1, host, 64, OCU_COPY_DEVICE_TO_DEVICE),
                 OCU_ERR_INVALID_VALUE);
        CHECK_ST(G.b->mem_copy(G.dev, q, back, d1, 64, OCU_COPY_DEVICE_TO_DEVICE),
                 OCU_ERR_INVALID_VALUE);
        /* out of range */
        set_case("rule 3: range checks (queue=%d)", qi);
        CHECK_ST(G.b->mem_copy(G.dev, q, d1, host, n + 1, OCU_COPY_HOST_TO_DEVICE),
                 OCU_ERR_INVALID_VALUE);
        CHECK_ST(G.b->mem_copy(G.dev, q, d1 + 200, host, 64, OCU_COPY_HOST_TO_DEVICE),
                 OCU_ERR_INVALID_VALUE);
        CHECK_ST(G.b->mem_copy(G.dev, q, back, d1 + 255, 2, OCU_COPY_DEVICE_TO_HOST),
                 OCU_ERR_INVALID_VALUE);
        CHECK_ST(G.b->mem_copy(G.dev, q, d2, d1 + 128, 256, OCU_COPY_DEVICE_TO_DEVICE),
                 OCU_ERR_INVALID_VALUE);
        void *before = (void *)((uintptr_t)d1 - 64u);
        CHECK_ST(G.b->mem_copy(G.dev, q, before, host, 64, OCU_COPY_HOST_TO_DEVICE),
                 OCU_ERR_INVALID_VALUE);
        CHECK_ST(G.b->mem_set(G.dev, q, host, 0, 64), OCU_ERR_INVALID_VALUE);
        CHECK_ST(G.b->mem_set(G.dev, q, d1 + 1, 0, n), OCU_ERR_INVALID_VALUE);
        CHECK_ST(G.b->queue_sync(G.q), OCU_OK);
    }
    set_case("rule 3: freed allocation is not live");
    CHECK_ST(G.b->mem_free(G.dev, d3), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, NULL, d3, host, 64, OCU_COPY_HOST_TO_DEVICE),
             OCU_ERR_INVALID_VALUE);
    CHECK_ST(G.b->mem_copy(G.dev, NULL, back, d3, 64, OCU_COPY_DEVICE_TO_HOST),
             OCU_ERR_INVALID_VALUE);
    set_case("rule 3: valid round trip, mem_set, offsets inside an allocation");
    CHECK_ST(G.b->mem_copy(G.dev, NULL, d1, host, n, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, G.q, d2, d1, n, OCU_COPY_DEVICE_TO_DEVICE), OCU_OK);
    CHECK_ST(G.b->mem_set(G.dev, G.q, d2 + 10, 0xAB, 20), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, G.q, back, d2, n, OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK_ST(G.b->queue_sync(G.q), OCU_OK);
    int ok = 1;
    for (size_t i = 0; i < n; ++i) {
        uint8_t want = (i >= 10 && i < 30) ? 0xABu : host[i];
        if (back[i] != want) {
            ok = 0;
        }
    }
    CHECK(ok);
    CHECK_ST(G.b->mem_copy(G.dev, NULL, back, d1 + 192, 64, OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK(memcmp(back, host + 192, 64) == 0);
    dfree(d1);
    dfree(d2);
}

/* ---- rule 9: DMA counters ------------------------------------------------ */

static void test_dma_counters(void)
{
    set_case("rule 9: DMA byte counters by direction");
    uint8_t *d1 = dalloc(1024), *d2 = dalloc(1024);
    if (!d1 || !d2) {
        return;
    }
    uint8_t host[1024];
    memset(host, 3, sizeof host);
    ocu_op_stats s0, s1, s2;
    CHECK(get_stats(&s0));
    CHECK_ST(G.b->mem_copy(G.dev, NULL, d1, host, 1000, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, G.q, d2, host, 24, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, NULL, host, d1, 300, OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, NULL, d2, d1, 128, OCU_COPY_DEVICE_TO_DEVICE), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, G.q, d2 + 512, d1, 72, OCU_COPY_DEVICE_TO_DEVICE), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, G.q, host, d2, 7, OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK_ST(G.b->queue_sync(G.q), OCU_OK);
    CHECK(get_stats(&s1));
    CHECK(s1.bytes_to_device - s0.bytes_to_device == 1024);
    CHECK(s1.bytes_to_host - s0.bytes_to_host == 307);
    CHECK(s1.bytes_on_device - s0.bytes_on_device == 200);
    set_case("rule 9: refused copies move no bytes");
    CHECK_ST(G.b->mem_copy(G.dev, NULL, host, host, 16, OCU_COPY_HOST_TO_DEVICE),
             OCU_ERR_INVALID_VALUE);
    CHECK_ST(G.b->mem_copy(G.dev, NULL, d1, host, 2048, OCU_COPY_HOST_TO_DEVICE),
             OCU_ERR_INVALID_VALUE);
    CHECK(get_stats(&s2));
    CHECK(s2.bytes_to_device == s1.bytes_to_device);
    CHECK(s2.bytes_to_host == s1.bytes_to_host);
    CHECK(s2.bytes_on_device == s1.bytes_on_device);
    dfree(d1);
    dfree(d2);
}

/* ---- rules 3, 4, 6: execute with host pointers, alignment ----------------- */

static void test_exec_pointers(void)
{
    set_case("rule 6: allocations are 64-byte aligned");
    static const size_t sizes[] = { 1, 3, 100, 4097, 65541 };
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; ++i) {
        void *p = dalloc(sizes[i]);
        CHECK(p != NULL && ((uintptr_t)p % 64u) == 0);
        dfree(p);
    }
    set_case("rule 6: support.align_bytes == 64, rule 11: accum f32, rule 12: var_rows");
    for (int oi = 0; oi < 5; ++oi) {
        for (int di = 0; di < 3; ++di) {
            ocu_op_support s;
            if (G.t->support(G.dev, OPS5[oi], FLOATS[di], &s) != OCU_OK) {
                continue;
            }
            CHECK(s.align_bytes == 64);
            CHECK(s.accum_dtype == OCU_DT_F32);
            CHECK(s.var_rows == 1);
        }
    }

    if (!g_allowed[OCU_DT_F32]) {
        return;
    }
    const ocu_dtype f = OCU_DT_F32;
    ocu_op_desc d = d_simple(OCU_OP_ADD, f, 2, 16); /* 128 bytes per tensor */
    ocu_op_plan *p = NULL;
    set_case("rule 3/6: plan add f32 [2,16]");
    if (!CHECK_ST(qp(&d, NULL, &p), OCU_OK) || !p) {
        return;
    }
    uint8_t *da = dalloc(1024), *db = dalloc(1024), *dy = dalloc(1024);
    static _Alignas(64) float host_buf[256]; /* aligned_alloc is missing on Windows */
    float *host = host_buf;
    if (!da || !db || !dy || !host) {
        return;
    }
    float ha[256], hb[256], hy[256];
    for (int i = 0; i < 256; ++i) {
        ha[i] = (float)urand(-4, 4);
        hb[i] = (float)urand(-4, 4);
        host[i] = 0.0f;
    }
    CHECK_ST(G.b->mem_copy(G.dev, NULL, da, ha, 1024, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, NULL, db, hb, 1024, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    ocu_tensor ts[3];
    for (int i = 0; i < 3; ++i) {
        ts[i].desc = d.tensors[i];
    }
    set_case("rule 3: execute with a host pointer");
    ts[0].data = host; ts[1].data = db; ts[2].data = dy;
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    ts[0].data = da; ts[1].data = db; ts[2].data = host;
    CHECK_ST(G.t->execute(p, G.q, ts, 3), OCU_ERR_INVALID_VALUE);
    ts[0].data = da; ts[1].data = ha; ts[2].data = dy;
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set_case("rule 3: execute with a freed device pointer");
    uint8_t *gone = dalloc(1024);
    dfree(gone);
    ts[0].data = gone; ts[1].data = db; ts[2].data = dy;
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set_case("rule 6: execute with pointers not 64-byte aligned");
    static const int offs[] = { 4, 16, 32, 60 };
    for (int i = 0; i < 4; ++i) {
        for (int slot = 0; slot < 3; ++slot) {
            ts[0].data = da; ts[1].data = db; ts[2].data = dy;
            ts[slot].data = (uint8_t *)ts[slot].data + offs[i];
            CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
        }
    }
    set_case("rule 6: 64-byte offsets inside an allocation are fine");
    ts[0].data = da + 128; ts[1].data = db + 64; ts[2].data = dy + 192;
    CHECK_ST(G.t->execute(p, G.q, ts, 3), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, G.q, hy, dy + 192, 128, OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK_ST(G.b->queue_sync(G.q), OCU_OK);
    int ok = 1;
    for (int i = 0; i < 32; ++i) {
        if (hy[i] != ha[32 + i] + hb[16 + i]) {
            ok = 0;
        }
    }
    CHECK(ok);
    CHECK_ST(G.t->destroy(p), OCU_OK);
    if (g_allowed[OCU_DT_F16]) {
        set_case("rule 6: f16 tensor at a 2-byte offset");
        ocu_op_desc dh = d_simple(OCU_OP_SOFTMAX, OCU_DT_F16, 2, 32);
        if (CHECK_ST(qp(&dh, NULL, &p), OCU_OK) && p) {
            ocu_tensor th[2];
            th[0].desc = dh.tensors[0];
            th[1].desc = dh.tensors[1];
            th[0].data = da + 2;
            th[1].data = dy;
            CHECK_ST(G.t->execute(p, NULL, th, 2), OCU_ERR_INVALID_VALUE);
            th[0].data = da;
            th[1].data = dy + 66;
            CHECK_ST(G.t->execute(p, NULL, th, 2), OCU_ERR_INVALID_VALUE);
            CHECK_ST(G.t->destroy(p), OCU_OK);
        }
    }
    dfree(da);
    dfree(db);
    dfree(dy);
}

/* ---- rule 5: scratchpad, row operators ----------------------------------- */

static void test_row_scratch(void)
{
    for (int di = 0; di < 3; ++di) {
        ocu_dtype dt = FLOATS[di];
        if (!g_allowed[dt]) {
            continue;
        }
        uint64_t sz = ocu_dtype_size(dt);
        for (int kind = 0; kind < 3; ++kind) { /* softmax, rms, rms + weight */
            uint64_t per = kind == 2 ? 3 * sz : 2 * sz;
            int64_t nmax = g_scratch > 64 ? (int64_t)((g_scratch - 64) / per) : 0;
            ocu_op op = kind == 0 ? OCU_OP_SOFTMAX : OCU_OP_RMS_NORM;
            for (int over = 0; over < 2; ++over) {
                int64_t n = nmax + over;
                if (n < 1) {
                    continue;
                }
                ocu_op_desc d = d_simple(op, dt, 2, n);
                if (kind == 2) {
                    d.num_tensors = 3;
                    d.tensors[1] = tdesc(dt, 1, SHP(n), 0);
                    d.tensors[2] = d.tensors[0];
                }
                set_case("rule 5: %s%s %s row %lld, S=%llu (largest fitting %lld)",
                         ocu_op_name(op), kind == 2 ? "+weight" : "", ocu_dtype_name(dt),
                         (long long)n, (unsigned long long)g_scratch, (long long)nmax);
                ocu_op_stats s0, s1;
                CHECK(get_stats(&s0));
                ocu_op_caps caps;
                ocu_status st = qp(&d, &caps, NULL);
                CHECK(get_stats(&s1));
                if (over) {
                    CHECK_ST(st, OCU_ERR_NOT_SUPPORTED);
                    CHECK(s1.plans_refused == s0.plans_refused + 1);
                } else {
                    CHECK_ST(st, OCU_OK);
                    if (st == OCU_OK) {
                        CHECK(caps.scratch_bytes == (uint64_t)n * per + 64);
                        CHECK(caps.scratch_bytes <= g_scratch);
                    }
                    /* same answer with VAR_ROWS planned at more rows (rule 12) */
                    ocu_op_desc v = d;
                    for (uint32_t i = 0; i < v.num_tensors; ++i) {
                        if (v.tensors[i].rank == 2) {
                            v.tensors[i].shape[0] = 7;
                        }
                    }
                    v.flags = OCU_OP_FLAG_VAR_ROWS;
                    ocu_op_caps vc;
                    CHECK_ST(qp(&v, &vc, NULL), OCU_OK);
                    CHECK(vc.scratch_bytes == caps.scratch_bytes);
                }
            }
            /* the largest fitting row runs and is correct */
            if (nmax >= 1 && nmax <= 70000) {
                row_case rc = { .dt = dt, .rank = 2, .shape = { 3, nmax }, .eps = 1e-5f,
                                .scale = 1.0f, .weight = kind == 2, .on_queue = 1 };
                ocu_status st = kind == 0 ? run_softmax(&rc) : run_rms(&rc);
                CHECK_ST(st, OCU_OK);
            }
        }
        set_case("rule 5: add/mul are streamed (%s, row 1<<20)", ocu_dtype_name(dt));
        ocu_op_desc d = d_simple(OCU_OP_ADD, dt, 4, (int64_t)1 << 20);
        EXPECT(d, OCU_OK);
        d = d_simple(OCU_OP_MUL, dt, 1, (int64_t)1 << 22);
        EXPECT(d, OCU_OK);
    }
}

/* ---- rule 5: GEMM tiling ------------------------------------------------- */

static uint64_t ws(uint64_t tm, uint64_t tn, uint64_t tk, uint64_t in)
{
    return (tm * tk + tk * tn) * in + tm * tn * 4u;
}

static int tile_formula(uint64_t sb, uint64_t in)
{
    for (uint64_t tm = 16; tm <= 128; tm *= 2) {
        for (uint64_t tn = 16; tn <= 128; tn *= 2) {
            for (uint64_t tk = 16; tk <= 128; tk *= 2) {
                if (ws(tm, tn, tk, in) == sb) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

static void test_gemm_tiling(void)
{
    for (int di = 0; di < 3; ++di) {
        ocu_dtype dt = FLOATS[di];
        if (!g_allowed[dt]) {
            continue;
        }
        uint64_t in = ocu_dtype_size(dt);
        ocu_dtype outs[2] = { OCU_DT_F32, dt };
        for (int oi = 0; oi < (dt == OCU_DT_F32 ? 1 : 2); ++oi) {
            ocu_dtype out = outs[oi];
            if (g_tiling) {
                int fits = ws(16, 16, 16, in) <= g_scratch;
                set_case("rule 5: gemm %s->%s tiled, S=%llu", ocu_dtype_name(dt),
                         ocu_dtype_name(out), (unsigned long long)g_scratch);
                ocu_op_support s;
                ocu_status ss = G.t->support(G.dev, OCU_OP_GEMM, dt, &s);
                if (!fits) {
                    CHECK_ST(ss, OCU_ERR_NOT_SUPPORTED);
                    ocu_op_desc d = d_gemm(dt, out, 1, 1, 1);
                    EXPECT(d, OCU_ERR_NOT_SUPPORTED);
                    d = d_gemm(dt, out, 64, 64, 64);
                    EXPECT(d, OCU_ERR_NOT_SUPPORTED);
                    continue;
                }
                CHECK_ST(ss, OCU_OK);
                static const int64_t sh[][3] = { { 256, 256, 256 }, { 200, 300, 260 },
                                                 { 1, 1000, 3 }, { 129, 1, 1000 } };
                for (size_t i = 0; i < sizeof sh / sizeof sh[0]; ++i) {
                    ocu_op_desc d = d_gemm(dt, out, sh[i][0], sh[i][1], sh[i][2]);
                    set_case("rule 5: gemm %s->%s %lldx%lldx%lld tiled, S=%llu",
                             ocu_dtype_name(dt), ocu_dtype_name(out), (long long)sh[i][0],
                             (long long)sh[i][1], (long long)sh[i][2],
                             (unsigned long long)g_scratch);
                    ocu_op_caps caps;
                    if (!CHECK_ST(qp(&d, &caps, NULL), OCU_OK)) {
                        continue;
                    }
                    CHECK(caps.scratch_bytes <= g_scratch);
                    CHECK(tile_formula(caps.scratch_bytes, in));
                    uint64_t mt = (uint64_t)(sh[i][0] + 127) / 128;
                    uint64_t nt = (uint64_t)(sh[i][1] + 127) / 128;
                    CHECK(caps.tiles >= mt * nt);
                    if (sh[i][0] > 128 || sh[i][1] > 128 || g_scratch < ws(128, 128, 128, in)) {
                        if (sh[i][0] * sh[i][1] > 256) {
                            CHECK(caps.tiles > 1);
                        }
                    }
                    if (g_scratch == 4096 && in == 4) {
                        CHECK(caps.scratch_bytes == ws(16, 16, 16, 4)); /* only fit */
                    }
                    /* VAR_ROWS: caps of the planned (largest) shape (rule 12) */
                    ocu_op_desc v = d;
                    v.flags = OCU_OP_FLAG_VAR_ROWS;
                    ocu_op_caps vc;
                    CHECK_ST(qp(&v, &vc, NULL), OCU_OK);
                    CHECK(vc.tiles == caps.tiles && vc.scratch_bytes == caps.scratch_bytes);
                }
                /* numbers across tiles */
                gemm_case gc = { .din = dt, .dout = out, .rank = 2, .alpha = 1.0f, .beta = 0.5f,
                                 .on_queue = 1 };
                if (g_scratch >= 64 * 1024) {
                    gc.m = 200; gc.n = 150; gc.k = 300;
                } else {
                    gc.m = 40; gc.n = 37; gc.k = 50;
                }
                CHECK_ST(run_gemm(&gc), OCU_OK);
                CHECK(G.last_caps.tiles > 1);
                gc.var = 1; gc.m_exec = 3; gc.ta = 1;
                CHECK_ST(run_gemm(&gc), OCU_OK);
            } else {
                /* whole problem as one tile */
                int64_t dmax = 1;
                while (ws((uint64_t)dmax + 1, (uint64_t)dmax + 1, (uint64_t)dmax + 1, in) <= g_scratch) {
                    ++dmax;
                }
                for (int over = 0; over < 2; ++over) {
                    int64_t dd = dmax + over;
                    ocu_op_desc d = d_gemm(dt, out, dd, dd, dd);
                    set_case("rule 5: gemm %s->%s %lld^3 untiled, S=%llu (largest %lld)",
                             ocu_dtype_name(dt), ocu_dtype_name(out), (long long)dd,
                             (unsigned long long)g_scratch, (long long)dmax);
                    ocu_op_caps caps;
                    ocu_status st = qp(&d, &caps, NULL);
                    if (over) {
                        CHECK_ST(st, OCU_ERR_NOT_SUPPORTED);
                    } else if (CHECK_ST(st, OCU_OK)) {
                        CHECK(caps.tiles == 1);
                        CHECK(caps.scratch_bytes == ws((uint64_t)dd, (uint64_t)dd, (uint64_t)dd, in));
                    }
                }
                int64_t kmax = (int64_t)((g_scratch - 4) / (2 * in));
                for (int over = 0; over < 2; ++over) {
                    ocu_op_desc d = d_gemm(dt, out, 1, 1, kmax + over);
                    set_case("rule 5: gemm %s 1x1x%lld untiled", ocu_dtype_name(dt),
                             (long long)(kmax + over));
                    EXPECT(d, over ? OCU_ERR_NOT_SUPPORTED : OCU_OK);
                }
                set_case("rule 5: gemm %s 256^3 untiled is refused", ocu_dtype_name(dt));
                ocu_op_desc big = d_gemm(dt, out, 256, 256, 256);
                EXPECT(big, OCU_ERR_NOT_SUPPORTED);
                gemm_case gc = { .din = dt, .dout = out, .rank = 2, .alpha = 1.0f,
                                 .m = dmax, .n = dmax, .k = dmax };
                CHECK_ST(run_gemm(&gc), OCU_OK);
                CHECK(G.last_caps.tiles == 1);
            }
        }
    }
}

/* ---- rule 7: dtype set --------------------------------------------------- */

static void test_dtype_set(void)
{
    for (int oi = 0; oi < 5; ++oi) {
        for (int di = 0; di < 3; ++di) {
            ocu_op op = OPS5[oi];
            ocu_dtype dt = FLOATS[di];
            set_case("rule 7: %s %s allowed=%d", ocu_op_name(op), ocu_dtype_name(dt),
                     g_allowed[dt]);
            ocu_op_support s;
            ocu_status ss = G.t->support(G.dev, op, dt, &s);
            ocu_op_desc d = d_simple(op, dt, 4, 16);
            int gemm_blocked = op == OCU_OP_GEMM && g_tiling && ws(16, 16, 16, ocu_dtype_size(dt)) > g_scratch;
            if (g_allowed[dt] && !gemm_blocked) {
                CHECK_ST(ss, OCU_OK);
                EXPECT(d, OCU_OK);
            } else if (!g_allowed[dt]) {
                CHECK_ST(ss, OCU_ERR_NOT_SUPPORTED);
                EXPECT(d, OCU_ERR_NOT_SUPPORTED);
                /* malformed is still INVALID_VALUE (4.5 order of checks) */
                d.tensors[0].shape[0] = 0;
                EXPECT(d, OCU_ERR_INVALID_VALUE);
            }
        }
    }
    if (g_allowed[OCU_DT_F16] && !g_allowed[OCU_DT_F32]) {
        set_case("rule 7: f16 only: mixed f16 -> f32 GEMM and f16 ADD run");
        gemm_case gc = { .din = OCU_DT_F16, .dout = OCU_DT_F32, .rank = 2, .alpha = 1.0f,
                         .m = 9, .n = 7, .k = 33 };
        CHECK_ST(run_gemm(&gc), OCU_OK);
        elt_case ec = { .op = OCU_OP_ADD, .dt = OCU_DT_F16, .rank = 2, .a = { 5, 40 },
                        .b = { 1, 40 }, .on_queue = 1 };
        CHECK_ST(run_elt(&ec), OCU_OK);
        set_case("rule 7: mismatched dtypes still INVALID_VALUE");
        ocu_op_desc d = d_simple(OCU_OP_ADD, OCU_DT_F32, 4, 16);
        d.tensors[1].dtype = OCU_DT_F16;
        EXPECT(d, OCU_ERR_INVALID_VALUE);
    }
}

/* ---- rules 8, 10: plan latency, memory size ------------------------------ */

static double now_ms(void)
{
#if defined(_WIN32)
    /* MinGW's C library lacks timespec_get; Windows' clock() is wall time. */
    return (double)clock() * 1e3 / CLOCKS_PER_SEC;
#else
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
#endif
}

static void test_plan_latency(void)
{
    set_case("rule 8: OCU_NPUSIM_PLAN_MS=%ld", g_plan_ms);
    ocu_op_desc d = d_simple(OCU_OP_ADD, OCU_DT_F32, 4, 16);
    ocu_op_plan *p = NULL;
    double t0 = now_ms();
    CHECK_ST(G.t->plan(G.dev, &d, &p), OCU_OK);
    double t1 = now_ms();
    CHECK(t1 - t0 >= (double)g_plan_ms - 1.0);
    if (p) {
        CHECK_ST(G.t->destroy(p), OCU_OK);
    }
    ocu_op_support s;
    if (G.t->support(G.dev, OCU_OP_ADD, OCU_DT_F32, &s) == OCU_OK) {
        printf("NOTE support.needs_plan_ms = %u with OCU_NPUSIM_PLAN_MS=%ld; plan took %.1f ms\n",
               s.needs_plan_ms, g_plan_ms, t1 - t0);
    }
}

static void test_mem_size(void)
{
    set_case("rule 10: OCU_NPUSIM_MEM_MB=%llu", (unsigned long long)g_mem_mb);
    void *p = NULL;
    CHECK_ST(G.b->mem_alloc(G.dev, (size_t)(g_mem_mb * 1024u * 1024u + 1u), &p),
             OCU_ERR_OUT_OF_MEMORY);
    void *h = dalloc((size_t)(g_mem_mb * 1024u * 1024u / 2u));
    dfree(h);
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "basic";
    int env_ok = 1;
    if (strcmp(mode, "basic") == 0) {
        env_ok = env_is("OCU_NPUSIM_SCRATCH_KB", NULL) && env_is("OCU_NPUSIM_TILING", NULL) &&
                 env_is("OCU_NPUSIM_DTYPES", NULL) && env_is("OCU_NPUSIM_MEM_MB", NULL);
    } else if (strcmp(mode, "scratch4") == 0) {
        env_ok = env_is("OCU_NPUSIM_SCRATCH_KB", "4");
        g_scratch = 4096;
    } else if (strcmp(mode, "scratch1") == 0) {
        env_ok = env_is("OCU_NPUSIM_SCRATCH_KB", "1");
        g_scratch = 1024;
    } else if (strcmp(mode, "tiling0") == 0) {
        env_ok = env_is("OCU_NPUSIM_TILING", "0");
        g_tiling = 0;
    } else if (strcmp(mode, "dtypes_f16") == 0) {
        env_ok = env_is("OCU_NPUSIM_DTYPES", "f16");
        g_allowed[OCU_DT_F32] = 0;
        g_allowed[OCU_DT_BF16] = 0;
    } else if (strcmp(mode, "env") == 0) {
        env_ok = env_is("OCU_NPUSIM_MEM_MB", "8") && env_is("OCU_NPUSIM_PLAN_MS", "40");
        g_mem_mb = 8;
        g_plan_ms = 40;
    } else {
        fprintf(stderr, "test_npusim: unknown mode '%s'\n", mode);
        return 2;
    }
    if (!env_ok) {
        fprintf(stderr, "test_npusim %s: environment does not match the mode\n", mode);
        return 2;
    }
    if (ocu_select_backend("npusim") != OCU_OK || (G.b = ocu_active_backend()) == NULL) {
        printf("test_npusim %s: SKIP (npusim not built in)\n", mode);
        return 77;
    }
    set_case("setup %s", mode);
    G.t = ocu_ops_of(G.b);
    if (!CHECK(G.t != NULL)) { /* rule 2: only operators -> the table exists */
        return 1;
    }
    CHECK(G.t->stats != NULL); /* rule 9 */
    if (!CHECK_ST(G.b->dev_open(0, &G.dev), OCU_OK) ||
        !CHECK_ST(G.b->queue_create(G.dev, &G.q), OCU_OK)) {
        return 1;
    }

    test_dev_info();
    if (strcmp(mode, "env") == 0) {
        /* every plan sleeps here: only the rules this mode is about */
        test_plan_latency();
        test_mem_size();
    } else {
        test_exec_pointers();
        test_row_scratch();
        test_gemm_tiling();
        test_dtype_set();
    }
    if (strcmp(mode, "basic") == 0) {
        test_no_kernels();
        test_memory_rules();
        test_dma_counters();
    }

    set_case("teardown");
    CHECK_ST(G.b->queue_sync(G.q), OCU_OK);
    CHECK_ST(G.b->queue_destroy(G.q), OCU_OK);
    CHECK_ST(G.b->dev_close(G.dev), OCU_OK);
    printf("test_npusim %s: %ld checks, %ld failures\n", mode, G.checks, G.failures);
    return G.failures == 0 ? 0 : 1;
}

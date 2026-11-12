/*
 * test_ops.c -- conformance of a backend's operator table
 * (docs/spec/op-table.md, version 1), written from the specification only.
 *
 * Usage: test_ops <backend name>. Prints SKIP and exits 77 when the backend
 * is not built in or has no operator table.
 *
 * Numbers are judged with the tolerance classes of spec section 6: the
 * reference is computed in binary64 from the already rounded inputs, with
 * the accumulation type the backend declares in caps.accum_dtype.
 */
#include "ops_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long g_skips;

static const ocu_dtype FLOATS[3] = { OCU_DT_F32, OCU_DT_F16, OCU_DT_BF16 };

/* A well-formed numeric case returned NOT_SUPPORTED / NOT_IMPLEMENTED:
 * the spec allows query to refuse a descriptor (5.1), so it is a skip. */
static void expect_run(ocu_status s)
{
    if (s == OCU_OK) {
        return;
    }
    if (s == OCU_ERR_NOT_SUPPORTED || s == OCU_ERR_NOT_IMPLEMENTED) {
        ++g_skips;
        printf("NOTE skipped (%s): %s\n", ocu_status_name(s), G.where);
        return;
    }
    check_st_at(s, OCU_OK, __FILE__, __LINE__, "well-formed case runs");
}

static int sup(ocu_op op, ocu_dtype dt, ocu_op_support *out)
{
    ocu_op_support s;
    ocu_status st = G.t->support(G.dev, op, dt, &s);
    if (out) {
        *out = s;
    }
    return st == OCU_OK;
}

/* ---- the converters this test uses (spec 2: ties to even) --------------- */

static void test_converters(void)
{
    set_case("test's own f16/bf16 conversions");
    CHECK(round_to(OCU_DT_F16, 1.0 + 0x1p-11) == 1.0);
    CHECK(round_to(OCU_DT_F16, 1.0 + 0x3p-11) == 1.0 + 0x1p-9);
    CHECK(round_to(OCU_DT_F16, 65519.0) == 65504.0);
    CHECK(isinf(round_to(OCU_DT_F16, 65520.0)));
    CHECK(round_to(OCU_DT_F16, 0x1p-25) == 0.0);
    CHECK(round_to(OCU_DT_F16, 0x3p-26) == 0x1p-24);
    CHECK(round_to(OCU_DT_BF16, 1.0 + 0x1p-8) == 1.0);
    CHECK(round_to(OCU_DT_BF16, 1.0 + 0x3p-8) == 1.0 + 0x1p-6);
    CHECK(round_to(OCU_DT_F32, 1.0 + 0x1p-24) == 1.0);
    CHECK(round_to(OCU_DT_F32, 0x1p-150) == 0.0);
    CHECK(round_to(OCU_DT_F32, 0x1.8p-149) == 0x1p-148);
    long bad16 = 0, badb = 0;
    for (uint32_t h = 0; h < 65536u; ++h) {
        double v = f16_decode((uint16_t)h);
        if (!isnan(v) && (f16_encode(v) != h || round_to(OCU_DT_F16, v) != v)) {
            ++bad16;
        }
        double w = bf16_decode((uint16_t)h);
        if (!isnan(w) && (bf16_encode(w) != h || round_to(OCU_DT_BF16, w) != w)) {
            ++badb;
        }
    }
    CHECK(bad16 == 0);
    CHECK(badb == 0);
}

/* ---- support() (5.1) ----------------------------------------------------- */

static int spec_undefined(ocu_op op, ocu_dtype dt)
{
    if (op == OCU_OP_QMATMUL) {
        return 1;
    }
    if (ocu_dtype_is_blocked(dt)) {
        return 1;
    }
    if (op == OCU_OP_ROPE) {
        return 0; /* dtype rules of ROPE are not given in version 1 */
    }
    return dt == OCU_DT_I8 || dt == OCU_DT_I32;
}

static void test_support(void)
{
    static const ocu_dtype dts[] = { OCU_DT_F32, OCU_DT_F16, OCU_DT_BF16, OCU_DT_I8,
                                     OCU_DT_I32, OCU_DT_Q8_B32, OCU_DT_Q4_B32 };
    for (uint32_t op = OCU_OP_GEMM; op <= OCU_OP_QMATMUL; ++op) {
        for (size_t i = 0; i < sizeof dts / sizeof dts[0]; ++i) {
            ocu_dtype dt = dts[i];
            set_case("support(%s, %s)", ocu_op_name((ocu_op)op), ocu_dtype_name(dt));
            ocu_op_support s;
            memset(&s, 0xAB, sizeof s);
            ocu_status st = G.t->support(G.dev, (ocu_op)op, dt, &s);
            CHECK(st == OCU_OK || st == OCU_ERR_NOT_SUPPORTED || st == OCU_ERR_NOT_IMPLEMENTED);
            if (spec_undefined((ocu_op)op, dt)) {
                if (op == OCU_OP_ROPE) {
                    /* Spec ambiguity: 4.5 says block types give NOT_SUPPORTED,
                     * 4.6 says a backend without ROPE gives NOT_IMPLEMENTED. */
                    CHECK(st == OCU_ERR_NOT_SUPPORTED || st == OCU_ERR_NOT_IMPLEMENTED);
                } else {
                    CHECK_ST(st, OCU_ERR_NOT_SUPPORTED);
                }
            }
            if (op == OCU_OP_ROPE && dt == OCU_DT_F32) {
                CHECK(st == OCU_OK || st == OCU_ERR_NOT_IMPLEMENTED);
            }
            if (st != OCU_OK) {
                ocu_op_support z;
                memset(&z, 0, sizeof z);
                CHECK(memcmp(&s, &z, sizeof s) == 0); /* *out zeroed */
                continue;
            }
            CHECK(s.layouts != 0 && (s.layouts & ~(OCU_LAYOUT_DENSE | OCU_LAYOUT_ROW_STRIDED)) == 0);
            CHECK(s.max_rank >= 1 && s.max_rank <= OCU_OPS_MAX_RANK);
            CHECK(s.max_row >= 0);
            CHECK(s.align_bytes != 0 && (s.align_bytes & (s.align_bytes - 1u)) == 0);
            CHECK(s.accum_dtype == OCU_DT_F32 || s.accum_dtype == OCU_DT_F16 ||
                  s.accum_dtype == OCU_DT_BF16);
            if (dt == OCU_DT_F32) {
                CHECK(s.accum_dtype == OCU_DT_F32);
            }
            CHECK(s.var_rows == 0 || s.var_rows == 1);
            uint32_t allowed = 1u << (unsigned)dt;
            if (op == OCU_OP_GEMM) {
                allowed |= 1u << OCU_DT_F32;
            }
            CHECK(s.out_dtypes != 0);
            if (op != OCU_OP_ROPE) {
                CHECK((s.out_dtypes & ~allowed) == 0);
            }
            if (op == OCU_OP_GEMM) {
                CHECK(s.max_rank >= 2);
            }
        }
    }
    set_case("support() argument errors");
    ocu_op_support s;
    CHECK_ST(G.t->support(NULL, OCU_OP_ADD, OCU_DT_F32, &s), OCU_ERR_INVALID_DEVICE);
    CHECK_ST(G.t->support(G.dev, OCU_OP_ADD, OCU_DT_F32, NULL), OCU_ERR_INVALID_VALUE);
    CHECK(G.t->support(G.dev, (ocu_op)0, OCU_DT_F32, &s) != OCU_OK);
    CHECK(G.t->support(G.dev, (ocu_op)99, OCU_DT_F32, &s) != OCU_OK);
    CHECK(G.t->support(G.dev, OCU_OP_ADD, OCU_DT_NONE, &s) != OCU_OK);
    CHECK(G.t->support(G.dev, OCU_OP_ADD, (ocu_dtype)77, &s) != OCU_OK);
}

/* ---- GEMM (4.1) ---------------------------------------------------------- */

static void test_gemm(void)
{
    for (int di = 0; di < 3; ++di) {
        ocu_dtype din = FLOATS[di];
        ocu_op_support s;
        if (!sup(OCU_OP_GEMM, din, &s)) {
            printf("NOTE gemm %s not supported: skipped\n", ocu_dtype_name(din));
            ++g_skips;
            continue;
        }
        ocu_dtype outs[2] = { din, OCU_DT_F32 };
        int nout = din == OCU_DT_F32 ? 1 : 2;
        for (int oi = 0; oi < nout; ++oi) {
            ocu_dtype dout = outs[oi];
            if (!(s.out_dtypes & (1u << (unsigned)dout))) {
                printf("NOTE gemm %s->%s not offered: skipped\n", ocu_dtype_name(din),
                       ocu_dtype_name(dout));
                ++g_skips;
                continue;
            }
            const gemm_case base = { .din = din, .dout = dout, .rank = 2, .alpha = 1.0f };
            static const int64_t sizes[][3] = {
                { 37, 29, 64 }, { 1, 1, 1 }, { 3, 1, 7 }, { 1, 129, 3 }, { 129, 3, 37 },
                { 17, 13, 1000 }, { 64, 64, 129 }, { 129, 129, 257 },
            };
            for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; ++i) {
                gemm_case gc = base;
                gc.m = sizes[i][0];
                gc.n = sizes[i][1];
                gc.k = sizes[i][2];
                gc.on_queue = (int)(i & 1u);
                expect_run(run_gemm(&gc));
            }
            for (uint32_t ta = 0; ta < 2; ++ta) {
                for (uint32_t tb = 0; tb < 2; ++tb) {
                    for (int64_t pad = 0; pad <= 5; pad += 5) {
                        gemm_case gc = base;
                        gc.m = 19; gc.n = 23; gc.k = 31;
                        gc.ta = ta; gc.tb = tb; gc.pad = pad;
                        gc.on_queue = (int)tb;
                        expect_run(run_gemm(&gc));
                    }
                }
            }
            /* alpha / beta */
            gemm_case gc = base;
            gc.m = 33; gc.n = 21; gc.k = 47; gc.alpha = 0.5f; gc.beta = -1.25f;
            expect_run(run_gemm(&gc));
            gc.ta = 1; gc.tb = 1; gc.pad = 3; gc.on_queue = 1;
            expect_run(run_gemm(&gc));
            /* beta == 0: C not read (NaN / Inf in C must not propagate) */
            gc = base;
            gc.m = 13; gc.n = 11; gc.k = 9; gc.c_fill = 1;
            expect_run(run_gemm(&gc));
            gc.c_fill = 2; gc.alpha = 2.0f;
            expect_run(run_gemm(&gc));
            /* alpha == 0: A and B not read; C = beta * C */
            gc = base;
            gc.m = 13; gc.n = 11; gc.k = 9; gc.alpha = 0.0f; gc.beta = 0.75f; gc.ab_nan = 1;
            expect_run(run_gemm(&gc));
            gc.beta = 0.0f; gc.c_fill = 1; /* neither read: C = 0 */
            expect_run(run_gemm(&gc));
            /* rank 3, batch broadcast */
            static const int64_t bat[][3] = { { 3, 3, 1 }, { 3, 1, 3 }, { 3, 3, 3 },
                                              { 1, 1, 1 }, { 4, 1, 1 } };
            for (size_t i = 0; i < sizeof bat / sizeof bat[0]; ++i) {
                gc = base;
                gc.rank = 3;
                gc.bt = bat[i][0]; gc.ba = bat[i][1]; gc.bb = bat[i][2];
                gc.m = 9; gc.n = 7; gc.k = 37;
                gc.ta = (uint32_t)(i & 1u);
                gc.tb = (uint32_t)((i >> 1) & 1u);
                gc.pad = (i % 3 == 2) ? 3 : 0;
                gc.alpha = 1.5f; gc.beta = (i & 1u) ? 0.5f : 0.0f;
                gc.on_queue = (int)(i & 1u);
                expect_run(run_gemm(&gc));
            }
            /* variable row count (4.7) */
            if (s.var_rows) {
                static const int64_t mex[] = { 1, 5, 37 };
                for (size_t i = 0; i < 3; ++i) {
                    gc = base;
                    gc.m = 37; gc.n = 29; gc.k = 64; gc.var = 1; gc.m_exec = mex[i];
                    gc.beta = 1.0f; gc.on_queue = (int)(i & 1u);
                    expect_run(run_gemm(&gc));
                }
                gc = base;
                gc.m = 37; gc.n = 29; gc.k = 64; gc.var = 1; gc.m_exec = 7; gc.ta = 1;
                gc.pad = 2;
                expect_run(run_gemm(&gc));
                gc = base;
                gc.rank = 3; gc.bt = 3; gc.ba = 3; gc.bb = 1;
                gc.m = 16; gc.n = 8; gc.k = 24; gc.var = 1; gc.m_exec = 4; gc.beta = 2.0f;
                expect_run(run_gemm(&gc));
            }
        }
        if (!s.var_rows) {
            ocu_op_desc d = op_desc(OCU_OP_GEMM);
            d.num_tensors = 3;
            d.flags = OCU_OP_FLAG_VAR_ROWS;
            d.tensors[0] = tdesc(din, 2, SHP(8, 16), 0);
            d.tensors[1] = tdesc(din, 2, SHP(16, 4), 0);
            d.tensors[2] = tdesc(din, 2, SHP(8, 4), 0);
            d.p.gemm.alpha = 1.0f;
            set_case("gemm %s VAR_ROWS without support.var_rows", ocu_dtype_name(din));
            EXPECT(d, OCU_ERR_NOT_SUPPORTED);
        }
    }
}

/* ---- ADD / MUL (4.2) ----------------------------------------------------- */

static void test_elementwise(void)
{
    static const ocu_op ops[2] = { OCU_OP_ADD, OCU_OP_MUL };
    for (int oi = 0; oi < 2; ++oi) {
        for (int di = 0; di < 3; ++di) {
            ocu_op op = ops[oi];
            ocu_dtype dt = FLOATS[di];
            ocu_op_support s;
            if (!sup(op, dt, &s)) {
                printf("NOTE %s %s not supported: skipped\n", ocu_op_name(op), ocu_dtype_name(dt));
                ++g_skips;
                continue;
            }
            const elt_case cases[] = {
                { .rank = 2, .a = { 37, 129 }, .b = { 37, 129 } },
                { .rank = 2, .a = { 37, 129 }, .b = { 1, 129 }, .on_queue = 1 },
                { .rank = 2, .a = { 37, 129 }, .b = { 37, 1 } },
                { .rank = 2, .a = { 37, 129 }, .b = { 1, 1 }, .on_queue = 1 },
                { .rank = 1, .a = { 1 }, .b = { 1 } },
                { .rank = 1, .a = { 1000 }, .b = { 1 } },
                { .rank = 1, .a = { 1000 }, .b = { 1000 }, .on_queue = 1 },
                { .rank = 3, .a = { 3, 5, 7 }, .b = { 3, 1, 7 } },
                { .rank = 3, .a = { 3, 5, 7 }, .b = { 1, 5, 1 } },
                { .rank = 4, .a = { 2, 3, 4, 5 }, .b = { 2, 1, 4, 1 } },
                { .rank = 4, .a = { 2, 3, 4, 5 }, .b = { 1, 1, 1, 1 }, .on_queue = 1 },
                { .rank = 2, .a = { 37, 129 }, .b = { 37, 129 }, .pad = 3 },
                { .rank = 3, .a = { 3, 5, 7 }, .b = { 1, 5, 7 }, .pad = 2, .on_queue = 1 },
                { .rank = 2, .a = { 37, 129 }, .b = { 1, 129 }, .inplace = 1 },
                { .rank = 3, .a = { 3, 5, 7 }, .b = { 3, 5, 1 }, .inplace = 1, .pad = 2 },
                { .rank = 2, .a = { 4, 16 }, .b = { 4, 16 }, .specials = 1 },
            };
            for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
                elt_case ec = cases[i];
                ec.op = op;
                ec.dt = dt;
                expect_run(run_elt(&ec));
            }
            if (s.var_rows) {
                const elt_case vcases[] = {
                    { .rank = 2, .a = { 37, 129 }, .b = { 37, 129 }, .m_exec = 5 },
                    { .rank = 2, .a = { 37, 129 }, .b = { 1, 129 }, .m_exec = 1, .on_queue = 1 },
                    { .rank = 2, .a = { 37, 129 }, .b = { 37, 129 }, .m_exec = 37 },
                    { .rank = 3, .a = { 6, 5, 7 }, .b = { 6, 1, 7 }, .m_exec = 2, .pad = 1 },
                    { .rank = 2, .a = { 9, 33 }, .b = { 9, 33 }, .m_exec = 3, .inplace = 1 },
                };
                for (size_t i = 0; i < sizeof vcases / sizeof vcases[0]; ++i) {
                    elt_case ec = vcases[i];
                    ec.op = op;
                    ec.dt = dt;
                    ec.var = 1;
                    expect_run(run_elt(&ec));
                }
            }
        }
    }
}

/* ---- RMS_NORM (4.3) and SOFTMAX (4.4) ------------------------------------ */

static void test_rows(void)
{
    for (int di = 0; di < 3; ++di) {
        ocu_dtype dt = FLOATS[di];
        ocu_op_support s;
        if (sup(OCU_OP_RMS_NORM, dt, &s)) {
            const row_case cases[] = {
                { .rank = 2, .shape = { 37, 129 }, .eps = 1e-5f },
                { .rank = 2, .shape = { 37, 129 }, .eps = 1e-5f, .weight = 1, .on_queue = 1 },
                { .rank = 2, .shape = { 1, 1 }, .eps = 1e-6f },
                { .rank = 1, .shape = { 129 }, .eps = 0.0f, .weight = 1 },
                { .rank = 2, .shape = { 3, 1000 }, .eps = 0.0f, .weight = 1 },
                { .rank = 3, .shape = { 5, 3, 64 }, .eps = 1e-5f, .on_queue = 1 },
                { .rank = 4, .shape = { 2, 2, 2, 17 }, .eps = 1e-3f, .weight = 1 },
                { .rank = 2, .shape = { 37, 129 }, .eps = 1e-5f, .pad = 3, .weight = 1 },
                { .rank = 2, .shape = { 37, 129 }, .eps = 1e-5f, .inplace = 1 },
                { .rank = 3, .shape = { 4, 3, 33 }, .eps = 1e-5f, .inplace = 1, .weight = 1,
                  .pad = 2, .on_queue = 1 },
                { .rank = 2, .shape = { 7, 64 }, .eps = 1e-5f, .mag = 300.0 },
            };
            for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
                row_case rc = cases[i];
                rc.dt = dt;
                expect_run(run_rms(&rc));
            }
            if (s.var_rows) {
                row_case rc = { .dt = dt, .rank = 2, .shape = { 37, 129 }, .eps = 1e-5f,
                                .weight = 1, .var = 1, .m_exec = 5 };
                expect_run(run_rms(&rc));
                rc.m_exec = 1; rc.inplace = 1; rc.on_queue = 1;
                expect_run(run_rms(&rc));
                row_case r3 = { .dt = dt, .rank = 3, .shape = { 6, 3, 40 }, .eps = 1e-5f,
                                .var = 1, .m_exec = 2, .pad = 1 };
                expect_run(run_rms(&r3));
            }
        } else {
            printf("NOTE rms_norm %s not supported: skipped\n", ocu_dtype_name(dt));
            ++g_skips;
        }

        if (sup(OCU_OP_SOFTMAX, dt, &s)) {
            double big = dt == OCU_DT_BF16 ? 3.0e4 : 1000.0;
            const row_case cases[] = {
                { .rank = 2, .shape = { 37, 129 }, .scale = 1.0f },
                { .rank = 2, .shape = { 37, 129 }, .scale = 0.125f, .on_queue = 1 },
                { .rank = 2, .shape = { 1, 1 }, .scale = 1.0f },
                { .rank = 1, .shape = { 7 }, .scale = -2.0f },
                { .rank = 2, .shape = { 4, 1000 }, .scale = 1.0f },
                { .rank = 3, .shape = { 2, 3, 17 }, .scale = -2.0f, .on_queue = 1 },
                { .rank = 4, .shape = { 2, 1, 3, 5 }, .scale = 0.5f },
                { .rank = 2, .shape = { 37, 129 }, .scale = 1.0f, .pad = 3 },
                { .rank = 2, .shape = { 37, 129 }, .scale = 1.0f, .inplace = 1 },
                { .rank = 2, .shape = { 9, 300 }, .scale = 1.0f, .mag = big },
                { .rank = 2, .shape = { 9, 300 }, .scale = 30.0f, .mag = 50.0, .on_queue = 1 },
                { .rank = 2, .shape = { 5, 33 }, .scale = 1.0f, .nan_row = 1 },
                { .rank = 3, .shape = { 3, 4, 129 }, .scale = 0.25f, .inplace = 1, .pad = 2 },
            };
            for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
                row_case rc = cases[i];
                rc.dt = dt;
                expect_run(run_softmax(&rc));
            }
            if (s.var_rows) {
                row_case rc = { .dt = dt, .rank = 2, .shape = { 37, 129 }, .scale = 1.0f,
                                .var = 1, .m_exec = 5 };
                expect_run(run_softmax(&rc));
                rc.m_exec = 1; rc.inplace = 1; rc.on_queue = 1;
                expect_run(run_softmax(&rc));
                row_case r3 = { .dt = dt, .rank = 3, .shape = { 6, 3, 40 }, .scale = 2.0f,
                                .var = 1, .m_exec = 2, .pad = 1 };
                expect_run(run_softmax(&r3));
            }
        } else {
            printf("NOTE softmax %s not supported: skipped\n", ocu_dtype_name(dt));
            ++g_skips;
        }
    }
}

/* ---- ROPE (4.6) ---------------------------------------------------------- */

static ocu_op_desc d_rope(void)
{
    ocu_op_desc d = op_desc(OCU_OP_ROPE);
    d.num_tensors = 3;
    d.tensors[0] = tdesc(OCU_DT_F32, 3, SHP(5, 3, 16), 0);
    d.tensors[1] = tdesc(OCU_DT_I32, 1, SHP(5), 0);
    d.tensors[2] = tdesc(OCU_DT_F32, 3, SHP(5, 3, 16), 0);
    d.p.rope.n_dims = 16;
    d.p.rope.theta_base = 10000.0f;
    d.p.rope.freq_scale = 1.0f;
    return d;
}

static void test_rope(void)
{
    ocu_op_support s;
    set_case("rope support");
    ocu_status st = G.t->support(G.dev, OCU_OP_ROPE, OCU_DT_F32, &s);
    CHECK(st == OCU_OK || st == OCU_ERR_NOT_IMPLEMENTED);
    ocu_op_desc d = d_rope();
    if (st != OCU_OK) {
        set_case("rope not implemented: query/plan agree");
        EXPECT(d, OCU_ERR_NOT_IMPLEMENTED);
        d.tensors[0].shape[1] = 0;
        set_case("rope not implemented: malformed reported first (4.5)");
        EXPECT(d, OCU_ERR_INVALID_VALUE);
        printf("NOTE rope: not implemented (allowed in version 1)\n");
        return;
    }
    const rope_case cases[] = {
        { .rank = 3, .t = 5, .h = 3, .n = 16, .mode = 0, .n_dims = 12, .theta = 10000.0f,
          .freq_scale = 1.0f },
        { .rank = 3, .t = 5, .h = 3, .n = 16, .mode = 1, .n_dims = 12, .theta = 10000.0f,
          .freq_scale = 1.0f, .on_queue = 1 },
        { .rank = 2, .t = 7, .h = 1, .n = 10, .mode = 0, .n_dims = 10, .theta = 500.0f,
          .freq_scale = 0.5f },
        { .rank = 2, .t = 7, .h = 1, .n = 64, .mode = 1, .n_dims = 64, .theta = 10000.0f,
          .freq_scale = 0.25f },
        { .rank = 3, .t = 1, .h = 1, .n = 2, .mode = 1, .n_dims = 2, .theta = 10000.0f,
          .freq_scale = 1.0f },
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        expect_run(run_rope(&cases[i]));
    }
    set_case("rope n_dims odd / larger than N");
    d = d_rope();
    d.p.rope.n_dims = 7;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.p.rope.n_dims = 18;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
}

/* ---- descriptor errors (3, 4, 4.5, 4.7) ---------------------------------- */

static ocu_op_desc d_add(ocu_op op, ocu_dtype dt)
{
    ocu_op_desc d = op_desc(op);
    d.num_tensors = 3;
    for (int i = 0; i < 3; ++i) {
        d.tensors[i] = tdesc(dt, 2, SHP(4, 16), 0);
    }
    return d;
}

static ocu_op_desc d_gemm(ocu_dtype in, ocu_dtype out)
{
    ocu_op_desc d = op_desc(OCU_OP_GEMM);
    d.num_tensors = 3;
    d.tensors[0] = tdesc(in, 2, SHP(8, 16), 0);
    d.tensors[1] = tdesc(in, 2, SHP(16, 4), 0);
    d.tensors[2] = tdesc(out, 2, SHP(8, 4), 0);
    d.p.gemm.alpha = 1.0f;
    return d;
}

static ocu_op_desc d_rms(ocu_dtype dt, int weight)
{
    ocu_op_desc d = op_desc(OCU_OP_RMS_NORM);
    d.num_tensors = weight ? 3 : 2;
    d.tensors[0] = tdesc(dt, 2, SHP(4, 16), 0);
    if (weight) {
        d.tensors[1] = tdesc(dt, 1, SHP(16), 0);
    }
    d.tensors[weight ? 2 : 1] = tdesc(dt, 2, SHP(4, 16), 0);
    d.p.rms_norm.eps = 1e-6f;
    return d;
}

static ocu_op_desc d_softmax(ocu_dtype dt)
{
    ocu_op_desc d = op_desc(OCU_OP_SOFTMAX);
    d.num_tensors = 2;
    d.tensors[0] = tdesc(dt, 2, SHP(4, 16), 0);
    d.tensors[1] = tdesc(dt, 2, SHP(4, 16), 0);
    d.p.softmax.scale = 1.0f;
    return d;
}

/* Applies malformation `m` to a tensor; returns 0 when it does not apply. */
static int malform(ocu_tensor_desc *t, int m)
{
    uint32_t r = t->rank;
    switch (m) {
    case 0: t->rank = 0; return 1;
    case 1: t->rank = 5; return 1;
    case 2: t->shape[0] = 0; return 1;
    case 3: t->shape[r - 1] = -1; return 1;
    case 4: t->stride[r - 1] = 2; return 1;
    case 5:
        if (r < 2) return 0;
        t->stride[r - 2] = t->shape[r - 1] - 1; /* rows overlap */
        return 1;
    case 6: t->dtype = OCU_DT_NONE; return 1;
    case 7: t->dtype = (ocu_dtype)77; return 1;
    case 8:
        if (r < 2) return 0;
        t->shape[0] = INT64_MAX / 8; /* element count overflows int64 */
        return 1;
    case 9:
        if (r < 2) return 0;
        t->shape[0] = 3;
        t->stride[0] = INT64_MAX / 2 + 1; /* span overflows int64 */
        return 1;
    case 10: t->shape[r - 1] = 0; return 1;
    default: return 0;
    }
}

static void test_descriptor_errors(void)
{
    const ocu_dtype f = OCU_DT_F32;
    ocu_op_desc bases[5] = { d_add(OCU_OP_ADD, f), d_add(OCU_OP_MUL, f), d_gemm(f, f),
                             d_rms(f, 1), d_softmax(f) };
    for (int bi = 0; bi < 5; ++bi) {
        set_case("valid base %s", ocu_op_name(bases[bi].op));
        EXPECT(bases[bi], OCU_OK);
        for (uint32_t slot = 0; slot < bases[bi].num_tensors; ++slot) {
            for (int m = 0; m <= 10; ++m) {
                ocu_op_desc d = bases[bi];
                if (!malform(&d.tensors[slot], m)) {
                    continue;
                }
                set_case("%s slot %u malformation %d (3)", ocu_op_name(d.op), slot, m);
                EXPECT(d, OCU_ERR_INVALID_VALUE);
            }
        }
    }

    ocu_op_desc d = d_add(OCU_OP_ADD, f);
    set_case("struct_size too small (4)");
    d.struct_size = 0;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.struct_size = (uint32_t)offsetof(ocu_op_desc, p);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.struct_size = (uint32_t)sizeof(ocu_op_desc) - 1u;
    EXPECT(d, OCU_ERR_INVALID_VALUE);

    set_case("unknown flag bits (4)");
    d = d_add(OCU_OP_ADD, f);
    d.flags = 2u;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.flags = 0x80000000u;
    EXPECT(d, OCU_ERR_INVALID_VALUE);

    set_case("operator value not in the header (4.5)");
    d = d_add(OCU_OP_ADD, f);
    d.op = (ocu_op)0;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.op = (ocu_op)8;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.op = (ocu_op)99;
    EXPECT(d, OCU_ERR_INVALID_VALUE);

    set_case("num_tensors other than listed (4.5)");
    static const uint32_t bad_add[] = { 0, 1, 2, 4, 5, 100 };
    for (size_t i = 0; i < sizeof bad_add / sizeof bad_add[0]; ++i) {
        d = d_add(OCU_OP_ADD, f);
        d.num_tensors = bad_add[i];
        if (bad_add[i] == 4) {
            d.tensors[3] = d.tensors[2];
        }
        EXPECT(d, OCU_ERR_INVALID_VALUE);
    }
    d = d_gemm(f, f);
    d.num_tensors = 2;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.num_tensors = 4;
    d.tensors[3] = d.tensors[2];
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_rms(f, 1);
    d.num_tensors = 1;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.num_tensors = 4;
    d.tensors[3] = d.tensors[2];
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_softmax(f);
    d.num_tensors = 1;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.num_tensors = 3;
    d.tensors[2] = d.tensors[1];
    EXPECT(d, OCU_ERR_INVALID_VALUE);

    set_case("entries at and after rank are ignored (3)");
    d = d_add(OCU_OP_ADD, f);
    for (int i = 0; i < 3; ++i) {
        d.tensors[i].shape[2] = 0;
        d.tensors[i].shape[3] = -7;
        d.tensors[i].stride[2] = -1;
        d.tensors[i].stride[3] = 12345;
    }
    EXPECT(d, OCU_OK);
    d = d_gemm(f, f);
    d.tensors[0].shape[3] = 0;
    d.tensors[2].stride[2] = -9;
    EXPECT(d, OCU_OK);

    /* GEMM rules (4.1) */
    set_case("gemm trans other than 0/1");
    d = d_gemm(f, f);
    d.p.gemm.trans_a = 2;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_gemm(f, f);
    d.p.gemm.trans_b = 7;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    set_case("gemm M/N/K mismatch");
    d = d_gemm(f, f);
    d.tensors[1] = tdesc(f, 2, SHP(15, 4), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_gemm(f, f);
    d.tensors[2] = tdesc(f, 2, SHP(8, 5), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_gemm(f, f);
    d.tensors[2] = tdesc(f, 2, SHP(9, 4), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_gemm(f, f);
    d.p.gemm.trans_a = 1; /* A [8,16] read as [K=8, M=16] */
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    set_case("gemm ranks differ");
    d = d_gemm(f, f);
    d.tensors[2] = tdesc(f, 3, SHP(1, 8, 4), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    set_case("gemm batch rule");
    d = d_gemm(f, f);
    d.tensors[0] = tdesc(f, 3, SHP(2, 8, 16), 0);
    d.tensors[1] = tdesc(f, 3, SHP(1, 16, 4), 0);
    d.tensors[2] = tdesc(f, 3, SHP(3, 8, 4), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.tensors[0] = tdesc(f, 3, SHP(4, 8, 16), 0);
    d.tensors[1] = tdesc(f, 3, SHP(2, 16, 4), 0);
    d.tensors[2] = tdesc(f, 3, SHP(4, 8, 4), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.tensors[0] = tdesc(f, 3, SHP(3, 8, 16), 0);
    d.tensors[1] = tdesc(f, 3, SHP(3, 16, 4), 0);
    d.tensors[2] = tdesc(f, 3, SHP(1, 8, 4), 0); /* C batch 1, A/B batch 3 */
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.tensors[2] = tdesc(f, 3, SHP(3, 8, 4), 0);
    set_case("gemm rank 3 valid");
    EXPECT(d, OCU_OK);
    set_case("gemm dtype matching rule (4.1, 4.5)");
    d = d_gemm(f, f);
    d.tensors[1].dtype = OCU_DT_F16;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_gemm(OCU_DT_F16, OCU_DT_BF16);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_gemm(OCU_DT_BF16, OCU_DT_F16);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_gemm(f, OCU_DT_F16);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    {
        /* Spec ambiguity: C i32 breaks the matching rule of 4.1 (INVALID_VALUE)
         * and is a dtype GEMM is not defined for (4.5, NOT_SUPPORTED). */
        d = d_gemm(f, OCU_DT_I32);
        ocu_status st = qp(&d, NULL, NULL);
        CHECK(st == OCU_ERR_INVALID_VALUE || st == OCU_ERR_NOT_SUPPORTED);
    }
    set_case("gemm int8 reserved (4.1)");
    d = d_gemm(OCU_DT_I8, OCU_DT_I8);
    EXPECT(d, OCU_ERR_NOT_SUPPORTED);
    set_case("gemm block-quantised (4.5)");
    d = op_desc(OCU_OP_GEMM);
    d.num_tensors = 3;
    d.tensors[0] = tdesc(OCU_DT_Q8_B32, 2, SHP(8, 32), 0);
    d.tensors[1] = tdesc(OCU_DT_Q8_B32, 2, SHP(32, 32), 0);
    d.tensors[2] = tdesc(f, 2, SHP(8, 32), 0);
    d.p.gemm.alpha = 1.0f;
    EXPECT(d, OCU_ERR_NOT_SUPPORTED);

    /* ADD / MUL rules (4.2) */
    for (int oi = 0; oi < 2; ++oi) {
        ocu_op op = oi ? OCU_OP_MUL : OCU_OP_ADD;
        set_case("%s rules (4.2)", ocu_op_name(op));
        d = d_add(op, f);
        d.tensors[1].dtype = OCU_DT_F16;
        EXPECT(d, OCU_ERR_INVALID_VALUE);
        d = d_add(op, f);
        d.tensors[2].dtype = OCU_DT_BF16;
        EXPECT(d, OCU_ERR_INVALID_VALUE);
        d = d_add(op, f);
        d.tensors[1] = tdesc(f, 1, SHP(16), 0);
        EXPECT(d, OCU_ERR_INVALID_VALUE);
        d = d_add(op, f);
        d.tensors[2] = tdesc(f, 2, SHP(4, 15), 0);
        EXPECT(d, OCU_ERR_INVALID_VALUE);
        d = d_add(op, f);
        d.tensors[1] = tdesc(f, 2, SHP(4, 3), 0);
        EXPECT(d, OCU_ERR_INVALID_VALUE);
        d = d_add(op, f);
        d.tensors[0] = tdesc(f, 2, SHP(4, 1), 0);
        d.tensors[2] = tdesc(f, 2, SHP(4, 1), 0); /* a broadcasting is not allowed */
        EXPECT(d, OCU_ERR_INVALID_VALUE);
        set_case("%s undefined dtypes (4.5)", ocu_op_name(op));
        d = d_add(op, OCU_DT_I32);
        EXPECT(d, OCU_ERR_NOT_SUPPORTED);
        d = d_add(op, OCU_DT_I8);
        EXPECT(d, OCU_ERR_NOT_SUPPORTED);
        d = op_desc(op);
        d.num_tensors = 3;
        for (int i = 0; i < 3; ++i) {
            d.tensors[i] = tdesc(OCU_DT_Q8_B32, 2, SHP(4, 32), 0);
        }
        EXPECT(d, OCU_ERR_NOT_SUPPORTED);
        set_case("%s malformed before NOT_SUPPORTED (4.5)", ocu_op_name(op));
        d = d_add(op, OCU_DT_I32);
        d.tensors[0].shape[0] = 0;
        EXPECT(d, OCU_ERR_INVALID_VALUE);
    }

    /* RMS_NORM rules (4.3) */
    set_case("rms_norm rules (4.3)");
    d = d_rms(f, 1);
    d.p.rms_norm.eps = -1.0f;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.p.rms_norm.eps = NAN;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.p.rms_norm.eps = 0.0f;
    EXPECT(d, OCU_OK);
    d = d_rms(f, 1);
    d.tensors[1] = tdesc(f, 1, SHP(17), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.tensors[1] = tdesc(f, 2, SHP(1, 16), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.tensors[1] = tdesc(OCU_DT_F16, 1, SHP(16), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_rms(f, 0);
    d.tensors[1] = tdesc(f, 2, SHP(4, 15), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_rms(f, 0);
    d.tensors[1].dtype = OCU_DT_F16;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_rms(f, 0);
    d.tensors[1] = tdesc(f, 3, SHP(1, 4, 16), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    set_case("rms_norm undefined dtypes (4.5)");
    d = d_rms(OCU_DT_I32, 0);
    EXPECT(d, OCU_ERR_NOT_SUPPORTED);
    d = d_rms(OCU_DT_I8, 1);
    EXPECT(d, OCU_ERR_NOT_SUPPORTED);

    /* SOFTMAX rules (4.4) */
    set_case("softmax rules (4.4)");
    d = d_softmax(f);
    d.p.softmax.scale = INFINITY;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.p.softmax.scale = -INFINITY;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d.p.softmax.scale = NAN;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_softmax(f);
    d.tensors[1] = tdesc(f, 2, SHP(4, 8), 0);
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = d_softmax(f);
    d.tensors[1].dtype = OCU_DT_BF16;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    set_case("softmax undefined dtype (4.5)");
    d = d_softmax(OCU_DT_I8);
    EXPECT(d, OCU_ERR_NOT_SUPPORTED);

    /* VAR_ROWS with a rank-1 output (4.7) */
    set_case("VAR_ROWS with rank-1 output (4.7)");
    d = op_desc(OCU_OP_ADD);
    d.num_tensors = 3;
    for (int i = 0; i < 3; ++i) {
        d.tensors[i] = tdesc(f, 1, SHP(16), 0);
    }
    d.flags = OCU_OP_FLAG_VAR_ROWS;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = op_desc(OCU_OP_SOFTMAX);
    d.num_tensors = 2;
    d.tensors[0] = d.tensors[1] = tdesc(f, 1, SHP(16), 0);
    d.p.softmax.scale = 1.0f;
    d.flags = OCU_OP_FLAG_VAR_ROWS;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    d = op_desc(OCU_OP_RMS_NORM);
    d.num_tensors = 2;
    d.tensors[0] = d.tensors[1] = tdesc(f, 1, SHP(16), 0);
    d.flags = OCU_OP_FLAG_VAR_ROWS;
    EXPECT(d, OCU_ERR_INVALID_VALUE);
    ocu_op_support s;
    if (sup(OCU_OP_ADD, f, &s) && !s.var_rows) {
        set_case("VAR_ROWS without support.var_rows (4.7)");
        d = d_add(OCU_OP_ADD, f);
        d.flags = OCU_OP_FLAG_VAR_ROWS;
        EXPECT(d, OCU_ERR_NOT_SUPPORTED);
    }

    /* QMATMUL reserved (4.5) */
    set_case("qmatmul reserved (4.5)");
    d = d_gemm(f, f);
    d.op = OCU_OP_QMATMUL;
    EXPECT(d, OCU_ERR_NOT_SUPPORTED);
    d.tensors[0].shape[0] = 0;
    set_case("qmatmul malformed before NOT_SUPPORTED (4.5)");
    EXPECT(d, OCU_ERR_INVALID_VALUE);

    /* NULL arguments (5) */
    set_case("NULL device / output pointers (5)");
    d = d_add(OCU_OP_ADD, f);
    ocu_op_caps caps;
    ocu_op_plan *p = (ocu_op_plan *)(uintptr_t)64;
    CHECK_ST(G.t->query(NULL, &d, &caps), OCU_ERR_INVALID_DEVICE);
    CHECK_ST(G.t->plan(NULL, &d, &p), OCU_ERR_INVALID_DEVICE);
    CHECK(p == NULL);
    CHECK_ST(G.t->query(G.dev, &d, NULL), OCU_ERR_INVALID_VALUE);
    CHECK_ST(G.t->plan(G.dev, &d, NULL), OCU_ERR_INVALID_VALUE);
    CHECK_ST(G.t->destroy(NULL), OCU_ERR_INVALID_HANDLE);
    ocu_tensor ts[3];
    memset(ts, 0, sizeof ts);
    CHECK_ST(G.t->execute(NULL, NULL, ts, 3), OCU_ERR_INVALID_HANDLE);
    if (G.t->stats) {
        ocu_op_stats st;
        CHECK_ST(G.t->stats(NULL, &st), OCU_ERR_INVALID_DEVICE);
        CHECK_ST(G.t->stats(G.dev, NULL), OCU_ERR_INVALID_VALUE);
        CHECK_ST(G.t->stats(G.dev, &st), OCU_OK);
    }
}

/* ---- execute() checks (5.4) ---------------------------------------------- */

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

static void upload_rand_f32(void *dptr, size_t n)
{
    float *h = malloc(n * sizeof *h);
    for (size_t i = 0; i < n; ++i) {
        h[i] = (float)urand(-1.0, 1.0);
    }
    CHECK_ST(G.b->mem_copy(G.dev, NULL, dptr, h, n * sizeof *h, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    free(h);
}

static void set3(ocu_tensor *ts, void *a, void *b, void *y, const ocu_op_desc *d)
{
    ts[0].data = a;
    ts[1].data = b;
    ts[2].data = y;
    for (int i = 0; i < 3; ++i) {
        ts[i].desc = d->tensors[i];
    }
}

static void test_execute_errors(void)
{
    const ocu_dtype f = OCU_DT_F32;
    ocu_op_support s;
    if (!sup(OCU_OP_ADD, f, &s)) {
        printf("NOTE add f32 not supported: execute checks skipped\n");
        ++g_skips;
        return;
    }
    uint32_t al = s.align_bytes ? s.align_bytes : 1u;
    ocu_op_desc d = d_add(OCU_OP_ADD, f);
    ocu_op_plan *p = NULL;
    set_case("execute checks: plan add f32 [4,16]");
    if (!CHECK_ST(qp(&d, NULL, &p), OCU_OK) || p == NULL) {
        return;
    }
    const size_t bytes = 4 * 16 * 4;
    void *da = dalloc(bytes), *db = dalloc(bytes), *dy = dalloc(bytes);
    void *big = dalloc(8192), *dc = dalloc(bytes);
    if (!da || !db || !dy || !big || !dc) {
        return;
    }
    upload_rand_f32(da, 64);
    upload_rand_f32(db, 64);
    upload_rand_f32(dy, 64);
    upload_rand_f32(big, 2048);
    float before[64], after[64];
    CHECK_ST(G.b->mem_copy(G.dev, NULL, before, dy, bytes, OCU_COPY_DEVICE_TO_HOST), OCU_OK);

    ocu_op_stats s0, s1;
    int have = get_stats(&s0);
    ocu_tensor ts[4];

    set_case("execute: shape differs from plan (5.4)");
    ocu_op_desc d2 = d_add(OCU_OP_ADD, f);
    for (int i = 0; i < 3; ++i) {
        d2.tensors[i] = tdesc(f, 2, SHP(4, 8), 0);
    }
    set3(ts, da, db, dy, &d2);
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set_case("execute: fewer rows without VAR_ROWS (5.4)");
    for (int i = 0; i < 3; ++i) {
        d2.tensors[i] = tdesc(f, 2, SHP(3, 16), 0);
    }
    set3(ts, da, db, dy, &d2);
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set_case("execute: stride differs from plan (5.4)");
    set3(ts, big, db, dy, &d);
    ts[0].desc = tdesc(f, 2, SHP(4, 16), 16);
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set_case("execute: dtype differs from plan (5.4)");
    set3(ts, da, db, dy, &d);
    ts[1].desc.dtype = OCU_DT_F16;
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set_case("execute: rank differs from plan (5.4)");
    set3(ts, da, db, dy, &d);
    ts[2].desc = tdesc(f, 3, SHP(1, 4, 16), 0);
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set_case("execute: num_tensors differs (5.4)");
    set3(ts, da, db, dy, &d);
    ts[3] = ts[2];
    CHECK_ST(G.t->execute(p, NULL, ts, 2), OCU_ERR_INVALID_VALUE);
    CHECK_ST(G.t->execute(p, NULL, ts, 4), OCU_ERR_INVALID_VALUE);
    CHECK_ST(G.t->execute(p, G.q, ts, 0), OCU_ERR_INVALID_VALUE);
    set_case("execute: host pointer (5.4)");
    float *host = malloc(bytes + 256);
    memset(host, 0, bytes + 256);
    set3(ts, host, db, dy, &d);
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set3(ts, da, db, host, &d);
    CHECK_ST(G.t->execute(p, G.q, ts, 3), OCU_ERR_INVALID_VALUE);
    set_case("execute: span leaves the allocation (5.4)");
    set3(ts, (char *)da + 64, db, dy, &d);
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set3(ts, da, db, (char *)dy + 256, &d);
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    if (al > 4) {
        set_case("execute: pointer breaks align_bytes=%u (5.4)", al);
        set3(ts, (char *)big + 4, db, dy, &d);
        CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
        set3(ts, da, db, (char *)big + al / 2, &d);
        CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    }
    set_case("execute: y overlaps b (4.2)");
    set3(ts, da, db, db, &d);
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set_case("execute: y partly overlaps a (4.2)");
    size_t step = al > 64 ? al : 64;
    set3(ts, big, db, (char *)big + step, &d);
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
    set_case("execute errors queue nothing (5.4)");
    CHECK_ST(G.b->queue_sync(G.q), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, NULL, after, dy, bytes, OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK(memcmp(before, after, bytes) == 0);
    if (have && get_stats(&s1)) {
        CHECK(s1.op_calls[OCU_OP_ADD] == s0.op_calls[OCU_OP_ADD]);
    }

    set_case("execute: valid calls (in place, aligned offset, aliased inputs)");
    set3(ts, dc, db, dc, &d); /* y is a: same pointer and descriptor */
    upload_rand_f32(dc, 64);
    CHECK_ST(G.t->execute(p, NULL, ts, 3), OCU_OK);
    set3(ts, (char *)big + (al > 64 ? al : 64), db, dy, &d);
    CHECK_ST(G.t->execute(p, G.q, ts, 3), OCU_OK);
    set3(ts, da, da, dy, &d); /* inputs may be the same */
    CHECK_ST(G.t->execute(p, G.q, ts, 3), OCU_OK);
    CHECK_ST(G.b->queue_sync(G.q), OCU_OK);
    if (have && get_stats(&s1)) {
        CHECK(s1.op_calls[OCU_OP_ADD] == s0.op_calls[OCU_OP_ADD] + 3);
    }

    set_case("execute: same pointer as a, different descriptor (4.2)");
    ocu_op_desc d3 = d_add(OCU_OP_ADD, f);
    d3.tensors[0] = tdesc(f, 2, SHP(4, 16), 16); /* a row-strided, y dense */
    ocu_op_plan *p3 = NULL;
    if (CHECK_ST(qp(&d3, NULL, &p3), OCU_OK) && p3) {
        set3(ts, big, db, big, &d3);
        CHECK_ST(G.t->execute(p3, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
        set3(ts, big, db, dy, &d3);
        CHECK_ST(G.t->execute(p3, NULL, ts, 3), OCU_OK);
        CHECK_ST(G.t->destroy(p3), OCU_OK);
    }

    set_case("execute: destroyed plan (5.4)");
    ocu_op_plan *p4 = NULL;
    if (CHECK_ST(qp(&d, NULL, &p4), OCU_OK) && p4) {
        CHECK_ST(G.t->destroy(p4), OCU_OK);
        set3(ts, da, db, dy, &d);
        CHECK_ST(G.t->execute(p4, NULL, ts, 3), OCU_ERR_INVALID_HANDLE);
    }
    CHECK_ST(G.t->destroy(p), OCU_OK);

    /* GEMM overlap (4.1) */
    set_case("execute: gemm C overlaps A or B (4.1)");
    ocu_op_desc g = op_desc(OCU_OP_GEMM);
    g.num_tensors = 3;
    for (int i = 0; i < 3; ++i) {
        g.tensors[i] = tdesc(f, 2, SHP(4, 16), 0);
    }
    g.tensors[1] = tdesc(f, 2, SHP(16, 16), 0);
    g.p.gemm.alpha = 1.0f;
    ocu_op_plan *pg = NULL;
    void *ga = dalloc(1024), *gb = dalloc(1024);
    if (ga && gb && CHECK_ST(qp(&g, NULL, &pg), OCU_OK) && pg) {
        upload_rand_f32(ga, 256);
        upload_rand_f32(gb, 256);
        set3(ts, ga, gb, ga, &g);
        CHECK_ST(G.t->execute(pg, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
        set3(ts, ga, gb, gb, &g);
        CHECK_ST(G.t->execute(pg, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
        set3(ts, gb, gb, dy, &g); /* A and B may share memory */
        CHECK_ST(G.t->execute(pg, NULL, ts, 3), OCU_OK);
        CHECK_ST(G.t->destroy(pg), OCU_OK);
    }
    dfree(ga);
    dfree(gb);

    /* VAR_ROWS execute rules (4.7) */
    if (s.var_rows) {
        ocu_op_desc v = d_add(OCU_OP_ADD, f);
        for (int i = 0; i < 3; ++i) {
            v.tensors[i] = tdesc(f, 2, SHP(8, 16), 0);
        }
        v.flags = OCU_OP_FLAG_VAR_ROWS;
        ocu_op_plan *pv = NULL;
        set_case("var rows add plan [8,16]");
        void *va = dalloc(16 * 16 * 4), *vb = dalloc(16 * 16 * 4), *vy = dalloc(16 * 16 * 4);
        if (va && vb && vy && CHECK_ST(qp(&v, NULL, &pv), OCU_OK) && pv) {
            upload_rand_f32(va, 256);
            upload_rand_f32(vb, 256);
            ocu_op_desc e = v;
            set_case("var rows: more rows than planned (4.7)");
            for (int i = 0; i < 3; ++i) e.tensors[i].shape[0] = 9;
            set3(ts, va, vb, vy, &e);
            CHECK_ST(G.t->execute(pv, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
            set_case("var rows: zero rows (3)");
            for (int i = 0; i < 3; ++i) e.tensors[i].shape[0] = 0;
            set3(ts, va, vb, vy, &e);
            CHECK_ST(G.t->execute(pv, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
            set_case("var rows: m' differs between slots (4.7)");
            for (int i = 0; i < 3; ++i) e.tensors[i].shape[0] = 3;
            e.tensors[2].shape[0] = 4;
            set3(ts, va, vb, vy, &e);
            CHECK_ST(G.t->execute(pv, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
            set_case("var rows: other dimension changed (4.7)");
            for (int i = 0; i < 3; ++i) e.tensors[i] = tdesc(f, 2, SHP(4, 8), 0);
            set3(ts, va, vb, vy, &e);
            CHECK_ST(G.t->execute(pv, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
            set_case("var rows: stride changed (4.7)");
            for (int i = 0; i < 3; ++i) e.tensors[i] = tdesc(f, 2, SHP(4, 16), 0);
            e.tensors[0].stride[0] = 17;
            set3(ts, va, vb, vy, &e);
            CHECK_ST(G.t->execute(pv, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
            set_case("var rows: 4 of 8 rows OK (4.7)");
            for (int i = 0; i < 3; ++i) e.tensors[i] = tdesc(f, 2, SHP(4, 16), 0);
            set3(ts, va, vb, vy, &e);
            CHECK_ST(G.t->execute(pv, NULL, ts, 3), OCU_OK);
            CHECK_ST(G.t->destroy(pv), OCU_OK);
        }
        ocu_op_desc gv = d_gemm(f, f); /* A [8,16] B [16,4] C [8,4] */
        gv.flags = OCU_OP_FLAG_VAR_ROWS;
        set_case("var rows gemm plan M=8");
        if (va && vb && vy && CHECK_ST(qp(&gv, NULL, &pv), OCU_OK) && pv) {
            ocu_op_desc e = gv;
            set_case("var rows gemm: A and C rows differ (4.7)");
            e.tensors[0].shape[0] = 5;
            e.tensors[2].shape[0] = 6;
            set3(ts, va, vb, vy, &e);
            CHECK_ST(G.t->execute(pv, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
            set_case("var rows gemm: more rows than planned (4.7)");
            e.tensors[0].shape[0] = 9;
            e.tensors[2].shape[0] = 9;
            set3(ts, va, vb, vy, &e);
            CHECK_ST(G.t->execute(pv, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
            set_case("var rows gemm: B does not vary (4.7)");
            e = gv;
            e.tensors[0].shape[0] = 5;
            e.tensors[2].shape[0] = 5;
            e.tensors[1].shape[0] = 8;
            e.tensors[0].shape[1] = 8;
            set3(ts, va, vb, vy, &e);
            CHECK_ST(G.t->execute(pv, NULL, ts, 3), OCU_ERR_INVALID_VALUE);
            set_case("var rows gemm: 5 of 8 rows OK (4.7)");
            e = gv;
            e.tensors[0].shape[0] = 5;
            e.tensors[2].shape[0] = 5;
            set3(ts, va, vb, vy, &e);
            CHECK_ST(G.t->execute(pv, G.q, ts, 3), OCU_OK);
            CHECK_ST(G.b->queue_sync(G.q), OCU_OK);
            CHECK_ST(G.t->destroy(pv), OCU_OK);
        }
        dfree(va);
        dfree(vb);
        dfree(vy);
    }

    free(host);
    dfree(da);
    dfree(db);
    dfree(dy);
    dfree(big);
    dfree(dc);
}

/* ---- ordering on a created queue (5.4) ----------------------------------- */

static void test_queue_order(void)
{
    const ocu_dtype f = OCU_DT_F32;
    if (!sup(OCU_OP_ADD, f, NULL) || !sup(OCU_OP_MUL, f, NULL) || !sup(OCU_OP_SOFTMAX, f, NULL)) {
        printf("NOTE queue ordering test skipped\n");
        ++g_skips;
        return;
    }
    set_case("queue ordering: mul -> add -> softmax, overwrite after execute");
    const int64_t R = 37, N = 129, E = R * N;
    const size_t bytes = (size_t)E * 4;
    ocu_op_desc dm = d_add(OCU_OP_MUL, f), da = d_add(OCU_OP_ADD, f), ds = d_softmax(f);
    for (int i = 0; i < 3; ++i) {
        dm.tensors[i] = da.tensors[i] = tdesc(f, 2, SHP(R, N), 0);
    }
    ds.tensors[0] = ds.tensors[1] = tdesc(f, 2, SHP(R, N), 0);
    ocu_op_plan *pm = NULL, *pa = NULL, *ps = NULL;
    if (!CHECK_ST(qp(&dm, NULL, &pm), OCU_OK) || !CHECK_ST(qp(&da, NULL, &pa), OCU_OK) ||
        !CHECK_ST(qp(&ds, NULL, &ps), OCU_OK)) {
        if (pm) G.t->destroy(pm);
        if (pa) G.t->destroy(pa);
        if (ps) G.t->destroy(ps);
        return;
    }
    float *ha = malloc(bytes), *hb = malloc(bytes), *hc = malloc(bytes), *hz = malloc(bytes);
    float *y3 = malloc(bytes), *y4 = malloc(bytes);
    for (int64_t i = 0; i < E; ++i) {
        ha[i] = (float)urand(-2, 2);
        hb[i] = (float)urand(-2, 2);
        hc[i] = (float)urand(-2, 2);
        hz[i] = 0.0f;
    }
    void *a = dalloc(bytes), *b = dalloc(bytes), *c = dalloc(bytes);
    void *t1 = dalloc(bytes), *t2 = dalloc(bytes), *t3 = dalloc(bytes), *t4 = dalloc(bytes);
    ocu_queue *q = G.q;
    ocu_tensor ts[3];
    CHECK_ST(G.b->mem_copy(G.dev, q, a, ha, bytes, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, q, b, hb, bytes, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, q, c, hc, bytes, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    set3(ts, a, b, t1, &dm);
    CHECK_ST(G.t->execute(pm, q, ts, 3), OCU_OK);
    set3(ts, t1, c, t2, &da);
    CHECK_ST(G.t->execute(pa, q, ts, 3), OCU_OK);
    ts[0].data = t2;
    ts[1].data = t3;
    ts[0].desc = ds.tensors[0];
    ts[1].desc = ds.tensors[1];
    CHECK_ST(G.t->execute(ps, q, ts, 2), OCU_OK);
    /* Overwrite a after the work that read it; then reuse it. */
    CHECK_ST(G.b->mem_copy(G.dev, q, a, hz, bytes, OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    set3(ts, a, b, t4, &dm);
    CHECK_ST(G.t->execute(pm, q, ts, 3), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, q, y3, t3, bytes, OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK_ST(G.b->mem_copy(G.dev, q, y4, t4, bytes, OCU_COPY_DEVICE_TO_HOST), OCU_OK);
    CHECK_ST(G.b->queue_sync(q), OCU_OK);

    double ua = unit_round(OCU_DT_F32), tn = tiny_of(OCU_DT_F32);
    long bad = 0, bad0 = 0;
    for (int64_t r = 0; r < R; ++r) {
        double t[129], mx = -INFINITY, amax = 0.0, sum = 0.0;
        for (int64_t j = 0; j < N; ++j) {
            double v1 = round_to(f, (double)ha[r * N + j] * hb[r * N + j]);
            t[j] = round_to(f, v1 + hc[r * N + j]);
            mx = t[j] > mx ? t[j] : mx;
            amax = fabs(t[j]) > amax ? fabs(t[j]) : amax;
        }
        for (int64_t j = 0; j < N; ++j) {
            sum += exp(t[j] - mx);
        }
        for (int64_t j = 0; j < N; ++j) {
            double ref = exp(t[j] - mx) / sum;
            double rtol = (2.0 * (double)N + 2.0 * (fabs(t[j]) + amax) + 16.0) * ua + 2.0 * ua;
            if (!ok_bound(y3[r * N + j], ref, rtol * ref + tn)) {
                ++bad;
            }
            if (y4[r * N + j] != 0.0f) {
                ++bad0;
            }
        }
    }
    CHECK(bad == 0);  /* chain saw the inputs uploaded before it */
    CHECK(bad0 == 0); /* second mul saw the overwrite queued before it */
    CHECK_ST(G.t->destroy(pm), OCU_OK);
    CHECK_ST(G.t->destroy(pa), OCU_OK);
    CHECK_ST(G.t->destroy(ps), OCU_OK);
    dfree(a); dfree(b); dfree(c); dfree(t1); dfree(t2); dfree(t3); dfree(t4);
    free(ha); free(hb); free(hc); free(hz); free(y3); free(y4);
}

/* ---- the plan outlives other plans (5.3) --------------------------------- */

static void test_plan_lifetime(void)
{
    const ocu_dtype f = OCU_DT_F32;
    if (!sup(OCU_OP_ADD, f, NULL)) {
        return;
    }
    set_case("plan stays valid while others are created and destroyed (5.3)");
    ocu_op_desc d = d_add(OCU_OP_ADD, f);
    ocu_op_plan *keep = NULL;
    if (!CHECK_ST(qp(&d, NULL, &keep), OCU_OK) || !keep) {
        return;
    }
    for (int i = 0; i < 20; ++i) {
        ocu_op_desc o = d_add(i & 1 ? OCU_OP_MUL : OCU_OP_ADD, f);
        for (int j = 0; j < 3; ++j) {
            o.tensors[j] = tdesc(f, 2, SHP(i + 1, 8), 0);
        }
        ocu_op_plan *tmp = NULL;
        if (G.t->plan(G.dev, &o, &tmp) == OCU_OK) {
            G.t->destroy(tmp);
        }
    }
    void *a = dalloc(256), *b = dalloc(256), *y = dalloc(256);
    if (a && b && y) {
        upload_rand_f32(a, 64);
        upload_rand_f32(b, 64);
        ocu_tensor ts[3];
        set3(ts, a, b, y, &d);
        /* the descriptor was copied by plan: changing ours has no effect */
        d.tensors[0].shape[0] = 99;
        ts[0].desc.shape[0] = 4;
        CHECK_ST(G.t->execute(keep, NULL, ts, 3), OCU_OK);
    }
    CHECK_ST(G.t->destroy(keep), OCU_OK);
    dfree(a);
    dfree(b);
    dfree(y);
}

int main(int argc, char **argv)
{
    const char *name = argc > 1 ? argv[1] : "cpu";
    if (ocu_select_backend(name) != OCU_OK || (G.b = ocu_active_backend()) == NULL) {
        printf("test_ops %s: SKIP (backend not available)\n", name);
        return 77;
    }
    G.t = ocu_ops_of(G.b);
    if (G.t == NULL) {
        printf("test_ops %s: SKIP (no operator table)\n", name);
        return 77;
    }
    set_case("setup");
    CHECK(G.t->version >= OCU_OPS_VERSION);
    CHECK(G.t->struct_size >= sizeof(ocu_op_table));
    if (!CHECK_ST(G.b->dev_open(0, &G.dev), OCU_OK)) {
        return 1;
    }
    if (!CHECK_ST(G.b->queue_create(G.dev, &G.q), OCU_OK)) {
        return 1;
    }

    test_converters();
    test_support();
    test_descriptor_errors();
    test_execute_errors();
    test_gemm();
    test_elementwise();
    test_rows();
    test_rope();
    test_queue_order();
    test_plan_lifetime();

    set_case("teardown");
    CHECK_ST(G.b->queue_sync(G.q), OCU_OK);
    CHECK_ST(G.b->queue_destroy(G.q), OCU_OK);
    CHECK_ST(G.b->dev_close(G.dev), OCU_OK);

    if (g_skips) {
        printf("test_ops %s: %ld cases skipped (NOT_SUPPORTED / NOT_IMPLEMENTED)\n", name, g_skips);
    }
    printf("test_ops %s: %ld checks, %ld failures\n", name, G.checks, G.failures);
    return G.failures == 0 ? 0 : 1;
}

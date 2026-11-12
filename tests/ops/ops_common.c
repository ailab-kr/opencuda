/*
 * ops_common.c -- helpers shared by the operator table tests
 * (docs/spec/op-table.md). See ops_common.h.
 */
#include "ops_common.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

test_env G = { .rng = 0x9E3779B97F4A7C15ull };

/* ---- checks -------------------------------------------------------------- */

void set_case(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(G.where, sizeof G.where, fmt, ap);
    va_end(ap);
    G.details = 4;
}

void detail(const char *fmt, ...)
{
    if (G.details <= 0) {
        return;
    }
    --G.details;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "    ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

int check_at(int ok, const char *file, int line, const char *expr)
{
    ++G.checks;
    if (!ok) {
        ++G.failures;
        const char *base = strrchr(file, '/');
        fprintf(stderr, "FAIL %s:%d [%s]: %s\n", base ? base + 1 : file, line,
                G.where, expr);
    }
    return ok;
}

int check_st_at(ocu_status got, ocu_status want, const char *file, int line,
                const char *expr)
{
    ++G.checks;
    if (got != want) {
        ++G.failures;
        const char *base = strrchr(file, '/');
        fprintf(stderr, "FAIL %s:%d [%s]: %s = %s, want %s\n",
                base ? base + 1 : file, line, G.where, expr,
                ocu_status_name(got), ocu_status_name(want));
        return 0;
    }
    return 1;
}

/* ---- random numbers ------------------------------------------------------ */

uint64_t rng_next(void)
{
    uint64_t x = G.rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    G.rng = x;
    return x;
}

double urand(double lo, double hi)
{
    return lo + (hi - lo) * ((double)(rng_next() >> 11) * 0x1.0p-53);
}

int64_t irand(int64_t lo, int64_t hi)
{
    return lo + (int64_t)(rng_next() % (uint64_t)(hi - lo + 1));
}

/* ---- number formats ------------------------------------------------------ */

typedef struct fpfmt { int p, emin, emax; } fpfmt;

static fpfmt fmt_of(ocu_dtype dt)
{
    fpfmt f = { 24, -126, 127 };
    if (dt == OCU_DT_F16) {
        f.p = 11;
        f.emin = -14;
        f.emax = 15;
    } else if (dt == OCU_DT_BF16) {
        f.p = 8;
    }
    return f;
}

double round_to(ocu_dtype dt, double x)
{
    if (dt == OCU_DT_I32 || dt == OCU_DT_I8) {
        return x;
    }
    if (x == 0.0 || isnan(x) || isinf(x)) {
        return x;
    }
    fpfmt f = fmt_of(dt);
    int e;
    (void)frexp(fabs(x), &e);
    int E = e - 1; /* |x| in [2^E, 2^(E+1)) */
    if (E < f.emin) {
        E = f.emin;
    }
    double q = ldexp(1.0, E - (f.p - 1));
    double v = nearbyint(x / q) * q; /* default mode: ties to even */
    double maxv = ldexp(2.0 - ldexp(1.0, 1 - f.p), f.emax);
    if (fabs(v) > maxv) {
        return copysign(INFINITY, x);
    }
    return v;
}

double unit_round(ocu_dtype dt) { return ldexp(1.0, -fmt_of(dt).p); }
double tiny_of(ocu_dtype dt) { return ldexp(1.0, fmt_of(dt).emin); }

double ulp_at(ocu_dtype dt, double r)
{
    fpfmt f = fmt_of(dt);
    int E = f.emin;
    if (r != 0.0 && isfinite(r)) {
        int e;
        (void)frexp(fabs(r), &e);
        if (e - 1 > E) {
            E = e - 1;
        }
    }
    return ldexp(1.0, E - (f.p - 1));
}

uint16_t f16_encode(double v)
{
    uint16_t s = signbit(v) ? 0x8000u : 0u;
    if (isnan(v)) {
        return 0x7e00u;
    }
    if (isinf(v)) {
        return (uint16_t)(s | 0x7c00u);
    }
    double a = fabs(v);
    if (a < 0x1p-14) {
        return (uint16_t)(s | (uint16_t)(a * 0x1p24));
    }
    int e;
    double m = frexp(a, &e); /* a = m 2^e, m in [0.5, 1) */
    uint16_t mant = (uint16_t)((m * 2.0 - 1.0) * 1024.0);
    return (uint16_t)(s | (uint16_t)((e - 1 + 15) << 10) | mant);
}

double f16_decode(uint16_t h)
{
    int e = (h >> 10) & 31;
    int m = h & 1023;
    double s = (h & 0x8000u) ? -1.0 : 1.0;
    if (e == 0) {
        return s * ldexp((double)m, -24);
    }
    if (e == 31) {
        return m ? (double)NAN : s * (double)INFINITY;
    }
    return s * ldexp((double)(m + 1024), e - 25);
}

uint16_t bf16_encode(double v)
{
    float f = (float)v; /* exact: v is a bfloat16 value */
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    if (isnan(f)) {
        return (uint16_t)(((u >> 16) & 0x8000u) | 0x7fc0u);
    }
    return (uint16_t)(u >> 16);
}

double bf16_decode(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

double load_el(ocu_dtype dt, const uint8_t *buf, int64_t i)
{
    size_t o = (size_t)i;
    switch (dt) {
    case OCU_DT_F32: { float f; memcpy(&f, buf + o * 4, 4); return f; }
    case OCU_DT_F16: { uint16_t h; memcpy(&h, buf + o * 2, 2); return f16_decode(h); }
    case OCU_DT_BF16: { uint16_t h; memcpy(&h, buf + o * 2, 2); return bf16_decode(h); }
    case OCU_DT_I32: { int32_t x; memcpy(&x, buf + o * 4, 4); return x; }
    case OCU_DT_I8: { int8_t x; memcpy(&x, buf + o, 1); return x; }
    default: return NAN;
    }
}

void store_el(ocu_dtype dt, uint8_t *buf, int64_t i, double v)
{
    size_t o = (size_t)i;
    switch (dt) {
    case OCU_DT_F32: {
        float f = (float)round_to(dt, v);
        memcpy(buf + o * 4, &f, 4);
        break;
    }
    case OCU_DT_F16: {
        uint16_t h = f16_encode(round_to(dt, v));
        memcpy(buf + o * 2, &h, 2);
        break;
    }
    case OCU_DT_BF16: {
        uint16_t h = bf16_encode(round_to(dt, v));
        memcpy(buf + o * 2, &h, 2);
        break;
    }
    case OCU_DT_I32: { int32_t x = (int32_t)v; memcpy(buf + o * 4, &x, 4); break; }
    case OCU_DT_I8: { int8_t x = (int8_t)v; memcpy(buf + o, &x, 1); break; }
    default: break;
    }
}

int ok_bound(double y, double r, double bound)
{
    if (isnan(r)) {
        return isnan(y);
    }
    if (isinf(r)) {
        return y == r;
    }
    if (!isfinite(y)) {
        return 0;
    }
    return fabs(y - r) <= bound;
}

int ok_exact(ocu_dtype dt, double y, double r)
{
    if (isnan(r)) {
        return isnan(y);
    }
    if (isinf(r)) {
        return y == r;
    }
    double rr = round_to(dt, r);
    if (isinf(rr)) {
        return y == rr;
    }
    if (!isfinite(y)) {
        return 0;
    }
    if (rr == 0.0) {
        return y == 0.0; /* both zeros accepted */
    }
    if (y == rr) {
        return 1;
    }
    return fabs(rr) < tiny_of(dt) && y == 0.0; /* flushed subnormal */
}

/* ---- descriptors --------------------------------------------------------- */

ocu_tensor_desc tdesc(ocu_dtype dt, uint32_t rank, const int64_t *shape, int64_t pad)
{
    ocu_tensor_desc d;
    memset(&d, 0, sizeof d);
    d.dtype = dt;
    d.rank = rank;
    for (uint32_t k = 0; k < rank && k < OCU_OPS_MAX_RANK; ++k) {
        d.shape[k] = shape[k];
    }
    if (rank >= 1 && rank <= OCU_OPS_MAX_RANK) {
        d.stride[rank - 1] = 1;
        for (int k = (int)rank - 2; k >= 0; --k) {
            d.stride[k] = d.stride[k + 1] * d.shape[k + 1] + pad;
        }
    }
    return d;
}

int64_t desc_span(const ocu_tensor_desc *d)
{
    int64_t s = 1;
    for (uint32_t k = 0; k < d->rank; ++k) {
        s += (d->shape[k] - 1) * d->stride[k];
    }
    return s;
}

int desc_dense(const ocu_tensor_desc *d)
{
    for (uint32_t k = 0; k + 1 < d->rank; ++k) {
        if (d->stride[k] != d->stride[k + 1] * d->shape[k + 1]) {
            return 0;
        }
    }
    return 1;
}

int64_t desc_offset(const ocu_tensor_desc *d, const int64_t *idx)
{
    int64_t o = 0;
    for (uint32_t k = 0; k < d->rank; ++k) {
        o += idx[k] * d->stride[k];
    }
    return o;
}

int next_index(const ocu_tensor_desc *d, int64_t *idx)
{
    for (int k = (int)d->rank - 1; k >= 0; --k) {
        if (++idx[k] < d->shape[k]) {
            return 1;
        }
        idx[k] = 0;
    }
    return 0;
}

int64_t row_count(const ocu_tensor_desc *d)
{
    int64_t n = 1;
    for (uint32_t k = 0; k + 1 < d->rank; ++k) {
        n *= d->shape[k];
    }
    return n;
}

int64_t row_base(const ocu_tensor_desc *d, int64_t row)
{
    int64_t off = 0;
    for (int k = (int)d->rank - 2; k >= 0; --k) {
        off += (row % d->shape[k]) * d->stride[k];
        row /= d->shape[k];
    }
    return off;
}

ocu_op_desc op_desc(ocu_op op)
{
    ocu_op_desc d;
    memset(&d, 0, sizeof d);
    d.struct_size = (uint32_t)sizeof d;
    d.op = op;
    return d;
}

const char *shape_str(const ocu_tensor_desc *d, char *buf, size_t n)
{
    size_t used = 0;
    buf[0] = '\0';
    for (uint32_t k = 0; k < d->rank && k < OCU_OPS_MAX_RANK; ++k) {
        int w = snprintf(buf + used, n - used, k ? "x%lld" : "%lld", (long long)d->shape[k]);
        if (w < 0 || (size_t)w >= n - used) {
            break;
        }
        used += (size_t)w;
    }
    return buf;
}

/* ---- host images --------------------------------------------------------- */

void tb_init(tbuf *t, ocu_tensor_desc d, double fill)
{
    memset(t, 0, sizeof *t);
    t->d = d;
    t->e = d;
    t->alias = -1;
    int64_t span = desc_span(&d);
    t->bytes = (size_t)span * ocu_dtype_size(d.dtype);
    t->h = malloc(t->bytes);
    for (int64_t i = 0; i < span; ++i) {
        store_el(d.dtype, t->h, i, fill);
    }
}

void tb_alias(tbuf *t, const tbuf *src, int slot)
{
    memset(t, 0, sizeof *t);
    t->d = src->d;
    t->e = src->e;
    t->bytes = src->bytes;
    t->alias = slot;
}

void tb_free(tbuf *t)
{
    free(t->h);
    free(t->res);
    t->h = NULL;
    t->res = NULL;
}

void tb_fill(tbuf *t, double v)
{
    int64_t idx[4] = { 0, 0, 0, 0 };
    do {
        store_el(t->d.dtype, t->h, desc_offset(&t->d, idx), v);
    } while (next_index(&t->d, idx));
}

void tb_rand(tbuf *t, double lo, double hi)
{
    int64_t idx[4] = { 0, 0, 0, 0 };
    do {
        store_el(t->d.dtype, t->h, desc_offset(&t->d, idx), urand(lo, hi));
    } while (next_index(&t->d, idx));
}

void tb_rand_exp(tbuf *t, int elo, int ehi)
{
    int64_t idx[4] = { 0, 0, 0, 0 };
    do {
        double v = urand(1.0, 2.0) * ldexp(1.0, (int)irand(elo, ehi));
        if (rng_next() & 1u) {
            v = -v;
        }
        store_el(t->d.dtype, t->h, desc_offset(&t->d, idx), v);
    } while (next_index(&t->d, idx));
}

/* ---- running operators --------------------------------------------------- */

int get_stats(ocu_op_stats *s)
{
    memset(s, 0, sizeof *s);
    if (G.t->stats == NULL) {
        return 0;
    }
    return G.t->stats(G.dev, s) == OCU_OK;
}

ocu_tol_class tol_of(ocu_op op)
{
    switch (op) {
    case OCU_OP_GEMM:     return OCU_TOL_DOT;
    case OCU_OP_ADD:
    case OCU_OP_MUL:      return OCU_TOL_EXACT;
    case OCU_OP_RMS_NORM:
    case OCU_OP_SOFTMAX:  return OCU_TOL_ROW;
    case OCU_OP_ROPE:     return OCU_TOL_ULP;
    case OCU_OP_QMATMUL:  break;
    }
    return (ocu_tol_class)0;
}

static int is_pow2(uint32_t x) { return x != 0 && (x & (x - 1u)) == 0; }

/* query() said OK: the caps and the support summary must agree (5.1, 5.2). */
static void check_caps(const ocu_op_desc *d, const ocu_op_caps *c)
{
    check_at(c->tol == tol_of(d->op), __FILE__, __LINE__,
             "caps.tol is the operator's class (6.1)");
    check_at(c->tiles >= 1, __FILE__, __LINE__, "caps.tiles >= 1 (5.2)");
    ocu_dtype in = d->tensors[0].dtype;
    if (in == OCU_DT_F32) {
        check_at(c->accum_dtype == OCU_DT_F32, __FILE__, __LINE__,
                 "caps.accum_dtype is f32 for f32 inputs (5.2)");
    }
    check_at(c->accum_dtype == OCU_DT_F32 || c->accum_dtype == OCU_DT_F16 ||
                 c->accum_dtype == OCU_DT_BF16,
             __FILE__, __LINE__, "caps.accum_dtype is f32, f16 or bf16 (5.2)");

    ocu_op_support s;
    ocu_status ss = G.t->support(G.dev, d->op, in, &s);
    if (!check_st_at(ss, OCU_OK, __FILE__, __LINE__,
                     "support() for an (op, dtype) that query() accepts (5.1)")) {
        return;
    }
    check_at(c->accum_dtype == s.accum_dtype, __FILE__, __LINE__,
             "caps.accum_dtype == support.accum_dtype");
    uint32_t n = d->num_tensors;
    if (n == 0 || n > OCU_OPS_MAX_TENSORS) {
        return;
    }
    ocu_dtype out = d->tensors[n - 1].dtype;
    if ((unsigned)out < 32u) {
        check_at((s.out_dtypes & (1u << (unsigned)out)) != 0, __FILE__, __LINE__,
                 "support.out_dtypes has the output dtype");
    }
    int dense = 1;
    for (uint32_t i = 0; i < n; ++i) {
        check_at(d->tensors[i].rank <= s.max_rank, __FILE__, __LINE__,
                 "tensor rank <= support.max_rank");
        if (!desc_dense(&d->tensors[i])) {
            dense = 0;
        }
    }
    if (dense) {
        check_at((s.layouts & OCU_LAYOUT_DENSE) != 0, __FILE__, __LINE__,
                 "dense descriptor accepted => support.layouts has DENSE");
    } else {
        check_at((s.layouts & OCU_LAYOUT_ROW_STRIDED) != 0, __FILE__, __LINE__,
                 "row-strided descriptor accepted => support.layouts has ROW_STRIDED");
    }
    const ocu_tensor_desc *o = &d->tensors[n - 1];
    if (s.max_row > 0 && o->rank >= 1 && o->rank <= OCU_OPS_MAX_RANK) {
        check_at(o->shape[o->rank - 1] <= s.max_row, __FILE__, __LINE__,
                 "output row length <= support.max_row");
    }
    if (d->flags & OCU_OP_FLAG_VAR_ROWS) {
        check_at(s.var_rows == 1, __FILE__, __LINE__,
                 "VAR_ROWS descriptor accepted => support.var_rows == 1");
    }
    check_at(is_pow2(s.align_bytes), __FILE__, __LINE__,
             "support.align_bytes is a power of two");
}

ocu_status qp(const ocu_op_desc *d, ocu_op_caps *caps_out, ocu_op_plan **plan_out)
{
    ocu_op_stats s0, s1;
    int have = get_stats(&s0);
    ocu_op_caps caps;
    memset(&caps, 0, sizeof caps);
    ocu_status sq = G.t->query(G.dev, d, &caps);
    ocu_op_plan *p = (ocu_op_plan *)(uintptr_t)64; /* must be overwritten */
    ocu_status sp = G.t->plan(G.dev, d, &p);
    if (sp != OCU_ERR_OUT_OF_MEMORY) {
        check_st_at(sp, sq, __FILE__, __LINE__, "plan() status == query() status (5.2)");
    }
    if (sp != OCU_OK) {
        check_at(p == NULL, __FILE__, __LINE__, "failed plan() sets *plan = NULL (5.3)");
        p = NULL;
    } else {
        check_at(p != NULL, __FILE__, __LINE__, "plan() OK gives a plan");
    }
    if (have && get_stats(&s1)) {
        if (sp == OCU_OK) {
            check_at(s1.plans_created == s0.plans_created + 1 &&
                         s1.plans_refused == s0.plans_refused,
                     __FILE__, __LINE__, "stats: plans_created +1 (5.6)");
        } else {
            check_at(s1.plans_refused == s0.plans_refused + 1 &&
                         s1.plans_created == s0.plans_created,
                     __FILE__, __LINE__, "stats: plans_refused +1 (5.6)");
        }
    }
    if (sq == OCU_OK) {
        check_caps(d, &caps);
    }
    if (caps_out) {
        *caps_out = caps;
    }
    if (plan_out) {
        *plan_out = p;
    } else if (p) {
        check_st_at(G.t->destroy(p), OCU_OK, __FILE__, __LINE__, "destroy()");
    }
    return sp == OCU_OK ? sq : sp;
}

void expect_at(const ocu_op_desc *d, ocu_status want, const char *file, int line)
{
    check_st_at(qp(d, NULL, NULL), want, file, line, "query/plan status");
}

ocu_status run_op(ocu_op_desc *pd, tbuf *t, uint32_t n, int on_queue)
{
    pd->struct_size = (uint32_t)sizeof *pd;
    pd->num_tensors = n;
    for (uint32_t i = 0; i < n; ++i) {
        pd->tensors[i] = t[i].d;
    }
    ocu_op_plan *plan = NULL;
    ocu_status s = qp(pd, &G.last_caps, &plan);
    if (s != OCU_OK) {
        if (plan) {
            G.t->destroy(plan);
        }
        return s;
    }
    ocu_queue *q = on_queue ? G.q : NULL;
    ocu_status result = OCU_OK;
    for (uint32_t i = 0; i < n; ++i) {
        t[i].dptr = NULL;
        if (t[i].alias >= 0) {
            continue;
        }
        if (!CHECK_ST(G.b->mem_alloc(G.dev, t[i].bytes, &t[i].dptr), OCU_OK)) {
            t[i].dptr = NULL;
            result = OCU_ERR_UNKNOWN;
            goto out;
        }
        CHECK_ST(G.b->mem_copy(G.dev, q, t[i].dptr, t[i].h, t[i].bytes,
                               OCU_COPY_HOST_TO_DEVICE), OCU_OK);
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (t[i].alias >= 0) {
            t[i].dptr = t[t[i].alias].dptr;
        }
    }
    ocu_tensor ts[OCU_OPS_MAX_TENSORS];
    for (uint32_t i = 0; i < n; ++i) {
        ts[i].data = t[i].dptr;
        ts[i].desc = t[i].e;
    }
    ocu_op_stats s0, s1;
    int have = get_stats(&s0);
    result = G.t->execute(plan, q, ts, n);
    CHECK_ST(result, OCU_OK);
    tbuf *o = &t[n - 1];
    free(o->res);
    o->res = malloc(o->bytes);
    memset(o->res, 0xEE, o->bytes);
    CHECK_ST(G.b->mem_copy(G.dev, q, o->res, o->dptr, o->bytes, OCU_COPY_DEVICE_TO_HOST),
             OCU_OK);
    if (q) {
        CHECK_ST(G.b->queue_sync(q), OCU_OK);
    }
    if (have && get_stats(&s1) && result == OCU_OK && (unsigned)pd->op < OCU_OP_COUNT) {
        check_at(s1.op_calls[pd->op] == s0.op_calls[pd->op] + 1, __FILE__, __LINE__,
                 "stats: op_calls[op] +1 per successful execute (5.6)");
    }
out:
    CHECK_ST(G.t->destroy(plan), OCU_OK);
    for (uint32_t i = 0; i < n; ++i) {
        if (t[i].alias < 0 && t[i].dptr) {
            CHECK_ST(G.b->mem_free(G.dev, t[i].dptr), OCU_OK);
        }
        t[i].dptr = NULL;
    }
    if (result != OCU_OK) {
        free(t[n - 1].res);
        t[n - 1].res = NULL;
    }
    return result == OCU_OK ? OCU_OK : OCU_ERR_UNKNOWN;
}

void check_untouched(const tbuf *o, const uint8_t *init)
{
    if (o->res == NULL) {
        return;
    }
    int64_t span = desc_span(&o->d);
    size_t es = ocu_dtype_size(o->d.dtype);
    char *mask = calloc((size_t)span, 1);
    int64_t idx[4] = { 0, 0, 0, 0 };
    do {
        mask[desc_offset(&o->e, idx)] = 1;
    } while (next_index(&o->e, idx));
    long bad = 0;
    for (int64_t i = 0; i < span; ++i) {
        if (!mask[i] && memcmp(o->res + (size_t)i * es, init + (size_t)i * es, es) != 0) {
            if (bad++ == 0) {
                detail("element %lld outside the output region was written",
                       (long long)i);
            }
        }
    }
    free(mask);
    check_at(bad == 0, __FILE__, __LINE__,
             "memory outside the output tensor (padding, rows beyond m') untouched");
}

/* ---- GEMM ---------------------------------------------------------------- */

static ocu_tensor_desc gemm_mat(ocu_dtype dt, uint32_t rank, int64_t batch,
                                int64_t rows, int64_t cols, int64_t pad)
{
    return rank == 3 ? tdesc(dt, 3, SHP(batch, rows, cols), pad)
                     : tdesc(dt, 2, SHP(rows, cols), pad);
}

static int64_t moff(const ocu_tensor_desc *d, int64_t b, int64_t r, int64_t c)
{
    int64_t off = r * d->stride[d->rank - 2] + c * d->stride[d->rank - 1];
    if (d->rank == 3 && d->shape[0] > 1) {
        off += b * d->stride[0];
    }
    return off;
}

ocu_status run_gemm(const gemm_case *gc)
{
    uint32_t rank = gc->rank ? gc->rank : 2;
    int64_t m = gc->m, n = gc->n, k = gc->k;
    int64_t me = gc->m_exec ? gc->m_exec : m;
    int64_t bt = rank == 3 ? gc->bt : 1;
    int64_t ba = rank == 3 ? gc->ba : 1;
    int64_t bb = rank == 3 ? gc->bb : 1;
    ocu_dtype din = gc->din;
    ocu_dtype dout = gc->dout ? gc->dout : din;
    set_case("gemm %s->%s rank%u batch C%lld/A%lld/B%lld m=%lld(exec %lld) n=%lld k=%lld "
             "ta=%u tb=%u alpha=%g beta=%g pad=%lld cfill=%d abnan=%d var=%d queue=%d",
             ocu_dtype_name(din), ocu_dtype_name(dout), rank, (long long)bt,
             (long long)ba, (long long)bb, (long long)m, (long long)me, (long long)n,
             (long long)k, gc->ta, gc->tb, (double)gc->alpha, (double)gc->beta,
             (long long)gc->pad, gc->c_fill, gc->ab_nan, gc->var, gc->on_queue);

    tbuf t[3];
    int64_t pad = gc->pad;
    tb_init(&t[0], gemm_mat(din, rank, ba, gc->ta ? k : m, gc->ta ? m : k, pad), NAN);
    tb_init(&t[1], gemm_mat(din, rank, bb, gc->tb ? n : k, gc->tb ? k : n,
                            pad ? pad + 1 : 0), NAN);
    tb_init(&t[2], gemm_mat(dout, rank, bt, m, n, pad ? pad + 2 : 0), -3.0);
    if (gc->ab_nan) {
        tb_fill(&t[0], NAN);
        tb_fill(&t[1], NAN);
    } else {
        tb_rand(&t[0], -1.0, 1.0);
        tb_rand(&t[1], -1.0, 1.0);
    }
    if (gc->c_fill == 1) {
        tb_fill(&t[2], NAN);
    } else if (gc->c_fill == 2) {
        tb_fill(&t[2], INFINITY);
    } else {
        tb_rand(&t[2], -2.0, 2.0);
    }
    if (me != m) {
        if (gc->ta) {
            t[0].e.shape[rank - 1] = me;
        } else {
            t[0].e.shape[rank - 2] = me;
        }
        t[2].e.shape[rank - 2] = me;
    }

    ocu_op_desc d = op_desc(OCU_OP_GEMM);
    d.flags = gc->var ? OCU_OP_FLAG_VAR_ROWS : 0u;
    d.p.gemm.trans_a = gc->ta;
    d.p.gemm.trans_b = gc->tb;
    d.p.gemm.alpha = gc->alpha;
    d.p.gemm.beta = gc->beta;
    ocu_status s = run_op(&d, t, 3, gc->on_queue);
    if (s == OCU_OK && t[2].res) {
        double ua = unit_round(G.last_caps.accum_dtype);
        double uo = unit_round(dout), tn = tiny_of(dout);
        double alpha = gc->alpha, beta = gc->beta;
        long bad = 0;
        for (int64_t b = 0; b < bt; ++b) {
            for (int64_t i = 0; i < me; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    double acc = 0.0, sabs = 0.0;
                    if (alpha != 0.0) {
                        for (int64_t q = 0; q < k; ++q) {
                            double av = load_el(din, t[0].h, gc->ta ? moff(&t[0].e, b, q, i)
                                                                    : moff(&t[0].e, b, i, q));
                            double bv = load_el(din, t[1].h, gc->tb ? moff(&t[1].e, b, j, q)
                                                                    : moff(&t[1].e, b, q, j));
                            acc += av * bv;
                            sabs += fabs(av * bv);
                        }
                    }
                    int64_t co = moff(&t[2].e, b, i, j);
                    double cold = load_el(dout, t[2].h, co);
                    double r = (alpha != 0.0 ? alpha * acc : 0.0) +
                               (beta != 0.0 ? beta * cold : 0.0);
                    double bound = (double)(k + 4) * ua *
                                       (fabs(alpha) * sabs +
                                        (beta != 0.0 ? fabs(beta) * fabs(cold) : 0.0)) +
                                   uo * fabs(r) + tn;
                    double y = load_el(dout, t[2].res, co);
                    if (!ok_bound(y, r, bound)) {
                        if (bad++ < 4) {
                            detail("C[%lld,%lld,%lld] = %.9g, reference %.9g, bound %.3g",
                                   (long long)b, (long long)i, (long long)j, y, r, bound);
                        }
                    }
                }
            }
        }
        if (bad) {
            detail("%ld elements out of bound", bad);
        }
        check_at(bad == 0, __FILE__, __LINE__, "GEMM within OCU_TOL_DOT (6.1)");
        check_untouched(&t[2], t[2].h);
    }
    for (int i = 0; i < 3; ++i) {
        tb_free(&t[i]);
    }
    return s;
}

/* ---- ADD / MUL ----------------------------------------------------------- */

ocu_status run_elt(const elt_case *ec)
{
    ocu_dtype dt = ec->dt;
    uint32_t rank = ec->rank;
    tbuf t[3];
    tb_init(&t[0], tdesc(dt, rank, ec->a, ec->pad), NAN);
    tb_init(&t[1], tdesc(dt, rank, ec->b, ec->pad ? ec->pad + 1 : 0), NAN);
    char sa[64], sb[64];
    set_case("%s %s a=[%s] b=[%s] pad=%lld inplace=%d var=%d m_exec=%lld specials=%d queue=%d",
             ocu_op_name(ec->op), ocu_dtype_name(dt), shape_str(&t[0].d, sa, sizeof sa),
             shape_str(&t[1].d, sb, sizeof sb), (long long)ec->pad, ec->inplace, ec->var,
             (long long)ec->m_exec, ec->specials, ec->on_queue);

    int elo = -12, ehi = 12; /* f32: a + b stays exact in binary64 */
    if (dt == OCU_DT_F16) {
        elo = -8;
        ehi = ec->op == OCU_OP_MUL ? 6 : 8;
    } else if (dt == OCU_DT_BF16) {
        elo = -20;
        ehi = 20;
    }
    tb_rand_exp(&t[0], elo, ehi);
    tb_rand_exp(&t[1], elo, ehi);
    if (ec->specials && dt == OCU_DT_F32) {
        static const double add_sp[][2] = {
            { INFINITY, 1.0 }, { INFINITY, -INFINITY }, { NAN, 1.0 }, { -0.0, -0.0 },
            { 0x1.8p-126, -0x1.4p-126 }, { 0x1p-126, -0x1p-126 }, { 3.0, -3.0 },
            { 0x1.fffffep127, -0x1p104 },
        };
        static const double mul_sp[][2] = {
            { 0.0, INFINITY }, { -0.0, 5.0 }, { 0x1p-70, 0x1.8p-60 }, { 0x1p-80, 0x1p-80 },
            { -INFINITY, -2.0 }, { NAN, 0.0 }, { 0x1p-63, 0x1p-63 }, { -1.0, 0x1p-149 },
        };
        const double (*sp)[2] = ec->op == OCU_OP_ADD ? add_sp : mul_sp;
        int64_t idx[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < 8; ++i) {
            store_el(dt, t[0].h, desc_offset(&t[0].d, idx), sp[i][0]);
            store_el(dt, t[1].h, desc_offset(&t[1].d, idx), sp[i][1]);
            if (!next_index(&t[0].d, idx)) {
                break;
            }
        }
    }
    if (ec->m_exec) {
        int64_t planned = t[0].d.shape[0];
        t[0].e.shape[0] = ec->m_exec;
        if (t[1].d.shape[0] == planned) {
            t[1].e.shape[0] = ec->m_exec;
        }
    }
    if (ec->inplace) {
        tb_alias(&t[2], &t[0], 0);
    } else {
        tb_init(&t[2], tdesc(dt, rank, ec->a, ec->pad ? ec->pad + 2 : 0), 5.0);
        t[2].e.shape[0] = t[0].e.shape[0];
    }

    ocu_op_desc d = op_desc(ec->op);
    d.flags = ec->var ? OCU_OP_FLAG_VAR_ROWS : 0u;
    ocu_status s = run_op(&d, t, 3, ec->on_queue);
    if (s == OCU_OK && t[2].res) {
        long bad = 0;
        int64_t idx[4] = { 0, 0, 0, 0 };
        do {
            int64_t bi[4] = { 0, 0, 0, 0 };
            for (uint32_t k2 = 0; k2 < rank; ++k2) {
                bi[k2] = t[1].d.shape[k2] == 1 ? 0 : idx[k2];
            }
            double av = load_el(dt, t[0].h, desc_offset(&t[0].e, idx));
            double bv = load_el(dt, t[1].h, desc_offset(&t[1].e, bi));
            double r = ec->op == OCU_OP_ADD ? av + bv : av * bv;
            double y = load_el(dt, t[2].res, desc_offset(&t[2].e, idx));
            if (!ok_exact(dt, y, r)) {
                if (bad++ < 4) {
                    detail("y[%lld,%lld,%lld,%lld] = %a, a %a, b %a, want %a",
                           (long long)idx[0], (long long)idx[1], (long long)idx[2],
                           (long long)idx[3], y, av, bv, round_to(dt, r));
                }
            }
        } while (next_index(&t[2].e, idx));
        if (bad) {
            detail("%ld elements differ", bad);
        }
        check_at(bad == 0, __FILE__, __LINE__, "element-wise result is OCU_TOL_EXACT (6.1)");
        check_untouched(&t[2], ec->inplace ? t[0].h : t[2].h);
    }
    for (int i = 0; i < 3; ++i) {
        tb_free(&t[i]);
    }
    return s;
}

/* ---- RMS_NORM / SOFTMAX -------------------------------------------------- */

static void row_set_case(const char *op, const row_case *rc, const tbuf *x)
{
    char sx[64];
    set_case("%s %s x=[%s] pad=%lld weight=%d eps=%g scale=%g mag=%g inplace=%d var=%d "
             "m_exec=%lld nan_row=%d queue=%d",
             op, ocu_dtype_name(rc->dt), shape_str(&x->d, sx, sizeof sx), (long long)rc->pad,
             rc->weight, (double)rc->eps, (double)rc->scale, rc->mag, rc->inplace, rc->var,
             (long long)rc->m_exec, rc->nan_row, rc->on_queue);
}

ocu_status run_rms(const row_case *rc)
{
    ocu_dtype dt = rc->dt;
    uint32_t rank = rc->rank;
    int64_t n = rc->shape[rank - 1];
    uint32_t nt = rc->weight ? 3u : 2u;
    tbuf t[3];
    memset(t, 0, sizeof t);
    tb_init(&t[0], tdesc(dt, rank, rc->shape, rc->pad), NAN);
    row_set_case("rms_norm", rc, &t[0]);
    double mag = rc->mag > 0 ? rc->mag : 2.0;
    tb_rand(&t[0], -mag, mag);
    if (rc->weight) {
        tb_init(&t[1], tdesc(dt, 1, SHP(n), 0), NAN);
        tb_rand(&t[1], -1.5, 1.5);
    }
    if (rc->m_exec) {
        t[0].e.shape[0] = rc->m_exec;
    }
    tbuf *o = &t[nt - 1];
    if (rc->inplace) {
        tb_alias(o, &t[0], 0);
    } else {
        tb_init(o, tdesc(dt, rank, rc->shape, rc->pad ? rc->pad + 3 : 0), 5.0);
        o->e.shape[0] = t[0].e.shape[0];
    }
    ocu_op_desc d = op_desc(OCU_OP_RMS_NORM);
    d.flags = rc->var ? OCU_OP_FLAG_VAR_ROWS : 0u;
    d.p.rms_norm.eps = rc->eps;
    ocu_status s = run_op(&d, t, nt, rc->on_queue);
    if (s == OCU_OK && o->res) {
        double ua = unit_round(G.last_caps.accum_dtype), uo = unit_round(dt), tn = tiny_of(dt);
        double rtol = (double)(n + 8) * ua + 2.0 * uo;
        long bad = 0;
        int64_t rows = row_count(&o->e);
        for (int64_t row = 0; row < rows; ++row) {
            int64_t xb = row_base(&t[0].e, row), yb = row_base(&o->e, row);
            double ss = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                double v = load_el(dt, t[0].h, xb + j);
                ss += v * v;
            }
            double inv = 1.0 / sqrt(ss / (double)n + (double)rc->eps);
            for (int64_t j = 0; j < n; ++j) {
                double r = load_el(dt, t[0].h, xb + j) * inv;
                if (rc->weight) {
                    r *= load_el(dt, t[1].h, j);
                }
                double y = load_el(dt, o->res, yb + j);
                if (!ok_bound(y, r, rtol * fabs(r) + tn)) {
                    if (bad++ < 4) {
                        detail("row %lld col %lld: y %.9g, reference %.9g, bound %.3g",
                               (long long)row, (long long)j, y, r, rtol * fabs(r) + tn);
                    }
                }
            }
        }
        if (bad) {
            detail("%ld elements out of bound", bad);
        }
        check_at(bad == 0, __FILE__, __LINE__, "RMS_NORM within OCU_TOL_ROW (6.1)");
        check_untouched(o, rc->inplace ? t[0].h : o->h);
    }
    for (uint32_t i = 0; i < nt; ++i) {
        tb_free(&t[i]);
    }
    return s;
}

ocu_status run_softmax(const row_case *rc)
{
    ocu_dtype dt = rc->dt;
    uint32_t rank = rc->rank;
    int64_t n = rc->shape[rank - 1];
    tbuf t[2];
    tb_init(&t[0], tdesc(dt, rank, rc->shape, rc->pad), NAN);
    row_set_case("softmax", rc, &t[0]);
    double mag = rc->mag > 0 ? rc->mag : 4.0;
    tb_rand(&t[0], -mag, mag);
    int64_t planned_rows = row_count(&t[0].d);
    if (rc->nan_row && planned_rows >= 4 && n >= 2) {
        store_el(dt, t[0].h, row_base(&t[0].d, 1), NAN);
        store_el(dt, t[0].h, row_base(&t[0].d, 2) + n / 2, INFINITY);
        store_el(dt, t[0].h, row_base(&t[0].d, 3) + n - 1, -INFINITY);
    }
    if (rc->m_exec) {
        t[0].e.shape[0] = rc->m_exec;
    }
    if (rc->inplace) {
        tb_alias(&t[1], &t[0], 0);
    } else {
        tb_init(&t[1], tdesc(dt, rank, rc->shape, rc->pad ? rc->pad + 3 : 0), 5.0);
        t[1].e.shape[0] = t[0].e.shape[0];
    }
    ocu_op_desc d = op_desc(OCU_OP_SOFTMAX);
    d.flags = rc->var ? OCU_OP_FLAG_VAR_ROWS : 0u;
    d.p.softmax.scale = rc->scale;
    ocu_status s = run_op(&d, t, 2, rc->on_queue);
    if (s == OCU_OK && t[1].res) {
        double ua = unit_round(G.last_caps.accum_dtype), uo = unit_round(dt), tn = tiny_of(dt);
        double sc = rc->scale;
        long bad = 0;
        int64_t rows = row_count(&t[1].e);
        double *tv = malloc((size_t)n * sizeof *tv);
        for (int64_t row = 0; row < rows; ++row) {
            if (rc->nan_row && (row == 1 || row == 2) && planned_rows >= 4) {
                continue; /* NaN / +Inf rows: unspecified (4.4) */
            }
            int64_t xb = row_base(&t[0].e, row), yb = row_base(&t[1].e, row);
            double mx = -INFINITY, amax = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                tv[j] = sc * load_el(dt, t[0].h, xb + j);
                if (tv[j] > mx) {
                    mx = tv[j];
                }
                if (isfinite(tv[j]) && fabs(tv[j]) > amax) {
                    amax = fabs(tv[j]); /* -Inf elements excluded from T_i */
                }
            }
            double sum = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                sum += exp(tv[j] - mx);
            }
            for (int64_t j = 0; j < n; ++j) {
                double r = exp(tv[j] - mx) / sum;
                double ti = (isfinite(tv[j]) ? fabs(tv[j]) : 0.0) + amax;
                double rtol = (2.0 * (double)n + 2.0 * ti + 16.0) * ua + 2.0 * uo;
                double y = load_el(dt, t[1].res, yb + j);
                if (!ok_bound(y, r, rtol * fabs(r) + tn)) {
                    if (bad++ < 4) {
                        detail("row %lld col %lld: y %.9g, reference %.9g, bound %.3g",
                               (long long)row, (long long)j, y, r, rtol * fabs(r) + tn);
                    }
                }
            }
        }
        free(tv);
        if (bad) {
            detail("%ld elements out of bound", bad);
        }
        check_at(bad == 0, __FILE__, __LINE__, "SOFTMAX within OCU_TOL_ROW (6.1)");
        check_untouched(&t[1], rc->inplace ? t[0].h : t[1].h);
    }
    tb_free(&t[0]);
    tb_free(&t[1]);
    return s;
}

/* ---- ROPE (4.6) ---------------------------------------------------------- */

ocu_status run_rope(const rope_case *rc)
{
    const ocu_dtype dt = OCU_DT_F32;
    tbuf t[3];
    ocu_tensor_desc xd = rc->rank == 3 ? tdesc(dt, 3, SHP(rc->t, rc->h, rc->n), 0)
                                       : tdesc(dt, 2, SHP(rc->t, rc->n), 0);
    tb_init(&t[0], xd, NAN);
    tb_init(&t[1], tdesc(OCU_DT_I32, 1, SHP(rc->t), 0), 0.0);
    tb_init(&t[2], xd, 5.0);
    set_case("rope f32 rank%u T=%lld H=%lld N=%lld mode=%u n_dims=%u theta=%g fs=%g queue=%d",
             rc->rank, (long long)rc->t, (long long)rc->h, (long long)rc->n, rc->mode,
             rc->n_dims, (double)rc->theta, (double)rc->freq_scale, rc->on_queue);
    tb_rand(&t[0], -1.0, 1.0);
    for (int64_t i = 0; i < rc->t; ++i) {
        store_el(OCU_DT_I32, t[1].h, i, (double)irand(0, 200));
    }
    ocu_op_desc d = op_desc(OCU_OP_ROPE);
    d.p.rope.mode = rc->mode;
    d.p.rope.n_dims = rc->n_dims;
    d.p.rope.theta_base = rc->theta;
    d.p.rope.freq_scale = rc->freq_scale;
    ocu_status s = run_op(&d, t, 3, rc->on_queue);
    if (s == OCU_OK && t[2].res) {
        long bad = 0;
        int64_t rows = row_count(&xd);
        int64_t heads = rc->rank == 3 ? rc->h : 1;
        int64_t dd = rc->n_dims;
        for (int64_t row = 0; row < rows; ++row) {
            int64_t tok = row / heads;
            double p = load_el(OCU_DT_I32, t[1].h, tok);
            int64_t xb = row_base(&xd, row);
            for (int64_t j = 0; j < rc->n; ++j) {
                double r;
                if (j >= dd) {
                    r = load_el(dt, t[0].h, xb + j);
                } else {
                    int64_t i, iu, iv;
                    int first;
                    if (rc->mode == 0) {
                        i = j / 2;
                        iu = 2 * i;
                        iv = 2 * i + 1;
                        first = (j % 2) == 0;
                    } else {
                        i = j < dd / 2 ? j : j - dd / 2;
                        iu = i;
                        iv = i + dd / 2;
                        first = j < dd / 2;
                    }
                    double a = p * (double)rc->freq_scale *
                               pow((double)rc->theta, -2.0 * (double)i / (double)dd);
                    double u = load_el(dt, t[0].h, xb + iu), v = load_el(dt, t[0].h, xb + iv);
                    r = first ? u * cos(a) - v * sin(a) : u * sin(a) + v * cos(a);
                }
                double y = load_el(dt, t[2].res, xb + j);
                double bound = j >= dd ? 0.0 : 8.0 * ulp_at(dt, r);
                if (!ok_bound(y, r, bound)) {
                    if (bad++ < 4) {
                        detail("row %lld col %lld: y %.9g, reference %.9g, bound %.3g",
                               (long long)row, (long long)j, y, r, bound);
                    }
                }
            }
        }
        if (bad) {
            detail("%ld elements out of bound", bad);
        }
        check_at(bad == 0, __FILE__, __LINE__, "ROPE within ULP(8), copies exact (4.6)");
    }
    for (int i = 0; i < 3; ++i) {
        tb_free(&t[i]);
    }
    return s;
}

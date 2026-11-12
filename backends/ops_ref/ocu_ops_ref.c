/*
 * Reference for the operator table (docs/spec/op-table.md): the checks all
 * backends share and plain C compute with binary32 accumulation.
 */
#include "ocu_ops_ref.h"

#include <math.h>
#include <string.h>

/* ---- element conversions -------------------------------------------------- */

static uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bits_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

float ocu_ref_f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0) {
        float v = (float)m * 0x1p-24f; /* exact: m < 2^10 */
        return sign ? -v : v;
    }
    if (e == 31) {
        return bits_f32(sign | 0x7f800000u | (m << 13) | (m ? 0x400000u : 0u));
    }
    return bits_f32(sign | ((e + 112u) << 23) | (m << 13));
}

uint16_t ocu_ref_f32_to_f16(float f)
{
    uint32_t x = f32_bits(f);
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) {
        return (uint16_t)(sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));
    }
    if (ax >= 0x477ff000u) { /* >= 65520 rounds to infinity */
        return (uint16_t)(sign | 0x7c00u);
    }
    if (ax < 0x38800000u) { /* below 2^-14: subnormal or zero */
        float s = bits_f32(ax) * 0x1p24f; /* exact scaling */
        return (uint16_t)(sign | (uint16_t)nearbyintf(s));
    }
    ax += 0xfffu + ((ax >> 13) & 1u);
    return (uint16_t)(sign | ((ax - 0x38000000u) >> 13));
}

float ocu_ref_bf16_to_f32(uint16_t h) { return bits_f32((uint32_t)h << 16); }

uint16_t ocu_ref_f32_to_bf16(float f)
{
    uint32_t x = f32_bits(f);
    if ((x & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t)((x >> 16) | 0x40u); /* quiet NaN */
    }
    x += 0x7fffu + ((x >> 16) & 1u);
    return (uint16_t)(x >> 16);
}

static inline float load(ocu_dtype dt, const void *base, int64_t i)
{
    switch (dt) {
    case OCU_DT_F16:  return ocu_ref_f16_to_f32(((const uint16_t *)base)[i]);
    case OCU_DT_BF16: return ocu_ref_bf16_to_f32(((const uint16_t *)base)[i]);
    default:          return ((const float *)base)[i];
    }
}

static inline void store(ocu_dtype dt, void *base, int64_t i, float v)
{
    switch (dt) {
    case OCU_DT_F16:  ((uint16_t *)base)[i] = ocu_ref_f32_to_f16(v); break;
    case OCU_DT_BF16: ((uint16_t *)base)[i] = ocu_ref_f32_to_bf16(v); break;
    default:          ((float *)base)[i] = v; break;
    }
}

/* ---- descriptor checks ---------------------------------------------------- */

static int dtype_known(ocu_dtype dt)
{
    switch (dt) {
    case OCU_DT_F32: case OCU_DT_F16: case OCU_DT_BF16: case OCU_DT_I8:
    case OCU_DT_I32: case OCU_DT_Q8_B32: case OCU_DT_Q4_B32:
        return 1;
    case OCU_DT_NONE:
        return 0;
    }
    return 0;
}

static int is_float(ocu_dtype dt)
{
    return dt == OCU_DT_F32 || dt == OCU_DT_F16 || dt == OCU_DT_BF16;
}

static int mul_ok(int64_t a, int64_t b, int64_t *r)
{
    return !__builtin_mul_overflow(a, b, r);
}

/* Section 3: well-formed tensor. */
static int tensor_ok(const ocu_tensor_desc *t)
{
    if (!dtype_known(t->dtype) || t->rank < 1 || t->rank > OCU_OPS_MAX_RANK) {
        return 0;
    }
    uint32_t r = t->rank;
    int64_t count = 1, span = 1;
    for (uint32_t k = 0; k < r; ++k) {
        if (t->shape[k] < 1 || t->stride[k] < 1) return 0;
        int64_t x;
        if (!mul_ok(count, t->shape[k], &count)) return 0;
        if (!mul_ok(t->shape[k] - 1, t->stride[k], &x)) return 0;
        if (__builtin_add_overflow(span, x, &span)) return 0;
    }
    if (t->stride[r - 1] != 1) return 0;
    for (uint32_t k = 0; k + 1 < r; ++k) {
        int64_t inner;
        if (!mul_ok(t->stride[k + 1], t->shape[k + 1], &inner)) return 0;
        if (t->stride[k] < inner) return 0;
    }
    if (ocu_dtype_is_blocked(t->dtype)) {
        int64_t blk = ocu_dtype_block(t->dtype);
        if (t->shape[r - 1] % blk) return 0;
        for (uint32_t k = 0; k + 1 < r; ++k) {
            if (t->stride[k] % blk) return 0;
        }
    }
    return 1;
}

static int same_shape(const ocu_tensor_desc *a, const ocu_tensor_desc *b)
{
    if (a->rank != b->rank) return 0;
    for (uint32_t k = 0; k < a->rank; ++k) {
        if (a->shape[k] != b->shape[k]) return 0;
    }
    return 1;
}

int ocu_ref_desc_equal(const ocu_tensor_desc *a, const ocu_tensor_desc *b)
{
    if (a->dtype != b->dtype || !same_shape(a, b)) return 0;
    for (uint32_t k = 0; k < a->rank; ++k) {
        if (a->stride[k] != b->stride[k]) return 0;
    }
    return 1;
}

uint64_t ocu_ref_span_bytes(const ocu_tensor_desc *t)
{
    int64_t span = 1;
    for (uint32_t k = 0; k < t->rank && k < OCU_OPS_MAX_RANK; ++k) {
        span += (t->shape[k] - 1) * t->stride[k];
    }
    if (ocu_dtype_is_blocked(t->dtype)) {
        return (uint64_t)(span / ocu_dtype_block(t->dtype)) * ocu_dtype_size(t->dtype);
    }
    return (uint64_t)span * ocu_dtype_size(t->dtype);
}

static ocu_status check_gemm(const ocu_op_desc *d, int listed)
{
    const ocu_tensor_desc *a = &d->tensors[0], *b = &d->tensors[1], *c = &d->tensors[2];
    if (d->p.gemm.trans_a > 1 || d->p.gemm.trans_b > 1) return OCU_ERR_INVALID_VALUE;
    if (a->rank != b->rank || a->rank != c->rank || a->rank < 2 || a->rank > 3) {
        return OCU_ERR_INVALID_VALUE;
    }
    uint32_t o = a->rank - 2; /* index of the row dim */
    int64_t m  = d->p.gemm.trans_a ? a->shape[o + 1] : a->shape[o];
    int64_t ka = d->p.gemm.trans_a ? a->shape[o] : a->shape[o + 1];
    int64_t kb = d->p.gemm.trans_b ? b->shape[o + 1] : b->shape[o];
    int64_t n  = d->p.gemm.trans_b ? b->shape[o] : b->shape[o + 1];
    if (ka != kb || c->shape[o] != m || c->shape[o + 1] != n) return OCU_ERR_INVALID_VALUE;
    if (o == 1) {
        int64_t bt = c->shape[0];
        if ((a->shape[0] != bt && a->shape[0] != 1) || (b->shape[0] != bt && b->shape[0] != 1)) {
            return OCU_ERR_INVALID_VALUE;
        }
    }
    if (!listed) return OCU_ERR_NOT_SUPPORTED;
    if (a->dtype != b->dtype) return OCU_ERR_INVALID_VALUE;
    if (c->dtype != OCU_DT_F32 && c->dtype != a->dtype) return OCU_ERR_INVALID_VALUE;
    return OCU_OK;
}

ocu_status ocu_ref_check_desc(const ocu_op_desc *d)
{
    if (d == NULL || d->struct_size < sizeof(ocu_op_desc) ||
        (d->flags & ~OCU_OP_FLAG_VAR_ROWS) != 0) {
        return OCU_ERR_INVALID_VALUE;
    }
    uint32_t want_lo, want_hi;
    switch (d->op) {
    case OCU_OP_GEMM: case OCU_OP_ADD: case OCU_OP_MUL: case OCU_OP_ROPE:
        want_lo = want_hi = 3; break;
    case OCU_OP_RMS_NORM: want_lo = 2; want_hi = 3; break;
    case OCU_OP_SOFTMAX:  want_lo = want_hi = 2; break;
    case OCU_OP_QMATMUL:  want_lo = 1; want_hi = OCU_OPS_MAX_TENSORS; break;
    default:              return OCU_ERR_INVALID_VALUE;
    }
    uint32_t n = d->num_tensors;
    if (n < want_lo || n > want_hi) return OCU_ERR_INVALID_VALUE;
    int listed = 1; /* every slot has a dtype the operator is defined for */
    for (uint32_t i = 0; i < n; ++i) {
        if (!tensor_ok(&d->tensors[i])) return OCU_ERR_INVALID_VALUE;
        if (!is_float(d->tensors[i].dtype)) listed = 0;
    }
    if (d->op == OCU_OP_QMATMUL) return OCU_ERR_NOT_SUPPORTED; /* reserved */
    const ocu_tensor_desc *t = d->tensors;
    /* Variable rows vary shape[0] of a row / element-wise output. */
    if ((d->flags & OCU_OP_FLAG_VAR_ROWS) && d->op != OCU_OP_GEMM && t[n - 1].rank < 2) {
        return OCU_ERR_INVALID_VALUE;
    }
    switch (d->op) {
    case OCU_OP_GEMM:
        return check_gemm(d, listed);
    case OCU_OP_ADD:
    case OCU_OP_MUL: {
        if (t[0].rank != t[1].rank || !same_shape(&t[0], &t[2])) return OCU_ERR_INVALID_VALUE;
        for (uint32_t k = 0; k < t[0].rank; ++k) {
            if (t[1].shape[k] != t[0].shape[k] && t[1].shape[k] != 1) return OCU_ERR_INVALID_VALUE;
        }
        if (!listed) return OCU_ERR_NOT_SUPPORTED;
        if (t[1].dtype != t[0].dtype || t[2].dtype != t[0].dtype) return OCU_ERR_INVALID_VALUE;
        return OCU_OK;
    }
    case OCU_OP_RMS_NORM: {
        const ocu_tensor_desc *y = &t[n - 1];
        if (!same_shape(&t[0], y)) return OCU_ERR_INVALID_VALUE;
        if (n == 3 && (t[1].rank != 1 || t[1].shape[0] != t[0].shape[t[0].rank - 1])) {
            return OCU_ERR_INVALID_VALUE;
        }
        if (!(d->p.rms_norm.eps >= 0.0f)) return OCU_ERR_INVALID_VALUE;
        if (!listed) return OCU_ERR_NOT_SUPPORTED;
        for (uint32_t i = 1; i < n; ++i) {
            if (t[i].dtype != t[0].dtype) return OCU_ERR_INVALID_VALUE;
        }
        return OCU_OK;
    }
    case OCU_OP_SOFTMAX:
        if (!same_shape(&t[0], &t[1])) return OCU_ERR_INVALID_VALUE;
        if (!isfinite(d->p.softmax.scale)) return OCU_ERR_INVALID_VALUE;
        if (!listed) return OCU_ERR_NOT_SUPPORTED;
        if (t[1].dtype != t[0].dtype) return OCU_ERR_INVALID_VALUE;
        return OCU_OK;
    case OCU_OP_ROPE:
        for (uint32_t i = 0; i < n; ++i) {
            if (ocu_dtype_is_blocked(t[i].dtype)) return OCU_ERR_NOT_SUPPORTED;
        }
        return OCU_ERR_NOT_IMPLEMENTED;
    default:
        return OCU_ERR_INVALID_VALUE;
    }
}

/* ---- execute-time checks -------------------------------------------------- */

static int overlaps(const ocu_tensor *a, const ocu_tensor *b)
{
    uintptr_t a0 = (uintptr_t)a->data, a1 = a0 + ocu_ref_span_bytes(&a->desc);
    uintptr_t b0 = (uintptr_t)b->data, b1 = b0 + ocu_ref_span_bytes(&b->desc);
    return a0 < b1 && b0 < a1;
}

/* The dimension of slot i that OCU_OP_FLAG_VAR_ROWS lets vary, or -1. */
static int var_dim(const ocu_op_desc *d, uint32_t i)
{
    if (!(d->flags & OCU_OP_FLAG_VAR_ROWS)) return -1;
    const ocu_tensor_desc *t = &d->tensors[i], *y = &d->tensors[d->num_tensors - 1];
    if (d->op == OCU_OP_GEMM) {
        int o = (int)t->rank - 2;
        if (i == 0) return d->p.gemm.trans_a ? o + 1 : o;
        if (i == 2) return o;
        return -1;
    }
    if (i == d->num_tensors - 1) return 0;
    return (t->rank == y->rank && t->shape[0] == y->shape[0]) ? 0 : -1;
}

ocu_status ocu_ref_check_tensors(const ocu_op_desc *d, const ocu_tensor *t, uint32_t n,
                                 ocu_op_desc *actual)
{
    if (t == NULL || actual == NULL || n != d->num_tensors) return OCU_ERR_INVALID_VALUE;
    *actual = *d;
    int64_t rows = -1; /* the one row count of this call (VAR_ROWS) */
    for (uint32_t i = 0; i < n; ++i) {
        const ocu_tensor_desc *g = &t[i].desc, *p = &d->tensors[i];
        if (t[i].data == NULL || g->dtype != p->dtype || g->rank != p->rank) {
            return OCU_ERR_INVALID_VALUE;
        }
        int vd = var_dim(d, i);
        for (uint32_t k = 0; k < p->rank; ++k) {
            if (g->stride[k] != p->stride[k]) return OCU_ERR_INVALID_VALUE;
            if ((int)k == vd) {
                if (g->shape[k] < 1 || g->shape[k] > p->shape[k]) return OCU_ERR_INVALID_VALUE;
                if (rows < 0) rows = g->shape[k];
                if (g->shape[k] != rows) return OCU_ERR_INVALID_VALUE;
            } else if (g->shape[k] != p->shape[k]) {
                return OCU_ERR_INVALID_VALUE;
            }
        }
        actual->tensors[i] = *g;
    }
    const ocu_tensor *out = &t[n - 1];
    for (uint32_t i = 0; i + 1 < n; ++i) {
        if (!overlaps(out, &t[i])) continue;
        /* In place is allowed for slot 0 of element-wise and row operators,
         * with the same pointer and descriptor; nothing else may overlap. */
        int in_place_op = d->op != OCU_OP_GEMM;
        if (!(in_place_op && i == 0 && out->data == t[0].data &&
              ocu_ref_desc_equal(&out->desc, &t[0].desc))) {
            return OCU_ERR_INVALID_VALUE;
        }
    }
    return OCU_OK;
}

/* ---- compute -------------------------------------------------------------- */

int64_t ocu_ref_rows(const ocu_op_desc *d)
{
    const ocu_tensor_desc *y = &d->tensors[d->num_tensors - 1];
    int64_t rows = 1;
    for (uint32_t k = 0; k + 1 < y->rank; ++k) rows *= y->shape[k];
    return rows;
}

/* Element offset of the start of row r of t, where rows are counted over
 * the outer dims of `like` (same rank); dims of size 1 in t broadcast. */
static int64_t row_offset(const ocu_tensor_desc *t, const ocu_tensor_desc *like, int64_t r)
{
    int64_t off = 0;
    for (int k = (int)like->rank - 2; k >= 0; --k) {
        int64_t idx = r % like->shape[k];
        r /= like->shape[k];
        if (t->shape[k] != 1) off += idx * t->stride[k];
    }
    return off;
}

static void run_gemm(const ocu_op_desc *d, void *const *data, int64_t r0, int64_t r1)
{
    const ocu_tensor_desc *a = &d->tensors[0], *b = &d->tensors[1], *c = &d->tensors[2];
    uint32_t o = a->rank - 2;
    int ta = (int)d->p.gemm.trans_a, tb = (int)d->p.gemm.trans_b;
    int64_t m = c->shape[o], n = c->shape[o + 1];
    int64_t k = ta ? a->shape[o] : a->shape[o + 1];
    int64_t sa = a->stride[o], sb = b->stride[o], sc = c->stride[o];
    float alpha = d->p.gemm.alpha, beta = d->p.gemm.beta;
    for (int64_t r = r0; r < r1; ++r) {
        int64_t bt = r / m, i = r % m;
        int64_t ab = 0, bb = 0, cb = 0;
        if (o == 1) {
            ab = (a->shape[0] == 1 ? 0 : bt) * a->stride[0];
            bb = (b->shape[0] == 1 ? 0 : bt) * b->stride[0];
            cb = bt * c->stride[0];
        }
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            if (alpha != 0.0f) {
                for (int64_t p = 0; p < k; ++p) {
                    float x = load(a->dtype, data[0], ab + (ta ? p * sa + i : i * sa + p));
                    float y = load(b->dtype, data[1], bb + (tb ? j * sb + p : p * sb + j));
                    acc += x * y;
                }
            }
            int64_t ci = cb + i * sc + j;
            float v = alpha == 0.0f ? 0.0f : alpha * acc;
            if (beta != 0.0f) v += beta * load(c->dtype, data[2], ci);
            store(c->dtype, data[2], ci, v);
        }
    }
}

static void run_binary(const ocu_op_desc *d, void *const *data, int64_t r0, int64_t r1)
{
    const ocu_tensor_desc *a = &d->tensors[0], *b = &d->tensors[1], *y = &d->tensors[2];
    int64_t n = y->shape[y->rank - 1];
    int bcast = b->shape[b->rank - 1] == 1;
    ocu_dtype dt = y->dtype;
    for (int64_t r = r0; r < r1; ++r) {
        int64_t ao = row_offset(a, y, r), bo = row_offset(b, y, r), yo = row_offset(y, y, r);
        for (int64_t i = 0; i < n; ++i) {
            float x = load(dt, data[0], ao + i);
            float z = load(dt, data[1], bo + (bcast ? 0 : i));
            store(dt, data[2], yo + i, d->op == OCU_OP_ADD ? x + z : x * z);
        }
    }
}

static void run_rms_norm(const ocu_op_desc *d, void *const *data, int64_t r0, int64_t r1)
{
    uint32_t nt = d->num_tensors;
    const ocu_tensor_desc *x = &d->tensors[0], *y = &d->tensors[nt - 1];
    int64_t n = y->shape[y->rank - 1];
    ocu_dtype dt = y->dtype;
    for (int64_t r = r0; r < r1; ++r) {
        int64_t xo = row_offset(x, y, r), yo = row_offset(y, y, r);
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float v = load(dt, data[0], xo + i);
            sum += v * v;
        }
        float inv = 1.0f / sqrtf(sum / (float)n + d->p.rms_norm.eps);
        for (int64_t i = 0; i < n; ++i) {
            float v = load(dt, data[0], xo + i) * inv;
            if (nt == 3) v *= load(dt, data[1], i);
            store(dt, data[nt - 1], yo + i, v);
        }
    }
}

static void run_softmax(const ocu_op_desc *d, void *const *data, int64_t r0, int64_t r1)
{
    const ocu_tensor_desc *x = &d->tensors[0], *y = &d->tensors[1];
    int64_t n = y->shape[y->rank - 1];
    ocu_dtype dt = y->dtype;
    float scale = d->p.softmax.scale;
    for (int64_t r = r0; r < r1; ++r) {
        int64_t xo = row_offset(x, y, r), yo = row_offset(y, y, r);
        float m = -INFINITY;
        for (int64_t i = 0; i < n; ++i) {
            float t = scale * load(dt, data[0], xo + i);
            if (t > m || isnan(t)) m = t;
        }
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            sum += expf(scale * load(dt, data[0], xo + i) - m);
        }
        for (int64_t i = 0; i < n; ++i) {
            store(dt, data[1], yo + i, expf(scale * load(dt, data[0], xo + i) - m) / sum);
        }
    }
}

void ocu_ref_run(const ocu_op_desc *d, void *const *data, int64_t r0, int64_t r1)
{
    switch (d->op) {
    case OCU_OP_GEMM:     run_gemm(d, data, r0, r1); break;
    case OCU_OP_ADD:
    case OCU_OP_MUL:      run_binary(d, data, r0, r1); break;
    case OCU_OP_RMS_NORM: run_rms_norm(d, data, r0, r1); break;
    case OCU_OP_SOFTMAX:  run_softmax(d, data, r0, r1); break;
    default:              break;
    }
}

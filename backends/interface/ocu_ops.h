/*
 * ocu_ops.h -- operator table: library-level operators a backend can run
 * without SIMT kernels (GEMM, normalisation, softmax, element-wise).
 *
 * The specification is docs/spec/op-table.md; this header follows it.
 *
 * A backend offers the table through the optional get_provider entry of
 * ocu_backend: get_provider(OCU_OPS_PROVIDER) returns a const ocu_op_table *
 * or NULL. The backend ABI (OCU_BACKEND_ABI) does not change. The table has
 * its own version and size; entries are only appended.
 *
 * Flow: support (coarse, per op x dtype) -> query (exact, per descriptor)
 * -> plan (shape-fixed, may compile) -> execute (on a queue) -> destroy.
 */
#ifndef OCU_OPS_H
#define OCU_OPS_H

#include "ocu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OCU_OPS_VERSION  1u
#define OCU_OPS_PROVIDER "ops"

#define OCU_OPS_MAX_RANK    4u
#define OCU_OPS_MAX_TENSORS 4u

/* ---- data types (spec section 2) ---------------------------------------- */

typedef enum ocu_dtype {
    OCU_DT_NONE   = 0,
    OCU_DT_F32    = 1,
    OCU_DT_F16    = 2,
    OCU_DT_BF16   = 3,
    OCU_DT_I8     = 4,
    OCU_DT_I32    = 5,
    /* block-quantised family (reserved: no version 1 operator takes them) */
    OCU_DT_Q8_B32 = 0x101, /* 32 values: binary16 scale + 32 x int8 (34 B) */
    OCU_DT_Q4_B32 = 0x102  /* 32 values: binary16 scale + 16 B of 4-bit (18 B) */
} ocu_dtype;

static inline int ocu_dtype_is_blocked(ocu_dtype dt) { return (int)dt >= 0x100; }

/* Bytes per element; for block types, bytes per block. 0 when unknown. */
static inline size_t ocu_dtype_size(ocu_dtype dt)
{
    switch (dt) {
    case OCU_DT_F32:    return 4;
    case OCU_DT_F16:    return 2;
    case OCU_DT_BF16:   return 2;
    case OCU_DT_I8:     return 1;
    case OCU_DT_I32:    return 4;
    case OCU_DT_Q8_B32: return 34;
    case OCU_DT_Q4_B32: return 18;
    case OCU_DT_NONE:   return 0;
    }
    return 0;
}

/* Values per block (1 for plain types, 0 when unknown). */
static inline uint32_t ocu_dtype_block(ocu_dtype dt)
{
    if (dt == OCU_DT_Q8_B32 || dt == OCU_DT_Q4_B32) return 32;
    return ocu_dtype_size(dt) ? 1u : 0u;
}

/* ---- tensors (spec section 3) ------------------------------------------- */

typedef struct ocu_tensor_desc {
    ocu_dtype dtype;
    uint32_t  rank;                    /* 1 .. OCU_OPS_MAX_RANK */
    int64_t   shape[OCU_OPS_MAX_RANK]; /* shape[0] is the outermost dim */
    int64_t   stride[OCU_OPS_MAX_RANK];/* in elements; stride[rank-1] == 1 */
} ocu_tensor_desc;

typedef struct ocu_tensor {
    void           *data; /* device pointer to the first element */
    ocu_tensor_desc desc;
} ocu_tensor;

/* Layout bits for ocu_op_support.layouts. */
#define OCU_LAYOUT_DENSE       0x1u /* packed row-major */
#define OCU_LAYOUT_ROW_STRIDED 0x2u /* padding between rows / matrices */

/* ---- operators (spec section 4) ----------------------------------------- */

typedef enum ocu_op {
    OCU_OP_GEMM     = 1, /* A, B, C:   C = alpha op(A) op(B) + beta C */
    OCU_OP_ADD      = 2, /* a, b, y:   y = a + b (b broadcasts) */
    OCU_OP_MUL      = 3, /* a, b, y:   y = a * b (b broadcasts) */
    OCU_OP_RMS_NORM = 4, /* x, [w,] y: per row */
    OCU_OP_SOFTMAX  = 5, /* x, y:      per row, of scale * x */
    OCU_OP_ROPE     = 6, /* x, pos, y: rotary embedding (optional in v1) */
    OCU_OP_QMATMUL  = 7  /* reserved */
} ocu_op;

#define OCU_OP_COUNT 8u /* op values are below this */

/* Descriptor flags (spec 4.7). VAR_ROWS: the planned row count (M of a
 * GEMM, shape[0] of a row or element-wise output) is the largest; execute
 * may pass fewer rows. */
#define OCU_OP_FLAG_VAR_ROWS 0x1u

typedef struct ocu_op_desc {
    uint32_t        struct_size; /* sizeof(ocu_op_desc) of the caller */
    ocu_op          op;
    uint32_t        num_tensors; /* inputs, then the output (last slot) */
    uint32_t        flags;       /* OCU_OP_FLAG_* bits */
    ocu_tensor_desc tensors[OCU_OPS_MAX_TENSORS];
    union {
        struct { uint32_t trans_a, trans_b; float alpha, beta; } gemm;
        struct { float eps; } rms_norm;
        struct { float scale; } softmax;
        struct { uint32_t mode, n_dims; float theta_base, freq_scale; } rope;
        double reserved[8];
    } p;
} ocu_op_desc;

/* ---- tolerance classes (spec section 6) --------------------------------- */

typedef enum ocu_tol_class {
    OCU_TOL_EXACT = 1, /* correctly rounded */
    OCU_TOL_ULP   = 2, /* k ulps of the output type */
    OCU_TOL_ROW   = 3, /* relative bound growing with the row length */
    OCU_TOL_DOT   = 4  /* inner-product bound growing with K */
} ocu_tol_class;

/* ---- capability answers (spec 5.1, 5.2) --------------------------------- */

typedef struct ocu_op_support {
    uint32_t  layouts;       /* OCU_LAYOUT_* bits */
    uint32_t  max_rank;
    int64_t   max_row;       /* largest innermost dim, 0 = no fixed limit */
    uint32_t  align_bytes;   /* required alignment of each data pointer */
    ocu_dtype accum_dtype;
    uint32_t  out_dtypes;    /* bit (1u << dtype) per accepted output dtype */
    uint32_t  needs_plan_ms; /* typical plan time in ms, 0 = negligible */
    uint32_t  var_rows;      /* 1 when OCU_OP_FLAG_VAR_ROWS plans are accepted */
} ocu_op_support;

typedef struct ocu_op_caps {
    ocu_tol_class tol;
    ocu_dtype     accum_dtype;
    uint64_t      scratch_bytes; /* on-chip working set of one tile, 0 = n/a */
    uint64_t      tiles;         /* tiles the plan runs, >= 1 */
} ocu_op_caps;

typedef struct ocu_op_stats {
    uint64_t op_calls[OCU_OP_COUNT]; /* successful execute calls per op */
    uint64_t plans_created;
    uint64_t plans_refused;
    uint64_t bytes_to_device;
    uint64_t bytes_to_host;
    uint64_t bytes_on_device;
    uint64_t kernel_launches_refused;
} ocu_op_stats;

typedef struct ocu_op_plan ocu_op_plan; /* owned by the backend */

/* ---- the table (spec section 5) ----------------------------------------- */

typedef struct ocu_op_table {
    uint32_t version;     /* OCU_OPS_VERSION the backend implements */
    uint32_t struct_size; /* sizeof(ocu_op_table) the backend was built with */

    ocu_status (*support)(ocu_device *dev, ocu_op op, ocu_dtype dtype,
                          ocu_op_support *out);
    ocu_status (*query)(ocu_device *dev, const ocu_op_desc *desc,
                        ocu_op_caps *caps);
    ocu_status (*plan)(ocu_device *dev, const ocu_op_desc *desc,
                       ocu_op_plan **plan);
    ocu_status (*execute)(ocu_op_plan *plan, ocu_queue *queue,
                          const ocu_tensor *tensors, uint32_t num_tensors);
    ocu_status (*destroy)(ocu_op_plan *plan);
    /* Optional: may be NULL. */
    ocu_status (*stats)(ocu_device *dev, ocu_op_stats *out);
} ocu_op_table;

/* The table of a backend, or NULL (no table, or an unusable version). */
static inline const ocu_op_table *ocu_ops_of(const ocu_backend *b)
{
    if (b == NULL || b->get_provider == NULL) return NULL;
    const ocu_op_table *t = (const ocu_op_table *)b->get_provider(OCU_OPS_PROVIDER);
    if (t == NULL || t->version < 1u ||
        t->struct_size < offsetof(ocu_op_table, stats) + sizeof(t->stats)) {
        return NULL;
    }
    return t;
}

static inline const char *ocu_op_name(ocu_op op)
{
    switch (op) {
    case OCU_OP_GEMM:     return "gemm";
    case OCU_OP_ADD:      return "add";
    case OCU_OP_MUL:      return "mul";
    case OCU_OP_RMS_NORM: return "rms_norm";
    case OCU_OP_SOFTMAX:  return "softmax";
    case OCU_OP_ROPE:     return "rope";
    case OCU_OP_QMATMUL:  return "qmatmul";
    }
    return "unknown";
}

static inline const char *ocu_dtype_name(ocu_dtype dt)
{
    switch (dt) {
    case OCU_DT_NONE:   return "none";
    case OCU_DT_F32:    return "f32";
    case OCU_DT_F16:    return "f16";
    case OCU_DT_BF16:   return "bf16";
    case OCU_DT_I8:     return "i8";
    case OCU_DT_I32:    return "i32";
    case OCU_DT_Q8_B32: return "q8_b32";
    case OCU_DT_Q4_B32: return "q4_b32";
    }
    return "unknown";
}

#ifdef __cplusplus
}
#endif

#endif /* OCU_OPS_H */

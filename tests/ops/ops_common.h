/*
 * ops_common.h -- helpers shared by the operator table tests.
 *
 * Written from docs/spec/op-table.md only (no implementation sources).
 * Device memory is reached only through the backend's mem_copy; the host
 * never dereferences a device pointer.
 */
#ifndef OCU_TESTS_OPS_COMMON_H
#define OCU_TESTS_OPS_COMMON_H

#include "ocu_core.h"
#include "ocu_ops.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__)
#define OCU_TEST_PRINTF(a, b) __attribute__((format(printf, a, b)))
#else
#define OCU_TEST_PRINTF(a, b)
#endif

/* ---- test state and checks ----------------------------------------------- */

typedef struct test_env {
    const ocu_backend  *b;
    const ocu_op_table *t;
    ocu_device         *dev;
    ocu_queue          *q;
    long                checks;
    long                failures;
    int                 details;    /* detail lines left for the current case */
    char                where[384]; /* description of the current case */
    uint64_t            rng;
    ocu_op_caps         last_caps;  /* caps of the last run_op */
} test_env;

extern test_env G;

void set_case(const char *fmt, ...) OCU_TEST_PRINTF(1, 2);
void detail(const char *fmt, ...) OCU_TEST_PRINTF(1, 2);
int  check_at(int ok, const char *file, int line, const char *expr);
int  check_st_at(ocu_status got, ocu_status want, const char *file, int line,
                 const char *expr);

#define CHECK(c)          check_at((c) ? 1 : 0, __FILE__, __LINE__, #c)
#define CHECK_ST(e, want) check_st_at((e), (want), __FILE__, __LINE__, #e)

/* ---- random numbers ------------------------------------------------------ */

uint64_t rng_next(void);
double   urand(double lo, double hi);
int64_t  irand(int64_t lo, int64_t hi); /* inclusive */

/* ---- number formats (spec 2 and 6) --------------------------------------- */

double   round_to(ocu_dtype dt, double x); /* round to nearest, ties to even */
double   unit_round(ocu_dtype dt);         /* u(dt) */
double   tiny_of(ocu_dtype dt);            /* smallest positive normal */
double   ulp_at(ocu_dtype dt, double r);   /* spacing at |r| (at tiny below it) */
uint16_t f16_encode(double v);             /* v must be a binary16 value */
double   f16_decode(uint16_t h);
uint16_t bf16_encode(double v);            /* v must be a bfloat16 value */
double   bf16_decode(uint16_t h);
double   load_el(ocu_dtype dt, const uint8_t *buf, int64_t i);
void     store_el(ocu_dtype dt, uint8_t *buf, int64_t i, double v); /* rounds */

int ok_bound(double y, double r, double bound); /* NaN/Inf rules + |y-r|<=bound */
int ok_exact(ocu_dtype dt, double y, double r); /* OCU_TOL_EXACT */

/* ---- descriptors --------------------------------------------------------- */

#define SHP(...) ((const int64_t[4]){__VA_ARGS__})

/* Row-major descriptor; `pad` elements are added to every outer stride. */
ocu_tensor_desc tdesc(ocu_dtype dt, uint32_t rank, const int64_t *shape, int64_t pad);
int64_t desc_span(const ocu_tensor_desc *d);
int     desc_dense(const ocu_tensor_desc *d);
int64_t desc_offset(const ocu_tensor_desc *d, const int64_t *idx);
int     next_index(const ocu_tensor_desc *d, int64_t *idx); /* 0 when done */
int64_t row_count(const ocu_tensor_desc *d);
int64_t row_base(const ocu_tensor_desc *d, int64_t row);
ocu_op_desc op_desc(ocu_op op); /* zeroed, struct_size set */
const char *shape_str(const ocu_tensor_desc *d, char *buf, size_t n);

/* ---- host images of tensors ---------------------------------------------- */

typedef struct tbuf {
    ocu_tensor_desc d;   /* planned descriptor (sizes the buffer) */
    ocu_tensor_desc e;   /* descriptor passed to execute (fewer rows: 4.7) */
    uint8_t        *h;   /* host image: desc_span(d) elements */
    uint8_t        *res; /* output only: device contents after the run */
    size_t          bytes;
    void           *dptr;
    int             alias; /* -1, or the slot whose device buffer this one is */
} tbuf;

void tb_init(tbuf *t, ocu_tensor_desc d, double fill); /* fill = padding too */
void tb_alias(tbuf *t, const tbuf *src, int slot);
void tb_free(tbuf *t);
void tb_fill(tbuf *t, double v);                 /* tensor elements of d */
void tb_rand(tbuf *t, double lo, double hi);
void tb_rand_exp(tbuf *t, int elo, int ehi);     /* +-[1,2) * 2^e */

/* ---- running operators --------------------------------------------------- */

int        get_stats(ocu_op_stats *s); /* 0 when the table has no stats entry */
ocu_tol_class tol_of(ocu_op op);

/*
 * query + plan with agreement checks (5.2, 5.3), support consistency (5.1)
 * and stats deltas (5.6). Returns the plan status (query's when they agree).
 * With plan_out == NULL a successful plan is destroyed.
 */
ocu_status qp(const ocu_op_desc *d, ocu_op_caps *caps_out, ocu_op_plan **plan_out);
void expect_at(const ocu_op_desc *d, ocu_status want, const char *file, int line);
#define EXPECT(d, want) expect_at(&(d), (want), __FILE__, __LINE__)

/*
 * Plans pd with t[i].d, uploads, executes with t[i].e (on G.q when
 * on_queue), downloads the last slot into t[n-1].res. Returns the query /
 * plan status (or the execute status). Caps are left in G.last_caps.
 */
ocu_status run_op(ocu_op_desc *pd, tbuf *t, uint32_t n, int on_queue);
/* Every output element outside the executed region still holds `init`. */
void check_untouched(const tbuf *o, const uint8_t *init);

/* ---- numeric cases ------------------------------------------------------- */

typedef struct gemm_case {
    ocu_dtype din, dout;   /* dout 0 = din */
    uint32_t  rank;        /* 2 or 3 (0 = 2) */
    int64_t   bt, ba, bb;  /* batch of C, A, B (rank 3) */
    int64_t   m, n, k;
    uint32_t  ta, tb;
    float     alpha, beta;
    int64_t   pad;         /* row padding (0 = dense) */
    int       c_fill;      /* 0 random, 1 NaN, 2 +Inf */
    int       ab_nan;      /* A and B all NaN */
    int       var;         /* OCU_OP_FLAG_VAR_ROWS */
    int64_t   m_exec;      /* rows given to execute (0 = m) */
    int       on_queue;
} gemm_case;

typedef struct elt_case {
    ocu_op    op;          /* ADD or MUL */
    ocu_dtype dt;
    uint32_t  rank;
    int64_t   a[4], b[4];
    int64_t   pad;
    int       inplace;
    int       var;
    int64_t   m_exec;
    int       specials;    /* f32 only, b of a's shape */
    int       on_queue;
} elt_case;

typedef struct row_case {
    ocu_dtype dt;
    uint32_t  rank;
    int64_t   shape[4];
    int64_t   pad;
    int       weight;      /* RMS_NORM */
    float     eps;         /* RMS_NORM */
    float     scale;       /* SOFTMAX */
    double    mag;         /* inputs in [-mag, mag] (0 = default) */
    int       inplace;
    int       var;
    int64_t   m_exec;
    int       nan_row;     /* SOFTMAX: rows with NaN / +Inf / -Inf */
    int       on_queue;
} row_case;

typedef struct rope_case {
    uint32_t rank;         /* 2: [T,N], 3: [T,H,N] */
    int64_t  t, h, n;
    uint32_t mode, n_dims;
    float    theta, freq_scale;
    int      on_queue;
} rope_case;

ocu_status run_gemm(const gemm_case *gc);
ocu_status run_elt(const elt_case *ec);
ocu_status run_rms(const row_case *rc);
ocu_status run_softmax(const row_case *rc);
ocu_status run_rope(const rope_case *rc);

#endif /* OCU_TESTS_OPS_COMMON_H */

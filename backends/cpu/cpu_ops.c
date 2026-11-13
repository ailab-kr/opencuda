/*
 * CPU backend: operator table (backends/interface/ocu_ops.h,
 * docs/spec/op-table.md).
 *
 * Every execute is a kernel run through cpu_kernel_launch, so it obeys
 * queue order like any other work. GEMM runs the blocked routine of
 * cpu_gemm.c; for the other operators each block of the kernel computes a
 * range of output rows with the shared reference (backends/ops_ref).
 * The descriptor and pointers travel in the launch arguments, so a plan
 * may be destroyed while work that used it is still queued.
 */
#include "cpu_gemm.h"
#include "cpu_internal.h"
#include "ocu_ops_ref.h"

#include <stdlib.h>
#include <string.h>

#define PLAN_MAGIC 0x6f707063u /* "oppc" */

struct ocu_op_plan {
    uint32_t     magic;
    ocu_op_plan *next; /* live plans, under g_plans_lock */
    ocu_op_desc  desc;
};

static pthread_mutex_t g_plans_lock = PTHREAD_MUTEX_INITIALIZER;
static ocu_op_plan    *g_plans;

static int plan_live_locked(const ocu_op_plan *p)
{
    for (const ocu_op_plan *q = g_plans; q != NULL; q = q->next) {
        if (q == p) return p->magic == PLAN_MAGIC;
    }
    return 0;
}

static int is_float(ocu_dtype dt)
{
    return dt == OCU_DT_F32 || dt == OCU_DT_F16 || dt == OCU_DT_BF16;
}

static ocu_status cpu_op_support(ocu_device *dev, ocu_op op, ocu_dtype dtype,
                                 ocu_op_support *out)
{
    if (dev == NULL) return OCU_ERR_INVALID_DEVICE;
    if (out == NULL) return OCU_ERR_INVALID_VALUE;
    memset(out, 0, sizeof(*out));
    switch (op) {
    case OCU_OP_GEMM: case OCU_OP_ADD: case OCU_OP_MUL:
    case OCU_OP_RMS_NORM: case OCU_OP_SOFTMAX:
        break;
    case OCU_OP_ROPE:    return OCU_ERR_NOT_IMPLEMENTED;
    case OCU_OP_QMATMUL: return OCU_ERR_NOT_SUPPORTED;
    default:             return OCU_ERR_INVALID_VALUE;
    }
    if (!is_float(dtype)) return OCU_ERR_NOT_SUPPORTED;
    out->layouts = OCU_LAYOUT_DENSE | OCU_LAYOUT_ROW_STRIDED;
    out->max_rank = op == OCU_OP_GEMM ? 3u : OCU_OPS_MAX_RANK;
    out->max_row = 0;
    out->align_bytes = (uint32_t)ocu_dtype_size(dtype);
    out->accum_dtype = OCU_DT_F32;
    out->out_dtypes = 1u << dtype;
    if (op == OCU_OP_GEMM) out->out_dtypes |= 1u << OCU_DT_F32;
    out->needs_plan_ms = 0;
    out->var_rows = 1;
    return OCU_OK;
}

static ocu_tol_class tol_of(ocu_op op)
{
    switch (op) {
    case OCU_OP_GEMM:                      return OCU_TOL_DOT;
    case OCU_OP_RMS_NORM: case OCU_OP_SOFTMAX: return OCU_TOL_ROW;
    case OCU_OP_ROPE:                      return OCU_TOL_ULP;
    default:                               return OCU_TOL_EXACT;
    }
}

static ocu_status cpu_op_query(ocu_device *dev, const ocu_op_desc *desc,
                               ocu_op_caps *caps)
{
    if (dev == NULL) return OCU_ERR_INVALID_DEVICE;
    if (desc == NULL || caps == NULL) return OCU_ERR_INVALID_VALUE;
    memset(caps, 0, sizeof(*caps));
    ocu_status st = ocu_ref_check_desc(desc);
    if (st != OCU_OK) return st;
    caps->tol = tol_of(desc->op);
    caps->accum_dtype = OCU_DT_F32;
    caps->scratch_bytes = 0;
    caps->tiles = 1;
    return OCU_OK;
}

static ocu_status cpu_op_plan(ocu_device *dev, const ocu_op_desc *desc,
                              ocu_op_plan **plan)
{
    if (plan != NULL) *plan = NULL;
    if (dev == NULL) return OCU_ERR_INVALID_DEVICE;
    if (desc == NULL || plan == NULL) return OCU_ERR_INVALID_VALUE;
    ocu_status st = ocu_ref_check_desc(desc);
    if (st != OCU_OK) return st;
    ocu_op_plan *p = calloc(1, sizeof(*p));
    if (p == NULL) return OCU_ERR_OUT_OF_MEMORY;
    p->magic = PLAN_MAGIC;
    p->desc = *desc;
    p->desc.struct_size = sizeof(ocu_op_desc);
    pthread_mutex_lock(&g_plans_lock);
    p->next = g_plans;
    g_plans = p;
    pthread_mutex_unlock(&g_plans_lock);
    *plan = p;
    return OCU_OK;
}

static ocu_status cpu_op_destroy(ocu_op_plan *plan)
{
    if (plan == NULL) return OCU_ERR_INVALID_HANDLE;
    pthread_mutex_lock(&g_plans_lock);
    ocu_op_plan **at = &g_plans;
    while (*at != NULL && *at != plan) at = &(*at)->next;
    if (*at == NULL) {
        pthread_mutex_unlock(&g_plans_lock);
        return OCU_ERR_INVALID_HANDLE;
    }
    *at = plan->next;
    pthread_mutex_unlock(&g_plans_lock);
    plan->magic = 0;
    free(plan);
    return OCU_OK;
}

/* Launch argument: everything the kernel needs, copied at submission. */
typedef struct op_launch {
    ocu_op_desc desc;
    void       *data[OCU_OPS_MAX_TENSORS];
    int64_t     rows;
    int64_t     rows_per_block;
} op_launch;

static void k_op(void **args)
{
    const op_launch *l = args[0];
    int64_t r0 = (int64_t)ocu_cpu_block_idx().x * l->rows_per_block;
    int64_t r1 = r0 + l->rows_per_block;
    if (r1 > l->rows) r1 = l->rows;
    if (r0 < r1) ocu_ref_run(&l->desc, l->data, r0, r1);
}

static ocu_status cpu_op_execute(ocu_op_plan *plan, ocu_queue *queue,
                                 const ocu_tensor *tensors, uint32_t num_tensors)
{
    op_launch l;
    ocu_op_desc planned;
    memset(&l, 0, sizeof(l));
    pthread_mutex_lock(&g_plans_lock);
    int live = plan != NULL && plan_live_locked(plan);
    if (live) planned = plan->desc;
    pthread_mutex_unlock(&g_plans_lock);
    if (!live) return OCU_ERR_INVALID_HANDLE;

    ocu_status st = ocu_ref_check_tensors(&planned, tensors, num_tensors, &l.desc);
    if (st != OCU_OK) return st;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        const ocu_tensor *t = &tensors[i];
        if ((uintptr_t)t->data % ocu_dtype_size(t->desc.dtype) != 0 ||
            cpu_check_device_range(t->data, ocu_ref_span_bytes(&t->desc)) != OCU_OK) {
            return OCU_ERR_INVALID_VALUE;
        }
        l.data[i] = t->data;
    }

    /* GEMM: the blocked routine (cpu_gemm.c) unless OCU_CPU_GEMM=ref. */
    if (l.desc.op == OCU_OP_GEMM && cpu_gemm_available()) {
        return cpu_gemm_launch(queue, &l.desc, l.data);
    }

    l.rows = ocu_ref_rows(&l.desc);
    /* About 256 blocks at most; a block is one fiber running whole rows. */
    l.rows_per_block = (l.rows + 255) / 256;
    if (l.rows_per_block < 1) l.rows_per_block = 1;
    int64_t blocks = (l.rows + l.rows_per_block - 1) / l.rows_per_block;

    struct ocu_kernel k = {NULL, k_op, NULL};
    void *args[1] = {&l};
    size_t sizes[1] = {sizeof(l)};
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = (uint32_t)blocks; p.grid[1] = 1; p.grid[2] = 1;
    p.block[0] = 1; p.block[1] = 1; p.block[2] = 1;
    p.args = args;
    p.arg_sizes = sizes;
    p.num_args = 1;
    return cpu_kernel_launch(queue, &k, &p);
}

static const ocu_op_table cpu_ops = {
    .version = OCU_OPS_VERSION,
    .struct_size = sizeof(ocu_op_table),
    .support = cpu_op_support,
    .query = cpu_op_query,
    .plan = cpu_op_plan,
    .execute = cpu_op_execute,
    .destroy = cpu_op_destroy,
    .stats = NULL,
};

const ocu_op_table *cpu_op_table(void) { return &cpu_ops; }

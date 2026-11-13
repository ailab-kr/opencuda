/*
 * CPU backend: the operator table's GEMM (docs/spec/op-table.md 4.1) as a
 * blocked routine instead of the row-by-row reference.
 *
 * Work is split into output tiles (up to TILE_M rows by TILE_N columns of
 * one batch entry); each tile is one block of a kernel launch with a single
 * GPU thread, so tiles run on the backend's worker pool in queue order like
 * any other work. Inside a tile the inner dimension is taken in chunks of
 * KC: the A rows and B columns of the chunk are copied into per-OS-thread
 * panels as binary32, contiguous along k (converting f16 and bf16 there),
 * and dot products over the panels run 4 x 3 at a time on 8-wide vectors.
 *
 * Accumulation is binary32. Each output element is accumulated in the same
 * order whatever the number of OS threads (tiles never split k), so results
 * are reproducible. The order differs from the reference (eight partial
 * sums per chunk), which the GEMM tolerance class allows (spec 6.1).
 */
#include "cpu_gemm.h"
#include "cpu_internal.h"
#include "ocu_ops_ref.h"

#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define GEMM_HAVE 1
#else
#define GEMM_HAVE 0
#endif

#define TILE_M 64   /* rows of C per tile (multiple of GEMM_MR) */
#define TILE_N 96   /* columns of C per tile (multiple of GEMM_NR) */
#define KC     512  /* inner-dimension chunk (multiple of 8) */
#define MIN_TILES 64 /* split finer below this many tiles */

typedef struct gemm_job {
    const void *a, *b;
    void       *c;
    ocu_dtype   in, out;
    int         ta, tb;
    int64_t     m, n, k, batch;
    int64_t     lda, ldb, ldc; /* row strides */
    int64_t     sa, sb, sc;    /* batch strides; 0 broadcasts */
    float       alpha, beta;
    int64_t     tm, tn;        /* tile size */
    int64_t     tiles_m, tiles_n;
    int         avx2;
} gemm_job;

static int64_t round_up(int64_t v, int64_t to) { return (v + to - 1) / to * to; }

/* ---- per-OS-thread panel memory ----------------------------------------- */

static pthread_key_t g_buf_key;
static pthread_once_t g_buf_once = PTHREAD_ONCE_INIT;
static int g_buf_key_ok;

typedef struct gemm_buf {
    void  *raw;
    float *p; /* 64-byte aligned */
    size_t floats;
} gemm_buf;

static void buf_free(void *v)
{
    gemm_buf *b = v;
    if (b != NULL) free(b->raw);
    free(b);
}

static void buf_key_init(void) { g_buf_key_ok = pthread_key_create(&g_buf_key, buf_free) == 0; }

/* At least `floats` floats for the calling OS thread, kept for reuse. */
static float *thread_buf(size_t floats)
{
    pthread_once(&g_buf_once, buf_key_init);
    if (!g_buf_key_ok) return NULL;
    gemm_buf *b = pthread_getspecific(g_buf_key);
    if (b == NULL) {
        b = calloc(1, sizeof(*b));
        if (b == NULL) return NULL;
        if (pthread_setspecific(g_buf_key, b) != 0) {
            free(b);
            return NULL;
        }
    }
    if (b->floats < floats) {
        free(b->raw);
        b->raw = malloc(floats * sizeof(float) + 64);
        if (b->raw == NULL) {
            b->p = NULL;
            b->floats = 0;
            return NULL;
        }
        b->p = (float *)(((uintptr_t)b->raw + 63) & ~(uintptr_t)63);
        b->floats = floats;
    }
    return b->p;
}

/* ---- one tile ------------------------------------------------------------ */

static float load_elem(ocu_dtype dt, const void *p, int64_t i)
{
    switch (dt) {
    case OCU_DT_F16:  return ocu_ref_f16_to_f32(((const uint16_t *)p)[i]);
    case OCU_DT_BF16: return ocu_ref_bf16_to_f32(((const uint16_t *)p)[i]);
    default:          return ((const float *)p)[i];
    }
}

static void store_elem(ocu_dtype dt, void *p, int64_t i, float v)
{
    switch (dt) {
    case OCU_DT_F16:  ((uint16_t *)p)[i] = ocu_ref_f32_to_f16(v); break;
    case OCU_DT_BF16: ((uint16_t *)p)[i] = ocu_ref_f32_to_bf16(v); break;
    default:          ((float *)p)[i] = v; break;
    }
}

static void k_gemm_tile(void **args)
{
    const gemm_job *j = args[0];
    int64_t t = (int64_t)ocu_cpu_block_idx().x;
    const int64_t tile_n = t % j->tiles_n;
    t /= j->tiles_n;
    const int64_t tile_m = t % j->tiles_m;
    const int64_t bt = t / j->tiles_m;

    const int64_t i0 = tile_m * j->tm, j0 = tile_n * j->tn;
    const int64_t rows = j->m - i0 < j->tm ? j->m - i0 : j->tm;
    const int64_t cols = j->n - j0 < j->tn ? j->n - j0 : j->tn;
    const int64_t rows_pad = round_up(rows, GEMM_MR), cols_pad = round_up(cols, GEMM_NR);
    const size_t a_bytes = ocu_dtype_size(j->in);
    const char *A = (const char *)j->a + (size_t)(bt * j->sa) * a_bytes;
    const char *B = (const char *)j->b + (size_t)(bt * j->sb) * a_bytes;
    char *C = (char *)j->c + (size_t)(bt * j->sc) * ocu_dtype_size(j->out);

    const int64_t kc_max = j->k < KC ? round_up(j->k, 8) : KC;
    float *mem = thread_buf((size_t)((rows_pad + cols_pad) * kc_max + rows_pad * cols_pad));
    if (mem == NULL) {
        /* No memory for panels: compute this tile directly. Slow, exact. */
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t c = 0; c < cols; ++c) {
                float acc = 0.0f;
                if (j->alpha != 0.0f) {
                    for (int64_t p = 0; p < j->k; ++p) {
                        const int64_t gi = i0 + i, gc = j0 + c;
                        float x = load_elem(j->in, A, j->ta ? p * j->lda + gi : gi * j->lda + p);
                        float y = load_elem(j->in, B, j->tb ? gc * j->ldb + p : p * j->ldb + gc);
                        acc += x * y;
                    }
                }
                const int64_t ci = (i0 + i) * j->ldc + j0 + c;
                float v = j->alpha == 0.0f ? 0.0f : j->alpha * acc;
                if (j->beta != 0.0f) v += j->beta * load_elem(j->out, C, ci);
                store_elem(j->out, C, ci, v);
            }
        }
        return;
    }
    float *ap = mem, *bp = ap + rows_pad * kc_max, *acc = bp + cols_pad * kc_max;
    memset(acc, 0, (size_t)(rows_pad * cols_pad) * sizeof(float));

    if (j->alpha != 0.0f) {
        for (int64_t k0 = 0; k0 < j->k; k0 += KC) {
            const int64_t kc = j->k - k0 < KC ? j->k - k0 : KC;
            const int64_t kcp = round_up(kc, 8);
            /* A(i, p) = A[i*lda + p], or A[p*lda + i] when transposed;
               B column c, p = B[c*ldb + p] when transposed, else B[p*ldb + c]. */
#if GEMM_HAVE
            if (j->avx2) {
                cpu_gemm_panel_avx2(ap, kcp, A, j->in, j->lda, !j->ta, i0, rows, rows_pad, k0, kc);
                cpu_gemm_panel_avx2(bp, kcp, B, j->in, j->ldb, j->tb, j0, cols, cols_pad, k0, kc);
                cpu_gemm_dots_avx2(acc, cols_pad, ap, bp, rows_pad, cols_pad, kcp);
            } else {
                cpu_gemm_panel_generic(ap, kcp, A, j->in, j->lda, !j->ta, i0, rows, rows_pad, k0, kc);
                cpu_gemm_panel_generic(bp, kcp, B, j->in, j->ldb, j->tb, j0, cols, cols_pad, k0, kc);
                cpu_gemm_dots_generic(acc, cols_pad, ap, bp, rows_pad, cols_pad, kcp);
            }
#endif
        }
    }

    for (int64_t i = 0; i < rows; ++i) {
        const int64_t row = (i0 + i) * j->ldc + j0;
        for (int64_t c = 0; c < cols; ++c) {
            float v = j->alpha == 0.0f ? 0.0f : j->alpha * acc[i * cols_pad + c];
            if (j->beta != 0.0f) v += j->beta * load_elem(j->out, C, row + c); /* beta 0: C not read */
            store_elem(j->out, C, row + c, v);
        }
    }
}

/* ---- dispatch ------------------------------------------------------------ */

static int g_mode = -1; /* 0: reference, 1: generic, 2: avx2 */
static pthread_once_t g_mode_once = PTHREAD_ONCE_INIT;

static void mode_init(void)
{
    const char *s = getenv("OCU_CPU_GEMM");
    if (!GEMM_HAVE || (s != NULL && strcmp(s, "ref") == 0)) {
        g_mode = 0;
        return;
    }
    g_mode = 1;
#if defined(OCU_CPU_GEMM_AVX2) && (defined(__GNUC__) || defined(__clang__))
    if (!(s != NULL && strcmp(s, "generic") == 0)) {
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
            __builtin_cpu_supports("f16c")) {
            g_mode = 2;
        }
    }
#endif
}

int cpu_gemm_available(void)
{
    pthread_once(&g_mode_once, mode_init);
    return g_mode > 0;
}

ocu_status cpu_gemm_launch(ocu_queue *queue, const ocu_op_desc *d, void *const *data)
{
    const ocu_tensor_desc *a = &d->tensors[0], *b = &d->tensors[1], *c = &d->tensors[2];
    const uint32_t o = a->rank - 2;
    gemm_job j;
    memset(&j, 0, sizeof(j));
    j.a = data[0]; j.b = data[1]; j.c = data[2];
    j.in = a->dtype; j.out = c->dtype;
    j.ta = (int)d->p.gemm.trans_a; j.tb = (int)d->p.gemm.trans_b;
    j.m = c->shape[o]; j.n = c->shape[o + 1];
    j.k = j.ta ? a->shape[o] : a->shape[o + 1];
    j.lda = a->stride[o]; j.ldb = b->stride[o]; j.ldc = c->stride[o];
    j.batch = 1;
    if (o == 1) {
        j.batch = c->shape[0];
        j.sa = a->shape[0] == 1 ? 0 : a->stride[0];
        j.sb = b->shape[0] == 1 ? 0 : b->stride[0];
        j.sc = c->stride[0];
    }
    j.alpha = d->p.gemm.alpha; j.beta = d->p.gemm.beta;
    j.avx2 = g_mode == 2;

    /* Tile size: as large as allowed, then finer while there are too few
       tiles to keep the worker threads busy. */
    j.tm = j.m < TILE_M ? round_up(j.m, GEMM_MR) : TILE_M;
    j.tn = j.n < TILE_N ? round_up(j.n, GEMM_NR) : TILE_N;
    for (;;) {
        int64_t tiles = j.batch * ((j.m + j.tm - 1) / j.tm) * ((j.n + j.tn - 1) / j.tn);
        if (tiles >= MIN_TILES) break;
        if (j.tn > 4 * GEMM_NR) j.tn = round_up(j.tn / 2, GEMM_NR);
        else if (j.tm > 4 * GEMM_MR) j.tm = round_up(j.tm / 2, GEMM_MR);
        else break;
    }
    j.tiles_m = (j.m + j.tm - 1) / j.tm;
    j.tiles_n = (j.n + j.tn - 1) / j.tn;
    const int64_t blocks = j.batch * j.tiles_m * j.tiles_n;
    if (blocks > 0x7fffffffLL) return OCU_ERR_NOT_SUPPORTED;

    struct ocu_kernel k = {NULL, k_gemm_tile};
    void *args[1] = {&j};
    size_t sizes[1] = {sizeof(j)};
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = (uint32_t)blocks; p.grid[1] = 1; p.grid[2] = 1;
    p.block[0] = 1; p.block[1] = 1; p.block[2] = 1;
    p.args = args;
    p.arg_sizes = sizes;
    p.num_args = 1;
    return cpu_kernel_launch(queue, &k, &p);
}

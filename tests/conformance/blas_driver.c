/*
 * BLAS conformance driver: reads cases on stdin, runs each through
 * libs/blas on the active backend, writes outputs on stdout.
 *
 * Protocol (one token per field, values as C99 hex floats, nan, inf):
 *   in:  case <id> <fn> <kind>        out: out <id> <status>
 *        i <name> <int>                    i <name> <int>
 *        c <name> <char>                   r <name> <value>
 *        r <name> <value>                  v <name> <count> <values...>
 *        v <name> <count> <values...>      end
 *        end
 * The Python side (conformance/harness/ocu_conf/blas_run.py) writes the
 * cases and grades the outputs. OCU_DRIVER_QUEUE=1 runs every call on a
 * queue instead of synchronously.
 */
#define _POSIX_C_SOURCE 200809L

#include "ocu_blas.h"
#include "ocu_core.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FIELDS 32

typedef struct field {
    char    kind;      /* 'i', 'c', 'r', 'v' */
    char    name[32];
    long    i;
    char    c;
    double  r;
    double *v;
    long    count;
} field;

typedef struct kase {
    char  id[256], fn[32], kind[8];
    field f[MAX_FIELDS];
    int   nf;
} kase;

static field *get(kase *k, const char *name)
{
    for (int i = 0; i < k->nf; ++i) {
        if (strcmp(k->f[i].name, name) == 0) return &k->f[i];
    }
    fprintf(stderr, "driver: case %s has no field %s\n", k->id, name);
    exit(2);
}

static long I(kase *k, const char *n) { return get(k, n)->i; }
static double R(kase *k, const char *n) { return get(k, n)->r; }
static char C(kase *k, const char *n) { return get(k, n)->c; }

static int read_case(kase *k)
{
    char tok[256];
    if (scanf("%255s", tok) != 1) return 0;
    if (strcmp(tok, "case") != 0) { fprintf(stderr, "driver: expected case\n"); exit(2); }
    if (scanf("%255s %31s %7s", k->id, k->fn, k->kind) != 3) exit(2);
    k->nf = 0;
    for (;;) {
        if (scanf("%255s", tok) != 1) exit(2);
        if (strcmp(tok, "end") == 0) return 1;
        if (k->nf == MAX_FIELDS) exit(2);
        field *f = &k->f[k->nf++];
        memset(f, 0, sizeof(*f));
        f->kind = tok[0];
        if (scanf("%31s", f->name) != 1) exit(2);
        switch (f->kind) {
        case 'i': if (scanf("%ld", &f->i) != 1) exit(2); break;
        case 'c': if (scanf(" %c", &f->c) != 1) exit(2); break;
        case 'r': if (scanf("%255s", tok) != 1) exit(2); f->r = strtod(tok, NULL); break;
        case 'v':
            if (scanf("%ld", &f->count) != 1 || f->count < 0) exit(2);
            f->v = calloc((size_t)(f->count ? f->count : 1), sizeof(double));
            for (long j = 0; j < f->count; ++j) {
                if (scanf("%255s", tok) != 1) exit(2);
                f->v[j] = strtod(tok, NULL);
            }
            break;
        default: exit(2);
        }
    }
}

static const ocu_backend *B;
static ocu_device *D;
static ocu_queue *Q;
static int f32;

/* Copy a vector field to device memory in the case's element type. */
static void *to_dev(kase *k, const char *name)
{
    field *f = get(k, name);
    size_t es = f32 ? sizeof(float) : sizeof(double);
    size_t bytes = (size_t)(f->count ? f->count : 1) * es;
    void *host = malloc(bytes);
    for (long j = 0; j < f->count; ++j) {
        if (f32) ((float *)host)[j] = (float)f->v[j];
        else ((double *)host)[j] = f->v[j];
    }
    void *dev = NULL;
    if (B->mem_alloc(D, bytes, &dev) != OCU_OK ||
        B->mem_copy(D, NULL, dev, host, bytes, OCU_COPY_HOST_TO_DEVICE) != OCU_OK) {
        exit(3);
    }
    free(host);
    return dev;
}

static void emit_vec(kase *k, const char *name, void *dev)
{
    field *f = get(k, name);
    size_t es = f32 ? sizeof(float) : sizeof(double);
    size_t bytes = (size_t)(f->count ? f->count : 1) * es;
    void *host = malloc(bytes);
    if (B->mem_copy(D, NULL, host, dev, bytes, OCU_COPY_DEVICE_TO_HOST) != OCU_OK) exit(3);
    printf("v %s %ld", name, f->count);
    for (long j = 0; j < f->count; ++j) {
        double v = f32 ? (double)((float *)host)[j] : ((double *)host)[j];
        printf(" %a", v);
    }
    printf("\n");
    free(host);
}

static ocu_blas_op op(char c) { return c == 'N' ? OCU_BLAS_OP_N : OCU_BLAS_OP_T; }
static ocu_blas_fill fill(char c) { return c == 'U' ? OCU_BLAS_UPPER : OCU_BLAS_LOWER; }
static ocu_blas_diag diag(char c) { return c == 'U' ? OCU_BLAS_UNIT : OCU_BLAS_NON_UNIT; }
static ocu_blas_side side(char c) { return c == 'L' ? OCU_BLAS_LEFT : OCU_BLAS_RIGHT; }

static void finish(ocu_status st)
{
    if (st == OCU_OK && Q != NULL) st = B->queue_sync(Q);
    (void)st;
}

static void run(kase *k)
{
    const char *fn = k->fn;
    ocu_status st = OCU_OK;
    char out[4][8] = {{0}};
    void *dev[4] = {0};
    int nout = 0;
    double rres = 0;
    int ires = 0, has_r = 0, has_i = 0;
#define OUT(name, ptr) do { strcpy(out[nout], name); dev[nout] = (ptr); ++nout; } while (0)
    if (strcmp(fn, "axpy") == 0) {
        void *x = to_dev(k, "x"), *y = to_dev(k, "y");
        st = f32 ? ocu_blas_saxpy(Q, I(k, "n"), (float)R(k, "alpha"), x, I(k, "incx"), y, I(k, "incy"))
                 : ocu_blas_daxpy(Q, I(k, "n"), R(k, "alpha"), x, I(k, "incx"), y, I(k, "incy"));
        finish(st); OUT("y", y); B->mem_free(D, x);
    } else if (strcmp(fn, "dot") == 0) {
        void *x = to_dev(k, "x"), *y = to_dev(k, "y");
        if (f32) { float r; st = ocu_blas_sdot(Q, I(k, "n"), x, I(k, "incx"), y, I(k, "incy"), &r); rres = r; }
        else st = ocu_blas_ddot(Q, I(k, "n"), x, I(k, "incx"), y, I(k, "incy"), &rres);
        has_r = 1; B->mem_free(D, x); B->mem_free(D, y);
    } else if (strcmp(fn, "scal") == 0) {
        void *x = to_dev(k, "x");
        st = f32 ? ocu_blas_sscal(Q, I(k, "n"), (float)R(k, "alpha"), x, I(k, "incx"))
                 : ocu_blas_dscal(Q, I(k, "n"), R(k, "alpha"), x, I(k, "incx"));
        finish(st); OUT("x", x);
    } else if (strcmp(fn, "nrm2") == 0 || strcmp(fn, "asum") == 0) {
        void *x = to_dev(k, "x");
        int nrm = fn[0] == 'n';
        if (f32) {
            float r;
            st = nrm ? ocu_blas_snrm2(Q, I(k, "n"), x, I(k, "incx"), &r)
                     : ocu_blas_sasum(Q, I(k, "n"), x, I(k, "incx"), &r);
            rres = r;
        } else {
            st = ocu_blas_dnrm2(Q, I(k, "n"), x, I(k, "incx"), &rres);
        }
        has_r = 1; B->mem_free(D, x);
    } else if (strcmp(fn, "iamax") == 0) {
        void *x = to_dev(k, "x");
        st = ocu_blas_isamax(Q, I(k, "n"), x, I(k, "incx"), &ires);
        has_i = 1; B->mem_free(D, x);
    } else if (strcmp(fn, "copy") == 0 || strcmp(fn, "swap") == 0) {
        void *x = to_dev(k, "x"), *y = to_dev(k, "y");
        if (fn[0] == 'c') {
            st = ocu_blas_scopy(Q, I(k, "n"), x, I(k, "incx"), y, I(k, "incy"));
            finish(st); OUT("y", y); B->mem_free(D, x);
        } else {
            st = ocu_blas_sswap(Q, I(k, "n"), x, I(k, "incx"), y, I(k, "incy"));
            finish(st); OUT("x", x); OUT("y", y);
        }
    } else if (strcmp(fn, "gemv") == 0) {
        void *a = to_dev(k, "a"), *x = to_dev(k, "x"), *y = to_dev(k, "y");
        st = f32 ? ocu_blas_sgemv(Q, op(C(k, "trans")), I(k, "m"), I(k, "n"), (float)R(k, "alpha"),
                                  a, I(k, "lda"), x, I(k, "incx"), (float)R(k, "beta"), y, I(k, "incy"))
                 : ocu_blas_dgemv(Q, op(C(k, "trans")), I(k, "m"), I(k, "n"), R(k, "alpha"),
                                  a, I(k, "lda"), x, I(k, "incx"), R(k, "beta"), y, I(k, "incy"));
        finish(st); OUT("y", y); B->mem_free(D, a); B->mem_free(D, x);
    } else if (strcmp(fn, "ger") == 0) {
        void *x = to_dev(k, "x"), *y = to_dev(k, "y"), *a = to_dev(k, "a");
        st = ocu_blas_sger(Q, I(k, "m"), I(k, "n"), (float)R(k, "alpha"), x, I(k, "incx"),
                           y, I(k, "incy"), a, I(k, "lda"));
        finish(st); OUT("a", a); B->mem_free(D, x); B->mem_free(D, y);
    } else if (strcmp(fn, "trsv") == 0) {
        void *a = to_dev(k, "a"), *x = to_dev(k, "x");
        st = ocu_blas_strsv(Q, fill(C(k, "uplo")), op(C(k, "trans")), diag(C(k, "diag")),
                            I(k, "n"), a, I(k, "lda"), x, I(k, "incx"));
        finish(st); OUT("x", x); B->mem_free(D, a);
    } else if (strcmp(fn, "gemm") == 0) {
        void *a = to_dev(k, "a"), *b = to_dev(k, "b"), *c = to_dev(k, "c");
        st = f32 ? ocu_blas_sgemm(Q, op(C(k, "transa")), op(C(k, "transb")), I(k, "m"), I(k, "n"),
                                  I(k, "k"), (float)R(k, "alpha"), a, I(k, "lda"), b, I(k, "ldb"),
                                  (float)R(k, "beta"), c, I(k, "ldc"))
                 : ocu_blas_dgemm(Q, op(C(k, "transa")), op(C(k, "transb")), I(k, "m"), I(k, "n"),
                                  I(k, "k"), R(k, "alpha"), a, I(k, "lda"), b, I(k, "ldb"),
                                  R(k, "beta"), c, I(k, "ldc"));
        finish(st); OUT("c", c); B->mem_free(D, a); B->mem_free(D, b);
    } else if (strcmp(fn, "syrk") == 0) {
        void *a = to_dev(k, "a"), *c = to_dev(k, "c");
        st = ocu_blas_ssyrk(Q, fill(C(k, "uplo")), op(C(k, "trans")), I(k, "n"), I(k, "k"),
                            (float)R(k, "alpha"), a, I(k, "lda"), (float)R(k, "beta"), c, I(k, "ldc"));
        finish(st); OUT("c", c); B->mem_free(D, a);
    } else if (strcmp(fn, "trsm") == 0) {
        void *a = to_dev(k, "a"), *b = to_dev(k, "b");
        st = ocu_blas_strsm(Q, side(C(k, "side")), fill(C(k, "uplo")), op(C(k, "trans")),
                            diag(C(k, "diag")), I(k, "m"), I(k, "n"), (float)R(k, "alpha"),
                            a, I(k, "lda"), b, I(k, "ldb"));
        finish(st); OUT("b", b); B->mem_free(D, a);
    } else {
        fprintf(stderr, "driver: unknown fn %s\n", fn);
        exit(2);
    }
    printf("out %s %d\n", k->id, (int)st);
    if (has_r) printf("r result %a\n", rres);
    if (has_i) printf("i result %d\n", ires);
    for (int i = 0; i < nout; ++i) {
        emit_vec(k, out[i], dev[i]);
        B->mem_free(D, dev[i]);
    }
    printf("end\n");
}

int main(void)
{
    if (ocu_core_device(&B, &D) != OCU_OK) {
        fprintf(stderr, "driver: no device\n");
        return 3;
    }
    const char *qmode = getenv("OCU_DRIVER_QUEUE");
    if (qmode != NULL && strcmp(qmode, "1") == 0 && B->queue_create(D, &Q) != OCU_OK) {
        return 3;
    }
    kase k;
    while (read_case(&k)) {
        f32 = strcmp(k.kind, "f32") == 0;
        run(&k);
        for (int i = 0; i < k.nf; ++i) free(k.f[i].v);
    }
    if (Q != NULL) B->queue_destroy(Q);
    return 0;
}

/*
 * CPU backend: queues, markers, copies, fills and launches.
 *
 * Each queue owns one worker thread that runs its operations in order.
 * `queue == NULL` runs the operation on the caller's thread before
 * returning. Kernel arguments are copied when the launch is submitted.
 */
#define _POSIX_C_SOURCE 200809L

#include "cpu_internal.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

double cpu_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* ---- markers ------------------------------------------------------------ */

/*
 * Markers are reference counted: the handle holds one reference and every
 * queued record holds one, so destroying a marker with a record still in a
 * queue is safe.
 */
typedef struct marker_box {
    ocu_marker marker; /* first member: the handle points here */
    unsigned   refs;   /* guarded by marker.lock */
} marker_box;

static void marker_unref(ocu_marker *m)
{
    marker_box *box = (marker_box *)m;
    pthread_mutex_lock(&m->lock);
    unsigned left = --box->refs;
    pthread_mutex_unlock(&m->lock);
    if (left == 0) {
        pthread_cond_destroy(&m->done_cv);
        pthread_mutex_destroy(&m->lock);
        free(box);
    }
}

static void marker_complete(ocu_marker *m, uint64_t generation)
{
    pthread_mutex_lock(&m->lock);
    if (m->generation == generation) { /* a later record supersedes this one */
        m->state = MARKER_DONE;
        m->time_ms = cpu_now_ms();
        pthread_cond_broadcast(&m->done_cv);
    }
    pthread_mutex_unlock(&m->lock);
}

ocu_status cpu_marker_create(ocu_device *dev, ocu_marker **marker)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_DEVICE;
    }
    if (marker == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    marker_box *box = calloc(1, sizeof(*box));
    if (box == NULL) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    pthread_mutex_init(&box->marker.lock, NULL);
    pthread_cond_init(&box->marker.done_cv, NULL);
    box->marker.state = MARKER_NEVER;
    box->refs = 1;
    *marker = &box->marker;
    return OCU_OK;
}

ocu_status cpu_marker_destroy(ocu_marker *marker)
{
    if (marker == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    marker_unref(marker);
    return OCU_OK;
}

ocu_status cpu_marker_wait(ocu_marker *m)
{
    if (m == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    pthread_mutex_lock(&m->lock);
    while (m->state == MARKER_PENDING) {
        pthread_cond_wait(&m->done_cv, &m->lock);
    }
    pthread_mutex_unlock(&m->lock);
    return OCU_OK;
}

ocu_status cpu_marker_query(ocu_marker *m)
{
    if (m == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    pthread_mutex_lock(&m->lock);
    ocu_status st = m->state == MARKER_PENDING ? OCU_ERR_NOT_READY : OCU_OK;
    pthread_mutex_unlock(&m->lock);
    return st;
}

ocu_status cpu_marker_elapsed_ms(ocu_marker *start, ocu_marker *end,
                                 float *milliseconds)
{
    if (start == NULL || end == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    if (milliseconds == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    ocu_marker *both[2] = {start, end};
    double t[2] = {0, 0};
    for (int i = 0; i < 2; ++i) {
        pthread_mutex_lock(&both[i]->lock);
        marker_state s = both[i]->state;
        t[i] = both[i]->time_ms;
        pthread_mutex_unlock(&both[i]->lock);
        if (s == MARKER_NEVER) {
            return OCU_ERR_INVALID_HANDLE;
        }
        if (s == MARKER_PENDING) {
            return OCU_ERR_NOT_READY;
        }
    }
    *milliseconds = (float)(t[1] - t[0]);
    return OCU_OK;
}

/* ---- operations ----------------------------------------------------------- */

typedef enum op_kind {
    OP_COPY = 0,
    OP_SET,
    OP_LAUNCH,
    OP_MARKER
} op_kind;

struct cpu_op {
    cpu_op *next;
    op_kind kind;
    union {
        struct {
            void       *dst;
            const void *src;
            size_t      bytes;
        } copy;
        struct {
            void  *dst;
            int    value;
            size_t bytes;
        } set;
        struct {
            ocu_cpu_kernel_fn fn;
            uint32_t          grid[3];
            uint32_t          block[3];
            size_t            shared_bytes;
            void            **args;     /* points into `storage` */
            unsigned char    *storage;  /* copied argument values */
        } launch;
        struct {
            ocu_marker *marker;
            uint64_t    generation;
        } mark;
    } u;
};

static ocu_status op_run(cpu_op *op)
{
    switch (op->kind) {
    case OP_COPY:
        memmove(op->u.copy.dst, op->u.copy.src, op->u.copy.bytes);
        return OCU_OK;
    case OP_SET:
        memset(op->u.set.dst, op->u.set.value, op->u.set.bytes);
        return OCU_OK;
    case OP_LAUNCH:
        return cpu_run_grid(op->u.launch.fn, op->u.launch.grid,
                            op->u.launch.block, op->u.launch.shared_bytes,
                            op->u.launch.args);
    case OP_MARKER:
        marker_complete(op->u.mark.marker, op->u.mark.generation);
        return OCU_OK;
    }
    return OCU_ERR_UNKNOWN;
}

static void op_free(cpu_op *op)
{
    if (op->kind == OP_LAUNCH) {
        free(op->u.launch.storage);
    } else if (op->kind == OP_MARKER) {
        marker_unref(op->u.mark.marker);
    }
    free(op);
}

/* ---- queues --------------------------------------------------------------- */

static void *queue_worker(void *arg)
{
    ocu_queue *q = arg;
    pthread_mutex_lock(&q->lock);
    for (;;) {
        while (q->head == NULL && !q->stopping) {
            pthread_cond_wait(&q->work_cv, &q->lock);
        }
        if (q->head == NULL) { /* stopping and drained */
            break;
        }
        cpu_op *op = q->head;
        q->head = op->next;
        if (q->head == NULL) {
            q->tail = NULL;
        }
        q->running = true;
        pthread_mutex_unlock(&q->lock);

        ocu_status st = op_run(op);
        op_free(op);

        pthread_mutex_lock(&q->lock);
        q->running = false;
        if (st != OCU_OK && q->sticky == OCU_OK) {
            q->sticky = st;
        }
        if (q->head == NULL) {
            pthread_cond_broadcast(&q->idle_cv);
        }
    }
    pthread_mutex_unlock(&q->lock);
    return NULL;
}

/* Run `op` now when there is no queue, else hand it to the queue. */
static ocu_status submit(ocu_queue *q, cpu_op *op)
{
    if (q == NULL) {
        ocu_status st = op_run(op);
        op_free(op);
        return st;
    }
    op->next = NULL;
    pthread_mutex_lock(&q->lock);
    if (q->tail != NULL) {
        q->tail->next = op;
    } else {
        q->head = op;
    }
    q->tail = op;
    pthread_cond_signal(&q->work_cv);
    pthread_mutex_unlock(&q->lock);
    return OCU_OK;
}

ocu_status cpu_queue_create(ocu_device *dev, ocu_queue **queue)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_DEVICE;
    }
    if (queue == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    ocu_queue *q = calloc(1, sizeof(*q));
    if (q == NULL) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->work_cv, NULL);
    pthread_cond_init(&q->idle_cv, NULL);
    q->sticky = OCU_OK;
    if (pthread_create(&q->worker, NULL, queue_worker, q) != 0) {
        pthread_cond_destroy(&q->idle_cv);
        pthread_cond_destroy(&q->work_cv);
        pthread_mutex_destroy(&q->lock);
        free(q);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    *queue = q;
    return OCU_OK;
}

/* Finishes queued work first, then stops the worker. */
ocu_status cpu_queue_destroy(ocu_queue *q)
{
    if (q == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    pthread_mutex_lock(&q->lock);
    q->stopping = true;
    pthread_cond_signal(&q->work_cv);
    pthread_mutex_unlock(&q->lock);
    pthread_join(q->worker, NULL);
    pthread_cond_destroy(&q->idle_cv);
    pthread_cond_destroy(&q->work_cv);
    pthread_mutex_destroy(&q->lock);
    free(q);
    return OCU_OK;
}

ocu_status cpu_queue_sync(ocu_queue *q)
{
    if (q == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    pthread_mutex_lock(&q->lock);
    while (q->head != NULL || q->running) {
        pthread_cond_wait(&q->idle_cv, &q->lock);
    }
    ocu_status st = q->sticky;
    q->sticky = OCU_OK;
    pthread_mutex_unlock(&q->lock);
    return st;
}

ocu_status cpu_queue_query(ocu_queue *q)
{
    if (q == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    pthread_mutex_lock(&q->lock);
    ocu_status st;
    if (q->head != NULL || q->running) {
        st = OCU_ERR_NOT_READY;
    } else {
        st = q->sticky;
        q->sticky = OCU_OK;
    }
    pthread_mutex_unlock(&q->lock);
    return st;
}

/* ---- copies and fills ----------------------------------------------------- */

ocu_status cpu_mem_copy(ocu_device *dev, ocu_queue *queue, void *dst,
                        const void *src, size_t bytes, ocu_copy_dir dir)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_DEVICE;
    }
    if (bytes == 0) {
        return OCU_OK;
    }
    if (dst == NULL || src == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    ocu_status st;
    switch (dir) {
    case OCU_COPY_HOST_TO_DEVICE:
        st = cpu_check_device_range(dst, bytes);
        break;
    case OCU_COPY_DEVICE_TO_HOST:
        st = cpu_check_device_range(src, bytes);
        break;
    case OCU_COPY_DEVICE_TO_DEVICE:
        st = cpu_check_device_range(dst, bytes);
        if (st == OCU_OK) {
            st = cpu_check_device_range(src, bytes);
        }
        break;
    default:
        st = OCU_ERR_INVALID_VALUE;
        break;
    }
    if (st != OCU_OK) {
        return st;
    }
    cpu_op *op = calloc(1, sizeof(*op));
    if (op == NULL) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    op->kind = OP_COPY;
    op->u.copy.dst = dst;
    op->u.copy.src = src;
    op->u.copy.bytes = bytes;
    return submit(queue, op);
}

ocu_status cpu_mem_set(ocu_device *dev, ocu_queue *queue, void *dst,
                       int value, size_t bytes)
{
    if (dev == NULL) {
        return OCU_ERR_INVALID_DEVICE;
    }
    if (bytes == 0) {
        return OCU_OK;
    }
    ocu_status st = cpu_check_device_range(dst, bytes);
    if (st != OCU_OK) {
        return st;
    }
    cpu_op *op = calloc(1, sizeof(*op));
    if (op == NULL) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    op->kind = OP_SET;
    op->u.set.dst = dst;
    op->u.set.value = value;
    op->u.set.bytes = bytes;
    return submit(queue, op);
}

/* ---- launches ------------------------------------------------------------- */

#define ARG_ALIGN 16u

ocu_status cpu_kernel_launch(ocu_queue *queue, ocu_kernel *kernel,
                             const ocu_launch_params *p)
{
    if (kernel == NULL || kernel->fn == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    ocu_status st = cpu_check_launch(p);
    if (st != OCU_OK) {
        return st;
    }
    if (queue == NULL) {
        return cpu_run_grid(kernel->fn, p->grid, p->block,
                            p->shared_memory_bytes, p->args);
    }

    /* Copy the argument values: the caller may reuse its storage. */
    size_t total = 0;
    for (uint32_t i = 0; i < p->num_args; ++i) {
        size_t sz = p->arg_sizes[i];
        total += (sz + ARG_ALIGN - 1) & ~(size_t)(ARG_ALIGN - 1);
    }
    size_t table = (size_t)p->num_args * sizeof(void *);
    table = (table + ARG_ALIGN - 1) & ~(size_t)(ARG_ALIGN - 1);
    unsigned char *storage = aligned_alloc(ARG_ALIGN, table + total + ARG_ALIGN);
    if (storage == NULL) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    void **args = (void **)storage;
    unsigned char *cursor = storage + table;
    for (uint32_t i = 0; i < p->num_args; ++i) {
        size_t sz = p->arg_sizes[i];
        if (sz > 0) {
            memcpy(cursor, p->args[i], sz);
        }
        args[i] = cursor;
        cursor += (sz + ARG_ALIGN - 1) & ~(size_t)(ARG_ALIGN - 1);
    }

    cpu_op *op = calloc(1, sizeof(*op));
    if (op == NULL) {
        free(storage);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    op->kind = OP_LAUNCH;
    op->u.launch.fn = kernel->fn;
    memcpy(op->u.launch.grid, p->grid, sizeof(op->u.launch.grid));
    memcpy(op->u.launch.block, p->block, sizeof(op->u.launch.block));
    op->u.launch.shared_bytes = p->shared_memory_bytes;
    op->u.launch.args = args;
    op->u.launch.storage = storage;
    return submit(queue, op);
}

ocu_status cpu_marker_record(ocu_marker *m, ocu_queue *queue)
{
    if (m == NULL) {
        return OCU_ERR_INVALID_HANDLE;
    }
    pthread_mutex_lock(&m->lock);
    uint64_t gen = ++m->generation;
    m->state = MARKER_PENDING;
    if (queue != NULL) {
        ((marker_box *)m)->refs += 1; /* held by the queued op */
    }
    pthread_mutex_unlock(&m->lock);
    if (queue == NULL) {
        marker_complete(m, gen);
        return OCU_OK;
    }
    cpu_op *op = calloc(1, sizeof(*op));
    if (op == NULL) {
        marker_complete(m, gen); /* do not leave it pending forever */
        marker_unref(m);
        return OCU_ERR_OUT_OF_MEMORY;
    }
    op->kind = OP_MARKER;
    op->u.mark.marker = m;
    op->u.mark.generation = gen;
    return submit(queue, op);
}

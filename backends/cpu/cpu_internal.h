/*
 * Internal declarations shared by the CPU backend's source files.
 */
#ifndef OCU_CPU_INTERNAL_H
#define OCU_CPU_INTERNAL_H

#include "ocu_backend.h"
#include "ocu_cpu_kernel.h"

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---- configuration (read once from the environment) ------------------- */

typedef enum cpu_order {
    CPU_ORDER_FIXED = 0,  /* threads of a block run 0, 1, 2, ... */
    CPU_ORDER_RANDOM      /* shuffled at every round, seeded */
} cpu_order;

typedef struct cpu_config {
    uint32_t  warp_size;      /* OCU_CPU_WARP: 32 (default) or 64 */
    cpu_order order;          /* OCU_CPU_ORDER: fixed (default) or random */
    uint64_t  seed;           /* OCU_CPU_SEED: for random order, default 1 */
    uint32_t  workers;        /* OCU_CPU_THREADS: OS threads per launch */
    size_t    stack_bytes;    /* OCU_CPU_STACK_KB: fiber stack size */
    bool      strict_barrier; /* OCU_CPU_STRICT=1: a barrier with exited
                                 threads is an error */
    uint32_t  spin_us;        /* OCU_CPU_SPIN_US: how long idle pool threads
                                 watch for work before sleeping (50) */
    bool      stats;          /* OCU_CPU_STATS=1: count launches, fiber
                                 switches and time; printed at exit */
} cpu_config;

const cpu_config *cpu_get_config(void);

/* ---- limits reported by dev_info and enforced by kernel_launch -------- */

#define CPU_MAX_THREADS_PER_BLOCK 1024u
#define CPU_MAX_BLOCK_X           1024u
#define CPU_MAX_BLOCK_Y           1024u
#define CPU_MAX_BLOCK_Z           64u
#define CPU_MAX_GRID_X            2147483647u
#define CPU_MAX_GRID_Y            65535u
#define CPU_MAX_GRID_Z            65535u
#define CPU_MAX_SHARED_BYTES      (48u * 1024u)
#define CPU_ALLOC_ALIGN           256u
#define CPU_SHARED_ALIGN          64u

/* ---- objects ---------------------------------------------------------- */

struct ocu_device {
    int unused;
};

struct ocu_module {
    const ocu_cpu_image *image;
};

struct ocu_kernel {
    ocu_module       *module;
    ocu_cpu_kernel_fn fn;
};

typedef enum marker_state {
    MARKER_NEVER = 0,  /* never recorded: counts as complete */
    MARKER_PENDING,
    MARKER_DONE
} marker_state;

struct ocu_marker {
    pthread_mutex_t lock;
    pthread_cond_t  done_cv;
    marker_state    state;
    uint64_t        generation; /* bumped by every record */
    uint64_t        completed;  /* highest generation whose record ran */
    double          time_ms;    /* monotonic clock when it completed */
};

typedef struct cpu_op cpu_op;

struct ocu_queue {
    ocu_queue      *next_live;  /* list of live queues, under g_queues_lock */
    pthread_mutex_t lock;
    pthread_cond_t  work_cv;   /* worker waits for ops */
    pthread_cond_t  idle_cv;   /* syncers wait for an empty, idle queue */
    pthread_t       worker;
    cpu_op         *head;
    cpu_op         *tail;
    bool            running;   /* worker is executing an op */
    bool            stopping;
    ocu_status      sticky;    /* first error from queued work */
    unsigned        generation; /* fork generation that created it */
};

/* ---- memory (cpu_memory.c) -------------------------------------------- */

ocu_status cpu_mem_alloc(ocu_device *dev, size_t bytes, void **ptr);
ocu_status cpu_mem_free(ocu_device *dev, void *ptr);
ocu_status cpu_mem_info(ocu_device *dev, uint64_t *free_bytes,
                        uint64_t *total_bytes);
ocu_status cpu_host_alloc_pinned(ocu_device *dev, size_t bytes, void **ptr);
ocu_status cpu_host_free_pinned(ocu_device *dev, void *ptr);
/* OCU_OK when [ptr, ptr + bytes) lies inside one device allocation. */
ocu_status cpu_check_device_range(const void *ptr, size_t bytes);
uint64_t   cpu_total_memory(void);

/* ---- execution (cpu_exec.c) ------------------------------------------- */

/* Validate launch dimensions against the limits above. */
ocu_status cpu_check_launch(const ocu_launch_params *params);
/* Run a whole grid now, on the calling thread plus worker threads. */
ocu_status cpu_run_grid(ocu_cpu_kernel_fn fn, const uint32_t grid[3],
                        const uint32_t block[3], size_t shared_bytes,
                        void **args);

/* ---- queues and markers (cpu_queue.c) --------------------------------- */

ocu_status cpu_queue_create(ocu_device *dev, ocu_queue **queue);
ocu_status cpu_queue_destroy(ocu_queue *queue);
ocu_status cpu_queue_sync(ocu_queue *queue);
ocu_status cpu_queue_query(ocu_queue *queue);
ocu_status cpu_mem_copy(ocu_device *dev, ocu_queue *queue, void *dst,
                        const void *src, size_t bytes, ocu_copy_dir dir);
ocu_status cpu_mem_set(ocu_device *dev, ocu_queue *queue, void *dst,
                       int value, size_t bytes);
ocu_status cpu_kernel_launch(ocu_queue *queue, ocu_kernel *kernel,
                             const ocu_launch_params *params);
ocu_status cpu_marker_create(ocu_device *dev, ocu_marker **marker);
ocu_status cpu_marker_destroy(ocu_marker *marker);
ocu_status cpu_marker_record(ocu_marker *marker, ocu_queue *queue);
ocu_status cpu_marker_wait(ocu_marker *marker);
ocu_status cpu_marker_query(ocu_marker *marker);
ocu_status cpu_marker_elapsed_ms(ocu_marker *start, ocu_marker *end,
                                 float *milliseconds);

double cpu_now_ms(void);

/* OCU_CPU_STATS: time spent in a queued copy or fill (cpu_exec.c). */
void cpu_stats_copy(double ms, size_t bytes);

/* Library providers (cpu_blas.c): "blas", "ops". NULL for anything else. */
const void *cpu_get_provider(const char *library);

/* Operator table (cpu_ops.c). */
struct ocu_op_table;
const struct ocu_op_table *cpu_op_table(void);

/* Wait until every live queue is empty and idle (errors stay sticky). */
void cpu_queues_drain(void);

/* True while the calling thread is running a kernel body. */
bool cpu_in_kernel(void);

#endif /* OCU_CPU_INTERNAL_H */

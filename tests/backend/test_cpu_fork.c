/*
 * fork() while another thread keeps launching (third review): every child
 * must be able to launch, with a working pool, and must not hang on a lock
 * that was held at the moment of fork.
 */
#define _POSIX_C_SOURCE 200809L

#include "ocu_backend.h"
#include "ocu_backend_cpu.h"
#include "ocu_cpu_kernel.h"

#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHILDREN 40

static const ocu_backend *B;
static ocu_device *D;
static ocu_kernel *K;
static _Atomic int g_stop;

static void k_fill(void **a)
{
    uint32_t *out = *(uint32_t **)a[0];
    uint32_t i = ocu_cpu_block_idx().x * ocu_cpu_block_dim().x + ocu_cpu_thread_idx().x;
    out[i] = i * 3u + 1u;
}

static const ocu_cpu_kernel_entry g_entries[] = {{"fill", k_fill}};
static const ocu_cpu_image g_image = {OCU_CPU_IMAGE_MAGIC, OCU_CPU_IMAGE_VERSION, 1, g_entries};

static int launch_and_check(uint32_t blocks)
{
    uint32_t n = blocks * 64u;
    uint32_t *out = NULL;
    if (B->mem_alloc(D, n * sizeof(uint32_t), (void **)&out) != OCU_OK) return 1;
    void *args[] = {&out};
    size_t sizes[] = {sizeof(out)};
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = blocks; p.grid[1] = p.grid[2] = 1;
    p.block[0] = 64; p.block[1] = p.block[2] = 1;
    p.args = args; p.arg_sizes = sizes; p.num_args = 1;
    int bad = B->kernel_launch(NULL, K, &p) != OCU_OK;
    for (uint32_t i = 0; i < n && !bad; ++i) bad += out[i] != i * 3u + 1u;
    B->mem_free(D, out);
    return bad;
}

static void *busy(void *unused)
{
    (void)unused;
    while (!atomic_load(&g_stop)) {
        if (launch_and_check(32)) atomic_store(&g_stop, 2);
    }
    return NULL;
}

static ocu_queue *g_parent_queue;

/* Keeps a queue busy and frees memory, so a fork can land while a queue
   works and while a free waits for it (fourth review). */
static void *queue_busy(void *unused)
{
    (void)unused;
    while (!atomic_load(&g_stop)) {
        uint32_t *buf = NULL;
        if (B->mem_alloc(D, 64 * 64 * sizeof(uint32_t), (void **)&buf) != OCU_OK) continue;
        void *args[] = {&buf};
        size_t sizes[] = {sizeof(buf)};
        ocu_launch_params p;
        memset(&p, 0, sizeof(p));
        p.grid[0] = 64; p.grid[1] = p.grid[2] = 1;
        p.block[0] = 64; p.block[1] = p.block[2] = 1;
        p.args = args; p.arg_sizes = sizes; p.num_args = 1;
        B->kernel_launch(g_parent_queue, K, &p);
        B->mem_free(D, buf); /* waits for the queue */
    }
    return NULL;
}

int main(void)
{
#if defined(__SANITIZE_THREAD__)
#define OCU_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define OCU_TEST_TSAN 1
#endif
#endif
#ifdef OCU_TEST_TSAN
    /* ThreadSanitizer cannot start threads after a multi-threaded fork. */
    printf("test_cpu_fork: skipped under ThreadSanitizer\n");
    return 0;
#endif
    B = ocu_backend_cpu();
    ocu_module *M = NULL;
    if (B->dev_open(0, &D) != OCU_OK ||
        B->module_load(D, &g_image, sizeof(g_image), &M) != OCU_OK ||
        B->module_get_kernel(M, "fill", &K) != OCU_OK) {
        return 1;
    }
    if (launch_and_check(64)) return 1; /* start the pool in the parent */
    if (B->queue_create(D, &g_parent_queue) != OCU_OK) return 1;
    pthread_t t, tq;
    pthread_create(&t, NULL, busy, NULL);
    pthread_create(&tq, NULL, queue_busy, NULL);
    int hung = 0, failed = 0;
    for (int c = 0; c < CHILDREN; ++c) {
        pid_t pid = fork();
        if (pid == 0) {
            alarm(20); /* a hang ends the child with SIGALRM */
            int bad = launch_and_check(64) + launch_and_check(7);
            /* Inherited queues are refused, new ones work. */
            bad += B->queue_sync(g_parent_queue) != OCU_ERR_NOT_SUPPORTED;
            ocu_queue *mine = NULL;
            bad += B->queue_create(D, &mine) != OCU_OK;
            bad += mine == NULL || B->queue_sync(mine) != OCU_OK;
            if (mine != NULL) B->queue_destroy(mine);
            _exit(bad ? 3 : 0);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFSIGNALED(status) && WTERMSIG(status) == SIGALRM) {
            ++hung;
        } else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            ++failed;
        }
    }
    int busy_bad = atomic_exchange(&g_stop, 1) == 2;
    pthread_join(t, NULL);
    pthread_join(tq, NULL);
    B->queue_destroy(g_parent_queue);
    printf("test_cpu_fork: %s (%d children: %d hung, %d failed; parent %s)\n",
           hung || failed || busy_bad ? "FAIL" : "ok", CHILDREN, hung, failed,
           busy_bad ? "wrong results" : "fine");
    return hung || failed || busy_bad ? 1 : 0;
}

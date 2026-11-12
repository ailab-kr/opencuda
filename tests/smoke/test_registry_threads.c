/*
 * Concurrent first use of the backend registry: many threads ask for the
 * active backend and the core device at the same time. Run under TSan by
 * tools/check.sh.
 */
#include "ocu_core.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define THREADS 32

static const ocu_backend *seen_backend[THREADS];
static ocu_device *seen_device[THREADS];
static size_t seen_count[THREADS];

static void *worker(void *arg)
{
    long i = (long)arg;
    seen_backend[i] = ocu_active_backend();
    seen_count[i] = ocu_backend_count();
    const ocu_backend *b = NULL;
    if (ocu_core_device(&b, &seen_device[i]) != OCU_OK) {
        seen_device[i] = NULL;
    }
    return NULL;
}

int main(void)
{
    pthread_t t[THREADS];
    for (long i = 0; i < THREADS; ++i) {
        pthread_create(&t[i], NULL, worker, (void *)i);
    }
    for (int i = 0; i < THREADS; ++i) {
        pthread_join(t[i], NULL);
    }
    int bad = 0;
    for (int i = 0; i < THREADS; ++i) {
        bad += seen_backend[i] == NULL || seen_backend[i] != seen_backend[0];
        bad += seen_device[i] == NULL || seen_device[i] != seen_device[0];
        bad += seen_count[i] != seen_count[0] || seen_count[i] < 1;
    }
    bad += strcmp(seen_backend[0]->name, "cpu") != 0;
    printf("test_registry_threads: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}

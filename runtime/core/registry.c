/*
 * Backend registry: the one place that knows which backends exist in this
 * build and which one is active. Nothing above this file names a chip.
 */
#include "ocu_core.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifdef OCU_HAVE_BACKEND_CPU
#include "ocu_backend_cpu.h"
#endif
#ifdef OCU_HAVE_BACKEND_NVIDIA
#include "ocu_backend_nvidia.h"
#endif
#ifdef OCU_HAVE_BACKEND_OPENCL
#include "ocu_backend_opencl.h"
#endif
#ifdef OCU_HAVE_BACKEND_METAL
#include "ocu_backend_metal.h"
#endif
#ifdef OCU_HAVE_BACKEND_NPUSIM
#include "ocu_backend_npusim.h"
#endif

#define OCU_MAX_BACKENDS 8

/*
 * All state below is guarded by g_lock. The built-in backends are
 * registered exactly once (pthread_once), so first use from several
 * threads at the same time is safe.
 */
static pthread_mutex_t    g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t     g_builtin_once = PTHREAD_ONCE_INIT;
static const ocu_backend *g_backends[OCU_MAX_BACKENDS];
static size_t             g_backend_count;
static const ocu_backend *g_active;
static int                g_active_chosen; /* the first-use choice was made */
static char               g_preferred[32]; /* default choice when OCU_BACKEND is unset */

/* Every entry except get_provider and the ABI 2 optional ones is mandatory. */
static int table_is_complete(const ocu_backend *b)
{
    return b->dev_count && b->dev_info && b->dev_open && b->dev_close &&
           b->mem_alloc && b->mem_free && b->mem_copy && b->mem_set &&
           b->mem_info && b->host_alloc_pinned && b->host_free_pinned &&
           b->queue_create && b->queue_destroy && b->queue_sync &&
           b->queue_query && b->module_load && b->module_unload &&
           b->module_get_kernel && b->kernel_launch && b->marker_create &&
           b->marker_destroy && b->marker_record && b->marker_wait &&
           b->marker_query && b->marker_elapsed_ms;
}

/* Caller holds g_lock. */
static const ocu_backend *find_backend(const char *name)
{
    for (size_t i = 0; i < g_backend_count; ++i) {
        if (strcmp(g_backends[i]->name, name) == 0) {
            return g_backends[i];
        }
    }
    return NULL;
}

/* Caller holds g_lock. */
static ocu_status register_locked(const ocu_backend *backend)
{
    if (backend == NULL || backend->name == NULL || backend->name[0] == '\0') {
        return OCU_ERR_INVALID_VALUE;
    }
    if (backend->abi != OCU_BACKEND_ABI) {
        return OCU_ERR_NOT_SUPPORTED;
    }
    if (!table_is_complete(backend)) {
        return OCU_ERR_INVALID_VALUE;
    }
    if (find_backend(backend->name) != NULL) {
        return OCU_ERR_INVALID_VALUE; /* duplicate name */
    }
    if (g_backend_count == OCU_MAX_BACKENDS) {
        return OCU_ERR_OUT_OF_MEMORY;
    }
    g_backends[g_backend_count++] = backend;
    return OCU_OK;
}

static void register_builtin_once(void)
{
    pthread_mutex_lock(&g_lock);
#ifdef OCU_HAVE_BACKEND_CPU
    (void)register_locked(ocu_backend_cpu());
#endif
#ifdef OCU_HAVE_BACKEND_NVIDIA
    (void)register_locked(ocu_backend_nvidia());
#endif
#ifdef OCU_HAVE_BACKEND_METAL
    (void)register_locked(ocu_backend_metal());
#endif
#ifdef OCU_HAVE_BACKEND_OPENCL
    (void)register_locked(ocu_backend_opencl());
#endif
#ifdef OCU_HAVE_BACKEND_NPUSIM
    /* Last: a simulator is never the default choice. */
    (void)register_locked(ocu_backend_npusim());
#endif
    pthread_mutex_unlock(&g_lock);
}

static void register_builtin_backends(void)
{
    pthread_once(&g_builtin_once, register_builtin_once);
}

ocu_status ocu_register_backend(const ocu_backend *backend)
{
    register_builtin_backends();
    pthread_mutex_lock(&g_lock);
    ocu_status st = register_locked(backend);
    pthread_mutex_unlock(&g_lock);
    return st;
}

ocu_status ocu_select_backend(const char *name)
{
    register_builtin_backends();
    if (name == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    const ocu_backend *b = find_backend(name);
    if (b != NULL) {
        g_active = b;
        g_active_chosen = 1;
    }
    pthread_mutex_unlock(&g_lock);
    return b != NULL ? OCU_OK : OCU_ERR_INVALID_VALUE;
}

const ocu_backend *ocu_active_backend(void)
{
    register_builtin_backends();
    pthread_mutex_lock(&g_lock);
    if (!g_active_chosen) {
        /* First use: OCU_BACKEND picks one; otherwise the first registered. */
        const char *want = getenv("OCU_BACKEND");
        if (want != NULL && want[0] != '\0') {
            g_active = find_backend(want); /* NULL when unknown: caller reports */
        } else if (g_preferred[0] != '\0' && find_backend(g_preferred) != NULL) {
            g_active = find_backend(g_preferred);
        } else if (g_backend_count > 0) {
            g_active = g_backends[0];
        }
        g_active_chosen = 1;
    }
    const ocu_backend *b = g_active;
    pthread_mutex_unlock(&g_lock);
    return b;
}

ocu_status ocu_prefer_backend(const char *name)
{
    register_builtin_backends();
    if (name == NULL || strlen(name) >= sizeof(g_preferred)) {
        return OCU_ERR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    ocu_status st = OCU_OK;
    if (find_backend(name) == NULL) {
        st = OCU_ERR_NOT_FOUND;
    } else if (!g_active_chosen) {
        memcpy(g_preferred, name, strlen(name) + 1);
    }
    pthread_mutex_unlock(&g_lock);
    return st;
}

ocu_status ocu_core_device(const ocu_backend **backend, ocu_device **device)
{
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    static ocu_device *dev;
    static const ocu_backend *dev_backend;
    ocu_status st = OCU_OK;
    pthread_mutex_lock(&lock);
    /* Read the active backend inside the lock so that the device and the
       backend handed out always belong together. */
    const ocu_backend *b = ocu_active_backend();
    if (b == NULL) {
        pthread_mutex_unlock(&lock);
        return OCU_ERR_INVALID_DEVICE;
    }
    if (dev == NULL || dev_backend != b) {
        /* The previous device is not closed: other threads may still use
           the handle they got. Switching backends after first use is rare;
           the cost is one open device per switch. */
        dev = NULL;
        st = b->dev_open(0, &dev);
        dev_backend = st == OCU_OK ? b : NULL;
    }
    ocu_device *d = dev;
    pthread_mutex_unlock(&lock);
    if (st == OCU_OK) {
        if (backend != NULL) *backend = b;
        if (device != NULL) *device = d;
    }
    return st;
}

size_t ocu_backend_count(void)
{
    register_builtin_backends();
    pthread_mutex_lock(&g_lock);
    size_t n = g_backend_count;
    pthread_mutex_unlock(&g_lock);
    return n;
}

const char *ocu_backend_name_at(size_t index)
{
    register_builtin_backends();
    pthread_mutex_lock(&g_lock);
    const char *name = index < g_backend_count ? g_backends[index]->name : NULL;
    pthread_mutex_unlock(&g_lock);
    return name;
}

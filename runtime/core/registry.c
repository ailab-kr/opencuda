/*
 * Backend registry: the one place that knows which backends exist in this
 * build and which one is active. Nothing above this file names a chip.
 */
#include "ocu_core.h"

#include <stdlib.h>
#include <string.h>

#ifdef OCU_HAVE_BACKEND_CPU
#include "ocu_backend_cpu.h"
#endif

#define OCU_MAX_BACKENDS 8

static const ocu_backend *g_backends[OCU_MAX_BACKENDS];
static size_t             g_backend_count;
static const ocu_backend *g_active;
static int                g_builtin_done;

/* Every entry except get_provider is mandatory. */
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

static const ocu_backend *find_backend(const char *name)
{
    for (size_t i = 0; i < g_backend_count; ++i) {
        if (strcmp(g_backends[i]->name, name) == 0) {
            return g_backends[i];
        }
    }
    return NULL;
}

ocu_status ocu_register_backend(const ocu_backend *backend)
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

static void register_builtin_backends(void)
{
    if (g_builtin_done) {
        return;
    }
    g_builtin_done = 1;
#ifdef OCU_HAVE_BACKEND_CPU
    (void)ocu_register_backend(ocu_backend_cpu());
#endif
}

ocu_status ocu_select_backend(const char *name)
{
    register_builtin_backends();
    if (name == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    const ocu_backend *b = find_backend(name);
    if (b == NULL) {
        return OCU_ERR_INVALID_VALUE;
    }
    g_active = b;
    return OCU_OK;
}

const ocu_backend *ocu_active_backend(void)
{
    register_builtin_backends();
    if (g_active != NULL) {
        return g_active;
    }
    /* First use: OCU_BACKEND picks one; otherwise the first registered. */
    const char *want = getenv("OCU_BACKEND");
    if (want != NULL && want[0] != '\0') {
        g_active = find_backend(want); /* NULL when unknown: caller reports */
        return g_active;
    }
    if (g_backend_count > 0) {
        g_active = g_backends[0];
    }
    return g_active;
}

size_t ocu_backend_count(void)
{
    register_builtin_backends();
    return g_backend_count;
}

const char *ocu_backend_name_at(size_t index)
{
    register_builtin_backends();
    if (index >= g_backend_count) {
        return NULL;
    }
    return g_backends[index]->name;
}

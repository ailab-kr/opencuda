/*
 * fatbin.c -- the registration and launch entry points a CUDA compiler's
 * host code calls (the ABI of the open-source Clang CUDA front end):
 *
 *   __cudaRegisterFatBinary(wrapper)      once per translation unit
 *   __cudaRegisterFunction(handle, stub, name, ...)   per kernel
 *   __cudaRegisterVar(handle, host_var, name, ...)    per device variable
 *   __cudaPushCallConfiguration / __cudaPopCallConfiguration   per <<<>>>
 *
 * The wrapper points at a device-code container written by the compiler
 * glue (compiler/probe/ocu_cc.py): images for one or more backends, chosen
 * here by the active backend's name. The container is ours; this file
 * never looks inside an image, so it stays chip-neutral.
 *
 * Container (little-endian):
 *   char     magic[8]  = "OCUFAT1\0"
 *   uint32_t count, header_bytes
 *   entries[count]: char backend[16]; uint64_t offset, size  (from the start)
 */
#include "ocu_core.h"
#include "ocu_rt.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif

#define WRAPPER_MAGIC 0x466243b1u /* what the compiler stores in the wrapper */

typedef struct fat_wrapper {
    uint32_t    magic, version;
    const void *data;
    const void *unused;
} fat_wrapper;

typedef struct fat_entry {
    char     backend[16];
    uint64_t offset, size;
} fat_entry;

typedef struct image {
    const unsigned char *base;
    uint32_t             count;
    const fat_entry     *entries;
    ocu_module          *module;      /* loaded on first use */
    const ocu_backend   *module_backend;
    ocu_status           load_status; /* OCU_OK until a load failed */
    bool                 loading;     /* a thread is in module_load for it (g_lock released) */
    struct image        *next;
} image;

typedef struct function {
    const void      *stub;
    image           *img;
    char            *name;
    ocu_kernel      *kernel;          /* looked up on first launch */
    uint32_t         nargs;
    size_t          *arg_sizes;
    struct function *next;
} function;

typedef struct variable {
    const void      *host;
    image           *img;
    char            *name;
    size_t           size;
    struct variable *next;
} variable;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static image          *g_images;
static function       *g_functions;
static variable       *g_variables;

/* Kernels by launch stub: llama.cpp-sized programs register thousands, and
   every launch looks one up, so a hash table (open addressing) rather than
   a walk of the list. Guarded by g_lock. */
static function **g_fn_tab;
static size_t     g_fn_cap, g_fn_n;
static size_t fn_slot(const void *stub, size_t cap)
{
    uintptr_t h = (uintptr_t)stub;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return (size_t)h & (cap - 1);
}
static int fn_tab_put(function *f)
{
    if ((g_fn_n + 1) * 2 > g_fn_cap) {
        size_t cap = g_fn_cap ? g_fn_cap * 2 : 1024;
        function **t = calloc(cap, sizeof(*t));
        if (t == NULL) return 0;
        for (size_t i = 0; i < g_fn_cap; ++i) {
            if (g_fn_tab[i] == NULL) continue;
            size_t j = fn_slot(g_fn_tab[i]->stub, cap);
            while (t[j] != NULL) j = (j + 1) & (cap - 1);
            t[j] = g_fn_tab[i];
        }
        free(g_fn_tab);
        g_fn_tab = t;
        g_fn_cap = cap;
    }
    size_t j = fn_slot(f->stub, g_fn_cap);
    /* The newest registration of a stub wins, as the list's head did. */
    while (g_fn_tab[j] != NULL && g_fn_tab[j]->stub != f->stub) j = (j + 1) & (g_fn_cap - 1);
    if (g_fn_tab[j] == NULL) g_fn_n++;
    g_fn_tab[j] = f;
    return 1;
}
static function *fn_tab_get(const void *stub)
{
    if (g_fn_cap == 0) return NULL;
    size_t j = fn_slot(stub, g_fn_cap);
    while (g_fn_tab[j] != NULL) {
        if (g_fn_tab[j]->stub == stub) return g_fn_tab[j];
        j = (j + 1) & (g_fn_cap - 1);
    }
    return NULL;
}

static char *dup_string(const char *s)
{
    size_t n = strlen(s) + 1;
    char *c = malloc(n);
    if (c != NULL) memcpy(c, s, n);
    return c;
}

/* ---- registration (runs before main) ------------------------------------- */

void **__cudaRegisterFatBinary(void *wrapper)
{
    const fat_wrapper *w = wrapper;
    image *img = calloc(1, sizeof(*img));
    if (img == NULL) return NULL;
    if (w != NULL && w->magic == WRAPPER_MAGIC && w->data != NULL) {
        const unsigned char *b = w->data;
        if (memcmp(b, "OCUFAT1", 8) == 0) {
            uint32_t count;
            memcpy(&count, b + 8, sizeof(count));
            img->base = b;
            img->count = count;
            img->entries = (const fat_entry *)(b + 16);
            /* The program's device code says which backend it is for. */
            for (uint32_t i = 0; i < count; ++i) {
                char name[17];
                memcpy(name, img->entries[i].backend, 16);
                name[16] = '\0';
                if (ocu_prefer_backend(name) == OCU_OK) break;
            }
        }
    }
    pthread_mutex_lock(&g_lock);
    img->next = g_images;
    g_images = img;
    pthread_mutex_unlock(&g_lock);
    return (void **)img;
}

void __cudaRegisterFatBinaryEnd(void **handle) { (void)handle; }

void __cudaUnregisterFatBinary(void **handle)
{
    /* At exit: modules go away with the process (the device may already be
       closing), so nothing is unloaded here. */
    (void)handle;
}

int __cudaRegisterFunction(void **handle, const char *host_fun, char *device_fun,
                           const char *device_name, int thread_limit, void *tid, void *bid,
                           void *bdim, void *gdim, int *wsize)
{
    (void)device_fun;
    (void)thread_limit;
    (void)tid;
    (void)bid;
    (void)bdim;
    (void)gdim;
    (void)wsize;
    if (handle == NULL || host_fun == NULL || device_name == NULL) return 1;
    function *f = calloc(1, sizeof(*f));
    char *name = dup_string(device_name);
    if (f == NULL || name == NULL) {
        free(f);
        free(name);
        return 1;
    }
    f->stub = host_fun;
    f->img = (image *)handle;
    f->name = name;
    pthread_mutex_lock(&g_lock);
    f->next = g_functions;
    g_functions = f;
    if (!fn_tab_put(f)) {
        g_functions = f->next;
        pthread_mutex_unlock(&g_lock);
        free(name);
        free(f);
        return 1;
    }
    pthread_mutex_unlock(&g_lock);
    return 0;
}

void __cudaRegisterVar(void **handle, char *host_var, char *device_address,
                       const char *device_name, int ext, size_t size, int constant, int global)
{
    (void)device_address;
    (void)ext;
    (void)constant;
    (void)global;
    if (handle == NULL || host_var == NULL || device_name == NULL) return;
    variable *v = calloc(1, sizeof(*v));
    char *name = dup_string(device_name);
    if (v == NULL || name == NULL) {
        free(v);
        free(name);
        return;
    }
    v->host = host_var;
    v->img = (image *)handle;
    v->name = name;
    v->size = size;
    pthread_mutex_lock(&g_lock);
    v->next = g_variables;
    g_variables = v;
    pthread_mutex_unlock(&g_lock);
}

/* ---- call configuration (per thread, nested launches are rare) ------------- */

#define CONFIG_DEPTH 16

typedef struct config {
    dim3         grid, block;
    size_t       shared;
    cudaStream_t stream;
} config;

static _Thread_local config tl_configs[CONFIG_DEPTH];
static _Thread_local int    tl_depth;

unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shared, void *stream)
{
    if (tl_depth >= CONFIG_DEPTH) return 1;
    config *c = &tl_configs[tl_depth++];
    c->grid = grid;
    c->block = block;
    c->shared = shared;
    c->stream = (cudaStream_t)stream;
    return 0;
}

int cudaConfigureCall(dim3 grid, dim3 block, size_t shared, cudaStream_t stream)
{
    return __cudaPushCallConfiguration(grid, block, shared, stream) == 0 ? 0 : 1;
}

int __cudaPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shared, void *stream)
{
    if (tl_depth <= 0) return 1;
    config *c = &tl_configs[--tl_depth];
    if (grid) *grid = c->grid;
    if (block) *block = c->block;
    if (shared) *shared = c->shared;
    if (stream) *(cudaStream_t *)stream = c->stream;
    return 0;
}

/* ---- lookups --------------------------------------------------------------- */

/* ---- module loading, and building the other modules ahead ------------------

   A module is loaded (for the OpenCL backend: compiled by the driver) when a
   kernel of its image is first needed. The first time that happens, the
   images not loaded yet are loaded by background threads, so that a program
   with many device-code files (llama.cpp: tens of modules, minutes of driver
   compilation the first time) compiles them side by side instead of one
   after the other as its kernels come up (OpenCL backend). OCU_PREBUILD_THREADS=0
   turns that off; =N uses N threads (default: half the processors, at most 8). Loading
   happens with g_lock released; a thread that needs an image another thread
   is loading waits for it. At exit no new load starts and the ones under way
   are waited for (the driver must not be torn down under them). */

static pthread_cond_t g_load_cond = PTHREAD_COND_INITIALIZER;
static bool           g_prebuild_started, g_prebuild_stop;
static int            g_prebuild_active;

static const fat_entry *image_entry(const image *img, const ocu_backend *b)
{
    for (uint32_t i = 0; i < img->count; ++i)
        if (strncmp(img->entries[i].backend, b->name, sizeof(img->entries[i].backend)) == 0) return &img->entries[i];
    return NULL;
}

/* Caller holds g_lock; img is not loaded, not failed, not being loaded. The
   lock is released during the backend's load. */
static void load_unlocked(image *img, const ocu_backend *b, ocu_device *dev, const fat_entry *e)
{
    img->loading = true;
    pthread_mutex_unlock(&g_lock);
    ocu_module *m = NULL;
    ocu_status st = b->module_load(dev, img->base + e->offset, (size_t)e->size, &m);
    pthread_mutex_lock(&g_lock);
    img->loading = false;
    if (st == OCU_OK) {
        img->module = m;
        img->module_backend = b;
    } else {
        img->load_status = st;
    }
    pthread_cond_broadcast(&g_load_cond);
}

static void *prebuild_worker(void *arg)
{
    (void)arg;
    const ocu_backend *b = ocu_rt_backend();
    ocu_device *dev = ocu_rt_device();
    pthread_mutex_lock(&g_lock);
    while (!g_prebuild_stop) {
        image *img = g_images;
        const fat_entry *e = NULL;
        for (; img != NULL; img = img->next)
            if (img->module == NULL && !img->loading && img->load_status == OCU_OK && (e = image_entry(img, b)) != NULL) break;
        if (img == NULL) break;
        load_unlocked(img, b, dev, e);
    }
    g_prebuild_active--;
    pthread_cond_broadcast(&g_load_cond);
    pthread_mutex_unlock(&g_lock);
    return NULL;
}

static void prebuild_at_exit(void)
{
    pthread_mutex_lock(&g_lock);
    g_prebuild_stop = true;
    while (g_prebuild_active > 0) pthread_cond_wait(&g_load_cond, &g_lock);
    pthread_mutex_unlock(&g_lock);
}

/* Caller holds g_lock: start the background loads once, if other images wait. */
static void prebuild_start(const ocu_backend *b)
{
    if (g_prebuild_started) return;
    g_prebuild_started = true;
    /* the OpenCL backend only: there a load is a driver compilation, and its
       module_load is safe to call from several threads */
    if (strcmp(b->name, "opencl") != 0) return;
    const char *env = getenv("OCU_PREBUILD_THREADS");
    long n = env != NULL && *env ? strtol(env, NULL, 10) : -1;
    if (n == 0) return;
    int waiting = 0;
    for (image *img = g_images; img != NULL; img = img->next)
        if (img->module == NULL && !img->loading && img->load_status == OCU_OK && image_entry(img, b) != NULL) waiting++;
    if (waiting == 0) return;
    if (n < 0) {
#if defined(_SC_NPROCESSORS_ONLN)
        long cpus = sysconf(_SC_NPROCESSORS_ONLN);
#else
        long cpus = 8;
#endif
        n = cpus > 1 ? cpus / 2 : 1;
        if (n > 8) n = 8;
    }
    if (n > waiting) n = waiting;
    atexit(prebuild_at_exit);
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    for (long i = 0; i < n; ++i) {
        pthread_t t;
        if (pthread_create(&t, &at, prebuild_worker, NULL) == 0) g_prebuild_active++;
    }
    pthread_attr_destroy(&at);
}

/* Caller holds g_lock (released while loading). Loads the image's entry for the active backend. */
static ocu_status image_module(image *img, ocu_module **out)
{
    const ocu_backend *b = ocu_rt_backend();
    ocu_device *dev = ocu_rt_device();
    if (b == NULL || dev == NULL) return OCU_ERR_INVALID_DEVICE;
    while (img->loading) pthread_cond_wait(&g_load_cond, &g_lock);
    if (img->module != NULL && img->module_backend == b) {
        *out = img->module;
        return OCU_OK;
    }
    if (img->load_status != OCU_OK) return img->load_status;
    const fat_entry *e = image_entry(img, b);
    if (e == NULL) return OCU_ERR_NOT_FOUND; /* no device code for this backend */
    load_unlocked(img, b, dev, e);
    if (img->load_status != OCU_OK) return img->load_status;
    prebuild_start(b);
    *out = img->module;
    return OCU_OK;
}

ocu_status ocu_rt_find_kernel(const void *stub, ocu_kernel **kernel, uint32_t *nargs,
                              const size_t **arg_sizes)
{
    pthread_mutex_lock(&g_lock);
    function *f = fn_tab_get(stub);
    if (f == NULL) {
        pthread_mutex_unlock(&g_lock);
        return OCU_ERR_NOT_FOUND;
    }
    ocu_status st = OCU_OK;
    if (f->kernel == NULL) {
        ocu_module *m = NULL;
        st = image_module(f->img, &m);
        const ocu_backend *b = ocu_rt_backend();
        /* image_module may have released g_lock: another thread may have
           looked the kernel up meanwhile */
        bool done = st == OCU_OK && f->kernel != NULL;
        if (st == OCU_OK && !done) st = b->module_get_kernel(m, f->name, &f->kernel);
        if (st == OCU_OK && !done) {
            if (b->kernel_arg_sizes == NULL) {
                st = OCU_ERR_NOT_SUPPORTED;
            } else {
                uint32_t n = 0;
                st = b->kernel_arg_sizes(f->kernel, &n, NULL, 0);
                size_t *sizes = n ? calloc(n, sizeof(size_t)) : NULL;
                if (st == OCU_OK && n > 0 && sizes == NULL) st = OCU_ERR_OUT_OF_MEMORY;
                if (st == OCU_OK && n > 0) st = b->kernel_arg_sizes(f->kernel, &n, sizes, n);
                if (st == OCU_OK) {
                    f->nargs = n;
                    f->arg_sizes = sizes;
                } else {
                    free(sizes);
                }
            }
            if (st != OCU_OK) f->kernel = NULL;
        }
    }
    if (st == OCU_OK) {
        *kernel = f->kernel;
        *nargs = f->nargs;
        *arg_sizes = f->arg_sizes;
    }
    pthread_mutex_unlock(&g_lock);
    return st;
}

ocu_status ocu_rt_find_symbol(const void *host, void **device, size_t *bytes)
{
    pthread_mutex_lock(&g_lock);
    variable *v = g_variables;
    while (v != NULL && v->host != host) v = v->next;
    if (v == NULL) {
        pthread_mutex_unlock(&g_lock);
        return OCU_ERR_NOT_FOUND;
    }
    ocu_module *m = NULL;
    ocu_status st = image_module(v->img, &m);
    const ocu_backend *b = ocu_rt_backend();
    if (st == OCU_OK && b->module_get_global == NULL) st = OCU_ERR_NOT_SUPPORTED;
    if (st == OCU_OK) st = b->module_get_global(m, v->name, device, bytes);
    pthread_mutex_unlock(&g_lock);
    return st;
}

int ocu_rt_has_device_code(void)
{
    pthread_mutex_lock(&g_lock);
    int any = g_images != NULL;
    pthread_mutex_unlock(&g_lock);
    return any;
}

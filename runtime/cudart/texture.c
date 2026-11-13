/*
 * CUDA arrays, texture objects and texture references on top of the rest
 * of the runtime API. Chip-neutral: arrays are device allocations made with
 * cudaMalloc, copies go through cudaMemcpy / cudaMemcpy2D, and a texture
 * object is a descriptor (ocu_texture_desc.h) placed in device memory so
 * that device code samples it with ordinary loads (ocu_texture_fetch.h).
 *
 * Written from the public CUDA Runtime API reference (memory management,
 * texture object management, texture reference management, C++ API
 * routines) and the CUDA C++ Programming Guide ("Texture and Surface
 * Memory"). Choices where the documents leave room are marked "choice".
 */
#define _POSIX_C_SOURCE 200809L

#include "cudart_internal.h"
#include "ocu_texture_desc.h"

#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- formats ------------------------------------------------------------------ */

/* A texel format from a channel descriptor. raw_ok: kind None is accepted
   (texture references bound with the descriptor of a type that has no
   specialisation, read as the fetch type). */
typedef struct tex_format {
    unsigned channels, channel_bytes, kind, texel_bytes, always_norm;
} tex_format;

static bool format_of(const struct cudaChannelFormatDesc *d, bool raw_ok, tex_format *f)
{
    memset(f, 0, sizeof *f);
    if (d->f == cudaChannelFormatKindNone) {
        if (!raw_ok || d->x != 0 || d->y != 0 || d->z != 0 || d->w != 0) return false;
        f->kind = OCU_TEXK_RAW;
        return true;
    }
    /* Components: a prefix x, xy or xyzw of equal width. */
    int bits[4] = {d->x, d->y, d->z, d->w};
    unsigned n = 0;
    while (n < 4 && bits[n] != 0) ++n;
    for (unsigned i = n; i < 4; ++i) {
        if (bits[i] != 0) return false;
    }
    if (n != 1 && n != 2 && n != 4) return false;
    for (unsigned i = 1; i < n; ++i) {
        if (bits[i] != bits[0]) return false;
    }
    int b = bits[0];
    switch (d->f) {
    case cudaChannelFormatKindSigned:
    case cudaChannelFormatKindUnsigned:
        if (b != 8 && b != 16 && b != 32) return false;
        f->kind = d->f == cudaChannelFormatKindSigned ? OCU_TEXK_SIGNED : OCU_TEXK_UNSIGNED;
        break;
    case cudaChannelFormatKindFloat:
        if (b != 16 && b != 32) return false;
        f->kind = b == 16 ? OCU_TEXK_HALF : OCU_TEXK_FLOAT;
        break;
    default: {
        /* The ...Normalized{8,16}X{1,2,4} kinds: always read as normalized
           floats; the kind fixes the width and the component count. */
        int k = (int)d->f;
        if (k < cudaChannelFormatKindUnsignedNormalized8X1 ||
            k > cudaChannelFormatKindSignedNormalized16X4) {
            return false;
        }
        int r = (k - cudaChannelFormatKindUnsignedNormalized8X1) % 6;
        int want_bits = r < 3 ? 8 : 16;
        unsigned want_n = (r % 3) == 0 ? 1u : (r % 3) == 1 ? 2u : 4u;
        if (b != want_bits || n != want_n) return false;
        f->kind = k < cudaChannelFormatKindSignedNormalized8X1 ? OCU_TEXK_UNSIGNED : OCU_TEXK_SIGNED;
        f->always_norm = 1;
        break;
    }
    }
    f->channels = n;
    f->channel_bytes = (unsigned)b / 8;
    f->texel_bytes = n * f->channel_bytes;
    return true;
}

/* ---- arrays ------------------------------------------------------------------- */

#define ARRAY_MAGIC 0x6f637561u /* "ocua" */

struct cudaArray {
    unsigned magic;
    struct cudaChannelFormatDesc desc;
    tex_format fmt;
    struct cudaExtent extent; /* as requested (0 for absent dimensions) */
    unsigned flags;
    size_t width, rows, slices; /* in elements, each at least 1 */
    size_t pitch;               /* row bytes, no padding */
    void *data;                 /* device memory */
    struct cudaArray *next;
};

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static struct cudaArray *g_arrays;

/* True when a is a live array; the lock must be held. */
static bool array_live_locked(const struct cudaArray *a)
{
    for (const struct cudaArray *p = g_arrays; p != NULL; p = p->next) {
        if (p == a) return true;
    }
    return false;
}

/* A copy of the live array a (its fields do not change after creation). */
static bool array_get(const struct cudaArray *a, struct cudaArray *out)
{
    bool ok = false;
    if (a == NULL) return false;
    pthread_mutex_lock(&g_lock);
    if (array_live_locked(a) && a->magic == ARRAY_MAGIC) {
        *out = *a;
        ok = true;
    }
    pthread_mutex_unlock(&g_lock);
    return ok;
}

bool ocu_rt_array_view_of(cudaArray_const_t array, ocu_rt_array_view *view)
{
    struct cudaArray a;
    if (!array_get(array, &a)) return false;
    view->ptr = a.data;
    view->pitch = a.pitch;
    view->rows = a.rows;
    view->width = a.width;
    view->slices = a.slices;
    view->elem_bytes = a.fmt.texel_bytes;
    return true;
}

static bool mul_ok(size_t a, size_t b, size_t *r) { return !__builtin_mul_overflow(a, b, r); }

static cudaError_t array_create(cudaArray_t *array, const struct cudaChannelFormatDesc *desc,
                                struct cudaExtent ext, unsigned flags)
{
    if (array == NULL || desc == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    *array = NULL;
    const unsigned known = cudaArrayLayered | cudaArraySurfaceLoadStore | cudaArrayCubemap |
                           cudaArrayTextureGather | cudaArrayColorAttachment | cudaArraySparse |
                           cudaArrayDeferredMapping;
    if (flags & ~known) return ocu_rt_set_error(cudaErrorInvalidValue);
    /* Arrays without backing memory need the driver's virtual memory calls. */
    if (flags & (cudaArraySparse | cudaArrayDeferredMapping)) {
        return ocu_rt_set_error(cudaErrorNotSupported);
    }
    tex_format f;
    if (!format_of(desc, false, &f)) return ocu_rt_set_error(cudaErrorInvalidValue);
    /* Valid shapes (cudaMalloc3DArray): 1D {w,0,0}, 2D {w,h,0}, 3D {w,h,d};
       layered 1D {w,0,layers}, layered 2D {w,h,layers}; cubemap {w,w,6}
       (layered: a multiple of 6). */
    bool layered = (flags & cudaArrayLayered) != 0, cube = (flags & cudaArrayCubemap) != 0;
    if (ext.width == 0) return ocu_rt_set_error(cudaErrorInvalidValue);
    if (ext.height == 0 && ext.depth != 0 && !layered) return ocu_rt_set_error(cudaErrorInvalidValue);
    if (layered && ext.depth == 0) return ocu_rt_set_error(cudaErrorInvalidValue);
    if (cube && (ext.width != ext.height || ext.depth == 0 ||
                 (layered ? ext.depth % 6 != 0 : ext.depth != 6))) {
        return ocu_rt_set_error(cudaErrorInvalidValue);
    }
    if ((flags & cudaArrayTextureGather) && (ext.height == 0 || ext.depth != 0)) {
        return ocu_rt_set_error(cudaErrorInvalidValue);
    }
    size_t rows = ext.height ? ext.height : 1, slices = ext.depth ? ext.depth : 1, pitch, bytes;
    if (ext.width > UINT_MAX || rows > UINT_MAX || slices > UINT_MAX ||
        !mul_ok(ext.width, f.texel_bytes, &pitch) || !mul_ok(pitch, rows, &bytes) ||
        !mul_ok(bytes, slices, &bytes)) {
        return ocu_rt_set_error(cudaErrorMemoryAllocation);
    }
    struct cudaArray *a = calloc(1, sizeof *a);
    if (a == NULL) return ocu_rt_set_error(cudaErrorMemoryAllocation);
    cudaError_t e = cudaMalloc(&a->data, bytes);
    if (e != cudaSuccess) {
        free(a);
        return e;
    }
    a->magic = ARRAY_MAGIC;
    a->desc = *desc;
    a->fmt = f;
    a->extent = ext;
    a->flags = flags;
    a->width = ext.width;
    a->rows = rows;
    a->slices = slices;
    a->pitch = pitch;
    pthread_mutex_lock(&g_lock);
    a->next = g_arrays;
    g_arrays = a;
    pthread_mutex_unlock(&g_lock);
    *array = a;
    return cudaSuccess;
}

struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
                                                   enum cudaChannelFormatKind f)
{
    struct cudaChannelFormatDesc d = {x, y, z, w, f};
    return d;
}

cudaError_t cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc,
                            size_t width, size_t height, unsigned int flags)
{
    if (flags & (cudaArrayLayered | cudaArrayCubemap)) return ocu_rt_set_error(cudaErrorInvalidValue);
    struct cudaExtent ext = {width, height, 0};
    return array_create(array, desc, ext, flags);
}

cudaError_t cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc,
                              struct cudaExtent extent, unsigned int flags)
{
    return array_create(array, desc, extent, flags);
}

cudaError_t cudaFreeArray(cudaArray_t array)
{
    if (array == NULL) return cudaSuccess;
    pthread_mutex_lock(&g_lock);
    struct cudaArray **pp = &g_arrays;
    while (*pp != NULL && *pp != array) pp = &(*pp)->next;
    if (*pp == NULL) {
        pthread_mutex_unlock(&g_lock);
        return ocu_rt_set_error(cudaErrorInvalidValue);
    }
    *pp = array->next;
    pthread_mutex_unlock(&g_lock);
    void *data = array->data;
    array->magic = 0;
    free(array);
    return cudaFree(data); /* waits for queued work using it */
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent,
                             unsigned int *flags, cudaArray_t array)
{
    struct cudaArray a;
    if (!array_get(array, &a)) return ocu_rt_set_error(cudaErrorInvalidValue);
    if (desc != NULL) *desc = a.desc;
    if (extent != NULL) *extent = a.extent;
    if (flags != NULL) *flags = a.flags;
    return cudaSuccess;
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array)
{
    struct cudaArray a;
    if (desc == NULL || !array_get(array, &a)) return ocu_rt_set_error(cudaErrorInvalidValue);
    *desc = a.desc;
    return cudaSuccess;
}

/* ---- copies to and from arrays --------------------------------------------------------
 * Offsets: wOffset in bytes within a row, hOffset in rows. The rows of all
 * slices count as one sequence of rows (choice: the 2D calls do not say how
 * they treat 3D and layered arrays). The 1D-style calls (cudaMemcpyToArray
 * and friends) copy `count` bytes starting at (wOffset, hOffset), running on
 * into the next rows.
 */

/* Address of byte (w, h) and the bytes from there to the end of the array;
   false when outside. */
static bool array_at(const struct cudaArray *a, size_t w, size_t h, char **p, size_t *left)
{
    size_t total = a->pitch * a->rows * a->slices, off;
    if (w > a->pitch || !mul_ok(h, a->pitch, &off) || __builtin_add_overflow(off, w, &off) ||
        off > total) {
        return false;
    }
    *p = (char *)a->data + off;
    *left = total - off;
    return true;
}

static cudaError_t copy_linear(cudaArray_t dst, size_t dw, size_t dh, cudaArray_const_t src,
                               size_t sw, size_t sh, void *dptr, const void *sptr, size_t count,
                               enum cudaMemcpyKind kind, cudaStream_t stream, bool async)
{
    struct cudaArray da, sa;
    size_t left;
    char *p;
    if (dst != NULL) {
        if (!array_get(dst, &da) || !array_at(&da, dw, dh, &p, &left) || count > left) {
            return ocu_rt_set_error(cudaErrorInvalidValue);
        }
        dptr = p;
    }
    if (src != NULL) {
        if (!array_get(src, &sa) || !array_at(&sa, sw, sh, &p, &left) || count > left) {
            return ocu_rt_set_error(cudaErrorInvalidValue);
        }
        sptr = p;
    }
    if (count == 0) return cudaSuccess;
    return async ? cudaMemcpyAsync(dptr, sptr, count, kind, stream)
                 : cudaMemcpy(dptr, sptr, count, kind);
}

cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src,
                              size_t count, enum cudaMemcpyKind kind)
{
    if (dst == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    return copy_linear(dst, wOffset, hOffset, NULL, 0, 0, NULL, src, count, kind, NULL, false);
}

cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset,
                                size_t count, enum cudaMemcpyKind kind)
{
    if (src == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    return copy_linear(NULL, 0, 0, src, wOffset, hOffset, dst, NULL, count, kind, NULL, false);
}

cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                   const void *src, size_t count, enum cudaMemcpyKind kind,
                                   cudaStream_t stream)
{
    if (dst == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    return copy_linear(dst, wOffset, hOffset, NULL, 0, 0, NULL, src, count, kind, stream, true);
}

cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset,
                                     size_t hOffset, size_t count, enum cudaMemcpyKind kind,
                                     cudaStream_t stream)
{
    if (src == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    return copy_linear(NULL, 0, 0, src, wOffset, hOffset, dst, NULL, count, kind, stream, true);
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                   cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc,
                                   size_t count, enum cudaMemcpyKind kind)
{
    if (dst == NULL || src == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    return copy_linear(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, NULL, NULL, count,
                       kind, NULL, false);
}

/* 2D: `height` rows of `width` bytes; the array side must hold the whole
   rectangle (wOffset + width <= row bytes, hOffset + height <= rows). */
static bool array_rect(cudaArray_const_t arr, size_t w, size_t h, size_t width, size_t height,
                       char **p, size_t *pitch)
{
    struct cudaArray a;
    if (!array_get(arr, &a)) return false;
    size_t rows = a.rows * a.slices;
    if (w > a.pitch || width > a.pitch - w || h > rows || height > rows - h) return false;
    *p = (char *)a.data + h * a.pitch + w;
    *pitch = a.pitch;
    return true;
}

static cudaError_t copy_2d(cudaArray_t dst, size_t dw, size_t dh, cudaArray_const_t src,
                           size_t sw, size_t sh, void *dptr, size_t dpitch, const void *sptr,
                           size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
                           cudaStream_t stream, bool async)
{
    char *p;
    if (dst != NULL) {
        if (!array_rect(dst, dw, dh, width, height, &p, &dpitch)) {
            return ocu_rt_set_error(cudaErrorInvalidValue);
        }
        dptr = p;
    }
    if (src != NULL) {
        if (!array_rect(src, sw, sh, width, height, &p, &spitch)) {
            return ocu_rt_set_error(cudaErrorInvalidValue);
        }
        sptr = p;
    }
    if (width == 0 || height == 0) return cudaSuccess;
    if (width > dpitch || width > spitch) return ocu_rt_set_error(cudaErrorInvalidPitchValue);
    return async ? cudaMemcpy2DAsync(dptr, dpitch, sptr, spitch, width, height, kind, stream)
                 : cudaMemcpy2D(dptr, dpitch, sptr, spitch, width, height, kind);
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src,
                                size_t spitch, size_t width, size_t height,
                                enum cudaMemcpyKind kind)
{
    if (dst == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    return copy_2d(dst, wOffset, hOffset, NULL, 0, 0, NULL, 0, src, spitch, width, height, kind,
                   NULL, false);
}

cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset,
                                  size_t hOffset, size_t width, size_t height,
                                  enum cudaMemcpyKind kind)
{
    if (src == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    return copy_2d(NULL, 0, 0, src, wOffset, hOffset, dst, dpitch, NULL, 0, width, height, kind,
                   NULL, false);
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                     const void *src, size_t spitch, size_t width, size_t height,
                                     enum cudaMemcpyKind kind, cudaStream_t stream)
{
    if (dst == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    return copy_2d(dst, wOffset, hOffset, NULL, 0, 0, NULL, 0, src, spitch, width, height, kind,
                   stream, true);
}

cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src,
                                       size_t wOffset, size_t hOffset, size_t width,
                                       size_t height, enum cudaMemcpyKind kind,
                                       cudaStream_t stream)
{
    if (src == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    return copy_2d(NULL, 0, 0, src, wOffset, hOffset, dst, dpitch, NULL, 0, width, height, kind,
                   stream, true);
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                     cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc,
                                     size_t width, size_t height, enum cudaMemcpyKind kind)
{
    if (dst == NULL || src == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    return copy_2d(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, NULL, 0, NULL, 0,
                   width, height, kind, NULL, false);
}

/* ---- descriptors ---------------------------------------------------------------- */

static void desc_format(struct ocu_tex_desc *d, const tex_format *f)
{
    d->channels = (unsigned char)f->channels;
    d->channel_bytes = (unsigned char)f->channel_bytes;
    d->kind = (unsigned char)f->kind;
    d->texel_bytes = f->texel_bytes;
    d->always_norm = (unsigned char)f->always_norm;
}

static void desc_from_array(struct ocu_tex_desc *d, const struct cudaArray *a)
{
    desc_format(d, &a->fmt);
    d->resource = OCU_TEXR_ARRAY;
    d->base = a->data;
    d->pitch = a->pitch;
    d->slice = (unsigned long long)a->pitch * a->rows;
    d->bytes = d->slice * a->slices;
    d->width = (unsigned)a->width;
    d->height = (unsigned)a->rows;
    d->depth = (unsigned)a->slices;
    d->layered = (a->flags & cudaArrayLayered) != 0;
}

/* Linear memory of `size` bytes at p, not past the end of its allocation
   when the runtime made it (choice: a size beyond it, such as the
   documented default UINT_MAX, is cut there). */
static void desc_linear(struct ocu_tex_desc *d, const void *p, size_t size, const tex_format *f)
{
    size_t avail;
    if (ocu_rt_alloc_avail(p, &avail) && size > avail) size = avail;
    desc_format(d, f);
    d->resource = OCU_TEXR_LINEAR;
    d->base = p;
    d->bytes = size;
    d->pitch = size;
    d->slice = size;
    size_t n = f->texel_bytes ? size / f->texel_bytes : size;
    d->width = n > UINT_MAX ? UINT_MAX : (unsigned)n;
    d->height = 1;
    d->depth = 1;
}

static bool desc_pitch2d(struct ocu_tex_desc *d, const void *p, size_t width, size_t height,
                         size_t pitch, const tex_format *f)
{
    size_t row;
    if (width == 0 || height == 0 || width > UINT_MAX || height > UINT_MAX) return false;
    if (f->texel_bytes != 0 && (!mul_ok(width, f->texel_bytes, &row) || row > pitch)) return false;
    desc_format(d, f);
    d->resource = OCU_TEXR_PITCH2D;
    d->base = p;
    d->pitch = pitch;
    d->slice = (unsigned long long)pitch * height;
    d->bytes = d->slice;
    d->width = (unsigned)width;
    d->height = (unsigned)height;
    d->depth = 1;
    return true;
}

/* ---- texture objects ------------------------------------------------------------- */

typedef struct tex_object {
    cudaTextureObject_t handle; /* device address of the descriptor */
    struct cudaResourceDesc res;
    struct cudaTextureDesc tex;
    struct cudaResourceViewDesc view;
    struct tex_object *next;
} tex_object;

static tex_object *g_objects;

static bool view_ok(const struct cudaResourceViewDesc *v, const struct cudaArray *a)
{
    /* Only a view that changes nothing: same format (or none), the whole
       array, no mipmap levels, all layers. */
    static const struct { int bits, n; enum cudaChannelFormatKind k; } fmt[] = {
        {0, 0, cudaChannelFormatKindNone},
        {8, 1, cudaChannelFormatKindUnsigned}, {8, 2, cudaChannelFormatKindUnsigned}, {8, 4, cudaChannelFormatKindUnsigned},
        {8, 1, cudaChannelFormatKindSigned}, {8, 2, cudaChannelFormatKindSigned}, {8, 4, cudaChannelFormatKindSigned},
        {16, 1, cudaChannelFormatKindUnsigned}, {16, 2, cudaChannelFormatKindUnsigned}, {16, 4, cudaChannelFormatKindUnsigned},
        {16, 1, cudaChannelFormatKindSigned}, {16, 2, cudaChannelFormatKindSigned}, {16, 4, cudaChannelFormatKindSigned},
        {32, 1, cudaChannelFormatKindUnsigned}, {32, 2, cudaChannelFormatKindUnsigned}, {32, 4, cudaChannelFormatKindUnsigned},
        {32, 1, cudaChannelFormatKindSigned}, {32, 2, cudaChannelFormatKindSigned}, {32, 4, cudaChannelFormatKindSigned},
        {16, 1, cudaChannelFormatKindFloat}, {16, 2, cudaChannelFormatKindFloat}, {16, 4, cudaChannelFormatKindFloat},
        {32, 1, cudaChannelFormatKindFloat}, {32, 2, cudaChannelFormatKindFloat}, {32, 4, cudaChannelFormatKindFloat},
    };
    unsigned vf = (unsigned)v->format;
    if (vf >= sizeof fmt / sizeof fmt[0]) return false;
    if (vf != 0) {
        bool same = a->desc.f == fmt[vf].k && (unsigned)fmt[vf].n == a->fmt.channels &&
                    (unsigned)fmt[vf].bits == a->fmt.channel_bytes * 8;
        if (!same) return false;
    }
    if (v->width != a->extent.width || v->height != a->extent.height ||
        v->depth != a->extent.depth) {
        return false;
    }
    if (v->firstMipmapLevel != 0 || v->lastMipmapLevel != 0 || v->firstLayer != 0) return false;
    return v->lastLayer == 0 || v->lastLayer + 1 == a->slices;
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                                    const struct cudaResourceDesc *pResDesc,
                                    const struct cudaTextureDesc *pTexDesc,
                                    const struct cudaResourceViewDesc *pResViewDesc)
{
    if (pTexObject == NULL || pResDesc == NULL || pTexDesc == NULL) {
        return ocu_rt_set_error(cudaErrorInvalidValue);
    }
    struct ocu_tex_desc d;
    memset(&d, 0, sizeof d);
    tex_format f;
    const struct cudaTextureDesc *t = pTexDesc;
    for (int i = 0; i < 3; ++i) {
        if ((unsigned)t->addressMode[i] > cudaAddressModeBorder) {
            return ocu_rt_set_error(cudaErrorInvalidValue);
        }
    }
    if ((unsigned)t->filterMode > cudaFilterModeLinear ||
        (unsigned)t->readMode > cudaReadModeNormalizedFloat) {
        return ocu_rt_set_error(cudaErrorInvalidValue);
    }
    if (t->sRGB) return ocu_rt_set_error(cudaErrorNotSupported);
    switch (pResDesc->resType) {
    case cudaResourceTypeArray: {
        struct cudaArray a;
        if (!array_get(pResDesc->res.array.array, &a)) return ocu_rt_set_error(cudaErrorInvalidValue);
        if (pResViewDesc != NULL && !view_ok(pResViewDesc, &a)) {
            return ocu_rt_set_error(cudaErrorNotSupported);
        }
        desc_from_array(&d, &a);
        f = a.fmt;
        break;
    }
    case cudaResourceTypeMipmappedArray:
        return ocu_rt_set_error(cudaErrorNotSupported);
    case cudaResourceTypeLinear:
        if (pResViewDesc != NULL || pResDesc->res.linear.devPtr == NULL ||
            !format_of(&pResDesc->res.linear.desc, false, &f) ||
            pResDesc->res.linear.sizeInBytes < f.texel_bytes) {
            return ocu_rt_set_error(cudaErrorInvalidValue);
        }
        desc_linear(&d, pResDesc->res.linear.devPtr, pResDesc->res.linear.sizeInBytes, &f);
        break;
    case cudaResourceTypePitch2D:
        if (pResViewDesc != NULL || pResDesc->res.pitch2D.devPtr == NULL ||
            !format_of(&pResDesc->res.pitch2D.desc, false, &f) ||
            !desc_pitch2d(&d, pResDesc->res.pitch2D.devPtr, pResDesc->res.pitch2D.width,
                          pResDesc->res.pitch2D.height, pResDesc->res.pitch2D.pitchInBytes, &f)) {
            return ocu_rt_set_error(cudaErrorInvalidValue);
        }
        break;
    default:
        return ocu_rt_set_error(cudaErrorInvalidValue);
    }
    /* Linear filtering only for results that are floating point: float
       texels, or 8/16-bit integers read as normalized floats. */
    bool float_result = f.kind == OCU_TEXK_FLOAT || f.kind == OCU_TEXK_HALF ||
                        ((t->readMode == cudaReadModeNormalizedFloat || f.always_norm) &&
                         f.channel_bytes <= 2);
    if (t->filterMode == cudaFilterModeLinear && !float_result &&
        pResDesc->resType != cudaResourceTypeLinear) {
        return ocu_rt_set_error(cudaErrorInvalidValue);
    }
    for (int i = 0; i < 3; ++i) d.address[i] = (unsigned char)t->addressMode[i];
    d.filter = (unsigned char)t->filterMode;
    d.read_norm = t->readMode == cudaReadModeNormalizedFloat;
    d.normalized = t->normalizedCoords != 0;
    for (int i = 0; i < 4; ++i) d.border[i] = t->borderColor[i];
    if (pResDesc->resType == cudaResourceTypeLinear) {
        /* Address and filter modes are ignored for linear memory. */
        d.filter = cudaFilterModePoint;
        d.normalized = 0;
    }

    tex_object *o = calloc(1, sizeof *o);
    if (o == NULL) return ocu_rt_set_error(cudaErrorMemoryAllocation);
    void *dev = NULL;
    cudaError_t e = cudaMalloc(&dev, sizeof d);
    if (e == cudaSuccess) e = cudaMemcpy(dev, &d, sizeof d, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        if (dev != NULL) (void)cudaFree(dev);
        free(o);
        return e;
    }
    o->handle = (cudaTextureObject_t)(uintptr_t)dev;
    o->res = *pResDesc;
    o->tex = *pTexDesc;
    if (pResViewDesc != NULL) o->view = *pResViewDesc;
    pthread_mutex_lock(&g_lock);
    o->next = g_objects;
    g_objects = o;
    pthread_mutex_unlock(&g_lock);
    *pTexObject = o->handle;
    return cudaSuccess;
}

/* A copy of the live object `h`. */
static bool object_get(cudaTextureObject_t h, tex_object *out)
{
    bool ok = false;
    pthread_mutex_lock(&g_lock);
    for (tex_object *o = g_objects; o != NULL; o = o->next) {
        if (o->handle == h && h != 0) {
            *out = *o;
            ok = true;
            break;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return ok;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject)
{
    pthread_mutex_lock(&g_lock);
    tex_object **pp = &g_objects;
    while (*pp != NULL && ((*pp)->handle != texObject || texObject == 0)) pp = &(*pp)->next;
    tex_object *o = *pp;
    if (o != NULL) *pp = o->next;
    pthread_mutex_unlock(&g_lock);
    if (o == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    void *dev = (void *)(uintptr_t)o->handle;
    free(o);
    return cudaFree(dev); /* waits for queued work that may still fetch */
}

cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc,
                                             cudaTextureObject_t texObject)
{
    tex_object o;
    if (pResDesc == NULL || !object_get(texObject, &o)) return ocu_rt_set_error(cudaErrorInvalidValue);
    *pResDesc = o.res;
    return cudaSuccess;
}

cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc,
                                            cudaTextureObject_t texObject)
{
    tex_object o;
    if (pTexDesc == NULL || !object_get(texObject, &o)) return ocu_rt_set_error(cudaErrorInvalidValue);
    *pTexDesc = o.tex;
    return cudaSuccess;
}

cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc,
                                                 cudaTextureObject_t texObject)
{
    tex_object o;
    if (pResViewDesc == NULL || !object_get(texObject, &o)) {
        return ocu_rt_set_error(cudaErrorInvalidValue);
    }
    *pResViewDesc = o.view; /* all zero when none was given */
    return cudaSuccess;
}

/* ---- texture references ----------------------------------------------------------- */

/* Kernels already queued keep the binding they were launched with: wait
   for them before changing it. */
static cudaError_t rebind(const struct textureReference *ref, const struct ocu_tex_desc *d)
{
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return e;
    ((struct textureReference *)ref)->ocu_bound = *d;
    return cudaSuccess;
}

cudaError_t cudaBindTexture(size_t *offset, const struct textureReference *texref,
                            const void *devPtr, const struct cudaChannelFormatDesc *desc,
                            size_t size)
{
    if (texref == NULL) return ocu_rt_set_error(cudaErrorInvalidTexture);
    tex_format f;
    if (devPtr == NULL || desc == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    if (!format_of(desc, true, &f)) return ocu_rt_set_error(cudaErrorInvalidChannelDescriptor);
    struct ocu_tex_desc d;
    memset(&d, 0, sizeof d);
    desc_linear(&d, devPtr, size, &f);
    cudaError_t e = rebind(texref, &d);
    if (e == cudaSuccess && offset != NULL) *offset = 0; /* no alignment needed */
    return e;
}

cudaError_t cudaBindTexture2D(size_t *offset, const struct textureReference *texref,
                              const void *devPtr, const struct cudaChannelFormatDesc *desc,
                              size_t width, size_t height, size_t pitch)
{
    if (texref == NULL) return ocu_rt_set_error(cudaErrorInvalidTexture);
    tex_format f;
    if (devPtr == NULL || desc == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    if (!format_of(desc, true, &f)) return ocu_rt_set_error(cudaErrorInvalidChannelDescriptor);
    struct ocu_tex_desc d;
    memset(&d, 0, sizeof d);
    if (!desc_pitch2d(&d, devPtr, width, height, pitch, &f)) {
        return ocu_rt_set_error(cudaErrorInvalidValue);
    }
    cudaError_t e = rebind(texref, &d);
    if (e == cudaSuccess && offset != NULL) *offset = 0;
    return e;
}

cudaError_t cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array,
                                   const struct cudaChannelFormatDesc *desc)
{
    if (texref == NULL) return ocu_rt_set_error(cudaErrorInvalidTexture);
    struct cudaArray a;
    if (desc == NULL || !array_get(array, &a)) return ocu_rt_set_error(cudaErrorInvalidValue);
    /* The texels are read in the array's format (the documents call a
       descriptor that differs undefined). */
    struct ocu_tex_desc d;
    memset(&d, 0, sizeof d);
    desc_from_array(&d, &a);
    return rebind(texref, &d);
}

cudaError_t cudaUnbindTexture(const struct textureReference *texref)
{
    if (texref == NULL) return ocu_rt_set_error(cudaErrorInvalidTexture);
    struct ocu_tex_desc d;
    memset(&d, 0, sizeof d);
    return rebind(texref, &d);
}

cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref)
{
    if (texref == NULL) return ocu_rt_set_error(cudaErrorInvalidTexture);
    if (offset == NULL) return ocu_rt_set_error(cudaErrorInvalidValue);
    if (texref->ocu_bound.base == NULL) return ocu_rt_set_error(cudaErrorInvalidTextureBinding);
    *offset = 0;
    return cudaSuccess;
}

/* The symbol of a texture reference is the reference itself. */
cudaError_t cudaGetTextureReference(const struct textureReference **texref, const void *symbol)
{
    if (texref == NULL || symbol == NULL) return ocu_rt_set_error(cudaErrorInvalidTexture);
    *texref = (const struct textureReference *)symbol;
    return cudaSuccess;
}

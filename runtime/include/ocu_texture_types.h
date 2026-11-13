/*
 * ocu_texture_types.h -- the C++ side of the texture API: the
 * cudaCreateChannelDesc<T>() specialisations, the texture<T, type, mode>
 * template of the legacy reference API, and the C++ overloads of the
 * binding and array calls (default arguments, references instead of
 * pointers).
 *
 * Written for this project from the public CUDA Runtime API reference
 * ("C++ API Routines", "Texture Reference Management") and the programming
 * guide ("Texture Reference API"). Not derived from any vendor header.
 *
 * Included by cuda_runtime_api.h in C++.
 */
#ifndef OCU_TEXTURE_TYPES_H
#define OCU_TEXTURE_TYPES_H

#if defined(__CUDA__)
#define OCU_TT_HD __attribute__((host)) __attribute__((device))
#else
#define OCU_TT_HD
#endif

/* ---- channel descriptors ------------------------------------------------- */

namespace ocu_tex_types_ {
inline cudaChannelFormatDesc desc(int x, int y, int z, int w, cudaChannelFormatKind f)
{
    cudaChannelFormatDesc d;
    d.x = x;
    d.y = y;
    d.z = z;
    d.w = w;
    d.f = f;
    return d;
}
} // namespace ocu_tex_types_

/* A type without a specialisation: kind None, no bits (the documented
   default). */
template <class T>
inline cudaChannelFormatDesc cudaCreateChannelDesc(void)
{
    return ocu_tex_types_::desc(0, 0, 0, 0, cudaChannelFormatKindNone);
}

#define OCU_TT_DESC(T, x, y, z, w, kind)                                                   \
    template <>                                                                            \
    inline cudaChannelFormatDesc cudaCreateChannelDesc<T>(void)                            \
    {                                                                                      \
        return ocu_tex_types_::desc(x, y, z, w, kind);                                     \
    }
#define OCU_TT_FAMILY(scalar, vec, kind)                                                   \
    OCU_TT_DESC(scalar, (int)sizeof(scalar) * 8, 0, 0, 0, kind)                            \
    OCU_TT_DESC(vec##1, (int)sizeof(scalar) * 8, 0, 0, 0, kind)                            \
    OCU_TT_DESC(vec##2, (int)sizeof(scalar) * 8, (int)sizeof(scalar) * 8, 0, 0, kind)      \
    OCU_TT_DESC(vec##4, (int)sizeof(scalar) * 8, (int)sizeof(scalar) * 8,                  \
                (int)sizeof(scalar) * 8, (int)sizeof(scalar) * 8, kind)

/* char's kind follows its signedness on the host compiler. */
#if defined(__CHAR_UNSIGNED__) || defined(_CHAR_UNSIGNED)
OCU_TT_DESC(char, 8, 0, 0, 0, cudaChannelFormatKindUnsigned)
#else
OCU_TT_DESC(char, 8, 0, 0, 0, cudaChannelFormatKindSigned)
#endif
OCU_TT_FAMILY(signed char, char, cudaChannelFormatKindSigned)
OCU_TT_FAMILY(unsigned char, uchar, cudaChannelFormatKindUnsigned)
OCU_TT_FAMILY(short, short, cudaChannelFormatKindSigned)
OCU_TT_FAMILY(unsigned short, ushort, cudaChannelFormatKindUnsigned)
OCU_TT_FAMILY(int, int, cudaChannelFormatKindSigned)
OCU_TT_FAMILY(unsigned int, uint, cudaChannelFormatKindUnsigned)
OCU_TT_FAMILY(long, long, cudaChannelFormatKindSigned)
OCU_TT_FAMILY(unsigned long, ulong, cudaChannelFormatKindUnsigned)
OCU_TT_FAMILY(float, float, cudaChannelFormatKindFloat)
#undef OCU_TT_FAMILY
#undef OCU_TT_DESC

/* 16-bit floating point (binary16) channels. */
inline cudaChannelFormatDesc cudaCreateChannelDescHalf(void)
{
    return ocu_tex_types_::desc(16, 0, 0, 0, cudaChannelFormatKindFloat);
}
inline cudaChannelFormatDesc cudaCreateChannelDescHalf1(void)
{
    return ocu_tex_types_::desc(16, 0, 0, 0, cudaChannelFormatKindFloat);
}
inline cudaChannelFormatDesc cudaCreateChannelDescHalf2(void)
{
    return ocu_tex_types_::desc(16, 16, 0, 0, cudaChannelFormatKindFloat);
}
inline cudaChannelFormatDesc cudaCreateChannelDescHalf4(void)
{
    return ocu_tex_types_::desc(16, 16, 16, 16, cudaChannelFormatKindFloat);
}

/* ---- texture references -------------------------------------------------- */

/*
 * texture<T, type, mode>: a file-scope texture reference. Its mutable
 * members (normalized, filterMode, addressMode) may be changed by host code
 * at any time; a fetch reads them as they are when it runs. Fetches:
 * ocu_texture_fetch.h (CPU path only: texture references were removed from
 * CUDA 12, and on the NVIDIA path device code cannot see host variables).
 */
template <class T, int texType = cudaTextureType1D,
          enum cudaTextureReadMode mode = cudaReadModeElementType>
struct texture : public textureReference {
    texture(int norm = 0, enum cudaTextureFilterMode fMode = cudaFilterModePoint,
            enum cudaTextureAddressMode aMode = cudaAddressModeClamp)
    {
        init(norm, fMode, aMode, cudaCreateChannelDesc<T>());
    }
    texture(int norm, enum cudaTextureFilterMode fMode, enum cudaTextureAddressMode aMode,
            struct cudaChannelFormatDesc desc)
    {
        init(norm, fMode, aMode, desc);
    }

private:
    void init(int norm, enum cudaTextureFilterMode fMode, enum cudaTextureAddressMode aMode,
              struct cudaChannelFormatDesc desc)
    {
        normalized = norm;
        filterMode = fMode;
        addressMode[0] = aMode;
        addressMode[1] = aMode;
        addressMode[2] = aMode;
        channelDesc = desc;
        sRGB = 0;
        maxAnisotropy = 0;
        mipmapFilterMode = cudaFilterModePoint;
        mipmapLevelBias = 0;
        minMipmapLevelClamp = 0;
        maxMipmapLevelClamp = 0;
        disableTrilinearOptimization = 0;
        ocu_bound = ocu_tex_desc();
    }
};

template <class T, int dim, enum cudaTextureReadMode m>
static inline cudaError_t cudaBindTexture(size_t *offset, const texture<T, dim, m> &tex,
                                          const void *devPtr, const cudaChannelFormatDesc &desc,
                                          size_t size = UINT_MAX)
{
    return cudaBindTexture(offset, &tex, devPtr, &desc, size);
}

template <class T, int dim, enum cudaTextureReadMode m>
static inline cudaError_t cudaBindTexture(size_t *offset, const texture<T, dim, m> &tex,
                                          const void *devPtr, size_t size = UINT_MAX)
{
    return cudaBindTexture(offset, &tex, devPtr, &tex.channelDesc, size);
}

/* The C form with its documented default size. */
static inline cudaError_t cudaBindTexture(size_t *offset, const struct textureReference *texref,
                                          const void *devPtr,
                                          const struct cudaChannelFormatDesc *desc)
{
    return cudaBindTexture(offset, texref, devPtr, desc, UINT_MAX);
}

template <class T, int dim, enum cudaTextureReadMode m>
static inline cudaError_t cudaBindTexture2D(size_t *offset, const texture<T, dim, m> &tex,
                                            const void *devPtr, const cudaChannelFormatDesc &desc,
                                            size_t width, size_t height, size_t pitch)
{
    return cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch);
}

template <class T, int dim, enum cudaTextureReadMode m>
static inline cudaError_t cudaBindTexture2D(size_t *offset, const texture<T, dim, m> &tex,
                                            const void *devPtr, size_t width, size_t height,
                                            size_t pitch)
{
    return cudaBindTexture2D(offset, &tex, devPtr, &tex.channelDesc, width, height, pitch);
}

template <class T, int dim, enum cudaTextureReadMode m>
static inline cudaError_t cudaBindTextureToArray(const texture<T, dim, m> &tex,
                                                 cudaArray_const_t array,
                                                 const cudaChannelFormatDesc &desc)
{
    return cudaBindTextureToArray(&tex, array, &desc);
}

/* Without a descriptor: the array's own. */
template <class T, int dim, enum cudaTextureReadMode m>
static inline cudaError_t cudaBindTextureToArray(const texture<T, dim, m> &tex,
                                                 cudaArray_const_t array)
{
    cudaChannelFormatDesc desc;
    cudaError_t e = cudaGetChannelDesc(&desc, array);
    if (e != cudaSuccess) return e;
    return cudaBindTextureToArray(&tex, array, &desc);
}

template <class T, int dim, enum cudaTextureReadMode m>
static inline cudaError_t cudaUnbindTexture(const texture<T, dim, m> &tex)
{
    return cudaUnbindTexture(&tex);
}

template <class T, int dim, enum cudaTextureReadMode m>
static inline cudaError_t cudaGetTextureAlignmentOffset(size_t *offset,
                                                        const texture<T, dim, m> &tex)
{
    return cudaGetTextureAlignmentOffset(offset, &tex);
}

/* ---- arrays: documented default arguments --------------------------------- */

static inline cudaError_t cudaMallocArray(cudaArray_t *array,
                                          const struct cudaChannelFormatDesc *desc, size_t width,
                                          size_t height = 0)
{
    return cudaMallocArray(array, desc, width, height, 0u);
}

static inline cudaError_t cudaMalloc3DArray(cudaArray_t *array,
                                            const struct cudaChannelFormatDesc *desc,
                                            struct cudaExtent extent)
{
    return cudaMalloc3DArray(array, desc, extent, 0u);
}

static inline cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst,
                                                 size_t hOffsetDst, cudaArray_const_t src,
                                                 size_t wOffsetSrc, size_t hOffsetSrc,
                                                 size_t count)
{
    return cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count,
                                  cudaMemcpyDeviceToDevice);
}

static inline cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst,
                                                   size_t hOffsetDst, cudaArray_const_t src,
                                                   size_t wOffsetSrc, size_t hOffsetSrc,
                                                   size_t width, size_t height)
{
    return cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc,
                                    width, height, cudaMemcpyDeviceToDevice);
}

static inline cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                                 const void *src, size_t count,
                                                 cudaMemcpyKind kind)
{
    return cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, (cudaStream_t)0);
}

static inline cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src,
                                                   size_t wOffset, size_t hOffset, size_t count,
                                                   cudaMemcpyKind kind)
{
    return cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, (cudaStream_t)0);
}

static inline cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset,
                                                   size_t hOffset, const void *src,
                                                   size_t spitch, size_t width, size_t height,
                                                   cudaMemcpyKind kind)
{
    return cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind,
                                    (cudaStream_t)0);
}

static inline cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
                                                     cudaArray_const_t src, size_t wOffset,
                                                     size_t hOffset, size_t width, size_t height,
                                                     cudaMemcpyKind kind)
{
    return cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind,
                                      (cudaStream_t)0);
}

#undef OCU_TT_HD

#endif /* OCU_TEXTURE_TYPES_H */

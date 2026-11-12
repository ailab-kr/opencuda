// ocu_metal_legacy_texture.h -- declarations of the legacy texture reference
// API, so that a .cu file that declares a texture reference still parses.
// Written for this project from the public CUDA Programming Guide (texture
// reference API): names and signatures only, no behavior.
//
// cu2msl.py force-includes this header. Textures are not translated: a kernel
// that uses a texture reference gets a "not supported" diagnostic, the other
// kernels of the file translate.
#ifndef OCU_METAL_LEGACY_TEXTURE_H
#define OCU_METAL_LEGACY_TEXTURE_H

#include <stddef.h>

enum cudaTextureAddressMode { cudaAddressModeWrap = 0, cudaAddressModeClamp = 1,
                              cudaAddressModeMirror = 2, cudaAddressModeBorder = 3 };
enum cudaTextureFilterMode { cudaFilterModePoint = 0, cudaFilterModeLinear = 1 };
enum cudaTextureReadMode { cudaReadModeElementType = 0, cudaReadModeNormalizedFloat = 1 };
enum cudaChannelFormatKind { cudaChannelFormatKindSigned = 0, cudaChannelFormatKindUnsigned = 1,
                             cudaChannelFormatKindFloat = 2, cudaChannelFormatKindNone = 3 };

struct cudaChannelFormatDesc {
    int x, y, z, w;
    enum cudaChannelFormatKind f;
};

struct cudaArray;
typedef struct cudaArray *cudaArray_t;
typedef const struct cudaArray *cudaArray_const_t;

struct textureReference {
    int normalized;
    enum cudaTextureFilterMode filterMode;
    enum cudaTextureAddressMode addressMode[3];
    struct cudaChannelFormatDesc channelDesc;
};

template <class T, int dim = 1, enum cudaTextureReadMode mode = cudaReadModeElementType>
struct __attribute__((device_builtin_texture_type)) texture : public textureReference {};

cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
template <class T> cudaChannelFormatDesc cudaCreateChannelDesc();

template <class T, int dim, enum cudaTextureReadMode mode>
cudaError_t cudaBindTexture(size_t *offset, const texture<T, dim, mode> &tex, const void *devPtr,
                            size_t size = (size_t)-1);
template <class T, int dim, enum cudaTextureReadMode mode>
cudaError_t cudaBindTexture(size_t *offset, const texture<T, dim, mode> &tex, const void *devPtr,
                            const cudaChannelFormatDesc &desc, size_t size = (size_t)-1);
template <class T, int dim, enum cudaTextureReadMode mode>
cudaError_t cudaBindTexture2D(size_t *offset, const texture<T, dim, mode> &tex, const void *devPtr,
                              const cudaChannelFormatDesc &desc, size_t width, size_t height,
                              size_t pitch);
template <class T, int dim, enum cudaTextureReadMode mode>
cudaError_t cudaBindTextureToArray(const texture<T, dim, mode> &tex, cudaArray_const_t array);
template <class T, int dim, enum cudaTextureReadMode mode>
cudaError_t cudaBindTextureToArray(const texture<T, dim, mode> &tex, cudaArray_const_t array,
                                   const cudaChannelFormatDesc &desc);
template <class T, int dim, enum cudaTextureReadMode mode>
cudaError_t cudaUnbindTexture(const texture<T, dim, mode> &tex);

// the C-style forms take the texture by pointer
cudaError_t cudaBindTexture(size_t *offset, const textureReference *tex, const void *devPtr,
                            const cudaChannelFormatDesc *desc, size_t size = (size_t)-1);
cudaError_t cudaUnbindTexture(const textureReference *tex);

#ifndef CUDA_SUCCESS
#  define CUDA_SUCCESS 0   // the driver API's result code for success (kmeans compares with it)
#endif

cudaError_t cudaMallocArray(cudaArray_t *array, const cudaChannelFormatDesc *desc, size_t width,
                            size_t height = 0, unsigned int flags = 0);
cudaError_t cudaFreeArray(cudaArray_t array);
cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src,
                              size_t count, enum cudaMemcpyKind kind);

// Fetches: declared so that device code parses; texture use is reported by cu2msl.py.
template <class T, enum cudaTextureReadMode mode>
__device__ T tex1Dfetch(texture<T, 1, mode> t, int x);
template <class T, enum cudaTextureReadMode mode>
__device__ T tex1D(texture<T, 1, mode> t, float x);
template <class T, enum cudaTextureReadMode mode>
__device__ T tex2D(texture<T, 2, mode> t, float x, float y);
template <class T, enum cudaTextureReadMode mode>
__device__ T tex3D(texture<T, 3, mode> t, float x, float y, float z);

#endif

/*
 * mma.h -- the warp matrix (WMMA) API, namespace nvcuda::wmma.
 *
 * Not implemented yet: the namespace exists so that programs which include
 * this header but do not use WMMA (bitsandbytes' kernels.cu, for one)
 * compile. Code that uses wmma::fragment, load_matrix_sync or mma_sync
 * fails to compile with "no member named ... in namespace nvcuda::wmma".
 */
#ifndef OCU_MMA_H
#define OCU_MMA_H
#include <cuda_runtime.h>
namespace nvcuda {
namespace wmma {
} // namespace wmma
} // namespace nvcuda
#endif

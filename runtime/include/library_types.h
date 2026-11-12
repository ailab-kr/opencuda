/*
 * library_types.h -- data type and library property enums shared by the
 * math libraries (cublas_v2.h and later ones).
 *
 * Written for this project from the public CUDA library documentation
 * (cudaDataType_t and libraryPropertyType, with their documented values).
 * Not derived from any vendor header. C and C++.
 */
#ifndef OCU_LIBRARY_TYPES_H
#define OCU_LIBRARY_TYPES_H

typedef enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_4I = 16,
    CUDA_C_4I = 17,
    CUDA_R_4U = 18,
    CUDA_C_4U = 19,
    CUDA_R_8I = 3,
    CUDA_C_8I = 7,
    CUDA_R_8U = 8,
    CUDA_C_8U = 9,
    CUDA_R_16I = 20,
    CUDA_C_16I = 21,
    CUDA_R_16U = 22,
    CUDA_C_16U = 23,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13,
    CUDA_R_64I = 24,
    CUDA_C_64I = 25,
    CUDA_R_64U = 26,
    CUDA_C_64U = 27,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29
} cudaDataType;

typedef cudaDataType cudaDataType_t;

typedef enum libraryPropertyType_t {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL = 2
} libraryPropertyType;

typedef libraryPropertyType libraryPropertyType_t;

#endif /* OCU_LIBRARY_TYPES_H */

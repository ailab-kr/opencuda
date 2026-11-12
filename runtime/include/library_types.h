/*
 * library_types.h -- data type names shared by the math libraries
 * (cudaDataType_t and the CUDA_R_* / CUDA_C_* values).
 *
 * Written for this project from the public CUDA library-types
 * documentation (the "cudaDataType" enumeration). Values are the
 * documented ones; not derived from any vendor header.
 *
 * Whether a library function accepts a given type is that function's
 * business; naming a type here promises nothing.
 */
#ifndef OCU_LIBRARY_TYPES_H
#define OCU_LIBRARY_TYPES_H

typedef enum cudaDataType_t {
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_R_16F = 2,
    CUDA_R_8I = 3,
    CUDA_C_32F = 4,
    CUDA_C_64F = 5,
    CUDA_C_16F = 6,
    CUDA_C_8I = 7,
    CUDA_R_8U = 8,
    CUDA_C_8U = 9,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_4I = 16,
    CUDA_C_4I = 17,
    CUDA_R_4U = 18,
    CUDA_C_4U = 19,
    CUDA_R_16I = 20,
    CUDA_C_16I = 21,
    CUDA_R_16U = 22,
    CUDA_C_16U = 23,
    CUDA_R_64I = 24,
    CUDA_C_64I = 25,
    CUDA_R_64U = 26,
    CUDA_C_64U = 27,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29
} cudaDataType;
typedef enum cudaDataType_t cudaDataType_t;

#endif /* OCU_LIBRARY_TYPES_H */

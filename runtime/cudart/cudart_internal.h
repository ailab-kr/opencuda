/*
 * cudart_internal.h -- what the parts of ocu_cudart share with each other
 * (cudart.c, texture.c). Not for user programs or other layers.
 */
#ifndef OCU_CUDART_INTERNAL_H
#define OCU_CUDART_INTERNAL_H

#include "cuda_runtime_api.h"

#include <stdbool.h>
#include <stddef.h>

/* Record e as the calling thread's last error (cudaSuccess is not
   recorded); returns e. */
cudaError_t ocu_rt_set_error(cudaError_t e);

/* Bytes from p to the end of the device allocation holding it, when the
   runtime made that allocation; false otherwise. */
bool ocu_rt_alloc_avail(const void *p, size_t *avail);

/* A CUDA array seen as pitched memory (cudaMemcpy3D). False when `array`
   is not a live array of this runtime. */
typedef struct ocu_rt_array_view {
    void  *ptr;        /* element (0, 0, 0) */
    size_t pitch;      /* bytes per row */
    size_t rows;       /* rows per slice */
    size_t width;      /* elements per row */
    size_t slices;     /* slices or layers */
    size_t elem_bytes; /* bytes per element */
} ocu_rt_array_view;
bool ocu_rt_array_view_of(cudaArray_const_t array, ocu_rt_array_view *view);

#endif /* OCU_CUDART_INTERNAL_H */

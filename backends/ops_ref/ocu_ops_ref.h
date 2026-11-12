/*
 * ocu_ops_ref.h -- chip-neutral reference for the operator table
 * (docs/spec/op-table.md): descriptor checks and plain C compute.
 *
 * This is not a backend. Backends that offer the op table link it for the
 * rules every backend shares (well-formedness, operator shape rules,
 * overlap rules) and, when they compute on host-addressable memory, for
 * the arithmetic. It depends only on backends/interface.
 */
#ifndef OCU_OPS_REF_H
#define OCU_OPS_REF_H

#include "ocu_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Check a descriptor against the spec (sections 3 and 4).
 * OCU_OK, OCU_ERR_INVALID_VALUE (malformed), or OCU_ERR_NOT_IMPLEMENTED
 * for cases the reference does not compute (ROPE, QMATMUL, int8 GEMM,
 * block-quantised tensors).
 */
ocu_status ocu_ref_check_desc(const ocu_op_desc *d);

/* Bytes from the first to one past the last element of a tensor. */
uint64_t ocu_ref_span_bytes(const ocu_tensor_desc *t);

/* 1 when two descriptors are identical up to their rank. */
int ocu_ref_desc_equal(const ocu_tensor_desc *a, const ocu_tensor_desc *b);

/*
 * Check the tensors of an execute call against a planned descriptor:
 * count, descriptors (identical, or fewer rows under OCU_OP_FLAG_VAR_ROWS,
 * spec 4.7), and the overlap rules (spec 4.1-4.4). On success *actual is
 * the descriptor of this call (the planned one with the passed shapes).
 * Pointers are only compared, never dereferenced.
 */
ocu_status ocu_ref_check_tensors(const ocu_op_desc *d, const ocu_tensor *t,
                                 uint32_t n, ocu_op_desc *actual);

/* Independent work items of an operator: output rows (all dims but the
 * innermost; for GEMM batch * M). */
int64_t ocu_ref_rows(const ocu_op_desc *d);

/*
 * Compute rows [r0, r1) of the operator. data[i] is a host-addressable
 * pointer for slot i. Accumulation is in binary32 (accum_dtype f32).
 * The descriptor must have passed ocu_ref_check_desc.
 */
void ocu_ref_run(const ocu_op_desc *d, void *const *data, int64_t r0, int64_t r1);

/* Element conversions (round to nearest even). */
float    ocu_ref_f16_to_f32(uint16_t h);
uint16_t ocu_ref_f32_to_f16(float f);
float    ocu_ref_bf16_to_f32(uint16_t h);
uint16_t ocu_ref_f32_to_bf16(float f);

#ifdef __cplusplus
}
#endif

#endif /* OCU_OPS_REF_H */

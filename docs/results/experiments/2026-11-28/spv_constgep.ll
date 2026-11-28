; llc -mtriple=spirv64-unknown-unknown -O2 spv_constgep.ll -o - (text) or -filetype=obj
; The constant getelementptr adds 52 bytes (element 13 of @h). The SPIR-V
; output indexes an i32 pointer with 52 (208 bytes) for the atomic and the
; load; the store through the instruction getelementptr is right.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64-unknown-unknown"

@h = internal addrspace(3) global [14 x i32] undef, align 4

define spir_kernel void @k(ptr addrspace(1) %out, i64 %i) {
entry:
  %p = getelementptr inbounds i8, ptr addrspace(3) @h, i64 52
  store i32 0, ptr addrspace(3) %p, align 4
  %old = atomicrmw add ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @h, i64 52), i32 1 seq_cst, align 4
  %v = load i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @h, i64 52), align 4
  %s = add i32 %old, %v
  store i32 %s, ptr addrspace(1) %out, align 4
  ret void
}

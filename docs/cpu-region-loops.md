# Design note: running CPU-path kernels as loops over threads

Status: proposal, not implemented (2026-11-13). Component: CPU path (`compiler/probe`, `backends/cpu`).

## Problem

The CPU backend runs one GPU thread as one fiber (ARCHITECTURE.md section 7). That is exact and simple, but in llama.cpp's generation step a GPU thread does a few multiplications between synchronisation points, while a fiber visit costs more than that work. Measured on Qwen2.5-0.5B Q8_0, 8 OS threads (`OCU_CPU_STATS=1`, one token):

| Kernel | Share of kernel time | Shape |
|---|---|---|
| `mul_mat_vec_q` (quantised matrix-vector) | 53 % | 128 threads per row block; per thread about 7 dot products, then 1 block barrier and 5 warp shuffles |
| `mul_mat_vec_f` (f16 matrix-vector) | 18 % | same pattern |
| `quantize_q8_1` (activations) | 15 % | warp reductions; 6 blocks only |

Generation makes about 1.5e8 fiber visits per token at about 25 ns each (kernel work included). Prompt processing after the GEMM change (cards `libs-cublas`, `backend-cpu`) is led by `dequantize_block` (no barriers, 2 values per thread, about 10 ns per thread) and `soft_max` (warp reductions).

## Idea

Compile a kernel a second time in a form where a block is one function call: the body is cut at every block barrier and warp operation into regions, and each region becomes a loop over the threads of the block (warp operations: over the lanes of each warp). Values that live across a cut are kept in per-thread arrays; shared memory is unchanged. The backend calls this form for a whole block when the module has it, and keeps fibers for every kernel the compiler could not convert.

This is the technique of MCUDA (Stratton, Stone, Hwu, LCPC 2008), of pocl's "loops" work-group method for OpenCL barriers (Jääskeläinen et al., IJPP 2015; pocl is MIT licensed), and of COX (Han et al., CGO 2022) for warp-level functions (shuffles and votes become array reads between two lane loops).

Warp shuffle `x = __shfl_xor_sync(m, v, o)` at a cut: the region before writes `v` into `lane_val[lane]`; the region after reads `lane_val[lane ^ o]`. Votes reduce `lane_val` once per warp.

## Why the semantics stay the same

The fixed-order fiber scheduler already runs the block region by region: thread 0 runs until its first synchronisation point, then thread 1, and so on; nobody passes a synchronisation point until every live thread reached it. A region loop is that same schedule decided at compile time. Under independent thread scheduling CUDA lets threads of a block interleave in any way between synchronisation points, so any program correct on a GPU is correct under this one schedule. The converted form therefore cannot show a behaviour that the fixed-order fiber run does not already show, provided the conversion is applied only where every live thread reaches the same sequence of cuts. Random order (`OCU_CPU_ORDER=random`) keeps using fibers, so order-dependence bugs can still be searched for.

A thread that returns before a cut counts as arrived (as now): it is marked inactive and skipped by later region loops.

## Conditions for converting a kernel (else: fibers)

1. Every call that can synchronise (`ocu_cpu_barrier`, `ocu_cpu_warp_*`) is in the kernel body after inlining; no call to an unknown function that might synchronise.
2. Each cut is reached uniformly: its control dependences are on values that are the same for every thread of the block (kernel arguments, `blockIdx`, `blockDim`, `gridDim`, constants, and values computed only from these), with one exception: an early `return` that depends on the thread index (`if (threadIdx.y > 0) return;`), which becomes "inactive from here".
3. Loops that contain a cut have a uniform trip count (condition 2 for the loop condition).
4. Warp operations use the full mask, or a mask the compiler can prove equals the set of active lanes; otherwise fibers (partial-mask code is rare and must keep the exact checks the fiber backend does).
5. No address of a thread-private variable escapes into shared or global memory across a cut (the private copy moves to an array slot; an escaped pointer would see the wrong copy). A compile-time check; else fibers.

`mul_mat_vec_q`, `mul_mat_vec_f`, `quantize_q8_1`, `soft_max` and `dequantize_block` in llama.cpp meet these conditions as far as read (uniform loops, one barrier, unrolled `warp_reduce_sum`, an early return on `threadIdx.y`).

## Where it goes

- An LLVM pass run on the CPU-path compile (Clang already compiles the rewritten `.cu`; the pass is loaded with `-fpass-plugin`). Working on IR after inlining makes conditions 1 to 5 checkable (uniformity analysis over the def-use graph, starting from the thread-index loads). A source-level rewrite in `cpu_rewrite.py` would have to handle templates, lambdas and references by hand and is not proposed.
- The CPU module image gains an optional per-kernel "whole block" entry (image version 2); `backends/cpu` calls it instead of starting fibers. Everything stays under `compiler/` and `backends/cpu`; `runtime` and `libs` do not change.
- After conversion the region loops are ordinary loops over the thread index, so the loop vectoriser can use SIMD across threads (the element-wise kernels: `dequantize_block`, conversions, `add`, `mul`).

## Estimate

- Work: pass with uniformity analysis, region formation and context arrays, the image extension and backend call, and tests (every kernel in `tests/` run both ways with identical results, plus mutation tests for each refused condition): about 4 to 6 weeks for one person. The hard part is condition 2 and 3 on real code (loops with barriers, `#pragma unroll` leftovers, `__syncthreads` behind `if` on uniform values).
- Gain (inference from the measurements above, not measured): if the three matrix-vector and quantisation kernels drop from about 25 ns to 2 to 3 ns per thread, a generated token goes from about 680 ms to about 150 ms of kernel time, then launch cost (about 700 launches per token at roughly 0.1 ms each) becomes the next limit. Expected generation 5 to 8 tokens/s from 1.25 (8 threads); reaching the stock CPU backend (74 tokens/s) is not expected from this alone.

## Open question

How to treat a cut whose uniformity cannot be proven statically but holds at run time (for example a barrier under `if (n > 0)` where `n` is loaded from global memory: the same address for every thread, but a load is not uniform by the simple rule). Options: (a) refuse and use fibers (safe, loses some kernels); (b) treat loads from addresses that are themselves uniform, with no store to that memory earlier in the kernel, as uniform (needs alias reasoning); (c) run the converted form with a check at every cut that all active threads arrived at the same cut, and fall back to fibers for the rest of the block when they did not. Option (c) needs the converted and fiber forms to share thread state at the cut, which the context arrays make possible but complicate.

## Review decisions (2026-11-13)

A design review settled the open question and adjusted the plan.

**Uniformity.** Option (c) is the default, with a different fallback. Each region is compiled once as `next_pc = region_k(ctx, tid)`. After a region loop, the fast path checks that all live threads return the same `next_pc`; this costs one compare per thread. If they differ, the block continues in a *pc dispatcher* that calls the same region functions per thread (`pc[tid] = R[pc[tid]](ctx, tid)`), which is exactly the fixed-order fiber schedule. Both forms read and write the same `ctx`, so no state has to be shared with fibers. Static uniformity (the simple rule: kernel arguments, `blockIdx`, constants) is only an optimisation that skips the check; it is never the correctness argument. Option (b) is rejected: a value loaded from memory can differ between threads when an earlier thread in the same region wrote it. Option (a) is only a stop-gap for the first stages.

**State.** Values that cross a cut live in per-cut arrays, one array per SSA value (structure of arrays). Allocas that cross a cut get a fixed per-thread frame, so their addresses stay valid even when they escape to shared memory. This replaces condition 5. Dynamic allocas across a cut are refused.

**Forms.** One set of region functions, three drivers:
1. the uniform fast loop;
2. the pc dispatcher;
3. random order, which is the same loop over a shuffled thread order.

Random order must exercise the loop form too, not only fibers. Fibers remain only for kernels that cannot be converted. Every module keeps the fiber entry, so differential testing and run-time choice stay possible.

**Warp operations.** Shuffles and votes read the per-cut lane arrays, using the same value-selection helpers as the fiber path (out-of-range source, short last warp, width, negative source). Stage 2 handles full masks only. Partial masks are checked at run time from stage 3: if the named lanes are at the same cut with the same mask, run them; otherwise hand over to the dispatcher, which reports the existing mismatch errors.

**Refused at compile time.**
- barriers inside non-inlined or indirect calls, or recursion;
- dynamic allocas across a cut;
- calls that change floating-point control state;
- external calls that look up the current thread.

**Stages and measures.**

| Stage | Work | Decides the next step |
|---|---|---|
| 0 | Hand-write `mul_mat_vec_q` in region-loop form and measure ns per thread. Add an analysis-only pass that counts convertible kernels and the reasons for refusing the rest, over llama.cpp, `tests/` and Rodinia. | 5 ns per thread or less; otherwise revisit |
| 1 | Kernels without cuts (dequantise, element-wise), image v2 and backend plumbing | prompt tokens/s; whether the loops vectorise |
| 2 | Statically uniform straight-line and loop cuts, block barriers, full-mask warp ops | `mul_mat_vec_q/f`, `quantize_q8_1`, `soft_max` converted; generation tokens/s (median, slow 10 %) |
| 3 | Run-time pc check and dispatcher | only if the refused kernels hold more than about 5 % of time |
| 4 | Partial masks at run time, launch cost, block batching | the remaining time breakdown |

**Tests.**
- **Differential.** `OCU_CPU_FORM=fiber|loop|dispatch` × fixed and several random seeds × warp 32/64 × 1 and many OS threads must give bit-identical results. This covers all of `tests/`, test-backend-ops and greedy generation.
- **Generated kernels.** A random kernel generator builds race-free kernels with regions, cuts, uniform ifs and loops.
- **Golden lists.** Golden files record which kernels convert and which are refused, with mutants for each refusal reason.
- **Adversarial kernels:**
  - barriers under memory-loaded conditions written in an earlier or the same region;
  - per-thread loop counts;
  - reduction trees;
  - early return inside loops with barriers;
  - escaped local arrays;
  - long shuffle chains;
  - odd block shapes;
  - partial and vote-derived masks;
  - `switch` and `goto` around barriers;
  - rounding-mode changes.

**Corrections to the estimate.** Vectorisation across threads may fail on aliasing between shared and global pointers, so it is measured separately and kept out of the gain estimate. Launch cost (about 700 launches per token) bounds the gain. `quantize_q8_1` has only 6 blocks, so it cannot use more OS threads. The estimate above is rewritten after stage 0. A pass plugin ties the CPU path to an LLVM version and loads poorly on Windows; the supported range has to be stated.

OpenCuda is not affiliated with or endorsed by NVIDIA. CUDA is a trademark of NVIDIA Corporation. No warranty.

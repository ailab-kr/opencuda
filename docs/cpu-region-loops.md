# Design note: running CPU-path kernels as loops over threads

Status: proposal; stage 0 (measurement) done, see "Stage 0 results" (2026-11-13). Component: CPU path (`compiler/probe`, `backends/cpu`).

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

`mul_mat_vec_q`, `mul_mat_vec_f`, `quantize_q8_1`, `soft_max` and `dequantize_block` in llama.cpp meet these conditions as far as read (uniform loops, one barrier, unrolled `warp_reduce_sum`, an early return on `threadIdx.y`). Correction after stage 0: `mul_mat_vec_f` does not (see "Census" below).

## Where it goes

- An LLVM pass run on the CPU-path compile (Clang already compiles the rewritten `.cu`; the pass is loaded with `-fpass-plugin`). Working on IR after inlining makes conditions 1 to 5 checkable (uniformity analysis over the def-use graph, starting from the thread-index loads). A source-level rewrite in `cpu_rewrite.py` would have to handle templates, lambdas and references by hand and is not proposed.
- The CPU module image gains an optional per-kernel "whole block" entry (image version 2); `backends/cpu` calls it instead of starting fibers. Everything stays under `compiler/` and `backends/cpu`; `runtime` and `libs` do not change.
- After conversion the region loops are ordinary loops over the thread index, so the loop vectoriser can use SIMD across threads (the element-wise kernels: `dequantize_block`, conversions, `add`, `mul`).

## Estimate

- Work: pass with uniformity analysis, region formation and context arrays, the image extension and backend call, and tests (every kernel in `tests/` run both ways with identical results, plus mutation tests for each refused condition): about 4 to 6 weeks for one person. The hard part is condition 2 and 3 on real code (loops with barriers, `#pragma unroll` leftovers, `__syncthreads` behind `if` on uniform values).
- Gain: replaced by the measured estimate in "Stage 0 results" below. (The first estimate assumed 25 ns falling to 2 to 3 ns per thread and gave 5 to 8 tokens/s.)

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

## Stage 0 results (2026-11-13)

### Hand-written `mul_mat_vec_q` in region-loop form

`tuning/region_loops/` (target `bench_region_loops`, not part of the runtime). The kernel is llama.cpp's `mul_mat_vec_q` for the generation case (Q8_0 weights, Q8_1 activations, one column, no fusion, the parameters the CPU path selects: 4 warps of 32, one row per block, two ints per thread and block). It is cut into 7 regions (entry to barrier; after barrier with the `threadIdx.y > 0` return; 4 shuffle regions; last shuffle and store), each a `next_pc = region_k(ctx, tid)` function. Values across cuts live in per-cut arrays, shuffles read the cut array at `lane ^ o`. The uniform fast loop inlines each region into a loop over the live lanes of each warp, stores each thread's next pc and compares it once with the expected one (exit always accepted). The pc dispatcher runs the same region functions one thread at a time. The fiber form is the same kernel compiled by the CPU path and launched through the runtime.

Same random inputs for all forms; every output row is compared bit for bit. Result: **bit-identical** for all three shapes, every form (fiber, loop scalar, loop vectorised, loop x86-64-v3, dispatcher), 1 and 8 OS threads, GCC 13 and Clang 18 builds. Sample: `q896x896` rows 0 to 3 = -756.976196, 761.097656, -499.329376, -253.905243 in every form.

Time per launch divided by GPU threads (rows × 128), ns. 41 timed launches per run after 3 warm-ups, 3 runs; median of the run medians / median of the run 90th percentiles. 24-core x86-64 (WSL2), load average 0.7 to 2.1. "1 thread" is core time per GPU thread; "8 threads" is wall time per GPU thread.

| Shape (rows × K) | Form | GCC 1 thr | GCC 8 thr | Clang 1 thr | Clang 8 thr |
|---|---|---|---|---|---|
| 896 × 896 (attention q/o) | fiber | 62.1 / 65.0 | 8.40 / 8.82 | 68.5 / 70.6 | 9.69 / 10.4 |
| | loop, vectoriser off | 5.53 / 5.60 | 0.77 / 0.81 | 4.72 / 4.79 | 0.68 / 0.71 |
| | loop, vectoriser on | 5.49 / 5.58 | 0.77 / 0.79 | 4.66 / 4.77 | 0.65 / 0.66 |
| | loop, x86-64-v3 | 5.62 / 5.73 | 0.79 / 0.81 | 4.63 / 4.69 | 0.67 / 0.72 |
| | dispatcher | 9.54 / 9.71 | 1.35 / 1.48 | 7.73 / 7.95 | 1.09 / 1.23 |
| 4864 × 896 (FFN up/gate) | fiber | 60.4 / 62.7 | 7.90 / 8.13 | 67.3 / 70.3 | 9.15 / 9.35 |
| | loop, vectoriser off | 5.50 / 5.72 | 0.77 / 0.79 | 4.77 / 4.92 | 0.67 / 0.68 |
| | loop, vectoriser on | 5.48 / 5.58 | 0.77 / 0.86 | 4.72 / 4.83 | 0.66 / 0.68 |
| | loop, x86-64-v3 | 5.62 / 5.72 | 0.79 / 0.80 | 4.67 / 4.79 | 0.65 / 0.66 |
| | dispatcher | 9.56 / 9.73 | 1.35 / 1.43 | 7.77 / 8.00 | 1.10 / 1.23 |
| 896 × 4864 (FFN down) | fiber | 74.9 / 77.5 | 10.2 / 11.0 | 80.8 / 83.2 | 11.3 / 11.8 |
| | loop, vectoriser off | 16.3 / 17.2 | 2.25 / 2.34 | 15.4 / 16.1 | 2.13 / 2.20 |
| | loop, vectoriser on | 16.3 / 16.7 | 2.25 / 2.32 | 15.2 / 15.6 | 2.13 / 2.22 |
| | loop, x86-64-v3 | 16.4 / 16.8 | 2.26 / 2.33 | 15.4 / 15.7 | 2.15 / 2.37 |
| | dispatcher | 20.8 / 21.4 | 2.95 / 3.07 | 18.7 / 19.3 | 2.61 / 2.71 |

Fixed launch cost (empty launch and synchronise, median / slow 10 %): runtime 75 to 120 / 126 to 296 µs with 1 OS thread, 54 to 77 / 84 to 110 µs with 8; the benchmark's own pool for the loop forms 0.7 to 0.8 µs. The loop numbers above therefore do not contain the runtime's launch cost; the fiber numbers do (it is under 1 % of these launches).

Reading:
- Most of the loop form's time is the thread's own work. With K = 896 a thread does 0.875 Q8_0 blocks on average, with K = 4864 4.75; the difference gives about 2.9 ns (GCC) / 2.8 ns (Clang) per block, so the cost of the region form itself (region loops, cut arrays, pc check, the 5 shuffle regions) is about 3 ns (GCC) / 2.2 ns (Clang) per thread. The fiber form costs about 55 to 65 ns more per thread.
- Vectorisation: no gain. GCC vectorised none of the region loops ("control flow in loop", "loop nest containing two or more consecutive inner loops"). Clang vectorised the loop of region 1 (sum of the shared partial sums) and the partial-warp loops, not region 0, whose per-thread loop over blocks makes the lane loop an outer loop (needs outer-loop vectorisation), and not the shuffle regions ("cannot identify array bounds": the read at `lane ^ o` of a cut array next to the write of the next one). A pass that emits shuffles as known permutations, and an outer-loop vectoriser, would be needed; neither is in this estimate.
- x86-64-v3 is built with `-ffp-contract=off`: with contraction the `tmp += d0 * d1 * sumi` step could become a fused multiply-add and the bits would differ from the fiber form compiled for baseline x86-64.
- The dispatcher (fallback schedule) is 1.4 to 1.7 times the fast loop on these shapes and still 6 to 8 times faster than fibers.

### Generation time by kernel (`OCU_CPU_STATS=1`)

Qwen2.5-0.5B Q8_0, `llama-bench -p 0 -n 16 -r 1 -t 8`, `OCU_CPU_THREADS=8` (17 tokens with the warm-up; 1.26 tokens/s). 11,917 launches = 701 per token; 13,379 ms in kernels = 787 ms per token.

| Kernel | Share | ms/token | Launches/token | Threads/block |
|---|---|---|---|---|
| `mul_mat_vec_q` (Q8_0, 1 column, no fusion) | 51.4 % | 404.7 | 169 | 128 |
| `mul_mat_vec_f` (f16 KV cache, attention) | 17.3 % | 136.3 | 48 | 128 |
| `quantize_q8_1` | 16.8 % | 132.4 | 169 | 256 |
| `rms_norm_f32` (one block per launch) | 4.0 % | 31.6 | 49 | 256 |
| `soft_max_f32` | 3.6 % | 28.0 | 24 | 256 |
| `swiglu` (by signature) | 2.1 % | 16.5 | 24 | 256 |
| `rope` (f32) | 1.9 % | 15.2 | 24 | 256 |
| `cpy_scalar` (copy into the KV cache) | 1.3 % | 10.2 | 24 | 64 |
| `rope` (to f16) | 0.8 % | 6.3 | 24 | 256 |
| `k_bin_bcast` (add) | 0.6 % | 5.0 | 120 | 128 |
| others | 0.1 % | 0.8 | 26 | |

### Census of convertible kernels (analysis only)

`tools/region_census.py` compiles each `.cu` to device LLVM IR with the nvidia target's device step at sm_50 (so `__CUDA_ARCH__` is 500 as on the CPU path) and applies the stage-2 rules to the `-O2` IR: control dependence from post-dominators, natural loops, a forward uniformity analysis (uniform: kernel arguments, block index and size, constants, loads through by-value parameters or from constant memory; every other load, atomics, warp-op results and calls are not), full constant masks, no dynamic allocas, no floating-point control changes. Kernels without a barrier or warp operation count as "no cuts" (stage 1). Commands and limits are in the script's header.

| Corpus | Files (failed to compile) | Kernels | No cuts | Convertible | Refused | Main refusal reasons |
|---|---|---|---|---|---|---|
| llama.cpp `ggml-cuda` (no flash attention; every instantiation) | 74 (0) | 2178 | 1092 (50 %) | 710 (33 %) | 376 (17 %) | divergent branch before a cut 325; divergent loop exit or trip count around a cut 41; barrier in a call 10 |
| `tests/` | 29 (1) | 410 | 159 (39 %) | 202 (49 %) | 49 (12 %) | computed (partial) mask 26; divergent branch or loop 23 (segmented sort whose loop bound is loaded from memory) |
| Rodinia (programs our path builds; kmeans missing) | 52 (17) | 64 | 17 (27 %) | 34 (53 %) | 13 (20 %) | barrier in a non-inlined call 12 (dwt2d); early return inside a loop with a barrier 1 (lavaMD) |

Weighted by generation time (table above):

| Kernel | Share | Verdict |
|---|---|---|
| `mul_mat_vec_q` | 51.4 % | convertible (all 271 instantiations) |
| `mul_mat_vec_f` | 17.3 % | **refused**: after a barrier, `if (tid < warp_size) { warp_reduce_sum }`, then another barrier. The condition is the same within a warp but not within the block, and the other threads go on to the next barrier, so the early-return exception does not apply. The note's claim above that it meets the conditions was wrong. |
| `quantize_q8_1` | 16.8 % | convertible |
| `rms_norm_f32` | 4.0 % | convertible |
| `soft_max_f32` | 3.6 % | convertible |
| `swiglu`, `rope`, `cpy_scalar`, `k_bin_bcast`, `k_set_rows` | 6.8 % | no cuts (stage 1) |
| others | 0.1 % | not resolved |

By time: 75.8 % convertible at stage 2, 6.8 % without cuts, 17.3 % refused. By count, of the 11 kernels that run in generation: 4 convertible, 6 without cuts, 1 refused.

Limits: the census reads NVPTX IR of our headers, not the CPU-path IR the pass will see (inlining differs); each template instantiation counts once, and most never run; it over-counts refusals where a branch is uniform per warp or a loaded value is uniform at run time (the stage-3 check accepts both), and under-counts them for allocas (no liveness check) and for code that sm_50 compiles as a stub. Early returns inside a loop that contains a barrier are refused (37 llama.cpp kernels, mostly `mul_mat_q`).

### Gain estimate from the measurements

Per generated token, 8 OS threads, ms. For `mul_mat_vec_q` the ratio is the benchmark's fiber time (GCC, as the CPU path builds today) over its loop time, weighted by the token's shapes (434,560 rows with K = 896 including the output layer, 21,504 with K = 4864): 9.7 with the GCC loop, 11.1 with the Clang loop. The other convertible kernels were not converted by hand; for them a ratio of 4.5 (the `mul_mat_vec_q` shape with the most work per thread) to 9.7 is assumed. The runtime's launch cost (about 60 µs per launch with 8 threads, measured above) stays for every launch: 701 launches are about 42 ms per token.

| Part | Now | After stage 2 | If `mul_mat_vec_f` converts too |
|---|---|---|---|
| `mul_mat_vec_q` (169 launches) | 404.7 | 46 to 51 (measured ratio) | same |
| `quantize_q8_1` (169) | 132.4 | 23 to 37 (assumed ratio) | same |
| `rms_norm_f32`, `soft_max_f32` (73) | 59.6 | 10 to 17 (assumed ratio) | same |
| `mul_mat_vec_f` (48) | 136.3 | 136.3 (refused) | 17 to 33 (assumed ratio) |
| kernels without cuts, others (242) | 54.8 | 54.8 (stage 1 not counted) | 54.8 |
| outside kernels | about 7 | 7 | 7 |
| **Token** | **794 (1.26 t/s)** | **277 to 303 (3.3 to 3.6 t/s)** | **158 to 200 (5.0 to 6.3 t/s)** |

- The launch path then holds 42 ms of a 160 to 300 ms token (15 to 25 %), and `quantize_q8_1` cannot use more than 6 OS threads (6 blocks).
- Not counted: vectorisation across threads (no gain measured) and stage 1 on the kernels without cuts (at most about 40 of their 55 ms). The stock CPU backend's 74 t/s is out of reach of this design alone.
- The first estimate (5 to 8 t/s) assumed that `mul_mat_vec_f` converts; at stage 2 as defined it does not.

### Decision for stages 1 and 2

Gate: 5 ns per thread or less for the loop form of `mul_mat_vec_q`. Measured with one OS thread: 4.7 ns (Clang) / 5.5 ns (GCC) for the K = 896 shapes, which hold 95 % of the kernel's threads in a token; 15 to 16 ns for K = 4864, where a thread does 5 times the work. The region form itself costs about 2 to 3 ns per thread, against about 60 ns for a fiber; the rest is the thread's own arithmetic. Token-weighted: 5.2 ns (Clang), 6.0 ns (GCC).

Recommendation: proceed with stages 1 and 2, with three changes to the plan.
1. Compile converted modules with Clang (the pass is an LLVM plugin anyway): the gate is met with Clang and missed by 10 % with GCC.
2. Add warp-level uniformity to stage 2: for a warp operation, a condition that is the same within each warp (`threadIdx.x / 32` with warp-aligned block shapes) is enough; the fast check compares the next pc per warp, and warps that skip the warp operation wait at the next block barrier. Without it `mul_mat_vec_f` (17 % of generation) stays on fibers and stage 2 gives 3.3 to 3.6 t/s instead of 5 to 6.
3. Bring the launch-cost work (stage 4) forward or run it alongside: after stage 2 it is the largest fixed cost left.

OpenCuda is not affiliated with or endorsed by NVIDIA. CUDA is a trademark of NVIDIA Corporation. No warranty.

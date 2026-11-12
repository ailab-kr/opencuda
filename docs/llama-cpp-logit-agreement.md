# llama.cpp through OpenCuda: why greedy output differs from the stock CPU backend

Date: 2026-11-13 · Subject: llama.cpp `0fb8f15` (MIT), CUDA backend built unmodified through OpenCuda (`cards/apps/llama-cpp.md`), CPU path and NVIDIA path, against llama.cpp's own CPU backend (gcc, `GGML_NATIVE=ON`) · Models: Qwen2.5-0.5B-Instruct Q8_0 and Q4_K_M · Tools: `tools/llama-logits/`.

## Question

Greedy 64-token generation (prompt "Write a short paragraph about the history of the city of Paris, …") agrees with the stock CPU backend on 47–48 of 64 tokens, while our CPU path and NVIDIA path agree with each other on 63/64. Is that ordinary floating-point divergence, or a defect in one of our operators?

## Answer

Ordinary divergence. With teacher forcing against an exact float32 oracle of the same weights, our builds are as close to the oracle as the stock CPU backend, and in the batch shape about 4 orders of magnitude closer. Every top-1 disagreement, for every build including the stock one, is at a near tie (oracle top-1/top-2 logit margin at most 0.27). Greedy divergence comes from where those near ties fall, not from a wrong operator.

## Method

- **Teacher forcing.** `llama_logits` feeds one fixed token sequence and writes float32 logits for every position. Two shapes: *batch* (the whole text in one decode call: llama.cpp's prompt path; at compute capability 5.0 this is dequantize + float32 GEMM) and *seq* (8 tokens as a batch, then one token per call: the generation path, matrix-vector kernels that quantize activations to 8 bits).
- **Oracle.** The same GGUF dequantized to F32 (`llama-quantize --allow-requantize … F32`, exact for Q8_0) and run on the stock CPU backend in batch shape: the float32 answer for these weights without activation quantization.
- **Metric.** KL(oracle ‖ test) of the softmax per position (float64), top-1 agreement, and the oracle margin at every top-1 flip.
- **Texts.** `paris` (prompt + the stock backend's 64 generated tokens, 90 tokens), `license` (llama.cpp LICENSE, 228), `code` (first 1500 bytes of `examples/simple/simple.cpp`, 404), `ko` (a Korean paragraph, 144).

## Results (KL against the F32 oracle)

Q8_0 (`b-` batch shape, `s-` seq shape; `ref` = stock CPU backend, `cpu` / `nv` = our two paths):

| dump | KL mean | KL p99 | KL max | top-1 | flips with margin ≥ 0.5 | largest flip margin |
|---|---|---|---|---|---|---|
| paris-b-ref | 1.88e-03 | 1.07e-02 | 4.95e-02 | 89/90 | 0 | 0.135 |
| paris-b-cpu | 2.79e-07 | 1.31e-06 | 1.35e-06 | 90/90 | 0 | – |
| paris-b-nv | 2.79e-07 | 1.17e-06 | 2.96e-06 | 90/90 | 0 | – |
| paris-s-ref | 1.81e-03 | 1.11e-02 | 4.95e-02 | 90/90 | 0 | – |
| paris-s-cpu | 5.08e-04 | 2.95e-03 | 3.69e-03 | 89/90 | 0 | 0.057 |
| paris-s-nv | 4.81e-04 | 2.52e-03 | 2.97e-03 | 88/90 | 0 | 0.069 |
| license-b-ref | 3.73e-04 | 4.50e-03 | 3.42e-02 | 227/228 | 0 | 0.042 |
| license-b-cpu | 5.07e-08 | 1.06e-06 | 1.42e-06 | 228/228 | 0 | – |
| license-b-nv | 4.19e-08 | 9.83e-07 | 1.13e-06 | 228/228 | 0 | – |
| license-s-ref | 4.63e-04 | 8.84e-03 | 3.42e-02 | 227/228 | 0 | 0.040 |
| license-s-cpu | 4.18e-04 | 6.42e-03 | 3.51e-02 | 228/228 | 0 | – |
| license-s-nv | 4.11e-04 | 6.97e-03 | 3.36e-02 | 227/228 | 0 | 0.042 |
| code-b-ref | 1.12e-03 | 5.95e-03 | 2.81e-02 | 398/404 | 0 | 0.103 |
| code-b-cpu | 5.95e-07 | 8.84e-06 | 2.31e-05 | 404/404 | 0 | – |
| code-b-nv | 4.33e-07 | 6.60e-06 | 1.50e-05 | 404/404 | 0 | – |
| ko-b-ref | 2.96e-03 | 1.02e-02 | 6.76e-02 | 140/144 | 0 | 0.121 |
| ko-b-cpu | 2.87e-07 | 1.57e-06 | 2.47e-06 | 144/144 | 0 | – |
| ko-b-nv | 3.17e-07 | 2.46e-06 | 5.35e-06 | 144/144 | 0 | – |
| ko-s-ref | 2.91e-03 | 1.37e-02 | 6.76e-02 | 135/144 | 0 | 0.173 |
| ko-s-cpu | 3.36e-03 | 2.10e-02 | 7.00e-02 | 137/144 | 0 | 0.181 |
| ko-s-nv | 3.17e-03 | 1.37e-02 | 7.00e-02 | 133/144 | 0 | 0.181 |

Q4_K_M (oracle: the Q4_K_M weights dequantized to F32; CPU path seq shape not run, it takes ~1.3 s per token):

| dump | KL mean | KL p99 | KL max | top-1 | flips with margin ≥ 0.5 | largest flip margin |
|---|---|---|---|---|---|---|
| license-b-ref | 6.99e-04 | 1.14e-02 | 4.70e-02 | 227/228 | 0 | 0.053 |
| license-b-cpu | 1.99e-07 | 2.97e-06 | 1.78e-05 | 228/228 | 0 | – |
| license-b-nv | 9.87e-08 | 1.48e-06 | 6.26e-06 | 228/228 | 0 | – |
| license-s-ref | 7.68e-04 | 8.36e-03 | 4.70e-02 | 228/228 | 0 | – |
| license-s-nv | 5.54e-04 | 7.44e-03 | 4.61e-02 | 227/228 | 0 | 0.053 |
| code-b-ref | 1.59e-03 | 9.96e-03 | 2.55e-02 | 395/404 | 0 | 0.269 |
| code-b-cpu | 4.42e-07 | 4.64e-06 | 1.18e-05 | 403/404 | 0 | 0.000 (exact tie) |
| code-b-nv | 4.41e-07 | 4.64e-06 | 1.12e-05 | 403/404 | 0 | 0.000 (exact tie) |
| code-s-ref | 1.75e-03 | 1.16e-02 | 2.55e-02 | 396/404 | 0 | 0.234 |
| code-s-nv | 1.21e-03 | 7.82e-03 | 3.00e-02 | 396/404 | 0 | 0.234 |
| ko-b-ref | 5.38e-03 | 2.75e-02 | 3.64e-02 | 139/144 | 0 | 0.107 |
| ko-b-nv | 3.31e-07 | 1.52e-06 | 2.29e-06 | 144/144 | 0 | – |
| ko-s-ref | 5.62e-03 | 3.01e-02 | 5.35e-02 | 139/144 | 0 | 0.107 |
| ko-s-nv | 3.07e-03 | 1.20e-02 | 4.01e-02 | 137/144 | 0 | 0.107 |

Reading:

- Batch shape: our two paths sit at 4e-8 – 6e-7 mean KL, the level of float32 summation order (the oracle's own batch-vs-seq difference is 2.9e-7). The stock backend is at 4e-4 – 5e-3 because it quantizes activations to 8 bits in every matrix product, also in batches.
- Seq shape: all three builds quantize activations (stock: q8_0 blocks, CUDA kernels: q8_1 blocks) and land in the same band, 4e-4 – 6e-3 mean KL; ours is lower in 7 of 9 cases and at most 1.16× higher in the other 2 (`ko`, Q8_0).
- Position 0 (only one token visible; the first token carries very large activations in this model) is the stock backend's worst position (KL 4.95e-2 on `paris`); our batch shape has 1e-6 there.

## Natural tolerance (stock backend against itself)

| comparison | KL mean | top-1 |
|---|---|---|
| stock, 8 threads vs 1 thread (seq) | bit-identical | 90/90 |
| stock, batch vs seq shape, Q8_0 paris / license / ko | 1.12e-03 / 2.31e-04 / 2.28e-03 | 89/90, 226/228, 135/144 |
| stock, batch vs seq shape, Q4_K_M license / code / ko | 4.67e-04 / 1.95e-03 / 4.90e-03 | 227/228, 395/404, 136/144 |
| ours, CPU path vs NVIDIA path, seq shape, Q8_0 paris / license / ko | 8.20e-04 / 1.78e-04 / 2.18e-03 | 87/90, 227/228, 134/144 |
| ours, CPU path vs NVIDIA path, batch shape, Q8_0 (4 texts) | 3.8e-08 – 5.6e-07 | all agree |

The same stock binary, just changing how many tokens go into one call, disagrees with itself on 1–6 % of top-1 choices. That is the scale of the 47/64 greedy result.

## Op-by-op check (seq shape, position 30 of `paris`, Q8_0)

`llama_logits --dump-pos 30` writes every float32 graph tensor of that one-token decode call; `cmp_dump.py` compares them in graph order.

CPU path vs NVIDIA path, layer 0: every MUL_MAT output and the RMS_NORM, MUL, ADD outputs are bit-identical. The first nonzero differences are ROPE (max relative 2.4e-8, sin/cos last-bit), the KQ product (1.95e-7, summation order), SOFT_MAX (3.0e-6, the KQ difference passed through exp with logits of that size) and KQV (4.7e-6). From layer 1 on, differences grow to ~1e-4 – 2e-3 relative, because a last-bit change before the 8-bit activation quantization or the F16 KV cache moves a value by a whole quantization step. Final logits: max relative 1.5e-2, NMSE 1.3e-4.

Stock vs our CPU path, layer 0: Q/K/V products already differ at 4e-5 – 4e-4 relative (q8_0 vs q8_1 activation blocks, different kernels), the FFN at ~1e-2 relative; the stock backend also uses its fused flash-attention op (FA is off in our build). No op stands out from this pattern.

Side note: the tensor named `node_1` (GET_ROWS, 896 values, followed by DUP and SET_ROWS) dumps inconsistent content in all three builds, including the stock one, and its DUP output is all zeros in each; it does not feed the logits. Treated as a dump artifact of that side graph, not followed further.

## Sample outputs (64-token greedy)

- Stock CPU backend, Q8_0: "… The city of Paris has a rich history that dates back to the 6th century. The city was founded by the Romans and was named after the city of Paris, which was the capital of the Roman Empire. The city was later conquered by the French and became the capital of France in 843. During"
- F32 oracle (stock backend, dequantized weights): identical to the line above, all 64 tokens.
- Our CPU path: "… dates back to the 6th century. The city was founded by the Romans, who built a city on the site of an ancient Roman settlement. The city was named after the French king Louis IX, …" Teacher-forced on the oracle sequence, its only top-1 flip is at position 50 ("," vs " and"), oracle margin 0.057 logits.
- Our NVIDIA path: "… dates back to the Middle Ages. The city was founded by the Romans in the 6th century, …" Flips at positions 38 and 82, oracle margins 0.069 and 0.032.

The stock Q8_0 backend happens to reproduce the oracle's greedy text here; on the same text in batch shape it flips at position 32 with a larger margin (0.135), and on `ko` / `code` it flips 4–9 times. Matching the oracle's 64 greedy tokens is luck at near ties, for every build.

## Proposed conformance metric for applications (L5)

Greedy token agreement is not a usable metric: one near tie decides the rest of the text. Proposed instead:

1. **Oracle:** the model's weights dequantized to F32, run on the stock CPU backend, whole text in one call.
2. **Inputs:** at least 3 texts (prose, code, a non-Latin script), at least 2 weight types (one 8-bit, one K-quant), both shapes (batch; seq = short prefix then one token per call).
3. **Per (model, text, shape), pass if all hold:**
   - mean KL(oracle ‖ test) ≤ max(2 × the stock CPU backend's mean KL for the same model, text and shape, 1e-4);
   - p99 KL ≤ max(2 × the stock p99, 1e-3);
   - no top-1 disagreement where the oracle's top-1/top-2 margin is ≥ 0.5 logits (largest flip margin observed above: 0.27).
4. **Same-source check (catches small operator defects that the band above would hide):** the same CUDA source built for two OpenCuda backends, batch shape: mean KL between them ≤ 1e-5 (observed 3.8e-8 – 5.6e-7).

Today all 35 rows above pass 3, and the 4 Q8_0 texts pass 4.

## Reproducing

```
tools/llama-logits/build.sh <llama.cpp src> <llama.cpp build dir> <out binary>   # reuses llama-simple's link line
tools/llama-logits/run.sh q8-license-b <model.gguf> license.txt all ref=ll-ref cpu=ll-cpu nv=ll-nv
tools/llama-logits/run.sh q8-license-s <model.gguf> license.txt 8   ref=ll-ref cpu=ll-cpu nv=ll-nv
tools/llama-logits/run.sh q8-license-f32 <model-f32.gguf> license.txt all oracle=ll-ref
python3 tools/llama-logits/table.py q8 license        # needs numpy
python3 tools/llama-logits/cmp_dump.py dumpA dumpB    # after llama_logits --dump-pos N --dump-dir dumpX
```

Wall time on this PC (one run each, other jobs running): Q8_0 `paris` seq shape 90 tokens, CPU path 51 s (64 one-token calls), stock backend 1.4 s. The CPU path in seq shape is the slow part (~1.3 s per token).

OpenCuda is not affiliated with or endorsed by NVIDIA. CUDA is a trademark of NVIDIA Corporation. No warranty.

#!/usr/bin/env python3
"""Tiny GPT-2 setup for the llm.c training programs (fixed seed, no download).

    gen.py <dir>

Writes into <dir>, under the names the programs open from their working
directory:
  gpt2_124M.bin                  a 2-layer, 4-head, 128-channel model with
                                 GPT-2's vocabulary (50257, padded 50304) and
                                 randomly initialised weights (std 0.02, layer
                                 norm weight 1, biases 0), in the checkpoint
                                 format both train_gpt2.c and
                                 train_gpt2_fp32.cu read;
  dev/data/tinyshakespeare/tiny_shakespeare_{train,val}.bin
                                 token files in the data format the programs'
                                 loaders read: a periodic pattern of 256
                                 random tokens, so every batch of 4 x 64 is the same whatever order the
                                 loader picks them in (the two programs shuffle with
                                 their own seeds) and a model can learn it.
The sizes are small on purpose (about 7 million parameters, 8 training
batches of 4 x 64 tokens); the real 124M checkpoint and the text data are
downloads of hundreds of MB and are not used.
"""
import os
import random
import struct
import sys

out = sys.argv[1]
rnd = random.Random(20261113)

V, VP, MAXT, L, NH, C = 50257, 50304, 128, 2, 4, 128  # the matmul kernel works in 128-wide tiles


def normal(n):
    return [rnd.gauss(0.0, 0.02) for _ in range(n)]


def const(n, v):
    return [v] * n


# The rows of the padded vocabulary (V..VP) are zero, as in the real
# checkpoint: the CUDA program treats them as logits and trains on them.
tensors = [
    normal(V * C) + const((VP - V) * C, 0.0),  # wte
    normal(MAXT * C),          # wpe
    const(L * C, 1.0),         # ln1w
    const(L * C, 0.0),         # ln1b
    normal(L * 3 * C * C),     # qkvw
    const(L * 3 * C, 0.0),     # qkvb
    normal(L * C * C),         # attprojw
    const(L * C, 0.0),         # attprojb
    const(L * C, 1.0),         # ln2w
    const(L * C, 0.0),         # ln2b
    normal(L * 4 * C * C),     # fcw
    const(L * 4 * C, 0.0),     # fcb
    normal(L * C * 4 * C),     # fcprojw
    const(L * C, 0.0),         # fcprojb
    const(C, 1.0),             # lnfw
    const(C, 0.0),             # lnfb
]

os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "gpt2_124M.bin"), "wb") as f:
    header = [0] * 256
    header[0], header[1] = 20240326, 3
    header[2:8] = [MAXT, V, L, NH, C, VP]
    f.write(struct.pack("<256i", *header))
    for t in tensors:
        f.write(struct.pack(f"<{len(t)}f", *t))

pattern = [rnd.randrange(0, V - 1) for _ in range(256)]


def tokens(n):
    return [pattern[i % len(pattern)] for i in range(n)]


data = os.path.join(out, "dev", "data", "tinyshakespeare")
os.makedirs(data, exist_ok=True)
for name, n in (("train", 8 * 4 * 64 + 1), ("val", 6 * 4 * 64 + 1)):
    with open(os.path.join(data, f"tiny_shakespeare_{name}.bin"), "wb") as f:
        header = [0] * 256
        header[0], header[1], header[2] = 20240520, 1, n
        f.write(struct.pack("<256i", *header))
        f.write(struct.pack(f"<{n}H", *tokens(n)))

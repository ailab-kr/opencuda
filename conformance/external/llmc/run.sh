#!/bin/bash
# Build llm.c's train_gpt2_fp32.cu, unmodified, with ocu-nvcc on the CPU path
# and train a tiny GPT-2 for 8 steps; compare the losses with the same
# project's plain C trainer (train_gpt2.c, built with gcc) on the same
# weights and data.
#
#   conformance/external/llmc/run.sh <cmake build dir> [work dir]
#
# No downloads except the llm.c sources: gen.py writes a small checkpoint
# and token files (see its header). Nothing is written inside the repository.
#
# Rows printed:
#   as is        the unmodified program.
#   race fixed   a copy with one __syncthreads() added after thread 0 reads
#                the target logit in fused_classifier_kernel3 (see below).
#
# Known, from the program: fused_classifier_kernel3 lets thread 0 read
# logits[idx * P + ix] for the loss while the other threads of the block
# overwrite the same array with the logit gradients; nothing orders the two.
# A GPU usually runs thread 0's read first, the CPU backend's fibers may not
# (the outcome depends on the order, OCU_CPU_ORDER=random changes it): the
# reported loss is then wrong for some rows (about 18 % of rows in the fixed
# order here) while the gradients, and so the training, are unaffected.
# With the barrier the losses equal the C trainer's to 6 digits.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=$(cd "$1" && pwd)
WORK=${2:-$HOME/.cache/ocu-llmc}
PIN=f1e2ace   # github.com/karpathy/llm.c
NVCC=$BUILD/toolkit/bin/ocu-nvcc
mkdir -p "$WORK"
if [ ! -d "$WORK/llm.c/.git" ]; then
    git clone -q https://github.com/karpathy/llm.c.git "$WORK/llm.c" || exit 2
fi
git -C "$WORK/llm.c" checkout -q "$PIN" 2>/dev/null || true
L=$WORK/llm.c
mkdir -p "$WORK/run" "$WORK/patched"
python3 "$HERE/gen.py" "$WORK/run" || exit 2

# the plain C trainer: reference
gcc -O2 -I "$L" "$L/train_gpt2.c" -lm -o "$WORK/ref_c" 2> "$WORK/ref_c.log" || { echo "reference did not build"; exit 2; }
# the CUDA trainer, as is
"$NVCC" -O3 -I "$L" "$L/train_gpt2_fp32.cu" -lcublas -o "$WORK/train_fp32" > "$WORK/train_fp32.build.log" 2>&1 || { echo "train_gpt2_fp32.cu: build FAILED"; tail -3 "$WORK/train_fp32.build.log"; exit 1; }
echo "train_gpt2_fp32.cu: built"
# a copy with the race ordered
python3 - "$L/train_gpt2_fp32.cu" "$WORK/patched/train_gpt2_fp32.cu" <<'PY'
import sys
s = open(sys.argv[1]).read()
old = """        losses[idx] = -logf(prob);
    }
"""
assert old in s
open(sys.argv[2], "w").write(s.replace(old, old + "    __syncthreads();\n", 1))
PY
"$NVCC" -O3 -I "$L" "$WORK/patched/train_gpt2_fp32.cu" -lcublas -o "$WORK/train_fp32_fixed" > "$WORK/train_fp32_fixed.build.log" 2>&1 || { echo "patched copy: build FAILED"; exit 1; }

cd "$WORK/run" || exit 2
ARGS="-b 4 -t 64 -v 100 -l 1e-4 -s 100000 -g 8"
timeout 300 stdbuf -oL "$WORK/ref_c" 2>&1 | grep -E '^step' | head -8 | sed 's/.*train loss \([0-9.]*\).*/\1/' > "$WORK/loss.ref"
"$WORK/train_fp32" $ARGS 2>&1 | grep -E '^step' | sed 's/.*train loss \([0-9.]*\).*/\1/' > "$WORK/loss.asis"
"$WORK/train_fp32_fixed" $ARGS 2>&1 | grep -E '^step' | sed 's/.*train loss \([0-9.]*\).*/\1/' > "$WORK/loss.fixed"
echo "step  C reference   as is     race fixed"
paste "$WORK/loss.ref" "$WORK/loss.asis" "$WORK/loss.fixed" | awk '{printf "%4d  %-12s %-12s %-12s\n", NR, $1, $2, $3}'
python3 - "$WORK/loss.ref" "$WORK/loss.asis" "$WORK/loss.fixed" <<'PY'
import sys
r, a, f = ([float(x) for x in open(p).read().split()] for p in sys.argv[1:4])
n = min(len(r), len(a), len(f))
print(f"steps compared: {n}; max |diff| to the C trainer: as is {max(abs(r[i]-a[i]) for i in range(n)):.6f}, race fixed {max(abs(r[i]-f[i]) for i in range(n)):.6f}")
PY

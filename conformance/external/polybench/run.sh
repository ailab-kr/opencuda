#!/bin/bash
# Build the public PolyBench-ACC CUDA programs, unmodified, with ocu-nvcc on
# the CPU path and run them. Each program computes the result on the
# "GPU" and with its own serial CPU code and prints how many elements differ
# beyond its threshold; that line is the check.
#
#   conformance/external/polybench/run.sh <cmake build dir> [work dir] [dataset]
#
# <build dir>: a CMake build of this repository (its toolkit/ folder holds
# ocu-nvcc). [work dir]: where the sources are fetched (default
# ~/.cache/ocu-polybench); nothing is written inside the repository.
# [dataset]: MINI | SMALL (default) | STANDARD.
set -u
BUILD=$(cd "$1" && pwd)
WORK=${2:-$HOME/.cache/ocu-polybench}
DS=${3:-SMALL}
PIN=70ea4ca   # github.com/cavazos-lab/PolyBench-ACC
NVCC=$BUILD/toolkit/bin/ocu-nvcc
mkdir -p "$WORK"
if [ ! -d "$WORK/PolyBench-ACC/.git" ]; then
    git clone -q https://github.com/cavazos-lab/PolyBench-ACC.git "$WORK/PolyBench-ACC" || exit 2
fi
git -C "$WORK/PolyBench-ACC" checkout -q "$PIN" 2>/dev/null || true
P=$WORK/PolyBench-ACC/CUDA
OUT=$WORK/out-$DS; mkdir -p "$OUT"
FLAGS=
[ "$DS" = STANDARD ] || FLAGS="-D${DS}_DATASET"
total=0; built=0; ran=0; match=0
printf '%-14s %-6s %-6s %s\n' program built ran check
for src in $(cd "$P" && find . -name '*.cu' | sort); do
    d=$P/$(dirname "$src"); f=$(basename "$src" .cu)
    exe=$OUT/$f.exe; log=$OUT/$f.log
    total=$((total + 1))
    if ! (cd "$d" && "$NVCC" -O3 $FLAGS "$f.cu" -I"$P/utilities" -o "$exe") > "$OUT/$f.build.log" 2>&1; then
        printf '%-14s %-6s %-6s %s\n' "$f" no - "$(grep -m1 -i 'error' "$OUT/$f.build.log" | cut -c1-100)"
        continue
    fi
    built=$((built + 1))
    if ! (cd "$d" && timeout 600 "$exe") > "$log" 2>&1; then
        printf '%-14s %-6s %-6s %s\n' "$f" yes no "$(tail -1 "$log" | cut -c1-100)"
        continue
    fi
    ran=$((ran + 1))
    res=$(grep -hE 'Non-Matching|Number of misses' "$log" | sed 's/.*: //' | tr '\n' ' ')
    if [ -z "$res" ]; then
        printf '%-14s %-6s %-6s %s\n' "$f" yes yes "no check line"
    elif echo "$res" | grep -qv '^[0 ]*$'; then
        printf '%-14s %-6s %-6s %s\n' "$f" yes yes "MISMATCH $res"
    else
        match=$((match + 1))
        printf '%-14s %-6s %-6s %s\n' "$f" yes yes "match (mismatches: $res)"
    fi
done
echo "summary[$DS]: $total programs, $built built, $ran ran, $match matched"
# Known, from the program: mvt launches 32x8 blocks but indexes only by
# threadIdx.x, so 8 threads add into every element of x1/x2. A GPU that runs
# those warps in lockstep hides the race; the CPU backend runs them one after
# the other and x1/x2 come out 8 times the serial value (4094 elements
# differ). With DIM_THREAD_BLOCK_Y=1 in a scratch copy it matches.

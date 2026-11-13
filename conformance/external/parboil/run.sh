#!/bin/bash
# Build the public Parboil CUDA programs, unmodified, with ocu-nvcc on the
# CPU path, and run those that can be given small generated inputs.
#
#   conformance/external/parboil/run.sh <cmake build dir> [work dir]
#
# Parboil's datasets are not part of the source tree, so inputs here come
# from gen.py (fixed seeds, a few KiB to a few MiB). The check is the suite's
# own CPU version of the same program (src/cpu or src/base, which is part of
# the suite) run on the same input; outputs are compared by cmp.py with a
# relative tolerance. Programs without a usable CPU version, or whose input
# format gen.py does not produce, are built only. Nothing is written inside
# the repository.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=$(cd "$1" && pwd)
WORK=${2:-$HOME/.cache/ocu-parboil}
PIN=ccf3d31   # github.com/yuhc/gpu-parboil
NVCC=$BUILD/toolkit/bin/ocu-nvcc
mkdir -p "$WORK"
if [ ! -d "$WORK/parboil/.git" ]; then
    git clone -q https://github.com/yuhc/gpu-parboil.git "$WORK/parboil" || exit 2
fi
git -C "$WORK/parboil" checkout -q "$PIN" 2>/dev/null || true
P=$WORK/parboil
B=$P/benchmarks
OUT=$WORK/out; mkdir -p "$OUT/in"
COM=$P/common
FLAGS="-O3 -I$COM/include -Xcompiler -Wno-c++11-narrowing -Xcompiler -fno-strict-return"

# The translation units of each program, as its Makefile lists them (the
# other .cu files are #included by these).
sources() {
    case $1 in
        bfs) echo main.cu ;;
        cutcp) echo main.c readatom.c output.c excl.c cutcpu.c cutoff6overlap.cu ;;
        histo) echo main.cu histo_final.cu histo_intermediates.cu histo_main.cu histo_prescan.cu util.cu ;;
        lbm) echo main.cc lbm.cu ;;
        mri-gridding) echo CPU_kernels.c CUDA_interface.cu scanLargeArray.cu sort.cu main.cu ;;
        mri-q) echo file.cc main.cu ;;
        sad) echo file.c image.c sad4.cu largerBlocks.cu main.cu ;;
        sgemm) echo main.cu io.cc ;;
        spmv) echo main.cu file.cc gpu_info.cc ;;
        stencil) echo main.cu file.cc ;;
        tpacf) echo args.cc model_io.cc main.cu ;;
    esac
}
# build_cuda <name> [extra flags]: every .c/.cc/.cu of src/cuda plus the
# suite's common timer/argument code.
build_cuda() {
    local n=$1; shift
    local s=$B/$n/src/cuda
    local srcs="" f
    for f in $(sources "$n"); do srcs="$srcs $s/$f"; done
    "$NVCC" $FLAGS "$@" $srcs "$COM/src/parboil_cuda.c" \
        -o "$OUT/$n.cuda" > "$OUT/$n.cuda.build.log" 2>&1
}
# build_ref <name> <dir>: the suite's CPU version, host compiler only.
build_ref() {
    local n=$1 d=$2; shift 2
    local s=$B/$n/src/$d
    local srcs
    srcs=""
    local o base
    for o in $(grep '^SRCDIR_OBJS' "$s/Makefile" | sed 's/^SRCDIR_OBJS *= *//; s/#.*//'); do
        base=${o%.o}
        for ext in c cc cpp; do [ -f "$s/$base.$ext" ] && srcs="$srcs $s/$base.$ext"; done
    done
    local cxx=${REFCXX:-$(command -v clang++ || echo /usr/lib/llvm-18/bin/clang++)}
    local objs="" f o
    : > "$OUT/$n.ref.build.log"
    for f in $srcs "$COM/src/parboil.c"; do
        o=$OUT/$n.ref.$(basename "$f").o
        case $f in
            *.c) "$cxx" -x c -O2 -w -I"$COM/include" "$@" -c "$f" -o "$o" ;;
            *) "$cxx" -O2 -fno-strict-return -w -include math.h -include malloc.h -include iostream                    -I"$COM/include" "$@" -c "$f" -o "$o" ;;
        esac >> "$OUT/$n.ref.build.log" 2>&1 || return 1
        objs="$objs $o"
    done
    "$cxx" $objs -o "$OUT/$n.ref" -lm >> "$OUT/$n.ref.build.log" 2>&1
}

python3 "$HERE/gen.py" "$OUT/in" || exit 2

printf '%-13s %-6s %-6s %s\n' program built ran check
total=0; built=0; ran=0; match=0
for n in bfs cutcp histo lbm mri-gridding mri-q sad sgemm spmv stencil tpacf; do
    total=$((total + 1))
    extra=
    [ "$n" = spmv ] && extra="-I$B/spmv/common_src/convert-dataset $B/spmv/common_src/convert-dataset/convert_dataset.c $B/spmv/common_src/convert-dataset/mmio.c"
    if ! build_cuda "$n" $extra; then
        printf '%-13s %-6s %-6s %s\n' "$n" no - "$(grep -m1 -i 'error' "$OUT/$n.cuda.build.log" | cut -c1-110)"
        continue
    fi
    built=$((built + 1))
    . "$HERE/cases.sh"
    case_$n   # sets RUN (command line for the cuda build), REF (for the reference) or SKIP (reason)
    if [ -n "${SKIP:-}" ]; then
        printf '%-13s %-6s %-6s %s\n' "$n" yes - "built only: $SKIP"; continue
    fi
    rm -f "$OUT/$n.out.cuda" "$OUT/$n.out.ref"
    if ! (cd "$OUT" && timeout 600 $RUN) > "$OUT/$n.run.log" 2>&1; then
        printf '%-13s %-6s %-6s %s\n' "$n" yes no "$(tail -1 "$OUT/$n.run.log" | cut -c1-100)"; continue
    fi
    ran=$((ran + 1))
    if [ -z "${REF:-}" ]; then
        printf '%-13s %-6s %-6s %s\n' "$n" yes yes "ran, no reference: $(ls -l "$OUT/$n.out.cuda" 2>/dev/null | awk '{print $5}') output bytes"; continue
    fi
    if ! (cd "$OUT" && timeout 600 $REF) > "$OUT/$n.ref.log" 2>&1; then
        printf '%-13s %-6s %-6s %s\n' "$n" yes yes "reference failed: $(tail -1 "$OUT/$n.ref.log" | cut -c1-80)"; continue
    fi
    res=$(python3 "$HERE/cmp.py" "$n" "$OUT/$n.out.cuda" "$OUT/$n.out.ref")
    case "$res" in SAME*) match=$((match + 1)) ;; esac
    printf '%-13s %-6s %-6s %s\n' "$n" yes yes "$res"
done
echo "summary: $total programs, $built built, $ran ran, $match matched the suite's CPU version"

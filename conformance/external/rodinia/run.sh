#!/bin/bash
# Build public Rodinia CUDA programs, unmodified, for the CPU path and check
# their results. Each check is independent of our code: the OpenMP version
# of the same program, the program's own check, or a serial reference
# written from the kernel's formula (hotspot_ref.c).
#
#   conformance/external/rodinia/run.sh <build dir> [work dir]
#
# <build dir>: a CMake build of this repository (libraries are linked from
# it). [work dir]: where the Rodinia sources are fetched (default
# ~/.cache/ocu-rodinia). Nothing is written inside the repository.
# Prints one line per program and a final summary line.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
BUILD=$(cd "$1" && pwd)
WORK=${2:-$HOME/.cache/ocu-rodinia}
PIN=9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4   # github.com/yuhc/gpu-rodinia

mkdir -p "$WORK"
if [ ! -d "$WORK/gpu-rodinia/.git" ]; then
    git clone -q https://github.com/yuhc/gpu-rodinia.git "$WORK/gpu-rodinia" || exit 2
fi
git -C "$WORK/gpu-rodinia" checkout -q "$PIN" || exit 2
R=$WORK/gpu-rodinia
TOOLS=$WORK/tools
mkdir -p "$TOOLS"
gcc -O2 -shared -fPIC "$HERE/fixtime.c" -o "$TOOLS/fixtime.so" || exit 2
gcc -O2 "$HERE/hotspot_ref.c" -o "$TOOLS/hotspot_ref" || exit 2

# OCU_TARGET=nvidia builds the same programs for the NVIDIA GPU.
TARGET=${OCU_TARGET:-cpu}
LIBS="$BUILD/runtime/libocu_cudart.a $BUILD/runtime/libocu_core.a $BUILD/backends/cpu/libocu_backend_cpu.a"
[ -f "$BUILD/backends/nvidia/libocu_backend_nvidia.a" ] && LIBS="$LIBS $BUILD/backends/nvidia/libocu_backend_nvidia.a -ldl"
# build <exe> <sources...> [-- flags]: every source through ocu_cc, as nvcc
# would take it.
build() {
    local out=$1; shift
    local srcs=() flags=()
    while [ $# -gt 0 ]; do
        if [ "$1" = "--" ]; then shift; flags=("$@"); break; fi
        srcs+=("$1"); shift
    done
    rm -rf .ocu; mkdir -p .ocu
    local objs=() s o
    for s in "${srcs[@]}"; do
        o=.ocu/$(echo "$s" | tr / _).o
        python3 "$REPO/compiler/probe/ocu_cc.py" --target "$TARGET" -O2 "${flags[@]}" -c "$s" -o "$o" \
            2> .ocu/build.log || return 1
        objs+=("$o")
    done
    g++ -o "$out" "${objs[@]}" $LIBS -lpthread -lm 2>> .ocu/build.log
}
cmp_num() { python3 "$HERE/numcmp.py" "$@"; }

pass=0; fail=0; built=0; total=0
report() { # report <name> <ok|FAIL|BUILD-FAIL|BUILT> <detail>
    total=$((total + 1))
    case "$2" in
        ok) pass=$((pass + 1)); built=$((built + 1)) ;;
        BUILT) built=$((built + 1)) ;;
        FAIL) fail=$((fail + 1)); built=$((built + 1)) ;;
        *) fail=$((fail + 1)) ;;
    esac
    printf '%-15s %-10s %s\n' "$1" "$2" "$3"
}
verdict() { case "$1" in SAME*) echo ok ;; *) echo FAIL ;; esac; }

C=$R/cuda; O=$R/openmp
export LD_PRELOAD_FIX=$TOOLS/fixtime.so

# pathfinder: result row against the OpenMP version
if (cd $C/pathfinder && build pf pathfinder.cu -- -DBENCH_PRINT); then
    (cd $C/pathfinder && ./pf 20000 200 20 | tail -n 1 > ocu.txt)
    (cd $O/pathfinder && g++ -O2 -fopenmp pathfinder.cpp -o pf_omp 2>/dev/null && ./pf_omp 20000 200 | tail -n 1 > ref.txt)
    r=$(cmp_num $C/pathfinder/ocu.txt $O/pathfinder/ref.txt 0); report pathfinder "$(verdict "$r")" "$r vs OpenMP"
else report pathfinder BUILD-FAIL "$(head -c 200 $C/pathfinder/.ocu/build.log)"; fi

# nw: traceback against the OpenMP version
if (cd $C/nw && build nw needle.cu -- -DTRACEBACK); then
    (cd $C/nw && ./nw 1024 10 > /dev/null && cp result.txt ocu.txt)
    (cd $O/nw && g++ -O2 -fopenmp needle.cpp -o nw_omp 2>/dev/null && ./nw_omp 1024 10 2 > /dev/null && cp result.txt ref.txt)
    r=$(cmp_num $C/nw/ocu.txt $O/nw/ref.txt 0); report nw "$(verdict "$r")" "$r vs OpenMP"
else report nw BUILD-FAIL "$(head -c 200 $C/nw/.ocu/build.log)"; fi

# bfs: levels on a generated graph against the OpenMP version
if (cd $C/bfs && build bfs bfs.cu -- -I../util); then
    python3 "$HERE/graphgen.py" 200000 $C/bfs/graph.txt
    (cd $C/bfs && ./bfs graph.txt > /dev/null && cp result.txt ocu.txt)
    (cd $O/bfs && g++ -O2 -fopenmp bfs.cpp -o bfs_omp 2>/dev/null && ./bfs_omp 4 $C/bfs/graph.txt > /dev/null && cp result.txt ref.txt)
    r=$(cmp_num $C/bfs/ocu.txt $O/bfs/ref.txt 0); report bfs "$(verdict "$r")" "$r vs OpenMP"
else report bfs BUILD-FAIL "$(head -c 200 $C/bfs/.ocu/build.log)"; fi

# lavaMD: forces against the OpenMP version, same inputs (time() fixed)
if (cd $C/lavaMD && build lavamd main.c kernel/kernel_gpu_cuda_wrapper.cu util/num/num.c util/timer/timer.c util/device/device.cu -- -DOUTPUT); then
    (cd $C/lavaMD && LD_PRELOAD=$TOOLS/fixtime.so ./lavamd -boxes1d 6 > /dev/null && cp result.txt ocu.txt)
    (cd $O/lavaMD && gcc -O2 -fopenmp -DOUTPUT main.c kernel/kernel_cpu.c util/num/num.c util/timer/timer.c -o lavamd_omp -lm 2>/dev/null \
        && LD_PRELOAD=$TOOLS/fixtime.so ./lavamd_omp -cores 4 -boxes1d 6 > /dev/null && cp result.txt ref.txt)
    r=$(cmp_num $C/lavaMD/ocu.txt $O/lavaMD/ref.txt 1e-5); report lavaMD "$(verdict "$r")" "$r vs OpenMP"
else report lavaMD BUILD-FAIL "$(head -c 200 $C/lavaMD/.ocu/build.log)"; fi

# hotspot: against a serial reference of the CUDA kernel's formula (the
# OpenMP version treats the chip edges differently)
if (cd $C/hotspot && build hotspot hotspot.cu); then
    python3 -c "
import random; random.seed(7)
open('$C/hotspot/t512','w').write(''.join('%f\n'%(320+random.random()*20) for _ in range(512*512)))
open('$C/hotspot/p512','w').write(''.join('%f\n'%(random.random()*0.5) for _ in range(512*512)))"
    (cd $C/hotspot && ./hotspot 512 2 20 t512 p512 ocu.txt > /dev/null && $TOOLS/hotspot_ref 512 20 t512 p512 > ref.txt)
    r=$(cmp_num $C/hotspot/ocu.txt $C/hotspot/ref.txt 1e-6); report hotspot "$(verdict "$r")" "$r vs serial reference"
else report hotspot BUILD-FAIL "$(head -c 200 $C/hotspot/.ocu/build.log)"; fi

# lud: the program's own check (L*U against the input)
if (cd $C/lud/cuda && build lud lud.cu lud_kernel.cu ../common/common.c -- -I../common); then
    m=$(cd $C/lud/cuda && ./lud -s 512 -v 2>&1 | grep -c dismatch)
    [ "$m" = 0 ] && report lud ok "0 mismatches, 512x512 (own check)" || report lud FAIL "$m mismatches (own check)"
else report lud BUILD-FAIL "$(head -c 200 $C/lud/cuda/.ocu/build.log)"; fi

# hotspot3D: the program's own accuracy against its CPU computation
if (cd $C/hotspot3D && build h3d 3D.cu); then
    python3 -c "
import random; random.seed(3)
open('$C/hotspot3D/p','w').write(''.join('%f\n'%(random.random()*0.5) for _ in range(512*512*8)))
open('$C/hotspot3D/t','w').write(''.join('%f\n'%(320+random.random()*20) for _ in range(512*512*8)))"
    a=$(cd $C/hotspot3D && ./h3d 512 8 20 p t o.txt 2>&1 | sed -n 's/.*Accuracy: *//p')
    python3 -c "import sys; sys.exit(0 if float('$a') < 1e-3 else 1)" 2>/dev/null \
        && report hotspot3D ok "accuracy $a (own check, < 1e-3)" || report hotspot3D FAIL "accuracy '$a'"
else report hotspot3D BUILD-FAIL "$(head -c 200 $C/hotspot3D/.ocu/build.log)"; fi

# nn: nearest records against an independent computation (nn_check.py
# reproduces the program's own selection, including its record swap)
if (cd $C/nn && build nn nn_cuda.cu); then
    (cd $C/nn && gcc -O2 hurricane_gen.c -o hgen -lm 2>/dev/null && ./hgen 20000 4 > /dev/null 2>&1 \
        && printf 'data/cane4_%s.db\n' 0 1 2 3 > flist && ./nn flist -r 10 -lat 30 -lng 90 | grep -- '-->' > ocu.txt)
    r=$(cd $C/nn && python3 "$HERE/nn_check.py" flist 10 30 90 ocu.txt); report nn "$(verdict "$r")" "$r"
else report nn BUILD-FAIL "$(head -c 200 $C/nn/.ocu/build.log)"; fi

# streamcluster: centres against the OpenMP version on one thread (both
# draw from lrand48 with the same fixed seed)
if (cd $C/streamcluster && build sc streamcluster_cuda_cpu.cpp streamcluster_cuda.cu streamcluster_header.cu); then
    (cd $C/streamcluster && ./sc 10 20 16 4096 4096 1000 none ocu.txt 1 > /dev/null 2>&1)
    (cd $O/streamcluster && g++ -O2 -fopenmp streamcluster_omp.cpp -o sc_omp -lpthread 2>/dev/null         && ./sc_omp 10 20 16 4096 4096 1000 none ref.txt 1 > /dev/null 2>&1)
    r=$(cmp_num $C/streamcluster/ocu.txt $O/streamcluster/ref.txt 1e-5); report streamcluster "$(verdict "$r")" "$r vs OpenMP"
else report streamcluster BUILD-FAIL "$(head -c 200 $C/streamcluster/.ocu/build.log)"; fi

# Built and run to completion; no independent check yet.
run_only() { # run_only <name> <dir> <exe> <run command> -- <build args>
    local name=$1 dir=$2 exe=$3 cmd=$4; shift 5
    if (cd $dir && build $exe "$@"); then
        if (cd $dir && timeout 300 sh -c "$cmd" > .ocu/run.log 2>&1); then
            report $name BUILT "ran to completion (no independent check yet)"
        else report $name FAIL "run failed: $(tail -c 150 $dir/.ocu/run.log)"; fi
    else report $name BUILD-FAIL "$(grep -m1 error $dir/.ocu/build.log | head -c 200)"; fi
}
run_only gaussian $C/gaussian gaussian "./gaussian -s 256 -q" -- gaussian.cu
run_only backprop $C/backprop backprop "./backprop 65536" -- backprop_cuda.cu backprop.c facetrain.c imagenet.c
run_only srad_v2 $C/srad/srad_v2 srad "./srad 512 512 0 127 0 127 0.5 2" -- srad.cu
run_only particlefilter $C/particlefilter pf "./pf -x 128 -y 128 -z 10 -np 1000" -- ex_particle_CUDA_float_seq.cu

# Built only (they need input files this script does not generate).
build_only() { # build_only <name> <dir> -- <build args>
    local name=$1 dir=$2; shift 3
    if (cd $dir && build ocu_exe "$@"); then report $name BUILT "built (needs input data to run)"
    else report $name BUILD-FAIL "$(grep -m1 error $dir/.ocu/build.log | head -c 200)"; fi
}
build_only srad_v1 $C/srad/srad_v1 -- main.cu
build_only b+tree $C/b+tree -- main.c kernel/kernel_gpu_cuda_wrapper.cu kernel/kernel_gpu_cuda_wrapper_2.cu util/num/num.c util/timer/timer.c util/cuda/cuda.cu
build_only myocyte $C/myocyte -- main.cu
build_only huffman $C/huffman -- main_test_cu.cu vlc_kernel_sm64huff.cu scan.cu pack_kernels.cu cpuencode.cpp -- -I.
build_only kmeans $C/kmeans -- kmeans_cuda.cu cluster.c getopt.c kmeans.c kmeans_clustering.c rmse.c

echo "summary: $total programs, $built built, $pass checked equal, $fail failed or not built"

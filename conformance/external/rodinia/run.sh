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
# OCU_PTX_SHIFT=1 builds every program with ocu_cc --ptx-shift (shifts in
# device code as PTX shl/shr define them; see README.md, "compat"). The
# default is standard C++.
SHIFT=
MODE=standard
if [ "${OCU_PTX_SHIFT:-0}" = 1 ]; then SHIFT=--ptx-shift; MODE=ptx-shift; fi
# Every library of the build in one group (the CPU backend needs ops_ref, the
# registry every backend that was built).
LIBS="-Wl,--start-group $BUILD/runtime/libocu_cudart.a $BUILD/runtime/libocu_core.a $(echo $BUILD/backends/*/libocu_*.a) -Wl,--end-group -ldl"
# OCU_TARGET=opencl: device code through cu2ocl.py, run on the OpenCL backend
# (OCU_OPENCL_LIBRARY / OCU_LLVM_SPIRV choose the library and translator).
[ "$TARGET" = opencl ] && export OCU_BACKEND=opencl
# build <exe> <sources...> [-- flags]: every source through ocu_cc, as nvcc
# would take it. With $HOSTCXX set, .cpp files go to that host compiler
# instead (programs whose Makefile builds them without nvcc). $EXTRA_LIBS and
# $LINK_FLAGS are added to the link.
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
        if [ -n "${HOSTCXX:-}" ] && [[ "$s" == *.cpp ]]; then
            $HOSTCXX -I"$REPO/runtime/include" "${flags[@]}" -c "$s" -o "$o" 2>> .ocu/build.log || return 1
            objs+=("$o"); continue
        fi
        python3 "$REPO/compiler/probe/ocu_cc.py" --target "$TARGET" -O2 $SHIFT "${flags[@]}" -c "$s" -o "$o" \
            2> .ocu/build.log || return 1
        objs+=("$o")
    done
    g++ -o "$out" "${objs[@]}" ${EXTRA_LIBS:-} $LIBS -lpthread -lm ${LINK_FLAGS:-} 2>> .ocu/build.log
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
# reproduces the program's own selection, including its record swap). The
# input generator seeds every file with time(NULL); fixtime.so fixes it, so
# the input is the same on every run (unfixed, runs differed from run to run
# and about one in five had ties the check orders differently).
if (cd $C/nn && build nn nn_cuda.cu); then
    (cd $C/nn && mkdir -p data && gcc -O2 hurricane_gen.c -o hgen -lm 2>/dev/null && LD_PRELOAD=$TOOLS/fixtime.so ./hgen 20000 4 > /dev/null 2>&1 \
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

# gaussian: the solution against a double-precision elimination with partial
# pivoting (gaussian_ref.py, input from the program's own matrix formula). The
# program prints its solution only in verbose mode, which no flag turns on, so
# gaussian_dump.cpp calls the program's solver and prints the result. Tolerance
# 2e-5: numcmp's bound is relative to max(1,|y|), so for these entries (all
# below 0.06) it is absolute; float elimination without pivoting differs from
# the reference by at most 1.03e-5.
if (cd $C/gaussian && build gaussian gaussian.cu "$HERE/gaussian_dump.cpp" -- -Dmain=ocu_prog_main); then
    python3 "$HERE/gaussian_ref.py" 256 $C/gaussian/input.txt $C/gaussian/ref.txt
    (cd $C/gaussian && ./gaussian input.txt > ocu.txt)
    r=$(cmp_num $C/gaussian/ocu.txt $C/gaussian/ref.txt 2e-5); report gaussian "$(verdict "$r")" "$r vs double-precision reference"
else report gaussian BUILD-FAIL "$(head -c 200 $C/gaussian/.ocu/build.log)"; fi

# backprop: the hidden and output layers after one training step against the
# OpenMP version (same seed, same input size). The program prints nothing
# about its results, so backprop_dump.cpp makes the calls its setup() makes.
# Tolerance 1e-6: both versions sum in float in a different order; observed
# difference 6e-8. The CUDA version does not write the input-layer weights
# back into the network, so those are not compared.
if (cd $C/backprop && build bp backprop_cuda.cu backprop.c facetrain.c imagenet.c "$HERE/backprop_dump.cpp" -- -Dmain=ocu_prog_main); then
    (cd $C/backprop && ./bp 4096 > ocu.txt)
    (cd $O/backprop && gcc -O2 -fopenmp -Dmain=ocu_prog_main -c backprop_kernel.c facetrain.c backprop.c imagenet.c 2>/dev/null \
        && g++ -O2 -fopenmp -DOMP_REF -c "$HERE/backprop_dump.cpp" -o dump_ref.o && g++ -fopenmp -o bp_omp dump_ref.o backprop_kernel.o facetrain.o backprop.o imagenet.o -lm \
        && ./bp_omp 4096 > ref.txt)
    r=$(cmp_num $C/backprop/ocu.txt $O/backprop/ref.txt 1e-6); report backprop "$(verdict "$r")" "$r vs OpenMP (after one step)"
else report backprop BUILD-FAIL "$(head -c 200 $C/backprop/.ocu/build.log)"; fi

# srad_v2: the image after two iterations against the OpenMP version (same
# seed, same input). Both print the image with 5 decimals, so the tolerance
# 1e-5 is about two units in the last printed digit; the observed difference is 0.
if (cd $C/srad/srad_v2 && build srad srad.cu -- -DOUTPUT); then
    (cd $C/srad/srad_v2 && ./srad 512 512 0 127 0 127 0.5 2 | grep -E '^[0-9.]+ [0-9.]+ ' > ocu.txt)
    (cd $O/srad/srad_v2 && g++ -O2 -fopenmp -DOUTPUT -o srad_omp srad.cpp -lm 2>/dev/null \
        && ./srad_omp 512 512 0 127 0 127 4 0.5 2 | grep -E '^[0-9.]+ [0-9.]+ ' > ref.txt)
    r=$(cmp_num $C/srad/srad_v2/ocu.txt $O/srad/srad_v2/ref.txt 1e-5); report srad_v2 "$(verdict "$r")" "$r vs OpenMP (image)"
else report srad_v2 BUILD-FAIL "$(head -c 200 $C/srad/srad_v2/.ocu/build.log)"; fi

# particlefilter: the estimates after one filter step against the OpenMP
# version (same seed: time() is fixed by fixtime.so). Estimates are printed
# with 6 decimals, so the tolerance 1e-7 relative is above the print rounding.
# Longer runs are not compared: the two versions sum the weights in a different
# order, which flips a resampling index now and then; at 10 frames the estimates
# then differ by up to about 0.2 (in 48-64).
if (cd $C/particlefilter && build pf ex_particle_CUDA_float_seq.cu); then
    (cd $C/particlefilter && LD_PRELOAD=$TOOLS/fixtime.so ./pf -x 128 -y 128 -z 2 -np 1000 | grep -E '^(XE|YE):' > ocu.txt)
    (cd $O/particlefilter && gcc -O2 -fopenmp ex_particle_OPENMP_seq.c -o pf_omp -lm 2>/dev/null \
        && LD_PRELOAD=$TOOLS/fixtime.so ./pf_omp -x 128 -y 128 -z 2 -np 1000 | grep -E '^(XE|YE):' > ref.txt)
    r=$(cmp_num $C/particlefilter/ocu.txt $O/particlefilter/ref.txt 1e-7); report particlefilter "$(verdict "$r")" "$r vs OpenMP (1 step)"
else report particlefilter BUILD-FAIL "$(head -c 200 $C/particlefilter/.ocu/build.log)"; fi

# b+tree: the batch searches of the GPU kernel (the 'k' command, 65,535 queries
# in all) against the OpenMP version's search on the same tree and queries.
# Integer results, tolerance 0.
if (cd $C/b+tree && build bt main.c kernel/kernel_gpu_cuda_wrapper.cu kernel/kernel_gpu_cuda_wrapper_2.cu util/num/num.c util/timer/timer.c util/cuda/cuda.cu); then
    python3 "$HERE/bpt_gen.py" 10000 $C/b+tree/tree.txt $C/b+tree/cmds.txt 2000 500 65535
    (cd $C/b+tree && rm -f output.txt && ./bt file tree.txt command cmds.txt > .ocu/run.log 2>&1 \
        && grep -E '^[0-9]+ +-?[0-9]+$' output.txt > ocu.txt)
    (cd $O/b+tree && gcc -O2 -fopenmp -D_GNU_SOURCE -o bt_omp main.c kernel/*.c util/num/num.c util/timer/timer.c -lm 2>/dev/null \
        && rm -f output.txt && ./bt_omp file $C/b+tree/tree.txt command $C/b+tree/cmds.txt > /dev/null 2>&1 \
        && grep -E '^[0-9]+ +-?[0-9]+$' output.txt > ref.txt)
    r=$(cmp_num $C/b+tree/ocu.txt $O/b+tree/ref.txt 0); report b+tree "$(verdict "$r")" "$r vs OpenMP (batch search)"
else report b+tree BUILD-FAIL "$(head -c 200 $C/b+tree/.ocu/build.log)"; fi

# huffman: the program's own check (its GPU bitstream against cpuencode.cpp,
# the program's CPU encoder). Reported as the program reports it. On the CPU
# path it FAILS by default because the program shifts 32-bit values by 32
# (undefined in C++; see README.md, "huffman"); with OCU_PTX_SHIFT=1 it
# passes. Below 16 blocks the program launches Pack2 with a grid of 0 blocks.
if (cd $C/huffman && build huff main_test_cu.cu vlc_kernel_sm64huff.cu scan.cu pack_kernels.cu cpuencode.cpp -- -I.); then
    python3 "$HERE/huffman_gen.py" 1048576 $C/huffman/data.bin
    (cd $C/huffman && ./huff data.bin > ocu.txt 2>&1)
    if grep -q '^PASS!' $C/huffman/ocu.txt; then report huffman ok "PASS (own check: GPU bitstream = CPU encoder)"
    else report huffman FAIL "own check: $(grep -c '^Diff' $C/huffman/ocu.txt) differing elements vs CPU encoder"; fi
else report huffman BUILD-FAIL "$(head -c 200 $C/huffman/.ocu/build.log)"; fi

# ---- programs that use texture references (README.md, "textures") ----

# hybridsort: the program's own check (-DVERIFY: its result against qsort on
# the same 4M random floats). It includes GL/glew.h, GL/glut.h and
# cuda_gl_interop.h without using them; empty ones are supplied when the
# system has none. checkCudaErrors comes from the program's bundled helper
# header only under a condition our headers do not meet, so it is given on the
# command line as the call itself (result not checked, as in a release build).
STUB=$TOOLS/stub; mkdir -p $STUB/GL
[ -f /usr/include/GL/glew.h ] || : > $STUB/GL/glew.h
[ -f /usr/include/GL/glut.h ] || : > $STUB/GL/glut.h
: > $STUB/cuda_gl_interop.h
if (cd $C/hybridsort && build hs bucketsort.cu mergesort.cu main.cu -- -I$STUB -DVERIFY "-DcheckCudaErrors(x)=(x)"); then
    (cd $C/hybridsort && ./hs r > ocu.txt 2>&1)
    if grep -q '^Checking result\.\.\.PASSED' $C/hybridsort/ocu.txt; then report hybridsort ok "PASSED (own check: 4,194,304 floats against qsort)"
    else report hybridsort FAIL "own check: $(grep -m1 -i 'missmatch\|FAILED' $C/hybridsort/ocu.txt | head -c 100)"; fi
else report hybridsort BUILD-FAIL "$(grep -m1 error $C/hybridsort/.ocu/build.log | head -c 200)"; fi

# kmeans: cluster centres against the serial version of the suite (openmp/
# kmeans/kmeans_serial: same first-k initial centres, same stopping rule) on
# 20,000 generated points with 16 features around 5 centres. The CUDA version
# passes the name of its __constant__ array as a string to
# cudaMemcpyToSymbol("c_clusters", ...), a form removed in CUDA 5.0; it is
# reported as it runs.
if (cd $C/kmeans && build km kmeans_cuda.cu cluster.c getopt.c kmeans.c kmeans_clustering.c rmse.c); then
    python3 -c "
import random
r = random.Random(5)
cs = [[r.uniform(0, 100) for _ in range(16)] for _ in range(5)]
with open('$C/kmeans/in.txt', 'w') as f:
    for i in range(20000):
        c = cs[r.randrange(5)]
        f.write(str(i + 1) + ' ' + ' '.join('%.4f' % (v + r.gauss(0, 4)) for v in c) + '\n')"
    (cd $C/kmeans && ./km -o -i in.txt > run.log 2>&1; grep -E '^[0-9]+:' run.log > ocu.txt)
    (cd $O/kmeans/kmeans_serial && gcc -O2 -w -fopenmp *.c -o km_ser -lm 2>/dev/null && ./km_ser -i $C/kmeans/in.txt | grep -E '^[0-9]+:' > ref.txt)
    if [ -s $C/kmeans/ocu.txt ]; then
        r=$(cmp_num $C/kmeans/ocu.txt $O/kmeans/kmeans_serial/ref.txt 1e-4); report kmeans "$(verdict "$r")" "$r vs serial version"
    else report kmeans FAIL "no centres printed: $(tail -c 120 $C/kmeans/run.log | tr '\n' ' ')"; fi
else report kmeans BUILD-FAIL "$(grep -m1 error $C/kmeans/.ocu/build.log | head -c 200)"; fi

# leukocyte: cells found in a generated video (leukocyte_avi.c, written with
# the program's own avilib.c; the suite's sample video is not in its source
# tree), tracked over 3 frames, against the OpenMP version. Both built with
# -DOUTPUT (result.txt: the cell positions after the last frame).
make -s -C $C/leukocyte/meschach_lib > /dev/null 2>&1
if (cd $C/leukocyte/CUDA && EXTRA_LIBS=../meschach_lib/meschach.a build lk detect_main.c avilib.c find_ellipse.c find_ellipse_kernel.cu track_ellipse.c track_ellipse_kernel.cu misc_math.c -- -I../meschach_lib -DOUTPUT); then
    gcc -O2 -w -I$C/leukocyte/CUDA "$HERE/leukocyte_avi.c" $C/leukocyte/CUDA/avilib.c -o $TOOLS/leukocyte_avi -lm \
        && $TOOLS/leukocyte_avi $TOOLS/cells.avi 8
    (cd $C/leukocyte/CUDA && rm -f result.txt && ./lk $TOOLS/cells.avi 3 > run.log 2>&1; cp result.txt ocu.txt 2>/dev/null)
    make -s -C $O/leukocyte/meschach_lib > /dev/null 2>&1
    (cd $O/leukocyte/OpenMP && gcc -O2 -w -fopenmp -DOUTPUT -I../meschach_lib *.c ../meschach_lib/meschach.a -o lk_omp -lm \
        && rm -f result.txt && ./lk_omp 3 4 $TOOLS/cells.avi > run.log 2>&1 && cp result.txt ref.txt)
    n1=$(grep -o 'Cells detected: [0-9]*' $C/leukocyte/CUDA/run.log); n2=$(grep -o 'Cells detected: [0-9]*' $O/leukocyte/OpenMP/run.log)
    r=$(cmp_num $C/leukocyte/CUDA/ocu.txt $O/leukocyte/OpenMP/ref.txt 1e-5); report leukocyte "$(verdict "$r")" "$n1 / OpenMP $n2; positions $r"
else report leukocyte BUILD-FAIL "$(grep -m1 error $C/leukocyte/CUDA/.ocu/build.log | head -c 200)"; fi

# mummergpu: the matches of the GPU kernels against the program's own CPU
# version (-C, mummergpu_gold.cpp), same binary, same generated reference
# (200,000 bases) and 2,000 reads of 100 bases with up to 3 changes; minimum
# match length 20. The program's host-only C++ files are built with the host
# compiler as its Makefile does, at -O0: with g++ -O2 they crash while
# flattening the suffix tree (in the -C CPU version too); they also need
# <unistd.h>, which they do not include.
if (cd $C/mummergpu/src && HOSTCXX="g++ -O0 -w -fopenmp -include unistd.h" LINK_FLAGS=-fopenmp build mg mummergpu.cu mummergpu_gold.cpp mummergpu_main.cpp suffix-tree.cpp PoolMalloc.cpp -- -I.); then
    python3 -c "
import random
r = random.Random(9)
ref = ''.join(r.choice('ACGT') for _ in range(200000))
with open('$C/mummergpu/src/ref.fna', 'w') as f:
    f.write('>ref synthetic\n')
    for i in range(0, len(ref), 70): f.write(ref[i:i + 70] + '\n')
with open('$C/mummergpu/src/q.fna', 'w') as f:
    for q in range(2000):
        p = r.randrange(len(ref) - 100)
        s = list(ref[p:p + 100])
        for _ in range(r.randrange(4)): s[r.randrange(100)] = r.choice('ACGT')
        f.write('>q%d\n%s\n' % (q, ''.join(s)))"
    (cd $C/mummergpu/src && ./mg -l 20 ref.fna q.fna > ocu.txt 2> run.log; ./mg -C -l 20 ref.fna q.fna > ref.txt 2> ref.log)
    if [ -s $C/mummergpu/src/ref.txt ] && cmp -s $C/mummergpu/src/ocu.txt $C/mummergpu/src/ref.txt; then
        report mummergpu ok "identical to its CPU version ($(grep -c '^>' $C/mummergpu/src/ocu.txt) reads, $(grep -vc '^>' $C/mummergpu/src/ocu.txt) matches)"
    else report mummergpu FAIL "differs from its CPU version: $(diff $C/mummergpu/src/ocu.txt $C/mummergpu/src/ref.txt | grep -c '^[<>]') lines"; fi
else report mummergpu BUILD-FAIL "$(grep -m1 error $C/mummergpu/src/.ocu/build.log | head -c 200)"; fi

# Built only (they need input files this script does not generate, or the
# program keeps its results inside the solver, see README.md).
build_only() { # build_only <name> <dir> -- <build args>
    local name=$1 dir=$2; shift 3
    if (cd $dir && build ocu_exe "$@"); then report $name BUILT "built only (no check yet, see README.md)"
    else report $name BUILD-FAIL "$(grep -m1 error $dir/.ocu/build.log | head -c 200)"; fi
}
build_only srad_v1 $C/srad/srad_v1 -- main.cu
build_only myocyte $C/myocyte -- main.cu

echo "summary: $total programs, $built built, $pass checked equal, $fail failed or not built ($TARGET, $MODE shifts)"

#!/bin/bash
# segsort-cuda: which OpenCuda step makes the GPU result wrong.
source ~/.opencuda/env-dev.sh >/dev/null; export OCU_BACKEND=opencl
cd /tmp/sym/w2/segsort-cuda_spirv
r() { echo "$1: $(env $2 timeout 300 ./main 2>&1 | grep -E 'PASS' | sed 's/checking //' | cut -c1-40 | tr '\n' ' ')"; }
r "runtime STAGE_SYNC=1" OCU_OPENCL_STAGE_SYNC=1
r "runtime INDIRECT=0" OCU_OPENCL_INDIRECT=0
r "runtime OPS=0" OCU_OPENCL_OPS=0
for kv in OCU_SPV_GLOBALS_BLOCK=0 OCU_SPV_SPIN_LOADS=0 OCU_SPV_INFER=0 OCU_SPV_DIV64=0; do
  d=/tmp/sym/w2/seg_$kv; rm -rf $d; cp -r ~/cmp/HeCBench/src/segsort-cuda $d
  ( cd $d && env $kv OCU_TARGET=spirv timeout 600 nice make -j8 CC=ocu-nvcc ARCH=sm_75 EXTRA_CFLAGS=--ocu-target=spirv > build.log 2>&1 || echo "build_fail $kv"
    echo "build $kv: $(timeout 300 ./main 2>&1 | grep -E 'PASS' | sed 's/checking //' | cut -c1-40 | tr '\n' ' ')" ); done

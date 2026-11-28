#!/bin/bash
# Cold start of llama.cpp (empty driver cache) with and without building the other modules ahead.
OUT=~/oc-repro/prebuild.log
exec > $OUT 2>&1
source ~/.opencuda/env-dev.sh >/dev/null
cd ~/opencuda-dev/build && nice cmake --build . -j4 --target ocu_toolkit > /tmp/pb_build.log 2>&1 && echo toolkit_ok || { grep -m5 error /tmp/pb_build.log; exit 1; }
ldd ~/llama-var/final/libggml-cuda.so 2>/dev/null | grep -i cudart
export OCU_BACKEND=opencl OCU_COMPUTE_CAPABILITY=7.0; M=~/models/qwen3-0.6b.gguf; V=~/llama-var/final
start() { s=$(date +%s.%N); env $2 LD_LIBRARY_PATH=$V timeout 1500 $V/llama-bench -m $M -p 16 -n 4 -r 1 > /tmp/pb_run.log 2>&1; rc=$?; e=$(date +%s.%N); printf "%s: %.1f s (rc %d), cache %s in %s files, %s\n" "$1" $(echo "$e - $s" | bc) $rc "$(du -sh $NEO_CACHE_DIR | cut -f1)" "$(ls $NEO_CACHE_DIR | wc -l)" "$(grep -E 'tg4|pp16' /tmp/pb_run.log | awk -F'|' '{print $(NF-1)}' | tr -s ' ' | tr '\n' ' ')"; }
for v in 0 default; do
  export NEO_CACHE_DIR=/tmp/neo_pb_$v; rm -rf $NEO_CACHE_DIR; mkdir -p $NEO_CACHE_DIR
  kv=OCU_PREBUILD_THREADS=$v; [ $v = default ] && kv=X_NONE=1
  start "prebuild $v, empty cache" $kv; start "prebuild $v, second start" $kv
done
export NEO_CACHE_DIR=/tmp/neo_pb_default
echo "ops with prebuild: MUL_MAT $(LD_LIBRARY_PATH=$V timeout 900 $V/test-backend-ops -b CUDA0 -o MUL_MAT 2>&1 | grep -E 'tests passed' | tail -1) FA128 $(LD_LIBRARY_PATH=$V timeout 900 $V/test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT -p hsk=128 2>&1 | grep -E 'tests passed' | tail -1)"
rm -rf /tmp/neo_pb_0 /tmp/neo_pb_default
echo "== done $(date -u +%T)"

#!/bin/bash
# Start-up time of llama.cpp (same binary) with the driver's compiler cache: empty, 1 GB limit, no limit.
OUT=~/oc-repro/jitcache.log
exec > $OUT 2>&1
while pgrep -f 'v4val.sh|v3val.sh|wrong2.sh' > /dev/null; do sleep 20; done
source ~/.opencuda/env-dev.sh >/dev/null
export OCU_BACKEND=opencl OCU_COMPUTE_CAPABILITY=7.0; M=~/models/qwen3-0.6b.gguf; V=~/llama-var/h16d
start() { s=$(date +%s.%N); LD_LIBRARY_PATH=$V timeout 1200 $V/llama-bench -m $M -p 16 -n 4 -r 1 > /dev/null 2>&1; rc=$?; e=$(date +%s.%N); printf "%s: %.1f s (rc %d), cache %s in %s files\n" "$1" $(echo "$e - $s" | bc) $rc "$(du -sh $NEO_CACHE_DIR | cut -f1)" "$(ls $NEO_CACHE_DIR | wc -l)"; }
for lim in default unlimited; do
  export NEO_CACHE_DIR=/tmp/neo_jit_$lim; rm -rf $NEO_CACHE_DIR; mkdir -p $NEO_CACHE_DIR
  if [ $lim = unlimited ]; then export NEO_CACHE_MAX_SIZE=0; else unset NEO_CACHE_MAX_SIZE; fi
  start "$lim, empty cache"; start "$lim, second start"; start "$lim, third start"
done
export NEO_CACHE_PERSISTENT=0; unset NEO_CACHE_DIR; start "no persistent cache"
OCU_OPENCL_VERBOSE=1 LD_LIBRARY_PATH=$V timeout 600 $V/llama-bench -m $M -p 16 -n 4 -r 1 2>&1 | grep -ciE 'build|module' | sed 's/^/verbose build lines: /'
rm -rf /tmp/neo_jit_default /tmp/neo_jit_unlimited
echo "== done $(date -u +%T)"

#!/bin/bash
# Which part of the native binary16 change breaks llama.cpp test-backend-ops: to_float, from_float, hfma.
OUT=~/oc-repro/bisect16.log
exec > $OUT 2>&1
source ~/.opencuda/env-dev.sh >/dev/null
H16=~/opencuda-dev/runtime/include/cuda_fp16.h
cp $H16 /tmp/cuda_fp16.h.keep
cd ~/llama-spvdev
build() { ninja -t clean ggml-cuda > /dev/null; OCU_SPV_KEEP_GOING=1 OCU_SPV_CLANG=$HOME/oc-repro/fa/bin/clang-fallback nice -n 19 cmake --build . -j4 --target test-backend-ops llama-bench > /tmp/b16_$1.log 2>&1 && echo "built $1"; rm -rf ~/llama-var/$1; mkdir -p ~/llama-var/$1; cp -a bin/llama-bench bin/test-backend-ops bin/*.so* ~/llama-var/$1/; }
# each variant keeps one native part: the others' #if defined(OCU_F16_NATIVE) become #if 0
variant() { cp /tmp/cuda_fp16.h.keep $H16; python3 - "$H16" "$1" <<'EOF'
import sys, re
p, keep = sys.argv[1], sys.argv[2]
s = open(p).read()
parts = s.split("#if defined(OCU_F16_NATIVE)")
# order in the file: 0 native_from_float decl, 1 from_double fast path, 2 to_float, 3 hfma
names = ["decl", "from", "to", "fma"]
out = parts[0]
for i, rest in enumerate(parts[1:]):
    out += ("#if defined(OCU_F16_NATIVE)" if names[i] in ("decl", keep) else "#if 0") + rest
open(p, "w").write(out)
EOF
grep -c '^#if 0$' $H16; }
for v in to from fma; do variant $v; build b16_$v; done
cp /tmp/cuda_fp16.h.keep $H16; echo "header restored: $(grep -c 'defined(OCU_F16_NATIVE)' $H16)"
echo "== builds done $(date -u +%T)"
HH=~/cmp/hec
running=0; pgrep -f hec_full.sh > /dev/null && running=1
resume() { [ $running = 1 ] || return; ( cd ~/cmp && exec setsid nohup ./hec_full.sh < /dev/null > /dev/null 2>&1 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&- & ); sleep 1; echo "resumed $(date -u '+%F %T UTC') at row $(($(wc -l < $HH/results_full.csv)-1)) (toolkit unchanged: 4f9930a)" >> $HH/RUN_NOTES.txt; echo "== RESUMED $(date -u +%T)"; }
if [ $running = 1 ]; then
for g in $(ps -eo pgid,cmd | grep -E 'hec_full.sh|timeout (180|300)' | grep -v grep | awk '{print $1}' | sort -u); do kill -TERM -- -$g 2>/dev/null; done
sleep 3; pkill -f 'cmp/hec/wfull' 2>/dev/null; sleep 1
echo "paused $(date -u '+%F %T UTC') at row $(($(wc -l < $HH/results_full.csv)-1)) for perf experiment 9 (fp16 bisect)" >> $HH/RUN_NOTES.txt
fi
trap resume EXIT
export OCU_BACKEND=opencl OCU_COMPUTE_CAPABILITY=7.0
for v in to from fma; do V=~/llama-var/b16_$v
  echo "$v: MUL_MAT $(LD_LIBRARY_PATH=$V timeout 900 $V/test-backend-ops -b CUDA0 -o MUL_MAT 2>&1 | grep -E 'tests passed' | tail -1) | FA128 $(LD_LIBRARY_PATH=$V timeout 900 $V/test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=128' 2>&1 | grep -E 'tests passed' | tail -1)"; done
M=~/models/qwen3-0.6b.gguf
for v in to from fma; do V=~/llama-var/b16_$v; r=$(LD_LIBRARY_PATH=$V timeout 600 $V/llama-bench -m $M -p 512 -n 128 -fa 1 -r 2 -o csv 2>/dev/null | grep -E '^"' | awk -F, '{printf "%s ", $(NF-1)}'); echo "$v: pp512/tg128 = $r"; done
echo "== done $(date -u +%T)"

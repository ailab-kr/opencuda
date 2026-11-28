#!/bin/bash
# segsort-cuda: the function whose infer-address-spaces result is wrong on the GPU (opt-bisect).
source ~/.opencuda/env-dev.sh >/dev/null; export OCU_BACKEND=opencl
REAL=$HOME/tools/LLVM-23.1.6-Linux-X64/bin/opt
cat > /tmp/optwrap2.sh <<EOF
#!/bin/bash
args=(); for a in "\$@"; do case "\$a" in -passes=*) args+=("-passes=infer-address-spaces");; *) args+=("\$a");; esac; done
exec $REAL -opt-bisect-limit=\$OCU_BISECT "\${args[@]}" 2>>/tmp/bisect_passes.txt
EOF
chmod +x /tmp/optwrap2.sh
try() {  # N -> 0 pass, 1 fail
  d=/tmp/sym/w2/seg_b; rm -rf $d; cp -r ~/cmp/HeCBench/src/segsort-cuda $d; : > /tmp/bisect_passes.txt
  ( cd $d && OCU_BISECT=$1 OCU_SPV_OPT=/tmp/optwrap2.sh OCU_TARGET=spirv timeout 600 make -j8 CC=ocu-nvcc ARCH=sm_75 EXTRA_CFLAGS=--ocu-target=spirv > build.log 2>&1 )
  out=$(cd $d && timeout 300 ./main 2>&1 | grep -c 'NOT PASSED')
  echo "limit $1: not_passed=$out" >&2
  [ "$out" = 0 ]
}
try -1 || true
total=$(grep -c 'BISECT: running pass' /tmp/bisect_passes.txt); echo "passes: $total"
lo=0; hi=$total   # try(lo) passes (assumed), try(hi) fails
try 0 && echo "limit 0 passes"
while [ $((hi - lo)) -gt 1 ]; do mid=$(((lo + hi) / 2)); if try $mid; then lo=$mid; else hi=$mid; fi; done
echo "first bad pass: $hi"
try -1 || true
grep "($hi)" /tmp/bisect_passes.txt | head -2

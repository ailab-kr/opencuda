#!/bin/bash
# segsort-cuda: which pass of cu2spv's opt step makes the GPU result wrong.
source ~/.opencuda/env-dev.sh >/dev/null; export OCU_BACKEND=opencl
REAL=$(dirname $(python3 -c "import sys; sys.path.insert(0,'$HOME/opencuda-dev/compiler/spirv'); import cu2spv" 2>/dev/null; command -v llc))/opt
[ -x "$REAL" ] || REAL=$HOME/tools/LLVM-23.1.6-Linux-X64/bin/opt
cat > /tmp/optwrap.sh <<EOF
#!/bin/bash
args=(); for a in "\$@"; do case "\$a" in -passes=*) args+=("-passes=\$OCU_PASSES");; *) args+=("\$a");; esac; done
exec $REAL "\${args[@]}"
EOF
chmod +x /tmp/optwrap.sh; echo "opt: $REAL"
for P in "always-inline,infer-address-spaces" "always-inline,instcombine<no-verify-fixpoint>" "always-inline,simplifycfg" "always-inline,infer-address-spaces,instcombine<no-verify-fixpoint>" "always-inline,infer-address-spaces,instcombine<no-verify-fixpoint>,simplifycfg"; do
  d=/tmp/sym/w2/seg_p; rm -rf $d; cp -r ~/cmp/HeCBench/src/segsort-cuda $d
  ( cd $d && OCU_PASSES="$P" OCU_SPV_OPT=/tmp/optwrap.sh OCU_TARGET=spirv timeout 600 nice make -j8 CC=ocu-nvcc ARCH=sm_75 EXTRA_CFLAGS=--ocu-target=spirv > build.log 2>&1 || echo "build_fail $P"
    echo "$P: $(timeout 300 ./main 2>&1 | grep -E 'PASS' | sed 's/checking //' | cut -c1-30 | tr '\n' ' ')" ); done

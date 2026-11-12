#!/bin/bash
# build.sh <llama.cpp source dir> <llama.cpp build dir> [out]
# Builds llama_logits against an existing llama.cpp build tree, reusing the
# link line of that tree's llama-simple (so the CUDA backend built through
# OpenCuda and its runtime libraries are linked exactly as for llama-simple).
set -e
SRC=$1; B=$2; OUT=${3:-$B/bin/llama_logits}
HERE=$(cd "$(dirname "$0")" && pwd)
LINK=$B/examples/simple/CMakeFiles/llama-simple.dir/link.txt
[ -f "$LINK" ] || { echo "no $LINK (build llama-simple in $B first)"; exit 1; }
OBJ=$(mktemp -d)/llama_logits.o
c++ -O2 -std=c++17 -I"$SRC/include" -I"$SRC/ggml/include" -c "$HERE/llama_logits.cpp" -o "$OBJ"
# link.txt is relative to the example's build dir; run it from there with our object and output.
cd "$B/examples/simple"
CMD=$(sed -e "s#\"CMakeFiles/llama-simple.dir/simple.cpp.o\"#$OBJ#" -e "s#CMakeFiles/llama-simple.dir/simple.cpp.o#$OBJ#" \
          -e "s#-o \.\./\.\./bin/llama-simple#-o $OUT#" "$LINK")
eval "$CMD"
echo "built $OUT"

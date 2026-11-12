#!/bin/sh
# One command to verify the repository: build, tests, sanitizers, the
# compiler probe, harness tests and the clean-room guard.
#
# Usage: tools/check.sh [--quick] [build_root]
#   --quick      skip the sanitizer builds
#   build_root   where build folders go (default /tmp/ocu-check)
#
# Exit status is 0 only when every step passes. Each step prints one
# "PASS"/"FAIL" line; details are in <build_root>/*.log.

set -u
quick=0
if [ "${1:-}" = "--quick" ]; then
    quick=1
    shift
fi
repo=$(cd "$(dirname "$0")/.." && pwd)
root=${1:-/tmp/ocu-check}
mkdir -p "$root"
failed=0

for d in /usr/lib/llvm-18/bin /usr/lib/llvm-19/bin /usr/lib/llvm-20/bin; do
    [ -d "$d" ] && PATH="$d:$PATH"
done
export PATH

step() {
    name=$1
    shift
    log="$root/$name.log"
    if "$@" >"$log" 2>&1; then
        echo "PASS  $name"
    else
        echo "FAIL  $name  (see $log)"
        tail -n 15 "$log" | sed 's/^/      /'
        failed=1
    fi
}

build_and_test() {
    dir=$1
    shift
    rm -rf "$dir"
    cmake -S "$repo" -B "$dir" -DCMAKE_BUILD_TYPE=Debug "$@" &&
        cmake --build "$dir" -j 4 2>&1 | tee "$dir.build.txt" &&
        ! grep -E 'warning:' "$dir.build.txt" &&
        ctest --test-dir "$dir" --output-on-failure
}

# 1. Default compiler (gcc on Ubuntu), warnings are failures.
step build_gcc build_and_test "$root/gcc"

# 2. clang, same tests.
if command -v clang >/dev/null 2>&1; then
    step build_clang build_and_test "$root/clang" \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
fi

# 3. Sanitizers (clang): address + undefined, then thread.
# Leak detection is not available in Apple's address sanitizer.
leaks=1
[ "$(uname -s)" = "Darwin" ] && leaks=0
if [ "$quick" -eq 0 ] && command -v clang >/dev/null 2>&1; then
    for san in address,undefined thread; do
        tag=$(echo "$san" | tr ',' '_')
        step "sanitize_$tag" env ASAN_OPTIONS=detect_leaks=$leaks \
            TSAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
            sh -c "$(cat <<EOF
set -e
rm -rf "$root/san_$tag"
cmake -S "$repo" -B "$root/san_$tag" -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  "-DCMAKE_C_FLAGS=-fsanitize=$san -fno-omit-frame-pointer -fno-sanitize-recover=all" \
  "-DCMAKE_CXX_FLAGS=-fsanitize=$san -fno-omit-frame-pointer -fno-sanitize-recover=all" \
  "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=$san"
cmake --build "$root/san_$tag" -j 4
ctest --test-dir "$root/san_$tag" --output-on-failure
EOF
)"
    done
fi

# 3b. AArch64 (the fast fiber switch has a separate implementation there):
#     cross-build statically and run every test under qemu-user, when the
#     cross compiler and qemu are installed.
if command -v aarch64-linux-gnu-g++ >/dev/null 2>&1 && command -v qemu-aarch64 >/dev/null 2>&1; then
    step build_aarch64_qemu build_and_test "$root/aarch64" \
        -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
        -DOCU_BACKEND_NVIDIA=OFF \
        -DCMAKE_ASM_COMPILER=aarch64-linux-gnu-gcc \
        -DCMAKE_EXE_LINKER_FLAGS=-static \
        -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-aarch64
fi

# 4. Conformance harness unit tests.
step harness sh -c "cd '$repo/conformance/harness' && python3 -m unittest discover -s tests"

# 4b. Scoreboard generator tests.
step scoreboard sh -c "cd '$repo' && python3 -m unittest discover -s tools/tests"

# 5. Compiler probe: every target must compile (needs an LLVM clang with the
#    NVPTX and AMDGPU back ends; skipped, and said so, when there is none).
if command -v clang++ >/dev/null 2>&1 &&
   clang++ --print-targets 2>/dev/null | grep -q nvptx64 &&
   clang++ --print-targets 2>/dev/null | grep -q amdgcn; then
    step compiler_probe sh -c "
        sh '$repo/compiler/probe/run_probe.sh' '$root/probe' > '$root/probe.txt' 2>&1
        cat '$root/probe.txt'
        ! grep -q 'result: FAILED' '$root/probe.txt'"
else
    echo "SKIP  compiler_probe  (no clang with NVPTX and AMDGPU targets)"
fi

# 6. Clean-room guard: no vendor SDK paths in any compile command or
#    dependency file of the builds above, no vendor headers in the tree.
guard() {
    found=0
    # Build trees inside the repository (they hold installed copies of our
    # own headers) are not sources: skip every folder with a CMakeCache.txt.
    # Hidden folders (local worktrees, tool state) are not sources: skip them.
    prune=""; for d in "$repo"/.[!.]*; do [ -d "$d" ] && prune="$prune -path $d -prune -o"; done
    for c in $(find "$repo" -path "$repo/.git" -prune -o -name CMakeCache.txt -print 2>/dev/null); do
        prune="$prune -path $(dirname "$c") -prune -o"
    done
    for f in $(find "$root" -name 'flags.make' -o -name '*.d' -o -name 'compile_commands.json' 2>/dev/null); do
        if grep -lE '/usr/local/cuda|/opt/cuda|NVIDIA GPU Computing Toolkit' "$f"; then
            found=1
        fi
    done
    # Our own CUDA-named headers live only in runtime/include and are
    # written from public documentation; the same names anywhere else, or
    # vendor libraries anywhere, are refused.
    if find "$repo" -path "$repo/.git" -prune -o $prune -path "$repo/runtime/include" -prune \
        -o -type f \( -name 'cuda_runtime*.h' -o -name 'cuda.h' -o -name 'cublas*.h' \
           -o -name 'cudnn*.h' \) -print | grep .; then
        found=1
    fi
    if find "$repo" -path "$repo/.git" -prune -o $prune -type f \
        \( -name 'libcuda*' -o -name 'libcudart*' -o -name 'libcublas*' \) -print | grep .; then
        found=1
    fi
    # Vendor header text: the licence and copyright lines vendor files carry.
    if find "$repo" -path "$repo/.git" -prune -o $prune -type f \
        \( -name '*.h' -o -name '*.hpp' -o -name '*.cuh' -o -name '*.inl' \) -print0 |
        xargs -0 grep -lE 'NVIDIA CORPORATION|NVIDIA Software License|Copyright \(c\) 1993-'; then
        found=1
    fi
    for f in "$root"/probe/*.log; do
        [ -f "$f" ] || continue
        if grep -lE '/usr/local/cuda' "$f"; then
            found=1
        fi
    done
    [ "$found" -eq 0 ]
}
step clean_room_guard guard

echo
if [ "$failed" -eq 0 ]; then
    echo "ALL PASS"
else
    echo "SOME STEPS FAILED"
fi
exit "$failed"

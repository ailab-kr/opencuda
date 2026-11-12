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
if [ "$quick" -eq 0 ] && command -v clang >/dev/null 2>&1; then
    for san in address,undefined thread; do
        tag=$(echo "$san" | tr ',' '_')
        step "sanitize_$tag" env ASAN_OPTIONS=detect_leaks=1 \
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

# 4. Conformance harness unit tests.
step harness sh -c "cd '$repo/conformance/harness' && python3 -m unittest discover -s tests"

# 5. Compiler probe: every target must compile.
if command -v clang++ >/dev/null 2>&1; then
    step compiler_probe sh -c "
        sh '$repo/compiler/probe/run_probe.sh' '$root/probe' > '$root/probe.txt' 2>&1
        cat '$root/probe.txt'
        ! grep -q 'result: FAILED' '$root/probe.txt'"
fi

# 6. Clean-room guard: no vendor SDK paths in any compile command or
#    dependency file of the builds above, no vendor headers in the tree.
guard() {
    found=0
    for f in $(find "$root" -name 'flags.make' -o -name '*.d' -o -name 'compile_commands.json' 2>/dev/null); do
        if grep -lE '/usr/local/cuda|/opt/cuda|NVIDIA GPU Computing Toolkit' "$f"; then
            found=1
        fi
    done
    if find "$repo" -path "$repo/.git" -prune -o -type f \
        \( -name 'cuda_runtime*.h' -o -name 'cuda.h' -o -name 'cublas*.h' \
           -o -name 'cudnn*.h' -o -name 'libcuda*' -o -name 'libcudart*' \) -print | grep .; then
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

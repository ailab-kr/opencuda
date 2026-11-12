#!/bin/bash
# run.sh <name> <model.gguf> <text file> <prefix> <build>...
# Teacher-forced logit dumps of one text with several llama_logits binaries.
#   prefix: tokens decoded as one batch before the one-token-at-a-time part
#           ("all" = the whole text as one batch).
#   build:  tag=binary, e.g. ref=~/ocu-kl/bin/ll-ref cpu=~/ocu-kl/bin/ll-cpu
# Writes <name>-<tag>.bin in the current directory and prints wall time.
# Waits for free memory with ~/ocu-tools/memgate.sh before each run.
NAME=$1; M=$2; TXT=$3; PFX=$4; shift 4
for b in "$@"; do
  tag=${b%%=*}; bin=${b#*=}
  [ -x ~/ocu-tools/memgate.sh ] && ~/ocu-tools/memgate.sh 2 "$NAME-$tag" > /dev/null
  args=()
  [ "$PFX" != all ] && args=(--prefix "$PFX")
  s=$(date +%s.%N)
  OCU_CPU_THREADS=8 "$bin" -m "$M" -t 8 "${args[@]}" -p "$(cat "$TXT")" -o "$NAME-$tag.bin" 2> "$NAME-$tag.err" \
    || { echo "$NAME-$tag: failed"; tail -3 "$NAME-$tag.err"; continue; }
  printf '%s-%s: %.1f s  %s\n' "$NAME" "$tag" "$(echo "$(date +%s.%N) - $s" | bc)" "$(tail -1 "$NAME-$tag.err")"
done

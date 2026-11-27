#!/bin/bash
# Build and run every HeCBench *-cuda program, unmodified, with OpenCuda
# (ocu-nvcc --ocu-target=spirv, OpenCL backend) and with chipStar (cucc),
# under the same rules, and record one CSV row per program and toolchain.
#
#   HECBENCH=<HeCBench checkout> OUT=<results folder> \
#   OCU_ENV=<script that puts ocu-nvcc on PATH> CHIPSTAR=<chipStar prefix> \
#   tools/bench/hecbench_compare.sh
#
# Rules: `make` with the toolchain (300 s), then `make run` twice (180 s
# each); the second run is timed. Status: build_fail, timeout, crash
# (non-zero exit), fail (printed FAIL), pass (printed PASS), ran (finished,
# printed neither). The two toolchains alternate per program, so a change in
# the machine's state affects both. Only logs are kept (OUT/logs/<tool>/),
# each work copy is deleted. Resumable: programs already in the CSV are
# skipped. Summary: tools/bench/hecbench_summary.py OUT.
set -u
H=${HECBENCH:?set HECBENCH to a HeCBench checkout}/src
OUT=${OUT:?set OUT to a results folder}
OCU_ENV=${OCU_ENV:-$HOME/.opencuda/env.sh}
CHIPSTAR=${CHIPSTAR:-$HOME/tools/chipstar}
ARCH=${ARCH:-sm_75}
CSV=$OUT/results_full.csv
mkdir -p $OUT/logs/ocu $OUT/logs/chip $OUT/work
[ -f $CSV ] || echo "bench,tool,status,seconds" > $CSV
for b in $(ls $H | grep -- '-cuda$'); do
  for tool in ocu chip; do
    grep -q "^$b,$tool," $CSV && continue
    avail=$(df --output=avail -k $OUT | tail -1)
    if [ "$avail" -lt 8000000 ]; then echo "STOP low disk ($avail KB)" >> $CSV; exit 1; fi
    d=$OUT/work/$tool/$b; rm -rf $d; mkdir -p $(dirname $d); cp -r $H/$b $d
    ( cd $d
      if [ $tool = ocu ]; then
        source $OCU_ENV >/dev/null; export OCU_BACKEND=opencl
        CCV="ocu-nvcc"; EX="--ocu-target=spirv"
      else
        export PATH=$CHIPSTAR/bin:$PATH LD_LIBRARY_PATH=$CHIPSTAR/lib CHIP_LOGLEVEL=err
        CCV="cucc"; EX=""
      fi
      if ! timeout 300 make -j8 CC="$CCV" ARCH=$ARCH EXTRA_CFLAGS="$EX" > build.log 2>&1; then echo "$b,$tool,build_fail,"; exit; fi
      timeout 180 make run CC="$CCV" ARCH=$ARCH EXTRA_CFLAGS="$EX" > run1.log 2>&1
      t0=$(date +%s.%N); timeout 180 make run CC="$CCV" ARCH=$ARCH EXTRA_CFLAGS="$EX" > run.log 2>&1; rc=$?; t1=$(date +%s.%N)
      s=$(echo "$t1 - $t0" | bc)
      if [ $rc = 124 ]; then st=timeout
      elif [ $rc != 0 ]; then st=crash
      elif grep -qiE "\bFAIL" run.log; then st=fail
      elif grep -qiE "\bPASS" run.log; then st=pass
      else st=ran; fi
      echo "$b,$tool,$st,$s" ) >> $CSV
    tail -c 200000 $d/build.log > $OUT/logs/$tool/$b.build.log 2>/dev/null
    tail -c 200000 $d/run.log > $OUT/logs/$tool/$b.run.log 2>/dev/null
    rm -rf $d
  done
done
echo DONE >> $CSV

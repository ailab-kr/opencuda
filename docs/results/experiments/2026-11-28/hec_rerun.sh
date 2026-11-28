#!/bin/bash
# HeCBench: the OpenCuda side again, with the development toolkit (main as of today), for the programs
# in ~/cmp/hec/rerun_list.txt. Same rules as hec_full.sh (unmodified sources, make 300 s, make run twice
# 180 s, second timed; NOT PASS counts as fail). Results: results_rerun.csv (tool "ocu_new"), logs in
# full_logs/ocu_new. Resumable.
H=$HOME/cmp/HeCBench/src; OUT=$HOME/cmp/hec; CSV=$OUT/results_rerun.csv; LOG=$OUT/full_logs/ocu_new
mkdir -p $LOG $OUT/wrerun
[ -f $CSV ] || echo "bench,tool,status,seconds" > $CSV
echo "toolkit: $(cd ~/opencuda-dev && git log --oneline -1) + $(cd ~/opencuda-dev && git status -s | wc -l) changed files" > $OUT/RERUN_NOTES.txt
for b in $(cat $OUT/rerun_list.txt); do
  grep -q "^$b,ocu_new," $CSV && continue
  avail=$(df --output=avail -k $HOME | tail -1)
  if [ "$avail" -lt 8000000 ]; then echo "STOP low disk ($avail KB)" >> $CSV; exit 1; fi
  d=$OUT/wrerun/$b; rm -rf $d; cp -r $H/$b $d
  ( cd $d
    source $HOME/.opencuda/env-dev.sh >/dev/null; export OCU_BACKEND=opencl OCU_TARGET=spirv
    if ! timeout 300 make -j8 CC=ocu-nvcc ARCH=sm_75 EXTRA_CFLAGS=--ocu-target=spirv > build.log 2>&1; then echo "$b,ocu_new,build_fail,"; exit; fi
    timeout 180 make run CC=ocu-nvcc ARCH=sm_75 EXTRA_CFLAGS=--ocu-target=spirv > run1.log 2>&1
    t0=$(date +%s.%N); timeout 180 make run CC=ocu-nvcc ARCH=sm_75 EXTRA_CFLAGS=--ocu-target=spirv > run.log 2>&1; rc=$?; t1=$(date +%s.%N)
    s=$(echo "$t1 - $t0" | bc)
    if [ $rc = 124 ]; then st=timeout
    elif [ $rc != 0 ]; then st=crash
    elif grep -qiE "\bFAIL|NOT PASS" run.log; then st=fail
    elif grep -qiE "\bPASS" run.log; then st=pass
    else st=ran; fi
    echo "$b,ocu_new,$st,$s" ) >> $CSV
  tail -c 200000 $d/build.log > $LOG/$b.build.log 2>/dev/null
  tail -c 200000 $d/run.log > $LOG/$b.run.log 2>/dev/null
  rm -rf $d
done
echo DONE >> $CSV

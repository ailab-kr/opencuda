# Per-program command lines for run.sh (sourced). Each case_<name> sets
#   RUN   the command for the build under test (writes $OUT/<name>.out.cuda)
#   REF   the command for the suite's CPU version (writes $OUT/<name>.out.ref)
#   SKIP  why the program is not run (empty when it is)
# $OUT is the work directory; $OUT/in holds gen.py's inputs.
I=$OUT/in
case_bfs() {
    SKIP=
    RUN="$OUT/bfs.cuda -i $I/bfs_graph.txt -o $OUT/bfs.out.cuda"
    build_ref bfs base || { SKIP="reference did not build"; return; }
    REF="$OUT/bfs.ref -i $I/bfs_graph.txt -o $OUT/bfs.out.ref"
}
case_sad() {
    SKIP=
    RUN="$OUT/sad.cuda -i $I/sad_ref.bin,$I/sad_cur.bin -o $OUT/sad.out.cuda"
    build_ref sad cpu || { SKIP="reference did not build"; return; }
    REF="$OUT/sad.ref -i $I/sad_ref.bin,$I/sad_cur.bin -o $OUT/sad.out.ref"
}
# spmv: the suite's own fidapm05.mtx (42 rows, 520 entries, in its source tree)
# and a generated vector. The pattern matrices next to it (jgl009, bcsstk32)
# crash the suite's shared converter in both versions ("free(): invalid
# pointer" from convert_dataset.c, also when built with the host compiler only).
case_spmv() {
    SKIP=
    local cv=$B/spmv/common_src/convert-dataset
    RUN="$OUT/spmv.cuda -i $cv/fidapm05.mtx,$I/spmv_vec.bin -o $OUT/spmv.out.cuda"
    REF_SRCS="$cv/convert_dataset.c $cv/mmio.c" build_ref spmv cpu -I"$cv" \
        || { SKIP="reference did not build"; return; }
    REF="$OUT/spmv.ref -i $cv/fidapm05.mtx,$I/spmv_vec.bin -o $OUT/spmv.out.ref"
}
case_cutcp() { SKIP=; REF=; RUN="$OUT/cutcp.cuda -i $I/cutcp.pqr -o $OUT/cutcp.out.cuda"; }
case_mri-gridding() { RUN=; REF=; SKIP="input is a trajectory file the suite does not ship with its source"; }
case_tpacf() { RUN=; REF=; SKIP="inputs are catalogue files the suite does not ship with its source"; }
case_sgemm() {
    SKIP=
    RUN="$OUT/sgemm.cuda -i $I/sgemm_a.txt,$I/sgemm_b.txt,$I/sgemm_bt.txt -o $OUT/sgemm.out.cuda"
    build_ref sgemm base || { SKIP="reference did not build"; return; }
    REF="$OUT/sgemm.ref -i $I/sgemm_a.txt,$I/sgemm_b.txt,$I/sgemm_bt.txt -o $OUT/sgemm.out.ref"
}
case_mri-q() {
    SKIP=
    RUN="$OUT/mri-q.cuda -i $I/mriq.bin -o $OUT/mri-q.out.cuda"
    build_ref mri-q cpu || { SKIP="reference did not build"; return; }
    REF="$OUT/mri-q.ref -i $I/mriq.bin -o $OUT/mri-q.out.ref"
}
case_stencil() {
    SKIP=
    RUN="$OUT/stencil.cuda -i $I/stencil.bin -o $OUT/stencil.out.cuda -- 16 16 16 8"
    build_ref stencil cpu || { SKIP="reference did not build"; return; }
    REF="$OUT/stencil.ref -i $I/stencil.bin -o $OUT/stencil.out.ref -- 16 16 16 8"
}
case_histo() {
    SKIP=
    RUN="$OUT/histo.cuda -i $I/histo.bin -o $OUT/histo.out.cuda -- 2"
    build_ref histo base || { SKIP="reference did not build"; return; }
    REF="$OUT/histo.ref -i $I/histo.bin -o $OUT/histo.out.ref -- 2"
}
case_lbm() {
    SKIP=
    RUN="$OUT/lbm.cuda -o $OUT/lbm.out.cuda -- 2"
    build_ref lbm cpu || { SKIP="reference did not build"; return; }
    REF="$OUT/lbm.ref -o $OUT/lbm.out.ref -- 2"
}

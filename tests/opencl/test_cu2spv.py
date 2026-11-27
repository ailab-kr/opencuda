#!/usr/bin/env python3
"""Host-side tests of compiler/spirv/cu2spv.py's IR rewrites (no device needed).

legalize_intrinsics: intrinsics LLVM's SPIR-V back end does not legalize are
written with ones it does, keeping their LangRef meaning:
  llvm.lrint / llvm.llrint   -> llvm.rint then fptosi
  llvm.lround / llvm.llround -> llvm.round then fptosi
  llvm.{u,s}{min,max}.i1     -> and / or
With OCU_SPV_LLC (and OCU_SPV_LLVM_AS) set, the rewritten IR also goes
through llc for spirv64, which refused the original.

    [OCU_SPV_LLC=<llc> OCU_SPV_LLVM_AS=<llvm-as>] python3 -m unittest test_cu2spv
"""
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("cu2spv", REPO / "compiler" / "spirv" / "cu2spv.py")
cu2spv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu2spv)
LLC = os.environ.get("OCU_SPV_LLC")
LLVM_AS = os.environ.get("OCU_SPV_LLVM_AS")

MODULE = """target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_kernel void @k(ptr addrspace(1) %o64, ptr addrspace(1) %o32, ptr addrspace(1) %ob, float %x, double %y, i32 %a, i32 %b) {
  %r1 = call i64 @llvm.llrint.i64.f32(float %x)
  %r2 = tail call i32 @llvm.lrint.i32.f64(double %y)
  %r3 = call i64 @llvm.llround.i64.f64(double noundef %y)
  %r4 = call i32 @llvm.lround.i32.f32(float %x) #0
  %p = icmp slt i32 %a, %b
  %q = icmp ugt i32 %a, 7
  %m1 = call i1 @llvm.umin.i1(i1 %p, i1 %q)
  %m2 = call i1 @llvm.umax.i1(i1 %p, i1 %q)
  %m3 = tail call i1 @llvm.smin.i1(i1 %p, i1 %q)
  %m4 = call i1 @llvm.smax.i1(i1 noundef %p, i1 noundef %q)
  %s1 = add i64 %r1, %r3
  %w2 = sext i32 %r2 to i64
  %w4 = sext i32 %r4 to i64
  %s2 = add i64 %w2, %w4
  %s = add i64 %s1, %s2
  store i64 %s, ptr addrspace(1) %o64
  %b1 = and i1 %m1, %m2
  %b2 = or i1 %m3, %m4
  %bb = xor i1 %b1, %b2
  %z = zext i1 %bb to i32
  store i32 %z, ptr addrspace(1) %ob
  ret void
}

declare i64 @llvm.llrint.i64.f32(float)
declare i32 @llvm.lrint.i32.f64(double)
declare i64 @llvm.llround.i64.f64(double)
declare i32 @llvm.lround.i32.f32(float)
declare i1 @llvm.umin.i1(i1, i1)
declare i1 @llvm.umax.i1(i1, i1)
declare i1 @llvm.smin.i1(i1, i1)
declare i1 @llvm.smax.i1(i1, i1)
attributes #0 = { nounwind }
"""


class Legalize(unittest.TestCase):
    def setUp(self):
        self.out = cu2spv.legalize_intrinsics(MODULE)

    def test_rint_round_to_int(self):
        o = self.out
        self.assertIn("%ocu_rint.1 = call float @llvm.rint.f32(float %x)", o)
        self.assertIn("%r1 = fptosi float %ocu_rint.1 to i64", o)
        self.assertIn("%ocu_rint.2 = call double @llvm.rint.f64(double %y)", o)
        self.assertIn("%r2 = fptosi double %ocu_rint.2 to i32", o)
        self.assertIn("%ocu_round.3 = call double @llvm.round.f64(double %y)", o)
        self.assertIn("%r3 = fptosi double %ocu_round.3 to i64", o)
        self.assertIn("%ocu_round.4 = call float @llvm.round.f32(float %x)", o)
        self.assertIn("%r4 = fptosi float %ocu_round.4 to i32", o)
        for d in ("declare float @llvm.rint.f32(float)", "declare double @llvm.rint.f64(double)",
                  "declare double @llvm.round.f64(double)", "declare float @llvm.round.f32(float)"):
            self.assertEqual(o.count(d), 1, d)
        for name in ("llrint", "lround", "llround"):
            self.assertNotIn(f"call i64 @llvm.{name}.", o)
            self.assertNotIn(f"call i32 @llvm.{name}.", o)
        self.assertNotIn("call i32 @llvm.lrint.", o)

    def test_i1_min_max(self):
        o = self.out
        # unsigned: true (1) > false (0); signed: true (-1) < false (0)
        self.assertIn("%m1 = and i1 %p, %q", o)
        self.assertIn("%m2 = or i1 %p, %q", o)
        self.assertIn("%m3 = or i1 %p, %q", o)
        self.assertIn("%m4 = and i1 %p, %q", o)
        self.assertNotRegex(o, r"call i1 @llvm\.[us](min|max)\.i1\(")

    def test_numbered_values(self):
        # clang's output numbers its values: the new value must be a named one
        src = "  %24 = call i64 @llvm.llrint.i64.f32(float %23)\n"
        o = cu2spv.legalize_intrinsics(src)
        self.assertIn("%ocu_rint.1 = call float @llvm.rint.f32(float %23)", o)
        self.assertIn("%24 = fptosi float %ocu_rint.1 to i64", o)

    def test_second_run_new_names(self):
        src = "  %ocu_rint.4 = call float @llvm.rint.f32(float %a)\n  %v = call i64 @llvm.llrint.i64.f32(float %b)\n"
        self.assertIn("%ocu_rint.5 = call float @llvm.rint.f32(float %b)", cu2spv.legalize_intrinsics(src))

    def test_logical_and_or(self):
        src = ("  %a = select i1 %p, i1 %q, i1 false\n"
               "  %o = select i1 %p, i1 true, i1 %q\n"
               "  %s = select i1 %p, i1 %q, i1 %r\n"
               "  %w = select i1 %p, i32 %x, i32 0\n")
        o = cu2spv.legalize_intrinsics(src)
        self.assertIn("%a = and i1 %p, %q", o)
        self.assertIn("%o = or i1 %p, %q", o)
        self.assertIn("%s = select i1 %p, i1 %q, i1 %r", o)   # not a logical and / or
        self.assertIn("%w = select i1 %p, i32 %x, i32 0", o)  # not i1

    @unittest.skipUnless(LLC and LLVM_AS, "OCU_SPV_LLC / OCU_SPV_LLVM_AS not set")
    def test_llc_logical_or(self):
        # llc -O2 for spirv64 made G_UMIN/G_UMAX on s1 from such selects and stopped
        # (HeCBench unfold-cuda); with plain or it compiles
        # (llvm-reduce of that program's kernel; llc 23.1.6 stops on the original)
        mod = MODULE.split("define spir_kernel")[0] + """
define spir_kernel void @l(i1 %a, i1 %b) {
entry:
  %x = select i1 %a, i1 true, i1 %b
  %w = zext i1 %x to i64
  br label %loop

loop:
  %i = phi i64 [ %w, %entry ], [ %n, %loop ]
  %n = add i64 %i, 1
  %e = icmp eq i64 %i, 1
  br label %loop
}
"""
        def llc(text, tmp, name):
            ll, bc, spv = (Path(tmp) / f"{name}{e}" for e in (".ll", ".bc", ".spv"))
            ll.write_text(text)
            r = subprocess.run([LLVM_AS, str(ll), "-o", str(bc)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            return subprocess.run([LLC, "-mtriple=spirv64-unknown-unknown", "-O2", "-filetype=obj", str(bc), "-o", str(spv)],
                                  capture_output=True, text=True)
        with tempfile.TemporaryDirectory() as tmp:
            before = llc(mod, tmp, "before")
            after = llc(cu2spv.legalize_intrinsics(mod), tmp, "after")
            self.assertEqual(after.returncode, 0, after.stderr[-2000:])
            if before.returncode == 0:
                self.skipTest("this llc compiles the logical or as it is (the rewrite is not needed there)")

    def test_wider_min_max_untouched(self):
        src = "  %v = call i32 @llvm.umin.i32(i32 %a, i32 %b)\n"
        self.assertEqual(cu2spv.legalize_intrinsics(src), src)

    def test_existing_declaration_kept_once(self):
        src = MODULE + "declare float @llvm.rint.f32(float)\n"
        self.assertEqual(cu2spv.legalize_intrinsics(src).count("declare float @llvm.rint.f32(float)"), 1)

    @unittest.skipUnless(LLC and LLVM_AS, "OCU_SPV_LLC / OCU_SPV_LLVM_AS not set")
    def test_llc_takes_the_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in (("orig", MODULE), ("new", self.out)):
                ll, bc, spv = (Path(tmp) / f"{name}{e}" for e in (".ll", ".bc", ".spv"))
                ll.write_text(text)
                r = subprocess.run([LLVM_AS, str(ll), "-o", str(bc)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                r = subprocess.run([LLC, "-mtriple=spirv64-unknown-unknown", "-O2", "-filetype=obj", str(bc), "-o", str(spv)],
                                   capture_output=True, text=True)
                if name == "new":
                    self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                    self.assertGreater(spv.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

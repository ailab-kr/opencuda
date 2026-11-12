"""The BLAS suite checks itself: a correct float32-rounding implementation
passes every case, and every deliberate mutant fails at least one."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # conformance/
sys.path.insert(0, str(ROOT))

from harness.ocu_conf import blas_check  # noqa: E402
from mutants.blas_candidates import MUTANTS, Naive  # noqa: E402
from oracle import cases as C  # noqa: E402

_CASES = None


def cases():
    global _CASES
    if _CASES is None:
        _CASES = C.all_cases()
    return _CASES


class SuiteShape(unittest.TestCase):
    def test_covers_all_twenty_functions(self):
        cards = {c["card"] for c in cases()}
        want = {f"blas-{n}" for n in (
            "saxpy daxpy sdot ddot sscal dscal snrm2 dnrm2 sasum isamax "
            "scopy sswap sgemv dgemv sger strsv sgemm dgemm ssyrk strsm").split()}
        self.assertEqual(cards, want)

    def test_ids_unique(self):
        ids = [c["id"] for c in cases()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_deterministic(self):
        self.assertEqual([c["id"] for c in C.all_cases()],
                         [c["id"] for c in cases()])


class SelfCheck(unittest.TestCase):
    def test_naive_correct_implementation_passes(self):
        results = blas_check.run_suite(cases(), Naive())
        failed = [r for r in results if r["status"] != "pass"]
        self.assertEqual(failed[:5], [], f"{len(failed)} of {len(results)} failed")
        self.assertLess(max(r["worst"] for r in results), 1.0)

    def test_every_mutant_is_caught(self):
        survivors = []
        for name, impl in MUTANTS:
            results = blas_check.run_suite(cases(), impl)
            if all(r["status"] == "pass" for r in results):
                survivors.append(name)
        self.assertEqual(survivors, [])

    def test_check_rejects_missing_and_wrong_length(self):
        case = next(c for c in cases() if c["fn"] == "axpy" and c["args"]["n"] == 7)
        tol = blas_check.load_tolerances()
        self.assertEqual(blas_check.check(case, {}, tol)["status"], "fail")
        self.assertEqual(blas_check.check(case, {"y": [0.0]}, tol)["status"], "fail")
        self.assertEqual(blas_check.check(case, dict(case["expect"]), tol)["status"], "pass")


class Subset(unittest.TestCase):
    def test_only_selects_functions_and_kinds(self):
        from harness.ocu_conf.blas_run import select
        sub = select(cases(), "gemm,dot:f32")
        self.assertTrue(sub)
        self.assertEqual({(c["fn"], c["kind"]) for c in sub},
                         {("gemm", "f32"), ("gemm", "f64"), ("dot", "f32")})
        self.assertIs(select(cases(), None), cases())
        with self.assertRaises(ValueError):
            select(cases(), "nosuchfn")
        with self.assertRaises(ValueError):
            select(cases(), "gemm,,dot")


if __name__ == "__main__":
    unittest.main()

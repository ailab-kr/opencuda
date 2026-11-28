import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))
import hecbench_xcheck as xc  # noqa: E402

REF = """Running the kernel
Average kernel execution time: 12.5 (us)
checksum = 1234.5678
max error 0.000012
result[0] = -3.25e-02 result[1] = 7
Bandwidth 512.3 GB/s
"""


class XcheckTests(unittest.TestCase):
    def test_timings_left_out(self):
        vals = [v for _, v, _ in xc.numbers(REF)]
        self.assertEqual(vals, [1234.5678, 0.000012, 0.0, -3.25e-02, 1.0, 7.0])

    def test_same_with_other_timings(self):
        test = REF.replace("12.5", "99.0").replace("512.3", "1100.0")
        self.assertEqual(xc.compare(REF, test, 1e-3, 1e-6, 3)[0], "same")

    def test_within_tolerance(self):
        test = REF.replace("1234.5678", "1234.6")
        self.assertEqual(xc.compare(REF, test, 1e-3, 1e-6, 3)[0], "same")

    def test_differs(self):
        test = REF.replace("1234.5678", "1300.1")
        st, notes = xc.compare(REF, test, 1e-3, 1e-6, 3)
        self.assertEqual(st, "differs")
        self.assertIn("1 of 6", notes[0])

    def test_count_mismatch(self):
        st, _ = xc.compare(REF, REF + "extra 5\n", 1e-3, 1e-6, 3)
        self.assertEqual(st, "differs")

    def test_nan(self):
        self.assertTrue(xc.same(float("nan"), float("nan"), 0, 0))
        self.assertFalse(xc.same(float("nan"), 1.0, 0, 0))
        self.assertEqual(xc.compare("x = nan\n", "x = 1.5\n", 1e-3, 1e-6, 3)[0], "differs")

    def test_seconds_and_performance_left_out(self):
        text = "0.046070967000 s\ndone in 1.27 s for 1 iterations\n# Average Performance : 1616.74 +- 56.8\nsum 3\n"
        self.assertEqual([v for _, v, _ in xc.numbers(text)], [3.0])

    def test_no_numbers(self):
        self.assertEqual(xc.compare("done\n", "done\n", 1e-3, 1e-6, 3)[0], "no_numbers")


if __name__ == "__main__":
    unittest.main()

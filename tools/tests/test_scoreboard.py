import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent))
import scoreboard as sb  # noqa: E402


class ReaderTests(unittest.TestCase):
    def test_record(self):
        meta, tests = sb.read_record(DATA / "record_cpu.json")
        self.assertEqual(meta["backend"], "cpu")
        self.assertEqual(len(tests), 5)
        self.assertEqual(tests[0].area, "conformance L4 libs/blas")
        self.assertEqual(sb.counts(tests)["fail"], 1)

    def test_ctest_text_backends(self):
        tests = sb.read_ctest(DATA / "ctest.txt", "cpu")
        by = {t.id: t for t in tests}
        self.assertEqual(by["cpu_backend"].backend, "cpu")
        self.assertEqual(by["gpu_rt_api"].backend, "nvidia")
        self.assertTrue(by["metal_cu2msl_host"].backend.startswith("metal"))
        self.assertEqual(by["metal_cu2msl_host"].status, "fail")
        self.assertEqual(by["nvidia_backend"].status, "skip")
        self.assertAlmostEqual(by["gpu_rt_api"].time_ms, 260)

    def test_ctest_junit(self):
        tests = sb.read_ctest(DATA / "ctest_junit.xml", "cpu")
        self.assertEqual([t.status for t in tests], ["pass", "fail", "skip"])
        self.assertEqual(tests[1].area, "ctest: libs (BLAS)")

    def test_rodinia(self):
        tests = sb.read_rodinia(DATA / "rodinia.txt", "cpu")
        self.assertEqual([t.status for t in tests], ["pass", "pass", "skip", "fail"])

    def test_rodinia_drops_folders(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.txt"
            p.write_text("kmeans BUILD-FAIL /home/someone/x/y/k.cu:15:1: error: bad\n")
            self.assertNotIn("/home", sb.read_rodinia(p, "cpu")[0].note)

    def test_empty_inputs_are_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.txt"
            p.write_text("nothing\n")
            with self.assertRaises(ValueError):
                sb.read_ctest(p, "cpu")
            with self.assertRaises(ValueError):
                sb.read_rodinia(p, "cpu")


class SummaryTests(unittest.TestCase):
    def test_speed_median_and_slow_tail(self):
        _, tests = sb.read_record(DATA / "record_cpu.json")
        s = sb.speed_summary(tests)
        self.assertEqual(s["time_median_ms"], 2.0)
        self.assertAlmostEqual(s["time_p90_ms"], 2.8)
        self.assertEqual(s["ratio_median"], 0.5)
        self.assertAlmostEqual(s["ratio_p10"], 0.3)
        self.assertEqual(s["slowest"], "L4.blas.sdot.a")

    def test_no_speed_data(self):
        s = sb.speed_summary(sb.read_rodinia(DATA / "rodinia.txt", "cpu"))
        self.assertIsNone(s["time_median_ms"])

    def test_samples_show_failures_first(self):
        _, tests = sb.read_record(DATA / "record_cpu.json")
        blas = [t for t in tests if t.area.endswith("blas")]
        self.assertEqual(sb.pick_samples(blas)[0].status, "fail")


class OutputTests(unittest.TestCase):
    def test_cli_writes_both_files(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [sys.executable, str(HERE.parent / "scoreboard.py"),
                 str(DATA / "record_cpu.json"),
                 "--ctest", f"cpu={DATA / 'ctest.txt'}",
                 "--rodinia", f"cpu={DATA / 'rodinia.txt'}",
                 "--out-dir", d],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            md = (Path(d) / "scoreboard.md").read_text(encoding="utf-8")
            page = (Path(d) / "scoreboard.html").read_text(encoding="utf-8")
            self.assertIn("conformance L4 libs/blas", md)
            self.assertIn("3 / 4 (1 fail)", md)
            self.assertIn("Sample outputs", md)
            self.assertNotIn("<script", page)
            self.assertNotIn("http://", page)
            self.assertNotIn("https://", page)

    def test_bad_record_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad_result.json"
            bad.write_text("{}")
            r = subprocess.run([sys.executable, str(HERE.parent / "scoreboard.py"),
                                str(bad), "--out-dir", d], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()

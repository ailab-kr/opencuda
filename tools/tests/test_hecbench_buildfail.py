import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))
import hecbench_buildfail as bf  # noqa: E402


class BuildFailTests(unittest.TestCase):
    def test_thrust(self):
        log = "main.cu:5:10: fatal error: 'thrust/transform.h' file not found\n"
        self.assertEqual(bf.classify(log)[0], "Thrust")

    def test_harness_copy(self):
        log = "make: *** No rule to make target '../include/SDKBitMap.h', needed by 'main.o'.  Stop.\n"
        self.assertTrue(bf.classify(log)[0].startswith("files outside the program's folder"))

    def test_missing_runtime_function(self):
        log = "main.cu:9:3: error: use of undeclared identifier 'cudaGraphCreate'\n"
        self.assertEqual(bf.classify(log)[0], "CUDA graphs / driver API")

    def test_first_error_and_other(self):
        log = "warning: unused\nmain.cu:1:1: error: something new\n"
        cause, err = bf.classify(log)
        self.assertEqual(cause, "other")
        self.assertEqual(err, "main.cu:1:1: error: something new")


if __name__ == "__main__":
    unittest.main()

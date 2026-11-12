import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from ocu_conf import results, score

DATA = Path(__file__).parent / "data" / "example_result.json"


def example() -> dict:
    with open(DATA, encoding="utf-8") as handle:
        return json.load(handle)


class ValidateTests(unittest.TestCase):
    def test_example_is_valid(self):
        self.assertEqual(results.validate(example()), [])
        self.assertEqual(results.load(DATA)["schema"], 1)

    def test_wrong_schema_version(self):
        record = example()
        record["schema"] = 99
        self.assertTrue(any("schema" in p for p in results.validate(record)))

    def test_missing_device_field(self):
        record = example()
        del record["device"]["driver"]
        self.assertIn("device.driver: missing or empty", results.validate(record))

    def test_unknown_status_and_level(self):
        record = example()
        record["tests"][0]["status"] = "maybe"
        record["tests"][1]["level"] = "L9"
        problems = results.validate(record)
        self.assertTrue(any("tests[0].status" in p for p in problems))
        self.assertTrue(any("tests[1].level" in p for p in problems))

    def test_skip_needs_reason(self):
        record = example()
        del record["tests"][3]["reason"]
        self.assertTrue(any("reason" in p for p in results.validate(record)))

    def test_duplicate_ids(self):
        record = example()
        record["tests"].append(copy.deepcopy(record["tests"][0]))
        self.assertTrue(any("duplicate" in p for p in results.validate(record)))

    def test_negative_and_boolean_numbers_rejected(self):
        record = example()
        record["tests"][0]["speed_ratio"] = -1
        record["tests"][1]["tolerance"] = True
        problems = results.validate(record)
        self.assertTrue(any("tests[0].speed_ratio" in p for p in problems))
        self.assertTrue(any("tests[1].tolerance" in p for p in problems))

    def test_not_an_object(self):
        self.assertEqual(results.validate([]), ["record: not an object"])

    def test_load_raises_on_invalid(self):
        record = example()
        record["run"]["set"] = "secret"
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bad.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(record, handle)
            with self.assertRaises(ValueError):
                results.load(path)


class ScoreTests(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(score.exact_match([1, 2, 3], [1, 2, 3]))
        self.assertFalse(score.exact_match([1, 2, 3], [1, 2, 4]))
        self.assertFalse(score.exact_match([1, 2], [1, 2, 3]))

    def test_relative_error(self):
        self.assertAlmostEqual(score.max_relative_error([1.01], [1.0]), 0.01)
        self.assertEqual(score.max_relative_error([1.0, 2.0], [1.0, 2.0]), 0.0)
        self.assertTrue(score.within_tolerance([1.0000001], [1.0], 1e-6))
        self.assertFalse(score.within_tolerance([1.001], [1.0], 1e-6))

    def test_relative_error_special_values(self):
        nan, inf = math.nan, math.inf
        self.assertEqual(score.max_relative_error([nan], [nan]), 0.0)
        self.assertEqual(score.max_relative_error([nan], [1.0]), inf)
        self.assertEqual(score.max_relative_error([inf], [inf]), 0.0)
        self.assertEqual(score.max_relative_error([inf], [-inf]), inf)
        self.assertEqual(score.max_relative_error([1.0], [inf]), inf)
        self.assertEqual(score.max_relative_error([1.0], [1.0, 2.0]), inf)

    def test_zero_expected_does_not_divide_by_zero(self):
        self.assertEqual(score.max_relative_error([0.0], [0.0]), 0.0)
        self.assertGreater(score.max_relative_error([1e-3], [0.0]), 1.0)

    def test_percentile(self):
        self.assertEqual(score.percentile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(score.percentile([0.5, 0.9], 0.1), 0.54)
        self.assertEqual(score.percentile([7], 0.1), 7)
        with self.assertRaises(ValueError):
            score.percentile([], 0.5)
        with self.assertRaises(ValueError):
            score.percentile([1], 1.5)

    def test_summarize(self):
        summary = score.summarize(example()["tests"])
        self.assertEqual(summary["pass"], 2)
        self.assertEqual(summary["fail"], 1)
        self.assertEqual(summary["skip"], 1)
        self.assertEqual(summary["unsupported"], 0)
        self.assertAlmostEqual(summary["speed_ratio_median"], 0.7)
        self.assertAlmostEqual(summary["speed_ratio_p10"], 0.54)
        self.assertEqual(summary["slowest_test"], "L4.blas.sdot.n1024_incx1")

    def test_summarize_without_timings(self):
        summary = score.summarize([{"id": "a", "status": "fail"}])
        self.assertIsNone(summary["speed_ratio_median"])
        self.assertIsNone(summary["slowest_test"])


if __name__ == "__main__":
    unittest.main()

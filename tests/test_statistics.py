import unittest

from src.statistics import paired_bootstrap_delta, paired_significance


class TestPairedStatistics(unittest.TestCase):
    def test_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap_delta([2, 3, 4], [1, 2, 3], seed=7, resamples=1000)
        second = paired_bootstrap_delta([2, 3, 4], [1, 2, 3], seed=7, resamples=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["mean_delta"], 1.0)

    def test_rejects_nonfinite_and_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_delta([1], [1, 2])
        with self.assertRaises(ValueError):
            paired_bootstrap_delta([float("nan")], [1])

    def test_zero_delta_significance(self) -> None:
        result = paired_significance([1, 2], [1, 2])
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["cohen_dz"], 0.0)


if __name__ == "__main__":
    unittest.main()

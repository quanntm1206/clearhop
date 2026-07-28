import unittest

import numpy as np


class TestCalibrationMetrics(unittest.TestCase):
    def test_projection_gain_and_polarity_for_scaled_outputs(self) -> None:
        from src.evaluate import calibration_metrics

        clean = np.asarray([1.0, -1.0])

        self.assertEqual(calibration_metrics(clean, clean)["projection_gain"], 1.0)
        self.assertEqual(calibration_metrics(clean, clean * 0.5)["projection_gain"], 0.5)
        self.assertEqual(calibration_metrics(clean, -clean)["projection_gain"], -1.0)
        self.assertEqual(calibration_metrics(clean, clean)["polarity_failure"], 0.0)
        self.assertEqual(calibration_metrics(clean, clean * 0.5)["polarity_failure"], 0.0)
        self.assertEqual(calibration_metrics(clean, -clean)["polarity_failure"], 1.0)

    def test_gain_error_uses_gain_magnitude(self) -> None:
        from src.evaluate import calibration_metrics

        clean = np.asarray([1.0, -1.0])

        self.assertAlmostEqual(calibration_metrics(clean, clean * 0.5)["gain_error_db"], -6.0206, places=4)
        self.assertEqual(calibration_metrics(clean, clean)["gain_error_db"], 0.0)
        self.assertEqual(calibration_metrics(clean, -clean)["gain_error_db"], 0.0)

    def test_aggregate_includes_distribution_statistics(self) -> None:
        from src.evaluate import _aggregate

        aggregate = _aggregate([{"projection_gain": 1.0}, {"projection_gain": 3.0}])

        self.assertEqual(
            set(aggregate["projection_gain"]),
            {"mean", "std", "median", "min", "max"},
        )


if __name__ == "__main__":
    unittest.main()

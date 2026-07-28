from __future__ import annotations

import unittest

from src.evaluate import evaluation_profile_options


class TestEvaluationProfiles(unittest.TestCase):
    def test_screen_profile_keeps_promotion_metrics_but_skips_expensive_pesq(self) -> None:
        outputs, compute_stoi, compute_pesq = evaluation_profile_options("screen")

        self.assertEqual(outputs, ("noisy", "enhanced"))
        self.assertTrue(compute_stoi)
        self.assertFalse(compute_pesq)

    def test_full_profile_preserves_existing_metric_surface(self) -> None:
        outputs, compute_stoi, compute_pesq = evaluation_profile_options("full")

        self.assertEqual(outputs, ("noisy", "mask_only", "enhanced"))
        self.assertTrue(compute_stoi)
        self.assertTrue(compute_pesq)

    def test_unknown_profile_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown evaluation profile"):
            evaluation_profile_options("fast-ish")


if __name__ == "__main__":
    unittest.main()

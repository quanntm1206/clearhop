import unittest


class TestFailureAnalysis(unittest.TestCase):
    def test_worst_cases_are_deterministic_and_classified(self) -> None:
        from scripts.analyze_failures import summarize_failures

        report = summarize_failures([
            {"index": 2, "si_sdr_delta": -0.4, "snr_delta": -0.1, "stoi_delta": -0.01, "metadata": {"snr_band": "low", "noise_family": "esc50", "speaker": "1"}},
            {"index": 1, "si_sdr_delta": 0.2, "snr_delta": 0.1, "stoi_delta": 0.0},
        ])
        self.assertEqual(report["worst_cases"][0]["index"], 2)
        self.assertEqual(report["failure_counts"]["si_sdr_delta"], 1)
        self.assertEqual(report["failure_counts"]["snr_non_positive"], 1)
        self.assertEqual(report["subgroup_failure_counts"]["snr_band:low"], 1)

    def test_rejects_malformed_items(self) -> None:
        from scripts.analyze_failures import summarize_failures

        with self.assertRaises(ValueError):
            summarize_failures([{"index": 0}], 1)


if __name__ == "__main__":
    unittest.main()

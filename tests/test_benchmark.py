import unittest


class TestBenchmarkHelpers(unittest.TestCase):
    def test_latency_summary_contains_realtime_factor(self) -> None:
        from scripts.benchmark import summarize_latencies

        summary = summarize_latencies([0.001, 0.002, 0.003], hop_seconds=0.01)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["mean_ms"], 2.0)
        self.assertEqual(summary["realtime_factor"], 5.0)


if __name__ == "__main__":
    unittest.main()

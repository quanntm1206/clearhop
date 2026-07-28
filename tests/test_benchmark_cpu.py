import unittest


class TestCpuBenchmark(unittest.TestCase):
    def test_latency_summary_has_p99_and_realtime_factor(self) -> None:
        from scripts.benchmark_cpu import summarize_cpu_latencies

        summary = summarize_cpu_latencies([0.001, 0.002, 0.003], 0.01)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["mean_ms"], 2.0)
        self.assertIn("p99_ms", summary)
        self.assertEqual(summary["realtime_factor"], 5.0)

    def test_latency_summary_rejects_nonfinite(self) -> None:
        from scripts.benchmark_cpu import summarize_cpu_latencies

        with self.assertRaises(ValueError):
            summarize_cpu_latencies([float("nan")], 0.01)

    def test_preferred_affinity_cpu_is_honored_when_available(self) -> None:
        from scripts.benchmark_cpu import select_affinity_cpu

        self.assertEqual(select_affinity_cpu(0b1111, 3), 3)
        self.assertEqual(select_affinity_cpu(0b0011, 3), 0)

    def test_onnx_session_is_single_threaded_for_single_cpu_benchmark(self) -> None:
        import onnxruntime as ort

        from scripts.benchmark_cpu import onnx_session_options

        options = onnx_session_options(ort)
        self.assertEqual(options.intra_op_num_threads, 1)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)


if __name__ == "__main__":
    unittest.main()

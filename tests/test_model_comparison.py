import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.model_comparison import (
    ArrayAdapter,
    BenchmarkItem,
    DeepFilterNet3Adapter,
    FrozenHistoricalAdapter,
    build_default_report,
    run_model_comparison,
    validate_model_comparison,
    load_research_baselines,
)


class TestModelComparison(unittest.TestCase):
    def _items(self):
        clean = np.sin(np.linspace(0, 2 * np.pi, 320, endpoint=False)).astype(np.float32)
        noisy = (clean + 0.1).astype(np.float32)
        return [
            BenchmarkItem("item-a", noisy, clean, 16000, "primary"),
            BenchmarkItem("item-b", noisy * 0.9, clean * 0.9, 16000, "audit"),
        ]

    def test_same_manifest_ids_and_finite_metrics(self):
        items = self._items()
        adapter = ArrayAdapter("identity", lambda x, _: x)
        report = run_model_comparison(items, [adapter], include_optional_metrics=False)
        self.assertEqual(report["models"][0]["item_ids"], ["item-a", "item-b"])
        self.assertTrue(report["models"][0]["metrics"]["snri_db"]["available"])
        self.assertTrue(validate_model_comparison(report))

    def test_native_rate_conversion_is_explicit(self):
        items = self._items()
        adapter = ArrayAdapter("native", lambda x, _: x, native_sample_rate=8000)
        report = run_model_comparison(items, [adapter], include_optional_metrics=False)
        conversion = report["models"][0]["sample_rate"]
        self.assertEqual(conversion["native"], 8000)
        self.assertEqual(conversion["input"]["from"], 16000)
        self.assertEqual(conversion["input"]["to"], 8000)
        self.assertEqual(conversion["method"], "scipy.resample_poly")

    def test_nonfinite_output_fails_not_passes(self):
        items = self._items()
        adapter = ArrayAdapter("nan", lambda x, _: np.full_like(x, np.nan))
        report = run_model_comparison(items, [adapter], include_optional_metrics=False)
        self.assertEqual(report["models"][0]["status"], "failed")
        self.assertFalse(validate_model_comparison(report))

    def test_missing_external_dependency_is_blocked(self):
        adapter = DeepFilterNet3Adapter(command=["definitely-missing-noise-reducer"])
        report = run_model_comparison(self._items(), [adapter], include_optional_metrics=False)
        row = report["models"][0]
        self.assertEqual(row["status"], "blocked")
        self.assertIn("not found", row["blocked_reason"])
        self.assertEqual(row["item_ids"], ["item-a", "item-b"])

    def test_latency_receipt_reproducible_with_injected_clock(self):
        items = self._items()
        adapter = ArrayAdapter("identity", lambda x, _: x)

        def run_once():
            ticks = iter([1.0, 1.002, 2.0, 2.004])
            report = run_model_comparison(items, [adapter], clock=lambda: next(ticks), include_optional_metrics=False)
            return report["models"][0]["latency_receipt_sha256"]

        self.assertEqual(run_once(), run_once())

    def test_frozen_receipt_adapter_preserves_manifest_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            manifest.write_text("{\"id\": 10}\n{\"id\": 11}\n", encoding="utf-8")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({
                "metadata": {"manifest": str(manifest), "slice_offset": 0, "slice_count": 2, "checkpoint_sha256": "abc"},
                "enhanced": {"snr_improvement_mean": 1.0, "si_sdr_improvement_mean": 2.0, "stoi": {"mean": 0.8}, "pesq": {"mean": 1.5}},
            }), encoding="utf-8")
            adapter = FrozenHistoricalAdapter(receipt, manifest)
            clean = self._items()[0].clean
            items = [BenchmarkItem("10", clean, clean, 16000, "primary"), BenchmarkItem("11", clean, clean, 16000, "primary")]
            report = run_model_comparison(items, [adapter], include_optional_metrics=False)
            self.assertEqual(report["models"][0]["status"], "reproduced_local")
            self.assertEqual(report["models"][0]["item_ids"], ["10", "11"])

    def test_default_report_marks_external_models_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            report = build_default_report(Path.cwd(), output=path)
            names = {row["name"]: row["status"] for row in report["models"]}
            self.assertEqual(names["DeepFilterNet3"], "blocked")
            self.assertTrue(path.exists())

    def test_baseline_config_contains_provenance_and_repro_recipe(self):
        config = load_research_baselines(Path.cwd() / "configs" / "research_baselines.json")
        self.assertGreaterEqual(len(config["baselines"]), 4)
        for baseline in config["baselines"]:
            self.assertTrue(baseline["source_url"].startswith("https://"))
            self.assertTrue(baseline["license"])
            self.assertIn("native_sample_rate", baseline)
            self.assertIn("provenance", baseline)
            provenance = baseline["provenance"]
            self.assertIn("commit", provenance)
            self.assertIn("retrieved_at", provenance)
            self.assertIn("weight_sha256", provenance)
            self.assertIn("recipe", baseline)

    def test_report_rows_expose_provenance_and_environment(self):
        report = build_default_report(Path.cwd())
        for row in report["models"]:
            self.assertIn("provenance", row)
            self.assertIn("environment", row)
            self.assertIn("command", row["environment"])
            self.assertIn("weights", row["provenance"])
            self.assertIn("weight_sha256", row["provenance"]["weights"])

    def test_public_report_has_no_failed_rows_and_schema(self):
        report = json.loads((Path.cwd() / "reports" / "public" / "model_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], 2)
        self.assertTrue(validate_model_comparison(report))
        self.assertTrue(all(row["status"] in {"reproduced_local", "literature_only", "blocked"} for row in report["models"]))


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.model_comparison import (
    ArrayAdapter,
    BenchmarkItem,
    ReproducedExternalAdapter,
    canonical_receipt_sha256,
    run_model_comparison,
)
from scripts.reproduce_deepfilternet3 import build_reproduction_receipt, installed_package_version, unpack_init_result


class TestDeepFilterNetReproduction(unittest.TestCase):
    def _items(self):
        clean = np.sin(np.linspace(0, 8 * np.pi, 1600, endpoint=False)).astype(np.float32)
        noisy = (clean + 0.05).astype(np.float32)
        return [BenchmarkItem("item-0", noisy, clean, 16000, "primary_comparison")]

    def test_receipt_binds_native_rate_metrics_ids_and_provenance(self):
        adapter = ArrayAdapter("DeepFilterNet3", lambda audio, _: audio, native_sample_rate=48000)
        report = run_model_comparison(self._items(), [adapter], include_optional_metrics=False)
        receipt = build_reproduction_receipt(
            report["models"][0],
            source_commit="d375b2d8309e0935d165700c91da9de862a99c31",
            model_archive_sha256="a" * 64,
            model_checkpoint_sha256="b" * 64,
            manifest_sha256="c" * 64,
            package_versions={"deepfilternet": "0.5.6", "torch": "test"},
        )
        row = receipt["model"]
        self.assertEqual(receipt["status"], "reproduced_local")
        self.assertEqual(row["item_ids"], ["item-0"])
        self.assertEqual(row["sample_rate"]["native"], 48000)
        self.assertEqual(row["sample_rate"]["input"], {"from": 16000, "to": 48000})
        self.assertEqual(row["sample_rate"]["output"], {"from": 48000, "to": 16000})
        self.assertEqual(row["provenance"]["commit"], "d375b2d8309e0935d165700c91da9de862a99c31")
        self.assertEqual(row["provenance"]["weights"]["weight_sha256"], "b" * 64)
        self.assertEqual(receipt["receipt_sha256"], canonical_receipt_sha256(receipt))

    def test_external_receipt_adapter_rejects_manifest_id_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deepfilternet3.json"
            payload = {
                "schema_version": 1,
                "status": "reproduced_local",
                "model": {
                    "name": "DeepFilterNet3",
                    "status": "reproduced_local",
                    "item_ids": ["different"],
                    "metrics": {},
                    "sample_rate": {"reference": 16000, "native": 48000},
                    "provenance": {},
                },
            }
            payload["receipt_sha256"] = canonical_receipt_sha256(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
            adapter = ReproducedExternalAdapter(path, name="DeepFilterNet3")
            report = run_model_comparison(self._items(), [adapter], include_optional_metrics=False)
            self.assertEqual(report["models"][0]["status"], "failed")

    def test_unpack_init_result_supports_pinned_package_contract(self):
        model, state, suffix, epoch = unpack_init_result(("model", "state", "suffix"))
        self.assertEqual((model, state, suffix), ("model", "state", "suffix"))
        self.assertIsNone(epoch)

    def test_external_receipt_adapter_rejects_hash_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deepfilternet3.json"
            payload = {
                "schema_version": 1,
                "status": "reproduced_local",
                "model": {
                    "name": "DeepFilterNet3",
                    "status": "reproduced_local",
                    "item_ids": ["item-0"],
                    "metrics": {},
                    "sample_rate": {"reference": 16000, "native": 48000},
                    "provenance": {},
                },
            }
            payload["receipt_sha256"] = canonical_receipt_sha256(payload)
            payload["model"]["item_ids"] = ["tampered"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            adapter = ReproducedExternalAdapter(path, name="DeepFilterNet3")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                adapter.summary_result(["tampered"])

    def test_package_version_falls_back_to_module_metadata(self):
        with patch("scripts.reproduce_deepfilternet3.importlib.metadata.version", return_value=None), patch(
            "scripts.reproduce_deepfilternet3.importlib.import_module",
            return_value=SimpleNamespace(__version__="2.6.0+cpu"),
        ):
            self.assertEqual(installed_package_version("torch"), "2.6.0+cpu")


if __name__ == "__main__":
    unittest.main()

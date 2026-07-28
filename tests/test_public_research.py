import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.model_comparison import canonical_receipt_sha256
from scripts.verify_public_research import audit_public_research


class TestPublicResearchAudit(unittest.TestCase):
    def setUp(self) -> None:
        root = Path.cwd()
        self.research = json.loads((root / "reports/public/research_readiness.json").read_text(encoding="utf-8"))
        self.comparison = json.loads((root / "reports/public/model_comparison.json").read_text(encoding="utf-8"))
        self.deepfilter = json.loads((root / "reports/public/deepfilternet3_reproduction.json").read_text(encoding="utf-8"))
        self.registry = (root / "configs/research_baselines.json").read_bytes()
        self.rnnoise_script = (root / "scripts/setup-rnnoise-baseline.ps1").read_bytes()
        self.rnnoise_dockerfile = (root / "containers/rnnoise/Dockerfile").read_bytes()
        self.rnnoise_toolchain = (root / "configs/rnnoise_toolchain.json").read_bytes()
        self.rnnoise_build = json.loads((root / "reports/public/rnnoise_build.json").read_text(encoding="utf-8"))
        self.ci_workflow = (root / ".github/workflows/ci.yml").read_bytes()
        self.manifest = (root / "manifests/v2/fold_0_test.jsonl").read_bytes()

    def _audit(self, research=None, comparison=None, deepfilter=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = root / "reports/public"
            public.mkdir(parents=True)
            (root / "configs").mkdir()
            (root / "scripts").mkdir()
            (root / "manifests/v2").mkdir(parents=True)
            (root / "containers/rnnoise").mkdir(parents=True)
            (root / ".github/workflows").mkdir(parents=True)
            (public / "research_readiness.json").write_text(
                json.dumps(self.research if research is None else research), encoding="utf-8"
            )
            (public / "model_comparison.json").write_text(
                json.dumps(self.comparison if comparison is None else comparison), encoding="utf-8"
            )
            (public / "deepfilternet3_reproduction.json").write_text(
                json.dumps(self.deepfilter if deepfilter is None else deepfilter), encoding="utf-8"
            )
            (public / "rnnoise_build.json").write_text(json.dumps(self.rnnoise_build), encoding="utf-8")
            (root / "configs/research_baselines.json").write_bytes(self.registry)
            (root / "scripts/setup-rnnoise-baseline.ps1").write_bytes(self.rnnoise_script)
            (root / "containers/rnnoise/Dockerfile").write_bytes(self.rnnoise_dockerfile)
            (root / "configs/rnnoise_toolchain.json").write_bytes(self.rnnoise_toolchain)
            (root / "manifests/v2/fold_0_test.jsonl").write_bytes(self.manifest)
            (root / ".github/workflows/ci.yml").write_bytes(self.ci_workflow)
            return audit_public_research(root)

    def test_current_public_evidence_passes(self):
        result = self._audit()
        self.assertEqual(result["status"], "pass", result)
        self.assertTrue(all(result["checks"].values()), result)
        self.assertEqual(result["coverage"], {
            "external_reproduced": 1,
            "verified_blocker_recipes": 1,
            "eligible": True,
            "tier": "one_plus_recipe",
        })

    def test_rejects_tampered_embedded_receipt_hashes(self):
        comparison = copy.deepcopy(self.comparison)
        comparison["models"][0]["metrics"]["snri_db"]["value"] += 0.01
        deepfilter = copy.deepcopy(self.deepfilter)
        deepfilter["model"]["metrics"]["snri_db"]["value"] += 0.01
        result = self._audit(comparison=comparison, deepfilter=deepfilter)
        self.assertFalse(result["checks"]["comparison_receipt_hash"])
        self.assertFalse(result["checks"]["deepfilter_receipt_hash"])

    def test_rejects_seed_and_evaluation_matrix_drift(self):
        receipt = copy.deepcopy(self.research)
        receipt["evidence"]["research_training"]["seeds"] = [17, 29, 43, 59]
        receipt["evidence"]["research_evaluations"]["runs"][0]["evaluations"].pop("audit")
        result = self._audit(research=receipt)
        self.assertFalse(result["checks"]["training_matrix"])
        self.assertFalse(result["checks"]["evaluation_matrix"])

    def test_rejects_invalid_hash_overlap_and_nonfinite_significance(self):
        receipt = copy.deepcopy(self.research)
        receipt["source_sha256"]["significance.json"] = "not-a-sha"
        receipt["manifest"]["summary"]["speaker_overlap"]["train_test"] = 1
        receipt["evidence"]["significance"]["metrics"]["snr"]["bootstrap"]["ci95_low"] = float("nan")
        result = self._audit(research=receipt)
        self.assertFalse(result["checks"]["source_hashes"])
        self.assertFalse(result["checks"]["manifest_overlap"])
        self.assertFalse(result["checks"]["significance"])
        self.assertFalse(result["checks"]["evidence_hashes"])
        self.assertFalse(result["checks"]["research_receipt_hash"])

    def test_rejects_missing_selection_robustness_and_failure_evidence(self):
        receipt = copy.deepcopy(self.research)
        receipt["evidence"]["research_selection"]["ranked_validation_metrics"].pop()
        receipt["evidence"]["robustness_matrix"]["rows"][0]["count"] = 499
        receipt["evidence"]["failure_analysis"]["worst_cases"] = []
        receipt["evidence"]["failure_audio"]["items"] = []
        result = self._audit(research=receipt)
        self.assertFalse(result["checks"]["selection"])
        self.assertFalse(result["checks"]["robustness"])
        self.assertFalse(result["checks"]["failure_analysis"])
        self.assertFalse(result["checks"]["failure_audio"])

    def test_rejects_model_id_metric_and_provenance_drift(self):
        receipt = copy.deepcopy(self.comparison)
        receipt["models"][0]["item_ids"] = receipt["models"][0]["item_ids"][:-1]
        receipt["models"][1]["metrics"]["snri_db"]["value"] = float("inf")
        blocked = next(row for row in receipt["models"] if row["name"] == "DeepFilterNet3")
        blocked["metrics"]["snri_db"].update({"available": True, "value": 99.0})
        blocked["provenance"]["commit"] = None
        result = self._audit(comparison=receipt)
        self.assertFalse(result["checks"]["model_item_ids"])
        self.assertFalse(result["checks"]["reproduced_metrics"])
        self.assertFalse(result["checks"]["blocked_metrics"])
        self.assertFalse(result["checks"]["model_provenance"])

    def test_accepts_pinned_reproduced_external_with_remaining_blockers(self):
        external = next(row for row in self.comparison["models"] if row["name"] == "DeepFilterNet3")
        self.assertEqual(external["status"], "reproduced_local")
        self.assertEqual(external["provenance"]["weights"]["hash_status"], "verified")
        self.assertTrue(any(row["status"] == "blocked" for row in self.comparison["models"]))
        result = self._audit()
        self.assertEqual(result["status"], "pass", result)

    def test_rejects_cross_receipt_checkpoint_drift(self):
        receipt = copy.deepcopy(self.research)
        receipt["evidence"]["research_selection"]["source_checkpoint_sha256"] = "0" * 64
        receipt["evidence"]["research_selection"]["production_checkpoint_sha256"] = "0" * 64
        result = self._audit(research=receipt)
        self.assertFalse(result["checks"]["checkpoint_binding"])

    def test_comparison_status_is_recomputed_from_rows(self):
        receipt = copy.deepcopy(self.comparison)
        receipt["status"] = "pass"
        result = self._audit(comparison=receipt)
        self.assertFalse(result["checks"]["comparison_status"])

    def test_comparison_binds_registry_and_manifest_files(self):
        result = self._audit()
        self.assertTrue(result["checks"]["comparison_inputs"])
        inputs = self.comparison["inputs"]
        self.assertEqual(inputs["registry"]["path"], "configs/research_baselines.json")
        self.assertEqual(inputs["manifest"]["path"], "manifests/v2/fold_0_test.jsonl")

    def test_rejects_registry_or_manifest_file_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = root / "reports/public"
            public.mkdir(parents=True)
            (root / "configs").mkdir()
            (root / "scripts").mkdir()
            (root / "manifests/v2").mkdir(parents=True)
            (root / "containers/rnnoise").mkdir(parents=True)
            (root / ".github/workflows").mkdir(parents=True)
            (public / "research_readiness.json").write_text(json.dumps(self.research), encoding="utf-8")
            (public / "model_comparison.json").write_text(json.dumps(self.comparison), encoding="utf-8")
            (public / "deepfilternet3_reproduction.json").write_text(json.dumps(self.deepfilter), encoding="utf-8")
            (public / "rnnoise_build.json").write_text(json.dumps(self.rnnoise_build), encoding="utf-8")
            (root / "configs/research_baselines.json").write_bytes(self.registry + b"\n")
            (root / "scripts/setup-rnnoise-baseline.ps1").write_bytes(self.rnnoise_script)
            (root / "containers/rnnoise/Dockerfile").write_bytes(self.rnnoise_dockerfile)
            (root / "configs/rnnoise_toolchain.json").write_bytes(self.rnnoise_toolchain)
            (root / "manifests/v2/fold_0_test.jsonl").write_bytes(self.manifest + b"\n")
            (root / ".github/workflows/ci.yml").write_bytes(self.ci_workflow)
            result = audit_public_research(root)
            self.assertFalse(result["checks"]["comparison_inputs"])

    def test_rejects_forged_but_internally_consistent_item_ids(self):
        receipt = copy.deepcopy(self.comparison)
        forged = [f"forged-{index}" for index in range(500)]
        receipt["protocol"]["item_ids"] = forged
        receipt["inputs"]["manifest"]["item_ids_sha256"] = hashlib.sha256("\n".join(forged).encode()).hexdigest()
        for row in receipt["models"]:
            row["item_ids"] = forged
        receipt["receipt_sha256"] = canonical_receipt_sha256(receipt)
        result = self._audit(comparison=receipt)
        self.assertTrue(result["checks"]["model_item_ids"])
        self.assertTrue(result["checks"]["comparison_receipt_hash"])
        self.assertFalse(result["checks"]["comparison_inputs"])


if __name__ == "__main__":
    unittest.main()

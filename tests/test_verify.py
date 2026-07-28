from __future__ import annotations

import tempfile
import unittest
import json
import hashlib
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.verify import (
    _finite_or_inf,
    artifact_audit,
    export_parity_audit,
    gain_calibration_audit,
)
from src.checkpoint import file_sha256


class TestVerificationGate(unittest.TestCase):
    def test_export_parity_audit_checks_recurrent_hops(self) -> None:
        try:
            import onnxruntime  # noqa: F401
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch and onnxruntime are required")
        from src.export import export_model
        from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints/best.pth"
            checkpoint.parent.mkdir(parents=True)
            model = MobileDeepFilterNet(
                MobileDeepFilterNetConfig(enc_channels=2, num_encoder_blocks=1, gru_hidden=2, k_tap=1)
            ).eval()
            torch.save(
                {
                    "schema_version": 2,
                    "model": model.state_dict(),
                    "model_cfg": model.cfg.__dict__,
                    "audio_cfg": {"sr": 16000, "n_fft": 320, "hop": 160, "freq_bins": 161},
                    "config": {},
                },
                checkpoint,
            )
            export_model(model, root / "checkpoints/full_best_export", export_onnx=True)

            result = export_parity_audit(root, steps=3)

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["steps"], 3)
            self.assertLess(max(result["eager_vs_onnx_max_abs_error"]), result["tolerance"])

    def _write_full_artifacts(self, root: Path) -> None:
        import torch
        from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig

        binary_paths = [
            "checkpoints/full_best_export.ts",
            "checkpoints/full_best_export.onnx",
        ]
        for relative in binary_paths:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode("ascii"))
        checkpoint_dir = root / "checkpoints"
        model = MobileDeepFilterNet(
            MobileDeepFilterNetConfig(enc_channels=2, num_encoder_blocks=1, gru_hidden=2, k_tap=1)
        )
        checkpoint_base = {
            "schema_version": 2,
            "model": model.state_dict(),
            "model_cfg": model.cfg.__dict__,
            "audio_cfg": {"sr": 16000, "n_fft": 320, "hop": 160, "freq_bins": 161},
            "config": {"epochs": 150, "steps_per_epoch": 200},
        }
        torch.save({**checkpoint_base, "global_step": 17400}, checkpoint_dir / "best.pth")
        torch.save(
            {
                **checkpoint_base,
                "global_step": 30000,
                "runtime": {"device": "cuda", "gpu": "test-gpu", "torch_version": "test"},
            },
            checkpoint_dir / "step_30000.pth",
        )
        checkpoint_hash = file_sha256(root / "checkpoints/best.pth")
        torchscript_hash = file_sha256(root / "checkpoints/full_best_export.ts")
        onnx_hash = file_sha256(root / "checkpoints/full_best_export.onnx")
        json_artifacts = {
            "checkpoints/full_best_export.json": {
                "source_checkpoint_sha256": checkpoint_hash,
            },
            "reports/generated/full_best_evaluation.json": {
                "metadata": {
                    "checkpoint_sha256": checkpoint_hash,
                    "config": "configs/train.yaml",
                    "manifest_fingerprint": "test-fingerprint",
                    "max_items": 500,
                    "device": "cuda",
                },
                **{
                    name: {
                        "n": 500,
                        **{
                            metric: {"mean": 1.0, "std": 0.1}
                            for metric in ("si_sdr", "snr", "stoi", "pesq")
                        },
                    }
                    for name in ("noisy", "mask_only", "enhanced")
                },
            },
            "reports/generated/full_best_benchmark.json": {
                "checkpoint_sha256": checkpoint_hash,
                "device": "cuda",
                "gpu": "test-gpu",
                "iterations": 500,
                "neural_core": {
                    "n": 500,
                    "mean_ms": 1.0,
                    "p95_ms": 2.0,
                    "max_ms": 3.0,
                    "realtime_factor": 10.0,
                },
                "streaming_end_to_end": {
                    "n": 500,
                    "mean_ms": 2.0,
                    "p95_ms": 3.0,
                    "max_ms": 4.0,
                    "realtime_factor": 5.0,
                },
            },
            "reports/generated/full_export_parity.json": {
                "status": "pass",
                "checkpoint_sha256": checkpoint_hash,
                "torchscript_sha256": torchscript_hash,
                "onnx_sha256": onnx_hash,
            },
        }
        for relative, payload in json_artifacts.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        report = root / "reports/full_training_report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("evidence", encoding="utf-8")
        summary = root / "manifests/v2/fold_0_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"fingerprints": {"test": "test-fingerprint"}}), encoding="utf-8")

    def test_nonfinite_parity_error_becomes_infinite(self) -> None:
        self.assertTrue(math.isinf(_finite_or_inf(float("nan"))))

    def test_full_artifact_audit_requires_bound_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_full_artifacts(root)

            result = artifact_audit(root, require_full=True)

            self.assertEqual(result["status"], "pass")
            self.assertIn("full_export_parity", result["artifacts"])

    def test_full_artifact_audit_rejects_mismatched_export_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_full_artifacts(root)
            parity_path = root / "reports/generated/full_export_parity.json"
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            parity["onnx_sha256"] = "0" * 64
            parity_path.write_text(json.dumps(parity), encoding="utf-8")

            result = artifact_audit(root, require_full=True)

            self.assertEqual(result["status"], "fail")

    def test_full_artifact_audit_rejects_incomplete_training(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_full_artifacts(root)
            torch.save(
                {"global_step": 29999, "runtime": {"device": "cuda", "gpu": "test-gpu"}},
                root / "checkpoints/step_30000.pth",
            )

            result = artifact_audit(root, require_full=True)

            self.assertEqual(result["status"], "fail")

    def test_full_artifact_audit_rejects_nonfinite_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_full_artifacts(root)
            evaluation_path = root / "reports/generated/full_best_evaluation.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            evaluation["enhanced"]["pesq"]["mean"] = float("nan")
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

            result = artifact_audit(root, require_full=True)

            self.assertEqual(result["status"], "fail")

    def _write_gain_calibration_artifacts(self, root: Path) -> None:
        import torch

        from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig
        from src.splits import manifest_fingerprint, slice_fingerprint

        manifest = root / "manifests/v2/fold_0_test.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            "".join(json.dumps({"id": index}) + "\n" for index in range(1000)),
            encoding="utf-8",
        )
        validation_manifest = root / "manifests/v2/fold_0_val.jsonl"
        validation_manifest.write_text(
            "".join(json.dumps({"id": index}) + "\n" for index in range(500)),
            encoding="utf-8",
        )
        manifest_hash = manifest_fingerprint(manifest)
        validation_manifest_hash = manifest_fingerprint(validation_manifest)
        validation_slice_hash = slice_fingerprint(validation_manifest, 0, 500)
        slice_hashes = {
            "comparison": slice_fingerprint(manifest, 0, 500),
            "audit": slice_fingerprint(manifest, 500, 500),
        }
        (root / "manifests/v2/fold_0_summary.json").write_text(
            json.dumps({"fingerprints": {"val": validation_manifest_hash, "test": manifest_hash}}),
            encoding="utf-8",
        )

        model = MobileDeepFilterNet(
            MobileDeepFilterNetConfig(enc_channels=2, num_encoder_blocks=1, gru_hidden=2, k_tap=1)
        )
        arm = "complex_nmse"
        generated = root / "reports/generated"
        generated.mkdir(parents=True)
        screen_path = generated / "gain_calibration_screen.json"
        screen_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "screen",
                    "status": "completed",
                    "promoted_arm": arm,
                    "runs": [
                        {
                            "arm": screen_arm,
                            "seed": seed,
                            "status": "completed",
                            "validation_metrics": {
                                "snri": 1.0 if screen_arm == "control" else (2.2 if screen_arm == arm else 1.2),
                                "si_sdri": 2.0 if screen_arm == "control" else (1.8 if screen_arm == arm else 1.0),
                                "stoi": 0.8 if screen_arm == "control" else (0.798 if screen_arm == arm else 0.79),
                                "projection_gain_median": 0.5 if screen_arm == "control" else (0.8 if screen_arm == arm else -0.1),
                                "gain_error_db_median": -6.0 if screen_arm == "control" else -1.0,
                            },
                        }
                        for screen_arm in ("control", "complex_nmse", "complex_nmse_sisdr", "compressed_complex")
                        for seed in (17, 29, 43)
                    ],
                }
            ),
            encoding="utf-8",
        )
        runs = []
        checkpoint_by_seed = {}
        for seed in (17, 29, 43):
            run_dir = root / f"checkpoints/gain_calibration/full/{arm}/seed_{seed}"
            run_dir.mkdir(parents=True)
            resolved = {
                "loss": arm,
                "alpha_loss": 1.0,
                "beta_si_sdr": 0.0,
                "loss_eps": 1e-8,
                "sisdr_warmup_start": 0,
                "sisdr_warmup_end": 0,
                "compression_exponent": 0.3,
                "compression_complex_weight": 0.3,
                "seed": seed,
                "max_steps": 30000,
                "scheduler_total_steps": 30000,
                "checkpoint_dir": str(run_dir),
                "experiment_id": f"gain_calibration_full_{arm}_seed_{seed}",
            }
            resolved_path = run_dir / "resolved_config.json"
            resolved_path.write_text(json.dumps(resolved, sort_keys=True), encoding="utf-8")
            config_hash = file_sha256(resolved_path)
            resolved_yaml = run_dir / "resolved_config.yaml"
            resolved_yaml.write_text(
                "".join(f"{key}: {json.dumps(value)}\n" for key, value in resolved.items()),
                encoding="utf-8",
            )
            resolved_yaml_hash = file_sha256(resolved_yaml)
            checkpoint = run_dir / "best.pth"
            torch.save(
                {
                    "schema_version": 2,
                    "model": model.state_dict(),
                    "model_cfg": model.cfg.__dict__,
                    "audio_cfg": {"sr": 16000, "n_fft": 320, "hop": 160, "freq_bins": 161},
                    "config": resolved,
                    "global_step": 30000,
                    "runtime": {"device": "cuda", "gpu": "test-gpu"},
                    "experiment_id": f"gain_calibration_full_{arm}_seed_{seed}",
                    "loss": arm,
                    "loss_config": {
                        key: resolved[key]
                        for key in (
                            "loss",
                            "alpha_loss",
                            "beta_si_sdr",
                            "loss_eps",
                            "sisdr_warmup_start",
                            "sisdr_warmup_end",
                            "compression_exponent",
                            "compression_complex_weight",
                        )
                    },
                    "manifest_fingerprints": {"val": validation_manifest_hash, "test": manifest_hash},
                },
                checkpoint,
            )
            checkpoint_hash = file_sha256(checkpoint)
            checkpoint_by_seed[seed] = checkpoint
            evaluations = {}
            for slice_name, offset in (("comparison", 0), ("audit", 500)):
                path = generated / f"gain_calibration_{arm}_seed_{seed}_{slice_name}.json"
                self._write_gain_evaluation(
                    path,
                    checkpoint_hash=checkpoint_hash,
                    config_hash=config_hash,
                    manifest_hash=manifest_hash,
                    slice_hash=slice_hashes[slice_name],
                    offset=offset,
                    candidate=True,
                )
                evaluations[slice_name] = {
                    "path": str(path.relative_to(root)),
                    "sha256": file_sha256(path),
                }
            validation_path = run_dir / "validation_evaluation.json"
            validation_path.write_bytes(
                (generated / f"gain_calibration_{arm}_seed_{seed}_comparison.json").read_bytes()
            )
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["enhanced"]["snr_improvement_mean"] = 3.0 - seed / 1000.0
            validation["metadata"].update(
                {
                    "manifest": str(validation_manifest),
                    "manifest_fingerprint": validation_manifest_hash,
                    "slice_offset": 0,
                    "slice_count": 500,
                    "slice_fingerprint": validation_slice_hash,
                    "max_items": 500,
                }
            )
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            run_summary = {
                "status": "completed",
                "arm": arm,
                "seed": seed,
                "global_step": 30000,
                "experiment_id": f"gain_calibration_full_{arm}_seed_{seed}",
                "max_steps": 30000,
                "scheduler_total_steps": 30000,
                "config_sha256": config_hash,
                "resolved_yaml_sha256": resolved_yaml_hash,
                "manifest_fingerprints": {"val": validation_manifest_hash, "test": manifest_hash},
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "validation_evaluation_sha256": file_sha256(validation_path),
                "validation_metrics": {
                    "snri": 3.0 - seed / 1000.0,
                    "si_sdri": 4.2,
                    "stoi": 0.8,
                    "projection_gain_median": 0.9,
                    "gain_error_db_median": -0.9,
                },
            }
            run_summary_path = run_dir / "run_summary.json"
            run_summary_path.write_text(json.dumps(run_summary), encoding="utf-8")
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "status": "completed",
                    "run_dir": str(run_dir.relative_to(root)),
                    "run_summary": {"path": str(run_summary_path.relative_to(root)), "sha256": file_sha256(run_summary_path)},
                    "resolved_config": {"path": str(resolved_path.relative_to(root)), "sha256": config_hash},
                    "resolved_yaml": {"path": str(resolved_yaml.relative_to(root)), "sha256": resolved_yaml_hash},
                    "checkpoint": {"path": str(checkpoint.relative_to(root)), "sha256": checkpoint_hash},
                    "validation_evaluation": {"path": str(validation_path.relative_to(root)), "sha256": file_sha256(validation_path)},
                    "validation_metrics": dict(run_summary["validation_metrics"]),
                    "evaluations": evaluations,
                    "acceptance": {
                        slice_name: {
                            "status": "pass",
                            "checks": {
                                "snri_positive": True,
                                "si_sdri_delta": True,
                                "stoi_delta": True,
                                "pesq_delta": True,
                                "positive_gain": True,
                                "gain_error": True,
                                "polarity": True,
                            },
                        }
                        for slice_name in ("comparison", "audit")
                    },
                }
            )

        baseline_checkpoint = root / "checkpoints/best.pth"
        baseline_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        baseline_checkpoint.write_bytes(checkpoint_by_seed[17].read_bytes())
        baseline_hash = file_sha256(baseline_checkpoint)
        baseline_evaluations = {}
        for slice_name, offset in (("comparison", 0), ("audit", 500)):
            path = generated / f"gain_calibration_baseline_{slice_name}.json"
            self._write_gain_evaluation(
                path,
                checkpoint_hash=baseline_hash,
                config_hash=None,
                manifest_hash=manifest_hash,
                slice_hash=slice_hashes[slice_name],
                offset=offset,
                candidate=False,
            )
            baseline_evaluations[slice_name] = {
                "path": str(path.relative_to(root)),
                "sha256": file_sha256(path),
            }

        selected = root / "checkpoints/gain_calibration/best.pth"
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes(checkpoint_by_seed[17].read_bytes())
        selected_hash = file_sha256(selected)
        selection_path = generated / "gain_calibration_selection.json"
        selection_payload = {
            "schema_version": 1,
            "selection_basis": "validation",
            "selected_arm": arm,
            "selected_seed": 17,
            "metric_order": ["validation_snri", "validation_si_sdri", "absolute_gain_error"],
            "ranked_validation_metrics": [
                {
                    "seed": seed,
                    "snri": 3.0 - seed / 1000.0,
                    "si_sdri": 4.2,
                    "stoi": 0.8,
                    "projection_gain_median": 0.9,
                    "gain_error_db_median": -0.9,
                }
                for seed in (17, 29, 43)
            ],
            "validation_evaluations": {
                str(row["seed"]): row["validation_evaluation"] for row in runs
            },
        }
        selection_payload["selection_contract_sha256"] = hashlib.sha256(
            (json.dumps(selection_payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        ).hexdigest()
        selection_path.write_text(json.dumps(selection_payload, sort_keys=True), encoding="utf-8")
        (generated / "gain_calibration_full.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "full",
                    "status": "completed",
                    "selection_basis": "validation",
                    "production_eligible": True,
                    "selected_arm": arm,
                    "selected_seed": 17,
                    "selected_checkpoint_path": selected.relative_to(root).as_posix(),
                    "selected_checkpoint_sha256": selected_hash,
                    "selected_checkpoint": {
                        "path": selected.relative_to(root).as_posix(),
                        "sha256": selected_hash,
                        "source_path": str(checkpoint_by_seed[17].relative_to(root)),
                        "source_sha256": file_sha256(checkpoint_by_seed[17]),
                        "production_eligible": True,
                    },
                    "screen_report": {
                        "path": str(screen_path.relative_to(root)),
                        "sha256": file_sha256(screen_path),
                        "promoted_arm": arm,
                    },
                    "selection_rationale": {
                        "selected_seed": 17,
                        "metric_order": ["validation_snri", "validation_si_sdri", "absolute_gain_error"],
                        "eligible_seeds": [17, 29, 43],
                        "ranked_validation_metrics": [
                            {
                                "seed": seed,
                                "snri": 3.0 - seed / 1000.0,
                                "si_sdri": 4.2,
                                "stoi": 0.8,
                                "projection_gain_median": 0.9,
                                "gain_error_db_median": -0.9,
                            }
                            for seed in (17, 29, 43)
                        ],
                    },
                    "selection_receipt": {
                        "path": selection_path.relative_to(root).as_posix(),
                        "sha256": file_sha256(selection_path),
                    },
                    "runs": runs,
                    "baseline_evaluations": baseline_evaluations,
                }
            ),
            encoding="utf-8",
        )

        torchscript = root / "checkpoints/gain_calibration/export.ts"
        onnx = root / "checkpoints/gain_calibration/export.onnx"
        torchscript.write_bytes(b"torchscript")
        onnx.write_bytes(b"onnx")
        (root / "checkpoints/gain_calibration/export.json").write_text(
            json.dumps({"source_checkpoint_sha256": selected_hash}), encoding="utf-8"
        )
        (generated / "gain_calibration_export_parity.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "steps": 20,
                    "checkpoint_sha256": selected_hash,
                    "torchscript_sha256": file_sha256(torchscript),
                    "onnx_sha256": file_sha256(onnx),
                }
            ),
            encoding="utf-8",
        )
        latency = {
            "n": 500,
            "mean_ms": 2.0,
            "p95_ms": 3.0,
            "max_ms": 4.0,
            "realtime_factor": 5.0,
        }
        (generated / "gain_calibration_benchmark.json").write_text(
            json.dumps(
                {
                    "checkpoint_sha256": selected_hash,
                    "device": "cuda",
                    "gpu": "test-gpu",
                    "iterations": 500,
                    "neural_core": latency,
                    "streaming_end_to_end": latency,
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_gain_evaluation(
        path: Path,
        *,
        checkpoint_hash: str,
        config_hash: str | None,
        manifest_hash: str,
        slice_hash: str,
        offset: int,
        candidate: bool,
    ) -> None:
        metadata = {
            "checkpoint_sha256": checkpoint_hash,
            "manifest_fingerprint": manifest_hash,
            "slice_fingerprint": slice_hash,
            "slice_offset": offset,
            "slice_count": 500,
            "max_items": 500,
            "device": "cuda",
        }
        if config_hash is not None:
            metadata["config_sha256"] = config_hash
        noisy = {
            "n": 500,
            "si_sdr": {"mean": 0.0, "std": 0.1},
            "snr": {"mean": 0.0, "std": 0.1},
            "stoi": {"mean": 0.7, "std": 0.1},
            "pesq": {"mean": 1.2, "std": 0.1},
        }
        enhanced = {
            "n": 500,
            "si_sdr": {"mean": 4.2 if candidate else 4.1, "std": 0.1},
            "snr": {"mean": 2.0 if candidate else 0.5, "std": 0.1},
            "stoi": {"mean": 0.80 if candidate else 0.801, "std": 0.1},
            "pesq": {"mean": 1.60 if candidate else 1.61, "std": 0.1},
            "projection_gain": {"mean": 0.9, "std": 0.1, "median": 0.9, "min": 0.7, "max": 1.1},
            "gain_error_db": {"mean": -0.9, "std": 0.1, "median": -0.9, "min": -1.1, "max": -0.7},
            "polarity_failure": {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
            "si_sdr_improvement_mean": 4.2 if candidate else 4.1,
            "snr_improvement_mean": 2.0 if candidate else 0.5,
        }
        path.write_text(
            json.dumps({"metadata": metadata, "noisy": noisy, "mask_only": enhanced, "enhanced": enhanced}),
            encoding="utf-8",
        )

    def test_gain_calibration_audit_checks_full_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)

            scripted = MagicMock()
            scripted.eval.return_value = scripted
            with (
                patch("torch.jit.load", return_value=scripted) as load_script,
                patch("onnxruntime.InferenceSession") as load_onnx,
                patch("scripts.verify.export_parity_audit", return_value={"status": "pass", "steps": 20}),
            ):
                result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "pass", result)
            self.assertTrue(all(result["checks"].values()), result)
            load_script.assert_called_once()
            load_onnx.assert_called_once()

    def test_gain_calibration_audit_rejects_slice_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            path = root / "reports/generated/gain_calibration_complex_nmse_seed_17_audit.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"]["slice_offset"] = 499
            path.write_text(json.dumps(payload), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["evaluation_slices"])

    def test_gain_calibration_audit_rejects_validation_manifest_binding_drift(self) -> None:
        mutations = {
            "manifest": ("manifest_fingerprint", "0" * 64),
            "offset": ("slice_offset", 1),
            "count": ("slice_count", 499),
            "slice": ("slice_fingerprint", "f" * 64),
        }
        for name, (key, value) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_gain_calibration_artifacts(root)
                validation_path = (
                    root / "checkpoints/gain_calibration/full/complex_nmse/seed_17/validation_evaluation.json"
                )
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
                validation["metadata"][key] = value
                validation_path.write_text(json.dumps(validation), encoding="utf-8")
                summary_path = validation_path.with_name("run_summary.json")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["validation_evaluation_sha256"] = file_sha256(validation_path)
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                full_path = root / "reports/generated/gain_calibration_full.json"
                full = json.loads(full_path.read_text(encoding="utf-8"))
                full["runs"][0]["validation_evaluation"]["sha256"] = file_sha256(validation_path)
                full["runs"][0]["run_summary"]["sha256"] = file_sha256(summary_path)
                full_path.write_text(json.dumps(full), encoding="utf-8")

                result = gain_calibration_audit(root)

                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["checks"]["validation_manifest_binding"])

    def test_gain_calibration_audit_rejects_unbound_baseline_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            path = root / "reports/generated/gain_calibration_baseline_comparison.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"]["checkpoint_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["baseline_checkpoint_binding"])

    def test_gain_calibration_audit_rejects_screen_promotion_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            screen = root / "reports/generated/gain_calibration_screen.json"
            payload = json.loads(screen.read_text(encoding="utf-8"))
            payload["promoted_arm"] = "compressed_complex"
            screen.write_text(json.dumps(payload), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["screen_promotion_binding"])

    def test_gain_calibration_audit_recomputes_screen_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            screen_path = root / "reports/generated/gain_calibration_screen.json"
            screen = json.loads(screen_path.read_text(encoding="utf-8"))
            for row in screen["runs"]:
                if row["arm"] == "complex_nmse":
                    row["validation_metrics"]["snri"] = 1.1
            screen_path.write_text(json.dumps(screen), encoding="utf-8")
            full_path = root / "reports/generated/gain_calibration_full.json"
            full = json.loads(full_path.read_text(encoding="utf-8"))
            full["screen_report"]["sha256"] = file_sha256(screen_path)
            full_path.write_text(json.dumps(full), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["screen_promotion_binding"])

    def test_gain_calibration_audit_requires_exact_loss_provenance_keys(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            checkpoint = root / "checkpoints/gain_calibration/full/complex_nmse/seed_17/best.pth"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state["loss_config"]["unreviewed_default"] = 1.0
            torch.save(state, checkpoint)
            summary = root / "checkpoints/gain_calibration/full/complex_nmse/seed_17/run_summary.json"
            row = json.loads(summary.read_text(encoding="utf-8"))
            row["checkpoint_sha256"] = file_sha256(checkpoint)
            summary.write_text(json.dumps(row), encoding="utf-8")
            full_path = root / "reports/generated/gain_calibration_full.json"
            full = json.loads(full_path.read_text(encoding="utf-8"))
            full["runs"][0]["checkpoint"]["sha256"] = file_sha256(checkpoint)
            full_path.write_text(json.dumps(full), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["loss_provenance"])

    def test_gain_calibration_audit_rejects_selected_source_binding_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            full_path = root / "reports/generated/gain_calibration_full.json"
            full = json.loads(full_path.read_text(encoding="utf-8"))
            full["selected_checkpoint"]["source_sha256"] = "0" * 64
            full_path.write_text(json.dumps(full), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["selected_run_binding"])

    def test_gain_calibration_audit_rejects_invalid_export_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["export_structure"])

    def test_all_finite_numeric_rejects_non_numeric_leaves(self) -> None:
        from scripts.verify import _all_finite_numeric

        self.assertTrue(_all_finite_numeric({"mean": 1.0, "values": [0, 2.0]}))
        for invalid in (True, "1.0", None, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                self.assertFalse(_all_finite_numeric({"value": invalid}))

    def test_gain_calibration_audit_recomputes_validation_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            validation_path = root / "checkpoints/gain_calibration/full/complex_nmse/seed_17/validation_evaluation.json"
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["enhanced"]["snr_improvement_mean"] = 0.1
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            summary_path = validation_path.with_name("run_summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["validation_evaluation_sha256"] = file_sha256(validation_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            full_path = root / "reports/generated/gain_calibration_full.json"
            full = json.loads(full_path.read_text(encoding="utf-8"))
            full["runs"][0]["validation_evaluation"]["sha256"] = file_sha256(validation_path)
            full["runs"][0]["run_summary"]["sha256"] = file_sha256(summary_path)
            full_path.write_text(json.dumps(full), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["validation_recomputation"])

    def test_gain_calibration_audit_parses_and_binds_resolved_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            yaml_path = root / "checkpoints/gain_calibration/full/complex_nmse/seed_17/resolved_config.yaml"
            text = yaml_path.read_text(encoding="utf-8").replace("seed: 17", "seed: 29")
            yaml_path.write_text(text, encoding="utf-8")
            summary_path = yaml_path.with_name("run_summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["resolved_yaml_sha256"] = file_sha256(yaml_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            full_path = root / "reports/generated/gain_calibration_full.json"
            full = json.loads(full_path.read_text(encoding="utf-8"))
            full["runs"][0]["resolved_yaml"]["sha256"] = file_sha256(yaml_path)
            full["runs"][0]["run_summary"]["sha256"] = file_sha256(summary_path)
            full_path.write_text(json.dumps(full), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["resolved_yaml_binding"])

    def test_gain_calibration_audit_accepts_equivalent_checkpoint_path_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            yaml_path = root / "checkpoints/gain_calibration/full/complex_nmse/seed_17/resolved_config.yaml"
            yaml_text = yaml_path.read_text(encoding="utf-8")
            checkpoint_line = next(line for line in yaml_text.splitlines() if line.startswith("checkpoint_dir:"))
            yaml_path.write_text(
                yaml_text.replace(checkpoint_line, r'checkpoint_dir: "C:\\Users\\RUNNER~1\\AppData\\Local\\Temp\\equivalent-run"'),
                encoding="utf-8",
            )
            summary_path = yaml_path.with_name("run_summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["resolved_yaml_sha256"] = file_sha256(yaml_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            full_path = root / "reports/generated/gain_calibration_full.json"
            full = json.loads(full_path.read_text(encoding="utf-8"))
            full["runs"][0]["resolved_yaml"]["sha256"] = file_sha256(yaml_path)
            full["runs"][0]["run_summary"]["sha256"] = file_sha256(summary_path)
            full_path.write_text(json.dumps(full), encoding="utf-8")

            scripted = MagicMock()
            scripted.eval.return_value = scripted
            with (
                patch("pathlib.Path.samefile", return_value=True),
                patch("torch.jit.load", return_value=scripted),
                patch("onnxruntime.InferenceSession"),
                patch("scripts.verify.export_parity_audit", return_value={"status": "pass", "steps": 20}),
            ):
                result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "pass", result)
            self.assertTrue(result["checks"]["resolved_yaml_binding"])

    def test_gain_calibration_audit_recomputes_recorded_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            full_path = root / "reports/generated/gain_calibration_full.json"
            full = json.loads(full_path.read_text(encoding="utf-8"))
            full["runs"][0]["acceptance"]["comparison"]["checks"]["snri_positive"] = False
            full_path.write_text(json.dumps(full), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["recorded_acceptance"])

    def test_gain_calibration_audit_requires_frozen_selected_seed_to_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            evaluation_path = root / "reports/generated/gain_calibration_complex_nmse_seed_17_comparison.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            evaluation["enhanced"]["snr_improvement_mean"] = -0.1
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
            full_path = root / "reports/generated/gain_calibration_full.json"
            full = json.loads(full_path.read_text(encoding="utf-8"))
            full["runs"][0]["evaluations"]["comparison"]["sha256"] = file_sha256(evaluation_path)
            full_path.write_text(json.dumps(full), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["selected_acceptance"])

    def test_gain_calibration_audit_requires_nonselected_seed_to_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_gain_calibration_artifacts(root)
            evaluation_path = root / "reports/generated/gain_calibration_complex_nmse_seed_29_comparison.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            evaluation["enhanced"]["snr_improvement_mean"] = -0.1
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
            full_path = root / "reports/generated/gain_calibration_full.json"
            full = json.loads(full_path.read_text(encoding="utf-8"))
            full["runs"][1]["evaluations"]["comparison"]["sha256"] = file_sha256(evaluation_path)
            full_path.write_text(json.dumps(full), encoding="utf-8")

            result = gain_calibration_audit(root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["checks"]["all_seed_acceptance"])


if __name__ == "__main__":
    unittest.main()
